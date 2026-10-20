#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
s2_apply_params_to_ssdna.py - 步骤2：将步骤1（DiI 图 QC + GEM 配准）确定的调整参数应用到 ssDNA 图像

用途
----
读取步骤1产出的调整参数（<chip>.adjust_params.json，或直接从 <chip>.ipr 的 Register 组读取），
对 ssDNA 图像执行与 DiI 图完全相同的几何变换序列，使 ssDNA 图与 DiI 图落在同一坐标系
（即与 GEM 矩阵对齐），供步骤3（cellbin2 圈细胞）直接使用。

变换序列（与 CellBin2 1.1.0 标准 10X 流程严格一致）
----------------------------------------------------
CellBin2 中 DiI 图经历的变换分为两个模块：
  1) run_transform        : scale=(1/ScaleX, 1/ScaleY)（注意取倒数）, rotate=+Rotation
  2) transform_to_register: flip_lr=Flip -> rot90=CounterRot90 -> offset=(OffsetX, OffsetY)
                            -> 输出画布 dst_size=MatrixShape
本脚本按同样的顺序对 ssDNA 应用：
  缩放+旋转（一步仿射） -> 水平翻转(Flip) -> 逆时针90°旋转(CounterRot90) -> 平移(Offset) -> 画布 dst_size

参数来源（按优先级）
--------------------
1) 显式传入 --params <json>：直接使用该 JSON 的 auto_params（或平铺键）。
2) 步骤1输出目录中的 <chip>.adjust_params.json（新格式：flip_h/flip_v/rotation_deg/
   scale_x/scale_y/dx/dy/counter_rot90/method/_source，来自 ipr Register）。
3) 直接从 <chip>.ipr 的 Register 组读取（最权威，含 MatrixShape / dst_size）。
4) CellBin2 运行日志解析兜底（scale/rotation，无 GEM 配准时）。

用法
----
# 典型：参数自动解析（步骤1输出目录 + ssDNA 图）
python s2_apply_params_to_ssdna.py \\
    -c Y40612D413 \\
    -i /data/input/Files/LHY_tif/Y40612D413-ssDNA.tif \\
    -q /data/work/DII_ssDNA/results/dii_qc_regist \\
    -o /data/work/DII_ssDNA/results/ssdna_adjusted

# 显式指定参数（不依赖步骤1目录）
python s2_apply_params_to_ssdna.py -c Y40612D413 \\
    -i ssDNA.tif -o out/ \\
    --rotation 1.7533 --scale-x 1.00716 --scale-y 1.00747

输出
----
<out>/<chip>_ssDNA_adjusted.tif        调整后的 ssDNA 图（步骤3输入）
<out>/<chip>.applied_params.json       实际应用的参数与变换记录
<out>/<chip>.ssdna_adjust.summary.txt  摘要
"""

import argparse
import json
import math
import os
import re
import sys
from glob import glob

import numpy as np

DEFAULT_FILL = 0          # 旋转/平移后边界填充值（0=黑）

# CellBin2 ipr 占位符（未配准/未检测时的默认值）
F_PLACEHOLDER = -999.999
I_PLACEHOLDER = -999


# ----------------------------------------------------------------------------
# 参数解析
# ----------------------------------------------------------------------------
def _is_placeholder(v, tol=1e-3):
    """判断是否为 CellBin2 占位符（-999.999 / -999）。"""
    try:
        return abs(float(v) - F_PLACEHOLDER) < tol or abs(float(v) - I_PLACEHOLDER) < tol
    except (TypeError, ValueError):
        return False


def parse_params_from_log(log_text):
    """
    从 CellBin2 运行日志文本中解析调整参数（无 GEM 配准时的兜底来源）。

    已知日志关键行（cellbin2 1.1.0 image_qc）：
      Template Scale is (1.0071593305798057, 1.0074742690737442),
      rotation is 1.7532599999999998
      ImageQC result is 1
    注意：这里的 ScaleX/ScaleY 是模板检测值，应用时需取倒数（与 run_transform 一致）。
    """
    params = {"flip_h": False, "flip_v": False,
              "rotation_deg": 0.0, "scale_x": 1.0, "scale_y": 1.0,
              "dx": 0.0, "dy": 0.0, "counter_rot90": 0, "source": "log"}

    m = re.search(
        r"Template\s+Scale\s+is\s*\(([0-9.eE+-]+),\s*([0-9.eE+-]+)\)\s*,\s*"
        r"rotation\s+is\s+([0-9.eE+-]+)",
        log_text, re.I)
    if m:
        params["scale_x"] = float(m.group(1))
        params["scale_y"] = float(m.group(2))
        params["rotation_deg"] = float(m.group(3))

    if re.search(r"\bflip\b|\bmirror\b|vertical\s*flip|horizontal\s*flip", log_text, re.I):
        if re.search(r"(?:flip|mirror)[^\n]{0,30}(?:x|h|horizontal)", log_text, re.I):
            params["flip_h"] = True
        if re.search(r"(?:flip|mirror)[^\n]{0,30}(?:y|v|vertical)", log_text, re.I):
            params["flip_v"] = True

    return params


def parse_register_from_ipr(output_dir, chip_id=None):
    """
    从 <chip>.ipr（HDF5）读取 ssDNA/Register 组下的配准结果（最权威来源）。
    返回 dict，键含：Flip/CounterRot90/OffsetX/OffsetY/Rotation/ScaleX/ScaleY/
                    Method/MatrixShape（dst_size）。
    """
    import h5py
    candidates = []
    if chip_id:
        candidates = sorted(glob(os.path.join(output_dir, f"{chip_id}.ipr")))
    if not candidates:
        candidates = sorted(glob(os.path.join(output_dir, "*.ipr")))
    for ipr_file in candidates:
        try:
            with h5py.File(ipr_file, "r") as fd:
                reg = None
                for ch in fd.keys():
                    if "Register" in fd[ch]:
                        reg = fd[ch]["Register"]
                        break
                if reg is None:
                    continue
                out = {}
                for key in ("Flip", "CounterRot90", "OffsetX", "OffsetY",
                            "Rotation", "ScaleX", "ScaleY", "Method"):
                    if key in reg.attrs:
                        v = reg.attrs[key]
                        if isinstance(v, bytes):
                            v = v.decode()
                        out[key] = v
                # MatrixShape（dst_size，配准后的目标画布）
                if "MatrixShape" in reg and reg["MatrixShape"].size >= 2:
                    out["MatrixShape"] = [float(reg["MatrixShape"][0]),
                                          float(reg["MatrixShape"][1])]
                if "MatrixShape" in reg.attrs:
                    v = reg.attrs["MatrixShape"]
                    if isinstance(v, bytes):
                        v = v.decode()
                    out["MatrixShape_attr"] = v
                out["_ipr_file"] = os.path.basename(ipr_file)
                # 占位符清洗
                flip = out.get("Flip")
                if flip in (None, -999, -999.0, -999.999, "-999", "False"):
                    out["Flip"] = False
                for k in ("CounterRot90", "OffsetX", "OffsetY", "Rotation",
                          "ScaleX", "ScaleY"):
                    if _is_placeholder(out.get(k)):
                        out[k] = 0.0 if k in ("OffsetX", "OffsetY", "Rotation") else (1.0 if k.startswith("Scale") else 0)
                return out
        except Exception:
            continue
    return None


def load_params(args, chip_id, qc_dir):
    """
    按优先级加载调整参数：
      显式 --params > adjust_params.json（新格式，来自 ipr Register）
      > 直接从 ipr 读取 > 日志解析兜底。
    返回 dict（键：flip_h/flip_v/rotation_deg/scale_x/scale_y/dx/dy/counter_rot90/
                method/dst_size/source）。
    注意：scale_x/scale_y 统一为模板检测值（应用时取倒数，见 apply_transform）。
    """
    # 1) 显式 JSON
    if args.params:
        with open(args.params, "r") as fh:
            j = json.load(fh)
        ap = j.get("auto_params", j)
        return _normalize_params(ap, "explicit_json")

    # 2) 步骤1输出目录 adjust_params.json（新格式）
    if qc_dir:
        candidates = sorted(glob(os.path.join(qc_dir, f"{chip_id}.adjust_params.json")))
        for fp in candidates:
            try:
                with open(fp, "r") as fh:
                    j = json.load(fh)
                ap = j.get("auto_params", j)
                p = _normalize_params(ap, "adjust_params_json")
                # adjust_params.json 通常不含 dst_size；从 ipr 补读 MatrixShape 作为输出画布
                if not p.get("dst_size"):
                    reg = parse_register_from_ipr(qc_dir, chip_id)
                    if reg and reg.get("MatrixShape"):
                        p["dst_size"] = [float(reg["MatrixShape"][0]),
                                         float(reg["MatrixShape"][1])]
                return p
            except Exception:
                continue

    # 3) 直接从 ipr Register 读取（最权威）
    if qc_dir:
        reg = parse_register_from_ipr(qc_dir, chip_id)
        if reg:
            p = _normalize_params({
                "flip_h": reg.get("Flip", False),
                "flip_v": False,
                "rotation_deg": reg.get("Rotation", 0.0),
                "scale_x": reg.get("ScaleX", 1.0),
                "scale_y": reg.get("ScaleY", 1.0),
                "dx": reg.get("OffsetX", 0.0),
                "dy": reg.get("OffsetY", 0.0),
                "counter_rot90": reg.get("CounterRot90", 0),
                "method": reg.get("Method", ""),
                "dst_size": reg.get("MatrixShape"),
                "_source": f"ipr_register:{reg.get('_ipr_file','')}",
            }, "ipr_register")
            return p

    # 4) 日志解析（无 GEM 配准时兜底）
    log_params = _parse_from_log_files(qc_dir, chip_id)
    if log_params:
        return log_params

    # 5) 全部失败：恒等变换并告警
    print("[WARN] 无法自动解析调整参数，将使用恒等变换（不调整）。", file=sys.stderr)
    return {"flip_h": False, "flip_v": False, "rotation_deg": 0.0,
            "scale_x": 1.0, "scale_y": 1.0, "dx": 0.0, "dy": 0.0,
            "counter_rot90": 0, "method": "", "dst_size": None,
            "source": "identity_fallback"}


def _normalize_params(ap, source):
    def _f(k, d):
        try:
            v = ap.get(k, d)
            return float(v) if not _is_placeholder(v) else d
        except (TypeError, ValueError):
            return d

    def _i(k, d):
        try:
            v = ap.get(k, d)
            return int(v) if not _is_placeholder(v) else d
        except (TypeError, ValueError):
            return d

    def _b(k, d):
        v = ap.get(k, d)
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes", "on")
        return bool(v)

    p = {
        "flip_h": _b("flip_h", False),
        "flip_v": _b("flip_v", False),
        "rotation_deg": _f("rotation_deg", _f("rotation", 0.0)),
        "scale_x": _f("scale_x", _f("scale", 1.0)),
        "scale_y": _f("scale_y", _f("scale", 1.0)),
        "dx": _f("dx", 0.0),
        "dy": _f("dy", 0.0),
        "counter_rot90": _i("counter_rot90", 0),
        "method": str(ap.get("method", "")),
        "dst_size": ap.get("dst_size") or ap.get("dst_shape") or ap.get("MatrixShape"),
        "source": source,
    }
    if p["dst_size"] is not None:
        try:
            p["dst_size"] = [float(p["dst_size"][0]), float(p["dst_size"][1])]
        except (TypeError, IndexError, ValueError):
            p["dst_size"] = None
    return p


def _parse_from_log_files(qc_dir, chip_id):
    """扫描 qc_dir（及一级子目录）中所有 .log/.txt 文本解析参数。"""
    if not qc_dir:
        return None
    files = []
    for ext in ("*.log", "*.txt", "*.out"):
        files.extend(glob(os.path.join(qc_dir, ext)))
    for sub in sorted(glob(os.path.join(qc_dir, "*"))):
        if os.path.isdir(sub):
            for ext in ("*.log", "*.txt", "*.out"):
                files.extend(glob(os.path.join(sub, ext)))
    for fp in files:
        try:
            with open(fp, "r", errors="ignore") as fh:
                text = fh.read()
        except Exception:
            continue
        p = parse_params_from_log(text)
        if p.get("scale_x") != 1.0 or p.get("rotation_deg") != 0.0 or \
           p.get("flip_h") or p.get("flip_v"):
            return p
    return None


# ----------------------------------------------------------------------------
# 图像变换（与 CellBin2 1.1.0 标准 10X 流程严格一致）
# ----------------------------------------------------------------------------
def apply_transform(img, params, fill=DEFAULT_FILL, prefer_cellbin2=True):
    """
    对图像执行与 CellBin2 完全一致的变换序列：
        1) 缩放+旋转（一步仿射，scale=(1/ScaleX, 1/ScaleY)，rotate=+Rotation）
        2) 水平翻转（flip_h，对应 Register.Flip）
        3) 逆时针90°旋转（counter_rot90，对应 Register.CounterRot90）
        4) 平移（dx/dy，对应 Register.OffsetX/Y）+ 输出画布 dst_size
    若环境有 cellbin2 且 prefer_cellbin2=True，直接调用 cellbin2 的 trans_image，
    保证与 DiI 图在 CellBin2 内的变换逐像素一致；否则用 OpenCV 等价实现。

    img: numpy 2D/3D 数组（灰度或 RGB）。
    params: dict（flip_h/flip_v/rotation_deg/scale_x/scale_y/dx/dy/counter_rot90/dst_size）。
    返回变换后的 numpy 数组。
    """
    scale_x = float(params.get("scale_x", 1.0))
    scale_y = float(params.get("scale_y", 1.0))
    rot = float(params.get("rotation_deg", 0.0))
    flip_h = bool(params.get("flip_h", False))
    flip_v = bool(params.get("flip_v", False))
    rot90 = int(params.get("counter_rot90", 0) or 0) % 4
    dx = float(params.get("dx", 0.0))
    dy = float(params.get("dy", 0.0))
    dst_size = params.get("dst_size")

    # -- 0) 优先使用 cellbin2 原生 trans_image ------------------------------
    if prefer_cellbin2:
        try:
            from cellbin2.image import cbimread, cbimwrite, CBImage
            cimg = CBImage(img) if isinstance(img, np.ndarray) else None
            if cimg is not None:
                # 模块1：run_transform 等价（scale 取倒数）
                cimg = cimg.trans_image(
                    scale=[1.0 / scale_x if scale_x else 1.0,
                           1.0 / scale_y if scale_y else 1.0],
                    rotate=rot,
                )
                # 模块2：transform_to_register 等价（flip -> rot90 -> offset -> dst_size）
                cimg = cimg.trans_image(
                    flip_lr=flip_h,
                    flip_ud=flip_v,
                    rot90=rot90,
                    offset=(dx, dy),
                    dst_size=tuple(int(v) for v in dst_size) if dst_size else None,
                )
                # CBImage 用 .image property 暴露 numpy 数组（无 get_image()/__array__ 协议）
                if isinstance(cimg, np.ndarray):
                    return cimg
                arr = cimg.image if hasattr(cimg, "image") else None
                if arr is not None and getattr(arr, "size", 0) > 0:
                    return np.asarray(arr)
                raise RuntimeError("cellbin2 trans_image 返回了空图像")
        except Exception as e:
            print(f"[s2] cellbin2 trans_image 不可用，改用 OpenCV 实现 ({e})", file=sys.stderr)

    # -- 1) OpenCV 等价实现 ------------------------------------------------
    import cv2
    out = img.copy()

    # 1.1) 缩放+旋转（一步仿射，绕中心；scale 取倒数，rotate 为正 Rotation）
    sx = 1.0 / scale_x if scale_x else 1.0
    sy = 1.0 / scale_y if scale_y else 1.0
    if abs(sx - 1.0) > 1e-9 or abs(sy - 1.0) > 1e-9 or abs(rot) > 1e-9:
        h, w = out.shape[:2]
        theta = math.radians(rot)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        # 输出画布：旋转后包围盒再缩放（与 cellbin2 __affine_transform 一致）
        new_w = int((abs(w * cos_t) + abs(h * sin_t)) * sx)
        new_h = int((abs(w * sin_t) + abs(h * cos_t)) * sy)
        new_w = max(1, new_w)
        new_h = max(1, new_h)
        # 仿射矩阵：M = R(θ)·S，映射原图中心到新画布中心
        M = np.float32([
            [sx * cos_t, -sy * sin_t],
            [sx * sin_t, sy * cos_t],
        ])
        # 目标中心相对源中心的偏移
        cx_src, cy_src = w / 2.0, h / 2.0
        cx_dst, cy_dst = new_w / 2.0, new_h / 2.0
        # 线性映射：先缩放到新画布中心，再应用仿射（原点为中心）
        M_full = np.zeros((2, 3), dtype=np.float32)
        M_full[0, 0] = M[0, 0]
        M_full[0, 1] = M[0, 1]
        M_full[1, 0] = M[1, 0]
        M_full[1, 1] = M[1, 1]
        # 平移项 = 目标中心 - M·源中心
        M_full[0, 2] = cx_dst - (M[0, 0] * cx_src + M[0, 1] * cy_src)
        M_full[1, 2] = cy_dst - (M[1, 0] * cx_src + M[1, 1] * cy_src)
        out = cv2.warpAffine(out, M_full, (new_w, new_h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=fill)

    # 1.2) 翻转（先 flip 再 rot90/offset，与 transform_to_register 一致）
    if flip_h:
        out = cv2.flip(out, 1)
    if flip_v:
        out = cv2.flip(out, 0)

    # 1.3) rot90（逆时针90°次数）
    if rot90 == 1:
        out = cv2.rotate(out, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rot90 == 2:
        out = cv2.rotate(out, cv2.ROTATE_180)
    elif rot90 == 3:
        out = cv2.rotate(out, cv2.ROTATE_90_CLOCKWISE)

    # 1.4) 平移 + 画布 dst_size
    if dst_size:
        th, tw = int(round(dst_size[0])), int(round(dst_size[1]))
        out = _shift_to_canvas(out, dx, dy, tw, th, fill)
    elif abs(dx) > 1e-9 or abs(dy) > 1e-9:
        M_tr = np.float32([[1, 0, dx], [0, 1, dy]])
        out = cv2.warpAffine(out, M_tr, (out.shape[1], out.shape[0]),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=fill)

    return out


def _shift_to_canvas(img, dx, dy, dst_w, dst_h, fill=0):
    """
    将图像平移到 (dx, dy) 并输出到指定画布 (dst_w, dst_h)（与 cellbin2 offset 语义一致：
    画布固定为 dst_size，图像内容平移到 offset 位置，越界部分裁掉/填充）。
    """
    import cv2
    h, w = img.shape[:2]
    canvas_shape = (dst_h, dst_w) + img.shape[2:]
    canvas = np.full(canvas_shape, fill, dtype=img.dtype)
    # 源图像在画布中的位置：偏移 (dx, dy)
    x0 = int(round(dx))
    y0 = int(round(dy))
    # 计算重叠区域
    src_x0, src_y0 = max(0, -x0), max(0, -y0)
    dst_x0, dst_y0 = max(0, x0), max(0, y0)
    w_copy = min(w - src_x0, dst_w - dst_x0)
    h_copy = min(h - src_y0, dst_h - dst_y0)
    if w_copy > 0 and h_copy > 0:
        if img.ndim == 2:
            canvas[dst_y0:dst_y0 + h_copy, dst_x0:dst_x0 + w_copy] = \
                img[src_y0:src_y0 + h_copy, src_x0:src_x0 + w_copy]
        else:
            canvas[dst_y0:dst_y0 + h_copy, dst_x0:dst_x0 + w_copy, :] = \
                img[src_y0:src_y0 + h_copy, src_x0:src_x0 + w_copy, :]
    return canvas


def read_image(path):
    """读取图像为 numpy 数组（保持通道）。"""
    import cv2
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"无法读取图像: {path}")
    return img


def write_image(path, img):
    """保存图像（tif 多页/多通道保持）。"""
    import cv2
    ext = os.path.splitext(path)[1] or ".tif"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise ValueError(f"图像编码失败: {path}")
    with open(path, "wb") as fh:
        fh.write(buf.tobytes())


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="步骤2：将 DiI 的调整参数（step1 QC+GEM 配准结果）原样应用到 ssDNA 图像",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-c", "--chip-id", required=True, help="芯片号 (e.g. Y40612D413)")
    parser.add_argument("-i", "--ssdna", required=True, help="ssDNA 染色图像路径 (.tif)")
    parser.add_argument("-q", "--qc-dir", default=None,
                        help="步骤1输出目录（含 adjust_params.json 与 *.ipr）；"
                             "省略时需用 --params 显式提供参数")
    parser.add_argument("-o", "--output-dir", required=True, help="输出目录")
    parser.add_argument("--params", default=None,
                        help="显式参数 JSON（含 auto_params 的 adjust_params.json 亦可）")
    # 显式参数覆盖（无需步骤1目录）
    parser.add_argument("--flip-h", action="store_true", help="水平翻转（左右镜像，对应 Register.Flip）")
    parser.add_argument("--flip-v", action="store_true", help="垂直翻转（上下颠倒）")
    parser.add_argument("--rotation", type=float, default=None, help="旋转角度（度，正=逆时针）")
    parser.add_argument("--scale-x", type=float, default=None, help="x 方向模板缩放因子（应用时取倒数）")
    parser.add_argument("--scale-y", type=float, default=None, help="y 方向模板缩放因子（应用时取倒数）")
    parser.add_argument("--dx", type=float, default=None, help="x 平移像素（Register.OffsetX）")
    parser.add_argument("--dy", type=float, default=None, help="y 平移像素（Register.OffsetY）")
    parser.add_argument("--rot90", type=int, default=None,
                        help="逆时针90°旋转次数（0/1/2/3，Register.CounterRot90）")
    parser.add_argument("--dst-w", type=int, default=None, help="输出画布宽度（Register.MatrixShape[1]）")
    parser.add_argument("--dst-h", type=int, default=None, help="输出画布高度（Register.MatrixShape[0]）")
    parser.add_argument("--fill", type=int, default=DEFAULT_FILL, help="边界填充值 (默认 0)")
    parser.add_argument("--no-cellbin2", action="store_true",
                        help="不尝试 cellbin2 原生 trans_image，强制 OpenCV 实现")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅打印参数与计划，不写图")
    args = parser.parse_args()

    if not os.path.isfile(args.ssdna):
        print(f"[ERROR] ssDNA image not found: {args.ssdna}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # -- 1. 确定调整参数 ------------------------------------------------------
    params = load_params(args, args.chip_id, args.qc_dir)

    # 显式 CLI 参数覆盖
    if args.flip_h:
        params["flip_h"] = True
    if args.flip_v:
        params["flip_v"] = True
    if args.rotation is not None:
        params["rotation_deg"] = args.rotation
    if args.scale_x is not None:
        params["scale_x"] = args.scale_x
    if args.scale_y is not None:
        params["scale_y"] = args.scale_y
    if args.dx is not None:
        params["dx"] = args.dx
    if args.dy is not None:
        params["dy"] = args.dy
    if args.rot90 is not None:
        params["counter_rot90"] = args.rot90 % 4
    if args.dst_w is not None and args.dst_h is not None:
        params["dst_size"] = [args.dst_h, args.dst_w]

    print("[s2] 调整参数（来自 %s）:" % params.get("source", "unknown"))
    print(json.dumps({k: v for k, v in params.items() if k != "source"},
                     indent=2, ensure_ascii=False))

    if args.dry_run:
        print("[s2] Dry run - image not written.")
        return

    # -- 2. 应用变换 ----------------------------------------------------------
    print(f"[s2] 读取 ssDNA 图像: {args.ssdna}")
    img = read_image(args.ssdna)
    print(f"[s2] 原始尺寸: {img.shape}")
    out = apply_transform(img, params, fill=args.fill,
                          prefer_cellbin2=not args.no_cellbin2)
    print(f"[s2] 调整后尺寸: {out.shape}")

    out_path = os.path.join(args.output_dir, f"{args.chip_id}_ssDNA_adjusted.tif")
    write_image(out_path, out)
    print(f"[s2] 已保存: {out_path}")

    # -- 3. 记录实际应用参数 --------------------------------------------------
    record = {
        "chip_id": args.chip_id,
        "input_ssdna": args.ssdna,
        "output_image": out_path,
        "params_source": params.get("source"),
        "applied_params": {k: v for k, v in params.items() if k != "source"},
        "input_shape": list(img.shape),
        "output_shape": list(out.shape),
        "note": "scale 应用时取倒数(1/ScaleX,1/ScaleY)，变换顺序与 CellBin2 run_transform+transform_to_register 一致",
    }
    record_path = os.path.join(args.output_dir, f"{args.chip_id}.applied_params.json")
    with open(record_path, "w") as fh:
        json.dump(record, fh, indent=2, ensure_ascii=False)
    print(f"[s2] 参数记录已保存: {record_path}")

    # -- 4. 摘要 -------------------------------------------------------------
    summary = [
        f"# 步骤2摘要 - {args.chip_id}",
        f"输入 ssDNA : {args.ssdna}",
        f"调整前尺寸 : {list(img.shape)}",
        f"调整后尺寸 : {list(out.shape)}",
        "",
        "## 应用参数（来自 DiI 图 QC+GEM 配准，原样套用）",
        json.dumps({k: v for k, v in params.items() if k != "source"},
                   indent=2, ensure_ascii=False),
        "",
        "## 变换顺序（与 CellBin2 1.1.0 标准 10X 流程一致）",
        "scale+rotate (取倒数) -> flip_h(Flip) -> rot90(CounterRot90) -> offset(dx,dy) -> canvas(dst_size)",
        "",
        "## 下一步（步骤3）",
        f"用 {out_path} 作为 cellbin2 圈细胞输入。",
    ]
    summary_path = os.path.join(args.output_dir, f"{args.chip_id}.ssdna_adjust.summary.txt")
    with open(summary_path, "w") as fh:
        fh.write("\n".join(summary))
    print(f"[s2] 摘要已保存: {summary_path}")


if __name__ == "__main__":
    main()
