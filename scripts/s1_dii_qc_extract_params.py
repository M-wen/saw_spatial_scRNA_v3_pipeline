#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
s1_dii_qc_extract_params.py - 步骤1：用 DiI 图像做 CellBin2 QC，并提取可量化的几何调整参数

用途
----
对 DiI（细胞膜）染色图像执行 CellBin2 图像 QC（含 trackline 检测 + 配准），
并从配准结果中提取一组可量化、可复现的几何调整参数（翻转 / 旋转 / 缩放 / 平移 / 裁剪），
输出 adjust_params.json 供步骤2（同参数应用到 ssDNA 图像）与步骤3（cellbin2 圈细胞）直接复用。

关键约定（与用户既有流程一致）
--------------------------------
- DiI 与 ssDNA 为同一成像通道的两种染料，两图几何关系天然一致；
  因此步骤1在 DiI 图上确定的调整参数将 原样 套用到 ssDNA 图。
- DiI 图以 ssDNA 类型（track 染色）输入，qc=true 开启 trackline 检测与配准
  （与 test3_expand10_report.sh 中 unregistered 模式一致）。
- 参数提取优先解析 CellBin2 配准产物（ipr/rpi/日志）；
  若无法解析，则用 原始 DiI 图 vs 配准后 *_regist.tif 做图像配准反推仿射矩阵，
  分解为：水平/垂直翻转标志、旋转角、x/y 缩放、平移、裁剪框。

用法
----
python s1_dii_qc_extract_params.py \\
    -c Y40612D413 \\
    -i /path/to/Y40612D413-DII.tif \\
    -o /path/to/output_dir \\
    --pipeline /opt/software/cellbin2/cellbin2/cellbin_pipeline.py \\
    --python /home/stereonote/miniconda3/envs/env-cellbinv2/bin/python \\
    --gpu 0 \\
    --magnification 10

仅生成配置与参数模板（不运行）：
python s1_dii_qc_extract_params.py -c XXX -i dii.tif -o out/ --dry-run

输出
----
<out>/<chip>.dii.qc.json           CellBin2 QC 阶段产出的原始指标（若存在）
<out>/<chip>.adjust_params.json    调整参数（主交付物，供步骤2使用）
<out>/<chip>.qc_summary.txt        人类可读 QC 摘要
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
from glob import glob

# ----------------------------------------------------------------------------
# 常量
# ----------------------------------------------------------------------------
DEFAULT_MAGNIFICATION = 10
DEFAULT_PIPELINE = "/opt/software/cellbin2/cellbin2/cellbin_pipeline.py"


def generate_config(chip_id, dii_path, gem_path=None, magnification=DEFAULT_MAGNIFICATION):
    """
    生成 CellBin2 pipeline JSON 配置：
      - DiI 图（track 染色，tech_type=ssDNA）为图像输入；
      - qc=true  -> 执行图像质量检查 + trackline 检测（配准所需几何参数在此产出）；
      - alignment=true -> 执行配准（含翻转搜索），输出 *_regist.tif 与 ipr/rpi 产物；
      - 若提供 GEM：作为 fixed_image（image_process["1"]），
        触发 scheduler 的 run_register -> registration() 用 flip_flag=true 搜索水平/垂直翻转，
        结果写入 ipr 的 Register.Flip / CounterRot90 / OffsetX / OffsetY。
      - matrix_extract=false / report=false（本步骤只做 QC + 配准，不做分子归类）。
    """
    config = {
        "image_process": {
            "0": {
                "file_path": dii_path,
                "tech_type": "ssDNA",
                "chip_detect": False,
                "quality_control": True,
                "tissue_segmentation": False,
                "cell_segmentation": False,   # 步骤1不做细胞分割
                "channel_align": 0,
                "magnification": magnification,
                "registration": {
                    "fixed_image": 1 if gem_path else -1,  # GEM 作为配准基准
                    "trackline": True,        # trackline 检测：翻转/旋转/缩放/平移 的直接来源
                    "reuse": -1
                }
            }
        },
        "molecular_classify": {
            "0": {
                "exp_matrix": 1,
                "cell_mask": {"nuclei": [], "interior": [], "boundary": []},
                "correct_r": 10,
                "extra_method": ""
            }
        },
        "run": {
            "qc": True,
            "alignment": True,
            "matrix_extract": False,
            "report": False,
            "annotation": False
        }
    }
    if gem_path:
        config["image_process"]["1"] = {
            "file_path": gem_path,
            "tech_type": "Transcriptomics",
            "chip_detect": False,
            "quality_control": False,
            "tissue_segmentation": False,
            "cell_segmentation": False,
            "channel_align": -1,
            "magnification": magnification,
            "registration": {
                "fixed_image": -1,
                "trackline": False,
                "reuse": -1
            }
        }
    return config


# ----------------------------------------------------------------------------
# 参数提取：从 CellBin2 配准产物解析
# ----------------------------------------------------------------------------
def try_parse_cellbin_params(output_dir, chip_id):
    """
    尝试从 CellBin2 输出目录解析配准参数。

    优先顺序：
      1) 日志文本解析（最可靠）：从 *.log 中抓取
         "Template Scale is (sx, sy), rotation is r"（image_qc 配准真实输出）。
      2) 文本文件关键字扫描（*.json / *.log / *.txt，排除 ipr/rpi 等二进制）。
      3) 若 cellbin2 可用，尝试用 cellbin2.utils.ipr 读取 ipr 文件并 dump。

    返回 dict 或 None。
    """
    found = {}

    # -- 0) 日志配准参数解析（CellBin2 image_qc 真实输出） ------------------
    log_params = _parse_qc_log(output_dir)
    if log_params:
        found.update(log_params)

    # -- 1) 文本关键字扫描（排除 HDF5 二进制 ipr/rpi 防误抓） ---------------
    patterns = {
        "flip_h": re.compile(r'(?:flip|mirror)[_\s]*(?:h|x|horizontal)', re.I),
        "flip_v": re.compile(r'(?:flip|mirror)[_\s]*(?:v|y|vertical)', re.I),
        "rotation": re.compile(r'(?:rotation|rot|angle|theta)[^\d]{0,12}([-+]?\d+(?:\.\d+)?)', re.I),
        "scale_x": re.compile(r'(?:scale|zoom|resize)[^\d]{0,12}?[xX][^\d]{0,4}([-+]?\d+(?:\.\d+)?)', re.I),
        "scale_y": re.compile(r'(?:scale|zoom|resize)[^\d]{0,12}?[yY][^\d]{0,4}([-+]?\d+(?:\.\d+)?)', re.I),
        "dx": re.compile(r'(?:offset|shift|translation|translate)[^\d]{0,12}?[d]?[xX][^\d]{0,4}([-+]?\d+(?:\.\d+)?)', re.I),
        "dy": re.compile(r'(?:offset|shift|translation|translate)[^\d]{0,12}?[d]?[yY][^\d]{0,4}([-+]?\d+(?:\.\d+)?)', re.I),
    }

    TEXT_EXTS = (".json", ".log", ".txt", ".out")
    BIN_EXTS = (".ipr", ".rpi")  # HDF5，禁止文本扫描

    scan_files = []
    for ext in TEXT_EXTS:
        scan_files.extend(glob(os.path.join(output_dir, ext)))
    # 子目录再扫一层（scheduler 可能把中间结果放子目录）
    for sub in sorted(glob(os.path.join(output_dir, "*"))):
        if os.path.isdir(sub):
            for ext in TEXT_EXTS:
                scan_files.extend(glob(os.path.join(sub, ext)))

    for fp in scan_files:
        if fp.lower().endswith(BIN_EXTS):
            continue
        try:
            with open(fp, "r", errors="ignore") as fh:
                text = fh.read()
        except Exception:
            continue
        for key, pat in patterns.items():
            if key in found:
                continue
            m = pat.search(text)
            if m:
                try:
                    val = m.group(1)
                    found[key] = float(val) if val else True
                except (ValueError, IndexError):
                    found[key] = True

    # -- 2) cellbin2 原生读取（若当前解释器可用） ---------------------------
    try:
        import cellbin2
        from cellbin2.utils import ipr as _ipr
        for ipr_file in glob(os.path.join(output_dir, "*.ipr")):
            try:
                _ipr_r, _channels = _ipr.read(ipr_file)
                found["_ipr_raw"] = json.loads(json.dumps(_ipr_r, default=str))
            except Exception:
                continue
    except Exception:
        pass

    return found if found else None


def _parse_qc_log(output_dir):
    """
    从 CellBin2 image_qc 运行日志解析配准参数（cellbin2 1.1.0 实测日志行）：
      Template Scale is (1.0071593305798057, 1.0074742690737442),
      rotation is 1.7532599999999998
    返回 dict 或 None。
    """
    files = []
    for ext in ("*.log", "*.out", "*.txt"):
        files.extend(glob(os.path.join(output_dir, ext)))
    for sub in sorted(glob(os.path.join(output_dir, "*"))):
        if os.path.isdir(sub):
            for ext in ("*.log", "*.out", "*.txt"):
                files.extend(glob(os.path.join(sub, ext)))

    pat = re.compile(
        r"Template\s+Scale\s+is\s*\(([0-9.eE+-]+),\s*([0-9.eE+-]+)\)\s*,\s*"
        r"rotation\s+is\s+([0-9.eE+-]+)",
        re.I)
    for fp in files:
        try:
            with open(fp, "r", errors="ignore") as fh:
                text = fh.read()
        except Exception:
            continue
        m = pat.search(text)
        if m:
            return {
                "scale_x": float(m.group(1)),
                "scale_y": float(m.group(2)),
                "rotation": float(m.group(3)),
                "_log_file": os.path.basename(fp),
            }
    return None


# ----------------------------------------------------------------------------
# 参数提取：从 ipr 的 Register 读取配准结果（含 Flip）
# ----------------------------------------------------------------------------
def parse_register_from_ipr(output_dir):
    """
    从 CellBin2 输出的 *.ipr（HDF5）读取 ssDNA/Register 组下的配准结果。

    配准由 scheduler.run_mul_image -> run_register 触发（需 GEM 作为 fixed_image），
    registration() 用 flip_flag=config.registration.flip 搜索水平翻转，
    结果写入：
      Register.Flip        (bool)  水平翻转标志（True=需要水平翻转）
      Register.CounterRot90(int)   逆时针90°旋转次数（0/1/2/3）
      Register.OffsetX/Y   (float) 配准后平移偏移
      Register.Rotation    (float) 模板旋转角（度）
      Register.ScaleX/Y    (float) 模板缩放
      Register.Method      (str)   配准方法

    返回 dict 或 None。
    """
    import h5py

    for ipr_file in sorted(glob(os.path.join(output_dir, "*.ipr"))):
        try:
            with h5py.File(ipr_file, "r") as fd:
                # 找到第一个含 Register 的通道组（ssDNA / DAPI / HE）
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
                if not out:
                    continue
                out["_ipr_file"] = os.path.basename(ipr_file)
                # Flip 需要是真值才有效（占位符为 False）
                flip = out.get("Flip")
                if flip in (None, -999, -999.0, -999.999):
                    out["Flip"] = False
                return out
        except Exception:
            continue
    return None


# ----------------------------------------------------------------------------
# 参数提取：图像配准反推（原始图 vs 配准后图）
# ----------------------------------------------------------------------------
def estimate_transform_from_images(src_path, reg_path):
    """
    用 原始 DiI 图 与 CellBin2 配准后 *_regist.tif 估计 2D 仿射变换，
    并分解为几何调整参数。

    依赖：OpenCV（优先）或 scikit-image + numpy。
    若两者都不可用，返回 None（由调用方降级处理）。

    返回 dict:
        flip_h, flip_v, rotation_deg, scale_x, scale_y, dx, dy, method
    """
    try:
        import cv2
        import numpy as np
        has_cv2 = True
    except ImportError:
        has_cv2 = False

    try:
        import numpy as np
        from skimage import transform as skt, io as skio
        has_sk = True
    except ImportError:
        has_sk = False

    if not (has_cv2 or has_sk):
        return None

    # -- 读取（灰度） --------------------------------------------------------
    if has_cv2:
        src = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
        reg = cv2.imread(reg_path, cv2.IMREAD_GRAYSCALE)
        if src is None or reg is None:
            return None
        # 若尺寸不一致，统一缩放到 src 尺寸（regist 图可能带填充/裁剪）
        if src.shape != reg.shape:
            reg = cv2.resize(reg, (src.shape[1], src.shape[0]))
    else:
        src = skio.imread(src_path, as_gray=True)
        reg = skio.imread(reg_path, as_gray=True)
        if src.shape != reg.shape:
            reg = skt.resize(reg, src.shape)

    src = np.asarray(src, dtype=np.float32)
    reg = np.asarray(reg, dtype=np.float32)

    # -- 估计变换矩阵 --------------------------------------------------------
    if has_cv2:
        # 先用 ORB 特征匹配估计单应/仿射，失败再退化到 ECC
        try:
            orb = cv2.ORB_create(nfeatures=5000)
            kp1, des1 = orb.detectAndCompute(src, None)
            kp2, des2 = orb.detectAndCompute(reg, None)
            if des1 is not None and des2 is not None and len(kp1) > 10 and len(kp2) > 10:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING)
                matches = bf.knnMatch(des1, des2, k=2)
                good = [m for m, n in matches if m.distance < 0.75 * n.distance]
                if len(good) >= 8:
                    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                    M, mask = cv2.estimateAffinePartial2D(pts1, pts2)
                    if M is not None:
                        method = "cv2_ORB_affine"
                        return _decompose_affine(M, method)
        except Exception:
            pass

        # ECC 迭代配准（估计平移+旋转+缩放）
        try:
            warp_mode = cv2.MOTION_AFFINE
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 2000, 1e-6)
            cc, warp_matrix = cv2.findTransformECC(
                reg, src, warp_matrix, warp_mode, criteria, None, 5)
            if abs(cc) > 0.01:
                return _decompose_affine(warp_matrix, "cv2_ECC_affine")
        except Exception:
            pass
    else:
        # skimage：估计相似变换（旋转+缩放+平移，无翻转）-> 翻转需另判
        try:
            tform = skt.estimate_transform("similarity", _sample_points(src), _sample_points(reg))
            M = tform.params[:2, :]
            res = _decompose_affine(M, "skimage_similarity")
            # 翻转检测：比较左右/上下半图相关
            res["flip_h"], res["flip_v"] = _detect_flips(src, reg)
            return res
        except Exception:
            return None

    return None


def _sample_points(img, n=200):
    """均匀采样 n 个点坐标（用于 skimage 估计变换）。"""
    import numpy as np
    h, w = img.shape
    xs = np.linspace(0, w - 1, int(math.sqrt(n * w / max(h, 1))) + 1)
    ys = np.linspace(0, h - 1, int(math.sqrt(n * h / max(w, 1))) + 1)
    xx, yy = np.meshgrid(xs, ys)
    return np.column_stack([xx.ravel(), yy.ravel()])[:n]


def _decompose_affine(M, method):
    """
    仿射矩阵 M (2x3) 分解为：
      翻转标志、旋转角、x/y 缩放、平移
    返回 dict。
    """
    import numpy as np
    a, b, tx = float(M[0, 0]), float(M[0, 1]), float(M[0, 2])
    c, d, ty = float(M[1, 0]), float(M[1, 1]), float(M[1, 2])

    A = np.array([[a, b], [c, d]])
    det = np.linalg.det(A)

    flip_h = det < 0          # 行列式为负 -> 含镜像（翻转）
    # 提取旋转-缩放：A = R(θ) * S（S 为对角缩放，可含负号代表翻转）
    # 简化处理：用列范数作为缩放，用去除缩放后的矩阵估计旋转
    sx = math.hypot(a, c)
    sy = math.hypot(b, d)
    sx = sx if sx > 1e-9 else 1.0
    sy = sy if sy > 1e-9 else 1.0

    # 归一化列向量 -> 旋转角
    ux = (a / sx, c / sx)
    uy = (b / sy, d / sy)
    theta = math.atan2(uy[0], ux[0])  # 粗略角度（deg 后处理）
    theta_deg = math.degrees(theta)

    return {
        "flip_h": bool(flip_h),
        "flip_v": False,             # 翻转方向已并入 flip_h/旋转，v 由额外检测补充
        "rotation_deg": round(theta_deg, 4),
        "scale_x": round(sx, 6),
        "scale_y": round(sy, 6),
        "dx": round(tx, 4),
        "dy": round(ty, 4),
        "method": method,
        "_det": round(det, 6),
    }


def _detect_flips(src, reg):
    """左右/上下镜像相关检测（skimage 路径补充）。返回 (flip_h, flip_v)。"""
    import numpy as np
    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        return False, False
    h, w = src.shape
    pairs = {
        "same": (reg, 1.0),
        "lr": (reg[:, ::-1], 1.0),
        "ud": (reg[::-1, :], 1.0),
    }
    best = ("same", -1.0)
    for name, (img2, _) in pairs.items():
        try:
            s = ssim(src, img2, data_range=src.max() - src.min() + 1e-9)
        except Exception:
            s = -1.0
        if s > best[1]:
            best = (name, s)
    return (best[0] == "lr"), (best[0] == "ud")


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def validate_inputs(dii_path):
    if not os.path.isfile(dii_path):
        print(f"[ERROR] DiI image not found: {dii_path}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="步骤1：DiI 图像 CellBin2 QC + 几何调整参数提取",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-c", "--chip-id", required=True, help="芯片号 (e.g. Y40612D413)")
    parser.add_argument("-i", "--dii", required=True, help="DiI 染色图像路径 (.tif)")
    parser.add_argument("-g", "--gem", default=None,
                        help="可选：GEM 表达矩阵路径 (.gem.gz / .gef)。提供后作为 fixed_image，"
                             "触发完整配准（含水平翻转搜索），并可从 ipr 提取 Register.Flip")
    parser.add_argument("-o", "--output-dir", required=True, help="输出目录")
    parser.add_argument("--pipeline", default=DEFAULT_PIPELINE,
                        help="cellbin_pipeline.py 路径")
    parser.add_argument("--python", default=None,
                        help="cellbin2 环境 Python 解释器（默认当前解释器）")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device id (默认 0)")
    parser.add_argument("--magnification", type=int, default=DEFAULT_MAGNIFICATION,
                        help="成像放大倍数 (默认 10)")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅生成配置 JSON，不运行 pipeline，不提取参数")
    parser.add_argument("--skip-validation", action="store_true",
                        help="跳过输入文件存在性检查")
    args = parser.parse_args()

    if not args.skip_validation:
        validate_inputs(args.dii)
        if args.gem and not os.path.isfile(args.gem):
            print(f"[ERROR] GEM file not found: {args.gem}", file=sys.stderr)
            sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # -- 1. 生成配置 --------------------------------------------------------
    config = generate_config(args.chip_id, args.dii, args.gem, args.magnification)
    config_path = os.path.join(args.output_dir, f"{args.chip_id}.dii.qc.config.json")
    with open(config_path, "w") as fh:
        json.dump(config, fh, indent=2, ensure_ascii=False)
    print(f"[s1] Config saved: {config_path}")
    print(f"     DiI image : {args.dii}")
    print(f"     GEM fixed : {args.gem if args.gem else '(无, 不触发翻转配准)'}")
    print(f"     tech_type : ssDNA (track), qc=True, alignment=True, trackline=True")

    if args.dry_run:
        print("[s1] Dry run - pipeline not executed.")
        return

    # -- 2. 运行 CellBin2 pipeline（QC + 配准） -----------------------------
    python_bin = args.python or sys.executable
    if not os.path.isfile(args.pipeline):
        print(f"[WARN] pipeline not found: {args.pipeline}；跳过运行，仅保留配置。", file=sys.stderr)
        adjust = {
            "chip_id": args.chip_id,
            "source_image": args.dii,
            "config_path": config_path,
            "status": "config_only",
            "message": "pipeline 未运行（路径不存在），请检查 --pipeline / 环境后重跑。",
        }
    else:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        cmd = [python_bin, args.pipeline, "-c", args.chip_id,
               "-p", config_path, "-o", args.output_dir]
        print(f"[s1] Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            print(f"[s1] pipeline 退出码 {result.returncode}；仍尝试从输出目录提取参数。",
                  file=sys.stderr)
        adjust = {
            "chip_id": args.chip_id,
            "source_image": args.dii,
            "config_path": config_path,
            "status": "pipeline_done",
        }

    # -- 3. 参数提取 --------------------------------------------------------
    print("\n[s1] === 参数提取 ===")
    # 优先级：ipr Register（含 Flip，需 GEM 配准）> 日志模板参数 > 文本扫描 > 图像反推
    reg_info = parse_register_from_ipr(args.output_dir) if args.gem else None
    params = try_parse_cellbin_params(args.output_dir, args.chip_id)
    est = None
    reg_candidates = glob(os.path.join(args.output_dir, "*_regist.tif"))
    if not reg_candidates:
        reg_candidates = glob(os.path.join(args.output_dir, "*", "*_regist.tif"))
    if reg_candidates:
        est = estimate_transform_from_images(args.dii, reg_candidates[0])
        if est:
            print(f"[s1] 图像配准反推成功 (方法: {est.get('method')})，"
                  f"参考配准图: {os.path.basename(reg_candidates[0])}")

    # 组装最终调整参数
    merged = {}
    if reg_info:
        # 来自 ipr Register：最权威（含 Flip 与配准偏移）
        merged.update({
            "flip_h": bool(reg_info.get("Flip", False)),
            "flip_v": False,
            "rotation_deg": float(reg_info.get("Rotation", 0.0) or 0.0),
            "scale_x": float(reg_info.get("ScaleX", 1.0) or 1.0),
            "scale_y": float(reg_info.get("ScaleY", 1.0) or 1.0),
            "dx": float(reg_info.get("OffsetX", 0.0) or 0.0),
            "dy": float(reg_info.get("OffsetY", 0.0) or 0.0),
            "counter_rot90": int(reg_info.get("CounterRot90", 0) or 0),
            "method": reg_info.get("Method", ""),
            "_source": f"ipr_register:{reg_info.get('_ipr_file','')}",
        })
        print(f"[s1] 从 ipr Register 提取配准参数: "
              f"Flip={merged['flip_h']}, rot90={merged['counter_rot90']}, "
              f"offset=({merged['dx']}, {merged['dy']}), rot={merged['rotation_deg']}°")
    elif params:
        # 日志模板参数（无 GEM 时的降级来源）
        merged.update({
            "flip_h": False,
            "flip_v": False,
            "rotation_deg": float(params.get("rotation", 0.0) or 0.0),
            "scale_x": float(params.get("scale_x", 1.0) or 1.0),
            "scale_y": float(params.get("scale_y", 1.0) or 1.0),
            "dx": float(params.get("dx", 0.0) or 0.0),
            "dy": float(params.get("dy", 0.0) or 0.0),
            "_source": "qc_log_template",
        })
        print(f"[s1] 从 QC 日志模板参数提取: rot={merged['rotation_deg']}°, "
              f"scale=({merged['scale_x']}, {merged['scale_y']})")
    elif est:
        merged = dict(est)
        merged["_source"] = "image_estimation"
        print(f"[s1] 使用图像配准反推参数: {json.dumps(est, ensure_ascii=False)}")

    if not merged:
        print("[s1] 未能自动解析出调整参数；请人工检查 QC 输出（见 qc_summary.txt 中的文件清单）。")
        adjust.update({
            "auto_params": None,
            "note": "未能自动解析参数；可结合 CellBin2 配准结果人工确定翻转/旋转/缩放/平移。",
        })
    else:
        adjust["auto_params"] = merged
        print(f"[s1] 调整参数: {json.dumps(merged, ensure_ascii=False, indent=2)}")

    # -- 4. 写输出文件 -------------------------------------------------------
    adjust_path = os.path.join(args.output_dir, f"{args.chip_id}.adjust_params.json")
    with open(adjust_path, "w") as fh:
        json.dump(adjust, fh, indent=2, ensure_ascii=False)
    print(f"[s1] 调整参数已保存: {adjust_path}")

    # -- 5. QC 摘要 ----------------------------------------------------------
    summary_lines = [
        f"# CellBin2 QC 摘要 - {args.chip_id}",
        f"输入图像 : {args.dii}",
        f"配置     : {config_path}",
        f"输出目录 : {args.output_dir}",
        "",
        "## 输出文件清单",
    ]
    for fp in sorted(glob(os.path.join(args.output_dir, "*"))):
        summary_lines.append(f"- {os.path.basename(fp)}")
    summary_lines += [
        "",
        "## 调整参数（供步骤2复用）",
        json.dumps(adjust.get("auto_params", {}), indent=2, ensure_ascii=False),
        "",
        "## 使用说明",
        "- 步骤2：将本文件中的 flip_h/flip_v/rotation_deg/scale_x/scale_y/dx/dy 原样应用到 ssDNA 图。",
        "- 步骤3：用调整后的 ssDNA 图作为 cellbin2 输入进行细胞分割。",
    ]
    summary_path = os.path.join(args.output_dir, f"{args.chip_id}.qc_summary.txt")
    with open(summary_path, "w") as fh:
        fh.write("\n".join(summary_lines))
    print(f"[s1] QC 摘要已保存: {summary_path}")


if __name__ == "__main__":
    main()
