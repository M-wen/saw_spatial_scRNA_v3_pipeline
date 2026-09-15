#!/usr/bin/env python3
"""
Extract cell labels and areas from a GEF file and save as a TSV file.

"""

import argparse
import h5py
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract cell labels and areas from a GEF file and save as a TSV file.'
    )

    # 必需参数：输入 gef 文件
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='path of cellbin2 gef file(*.gef)'
    )

    # 可选参数：输出文件
    parser.add_argument(
        '-o', '--output',
        default='cell_area.txt',
        help='output file'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # 读取 GEF
    with h5py.File(args.input, 'r') as f:
        cell = f['/cellBin/cell'][:]

    # 构造 DataFrame
    df_cell = pd.DataFrame({
        'label': cell['id'],
        'area':  cell['area']
    })

    df_cell.columns = df_cell.columns.str.strip()

    df_cell.to_csv(
        args.output,
        index=False,
        sep="\t",
        encoding='utf-8-sig'
    )

    print(f'have already saved {len(df_cell)} 个 cell 到 {.output}')


if __name__ == '__main__':
    main()
