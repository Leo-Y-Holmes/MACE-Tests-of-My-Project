"""
从 MPtrj 的 parquet 文件里筛纯 Si 条目，抽结构+能量+力 → 生成 si_train.xyz。

数据来源：https://gitee.com/hf-datasets/mptrj（HuggingFace datasets/nimashoghi/mptrj 的国内镜像）

列含义（已在 val 分片上验证）：
  - composition: 长度 120 的计数向量，索引 = 原子序数（第 14 位 = Si）
  - numbers: 每个原子的原子序数
  - positions: 笛卡尔坐标（Å）
  - cell: 3x3 晶格
  - pbc: 周期边界
  - energy: 总能量（eV）
  - forces: 力（eV/Å）

用法：python extract_si_mptrj.py mptrj/*.parquet
"""

import glob
import sys

import numpy as np
import pyarrow.parquet as pq
from ase import Atoms
from ase.io import write

OUT_XYZ = "si_train.xyz"


def is_pure_si(composition) -> bool:
    """composition 第 14 位 > 0 且只有 Si。"""
    return len(composition) > 14 and composition[14] > 0 and sum(composition) == composition[14]


def main() -> None:
    files = sorted(sys.argv[1:]) or sorted(glob.glob("mptrj/*.parquet"))
    if not files:
        raise SystemExit("用法: python extract_si_mptrj.py <parquet文件...>")

    total = 0
    first_write = True
    buffer: list[Atoms] = []

    def flush() -> None:
        nonlocal first_write, total
        if buffer:
            write(OUT_XYZ, buffer, format="extxyz", append=not first_write)
            first_write = False
            total += len(buffer)
            buffer.clear()

    for path in files:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=20000):
            d = batch.to_pydict()
            for i in range(len(d["numbers"])):
                if not is_pure_si(d["composition"][i]):
                    continue
                atoms = Atoms(
                    numbers=d["numbers"][i],
                    positions=d["positions"][i],
                    cell=d["cell"][i],
                    pbc=d["pbc"][i],
                )
                atoms.info["REF_energy"] = float(d["energy"][i])
                atoms.arrays["REF_forces"] = np.array(d["forces"][i], dtype=float)
                buffer.append(atoms)
                if len(buffer) >= 1000:
                    flush()
        flush()
        print(f"  处理完 {path}，累计 {total} 帧")

    print(f"\n完成！共 {total} 帧纯 Si 写入 {OUT_XYZ}")


if __name__ == "__main__":
    main()
