#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_si_dielectric.py
======================
从 Materials Project 抓取纯 Si 的 DFPT 介电数据，生成 MACE-Field 多头微调用的
"介电头"训练文件 si_dielectric.xyz。

数据来源：
  - summary:  结构 + energy_per_atom（DFT 能量）
  - dielectric: 电子介电张量 electronic (= eps_inf)、离子介电张量 ionic

MP 里纯 Si 带介电数据的条目很少（约 3 条）。Born 有效电荷（born）在当前 mp-api
版本里不直接提供，而硅的 Born 电荷≈0（金刚石结构严格为 0），故 REF_becs 置 0。

输出字段（extxyz，对齐 mace-field 的 universal_field 数据格式）：
  info:  REF_energy, REF_stress, REF_electric_field, REF_polarization, REF_polarizability
  arrays: REF_forces, REF_becs
"""
import numpy as np
from ase.io import write
from pymatgen.io.ase import AseAtomsAdaptor

from mp_api.client import MPRester

API_KEY = "mKG481zM7LqI6Dzrhai7qYHxMrSBMcTP"   # 可改成环境变量 MP_API_KEY
OUT_XYZ = "si_dielectric.xyz"
FIELDS = ["material_id", "formula_pretty", "energy_per_atom", "structure"]


def main() -> None:
    with MPRester(API_KEY) as mpr:
        # 1) 纯 Si 的 summary（结构 + 能量）
        si_summary = mpr.materials.summary.search(formula="Si", fields=FIELDS)
        print(f"纯 Si summary 条目: {len(si_summary)}")

        # 2) 这些条目里带介电数据的
        si_ids = [d.material_id for d in si_summary]
        diel_docs = mpr.materials.dielectric.search(material_ids=si_ids)
        print(f"带介电数据的条目: {len(diel_docs)}")

        # 3) 按 material_id 匹配 summary（拿结构和能量）
        summary_by_id = {d.material_id: d for d in si_summary}

    atoms_list = []
    for d in diel_docs:
        mid = d.material_id
        summ = summary_by_id.get(mid)
        if summ is None or summ.structure is None:
            print(f"跳过 {mid}：无结构")
            continue

        atoms = AseAtomsAdaptor.get_atoms(summ.structure)
        natoms = len(atoms)

        # 电子介电张量 eps_inf (3x3) -> 极化率 chi = eps_inf - I
        eps_inf = np.asarray(d.electronic, dtype=float)
        chi = eps_inf - np.eye(3)

        # 能量（eV）：energy_per_atom * 原子数
        energy = float(summ.energy_per_atom) * natoms

        # 弛豫结构：力≈0、应力≈0；中心对称硅：P=0；硅 Born 电荷≈0
        atoms.info["REF_energy"] = energy
        atoms.info["REF_stress"] = np.zeros(6)
        atoms.info["REF_electric_field"] = np.zeros(3)
        atoms.info["REF_polarization"] = np.zeros(3)
        atoms.info["REF_polarizability"] = chi.reshape(9)  # 展平成 9
        atoms.arrays["REF_forces"] = np.zeros((natoms, 3))
        atoms.arrays["REF_becs"] = np.zeros((natoms, 9))   # 展平成 (N,9)

        atoms.info["_material_id"] = mid
        atoms.info["_eps_inf_diag"] = list(np.diag(eps_inf))
        atoms_list.append(atoms)
        print(f"  {mid}  {summ.formula_pretty}:  natoms={natoms}  "
              f"eps_inf_diag={np.round(np.diag(eps_inf),3)}  energy={energy:.4f} eV")

    # 数据太少（3 条），train/valid 切分会强制留 1 条验证，可能把关键相（金刚石硅）
    # 挤到验证集。把每条复制 4 份，保证切分后三个相都在训练集里。
    N_COPIES = 4
    atoms_list = [a.copy() for a in atoms_list for _ in range(N_COPIES)]
    write(OUT_XYZ, atoms_list, format="extxyz")
    print(f"\n完成！{len(atoms_list)} 条（{len(atoms_list)//N_COPIES} 相 × {N_COPIES} 份）写入 {OUT_XYZ}")


if __name__ == "__main__":
    main()
