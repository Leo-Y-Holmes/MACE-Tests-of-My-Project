import os

# 跳过 HuggingFace 联网查询（matcalc import 时会查 HF 模型列表，本机网络下不可达，白等 40 秒）
os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
from pymatgen.core import Structure, Lattice
from matcalc import PESCalculator, RelaxCalc, MDCalc, PhononCalc

# 固定随机种子，让 MD 结果可复现
np.random.seed(42)

# 要用的模型（如 "MACE-OMAT-0-medium"）
MODEL = "MACE-OMAT-0-medium"

# ============ 1. 构建金刚石结构 Si（8 原子惯用胞，a = 5.43 Å）============
a = 5.43
lattice = Lattice.cubic(a)
species = ["Si"] * 8
coords = [
    [0.0, 0.0, 0.0],
    [0.0, 0.5, 0.5],
    [0.5, 0.0, 0.5],
    [0.5, 0.5, 0.0],
    [0.25, 0.25, 0.25],
    [0.25, 0.75, 0.75],
    [0.75, 0.25, 0.75],
    [0.75, 0.75, 0.25],
]
si = Structure(lattice, species, coords)
print(f"构建成功！体系包含 {len(si)} 个原子。")

# ============ 2. 加载 MACE 计算器 ============
calc = PESCalculator.load_universal(MODEL)
print(f"MACE 模型加载成功（{MODEL}）！")

# ============ 3. 静态能量与受力 ============
from pymatgen.io.ase import AseAtomsAdaptor

atoms = AseAtomsAdaptor.get_atoms(si)
atoms.calc = calc
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
print(f"体系总能量: {energy:.6f} eV  (每个原子 {energy / len(si):.6f} eV)")
print("前 3 个原子的受力 (eV/Angstrom):")
for i, f in enumerate(forces[:3]):
    print(f"  原子 {i}: {f}")

# ============ 4. 结构优化 ============
print("\n开始结构优化...")
relaxer = RelaxCalc(calc, fmax=0.01)
relax_result = relaxer.calc(si)
si_relaxed = relax_result["final_structure"]
print(
    f"结构优化完成！收敛: {relax_result['is_converged']}, "
    f"最大受力: {relax_result['max_force']:.6f} eV/A"
)

# ============ 5. NVT 分子动力学 ============
print("\n开始 NVT 分子动力学模拟...")
md = MDCalc(
    calc,
    ensemble="nvt_langevin",
    temperature=300,
    timestep=1.0,      # fs
    steps=1000,        # 1 fs * 1000 = 1 ps
    friction=1.0e-3,
)
md_result = md.calc(si_relaxed)
print("NVT MD 完成！")

# ============ 6. 声子计算 ============
print("\n开始声子计算（这可能需要几分钟）...")
ph = PhononCalc(
    calc,
    min_length=10.0,                 # 超胞最小边长（Å）
    write_band_structure=False,      # 不写能带文件
    write_total_dos=False,
    write_phonon=False,
)
ph_result = ph.calc(si_relaxed)

frequencies = ph_result["frequencies"]  # 单位 THz
omega_max_thz = float(np.max(frequencies))
omega_max_cm = omega_max_thz * 33.356  # 1 THz = 33.356 cm^-1
print("\n=== 声子最高频率 ===")
print(f"ω_max = {omega_max_cm:.2f} cm^-1  ≈  {omega_max_thz:.2f} THz")

print("\n全部计算完成！")
