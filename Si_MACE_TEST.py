import numpy as np
from ase import units
from ase.io import read, Trajectory
from ase.build import bulk
from ase.optimize import BFGS
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import thermalize_momenta

# 固定随机种子，让 MD 结果可复现
np.random.seed(42)

# 直接构建一个金刚石结构的Si晶体
atoms = bulk('Si', 'diamond', a=5.43)
print(f"读取成功！体系包含 {len(atoms)} 个原子。")

from mace.calculators import mace_mp
# 加载模型
mace_calc = mace_mp(model="MACE_Si_mh0_finetuned_2.0_stagetwo.model",default_dtype="float64")
atoms.calc = mace_calc
print("MACE 模型加载成功！")

# 计算总能量
total_energy = atoms.get_potential_energy()
print(f"体系总能量: {total_energy:.6f} eV  (每个原子 {total_energy/len(atoms):.6f} eV)")

# 获取原子受力 (单位: eV/Å)
forces = atoms.get_forces()
print("前5个原子的受力 (eV/Angstrom):")
for i, f in enumerate(forces[:5]):
    print(f"原子 {i}: {f}")
print("\n开始结构优化...")
# 使用 BFGS 算法，fmax=0.01 表示力的收敛标准为 0.01 eV/Å
opt = BFGS(atoms, trajectory='si_opt.traj')
opt.run(fmax=0.01)
print("结构优化完成！")
atoms_equil = atoms.copy()  # 保存优化后的平衡结构，供声子计算使用

print("\n开始 NVT 分子动力学模拟...")
T = 0  # 目标温度 300 K

# 1. 初始化速度 
thermalize_momenta(atoms, T)

# 2. 设置 Langevin 动力学引擎
# 参数：原子, 时间步长(1 fs), 目标温度, 摩擦系数(0.01)
dyn = Langevin(atoms, timestep=1.0 * units.fs, temperature_K=T, friction=0.01)

# 3. 打印模拟日志
def print_temp():
    print(f"当前温度: {atoms.get_temperature():.2f} K, 势能: {atoms.get_potential_energy():.2f} eV")
dyn.attach(print_temp, interval=50)

# 4. 运行 1000 步 (1 fs * 1000 = 1 ps)
dyn.run(1000)
print("NVT MD 完成！")

print("\n开始声子计算 (3x3x3 超胞，这可能需要几分钟)...")
import matplotlib.pyplot as plt
import shutil
from ase.phonons import Phonons

# 每次运行前清掉旧的声子缓存，避免复用上一个模型的力
shutil.rmtree("phonon", ignore_errors=True)

# 1. 正确的声子计算设置
# 传入优化后的平衡结构 atoms_equil，让 ASE 自动扩胞为 3x3x3
ph = Phonons(atoms_equil, mace_calc, supercell=(3, 3, 3), delta=0.03)
ph.run()

# 2. 读取力常数
ph.read(acoustic=True)

# 3. 定义高对称路径 (金刚石/面心立方结构标准路径: G-X-W-K-G-L-U-W-L-K)
# 使用 ASE 内置的 bandpath 自动生成路径和坐标
path = atoms_equil.cell.bandpath('GXWKLG', npoints=100)
qpts = path.kpts  # 获取路径上的 q 点坐标

# 4. 计算声子能带
# omega 形状为 (nqpts, 3N)，单位是 eV
omega = ph.band_structure(qpts)

# 5. 单位转换: eV -> cm^-1 (波数)
freqs_cm = omega / units.invcm

# 6 提取并输出 Γ 点 (G点) 的最高声子频率
# path.kpts 是路径上的 q 点坐标，找到第一个 [0,0,0] 的索引
gamma_indices = np.where(np.all(np.isclose(path.kpts, [0, 0, 0]), axis=1))[0]

if len(gamma_indices) > 0:
    gamma_idx = gamma_indices[0]  # 取第一个 Γ 点（路径起点）
    freqs_at_gamma = freqs_cm[gamma_idx, :]  # 提取该点的所有频率
    omega_max = np.max(freqs_at_gamma)       # 取最大值
    
    # 单位换算：1 THz = 33.356 cm^-1
    omega_max_thz = omega_max / 33.356
    
    print(f"\n=== Γ 点声子频率分析 ===")
    print(f"Γ 点所有频率 (cm^-1): {np.round(freqs_at_gamma, 2)}")
    print(f"Γ 点最高声子频率 ω_max = {omega_max:.2f} cm^-1")
    print(f"                      ≈ {omega_max_thz:.2f} THz")
else:
    print("\n未找到 Γ 点，请检查路径设置。")

# 7. 开始绘图 
plt.figure(figsize=(10, 6))

# 获取正确的线性坐标 (x) 和高对称点刻度 (xticks)
x, xticks, xlabels = path.get_linear_kpoint_axis()

# 绘制每一条能带
for i in range(freqs_cm.shape[1]):
    plt.plot(x, freqs_cm[:, i], color='b', linewidth=1.5)

# 标注高对称点
plt.xticks(xticks, xlabels, fontsize=12)
for idx in xticks:
    plt.axvline(x=idx, color='gray', linestyle='--', linewidth=0.8)

plt.axhline(y=0, color='black', linewidth=1) # 画一条 0 频率基准线
plt.ylabel(r'Frequency (cm$^{-1}$)', fontsize=12)
plt.xlabel('Wave vector', fontsize=12)
plt.title('Phonon dispersion of Si (MACE)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('si_phonon_band_fixed.png', dpi=300)
plt.show()

print("声子谱绘制完成！图片已保存为 si_phonon_band_fixed.png")