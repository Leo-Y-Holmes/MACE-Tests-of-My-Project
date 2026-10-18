import os
import numpy as np
from ase.build import bulk

# ============ 1. 物理常数 (SI) ============
EPS0 = 8.8541878128e-12        # 真空介电常数 (F/m)
E_CHG = 1.602176634e-19        # 元电荷 (C)
EV = 1.602176634e-19           # 1 eV 的焦耳数 (J)
ANG = 1e-10                    # 1 Angstrom = 1e-10 m
EV_PER_A2_TO_J_PER_M2 = EV / ANG ** 2   # 1 eV/Angstrom^2 -> 16.02 J/m^2

# ============ 2. 模型与结构设置 ============
MODEL_PATH =  "MACE_Si_macefield_finetuned.model" 
MODEL_URL = ("https://github.com/mdi-group/mace-field/releases/download/"
             "1.0.2/MACEField-MH-0-omat-dielectric.model")
HEAD = "si_dielectric"                # 该预训练模型的介电/回放头
DEVICE = "cuda"                # 无 GPU 时改成 "cpu"
DEFAULT_DTYPE = "float32"      # 与预训练模型一致

A_LATT = 5.43                  # Si的初始晶格常数（Angstrom）
RELAX_CELL = True              # 是否先弛豫晶胞（晶格常数 + 原子位置）
FMAX = 0.01                    # 弛豫收敛标准 (eV/Angstrom)


# ============ 3. 张量整理与介电常数公式 ============
def reshape_sym(mat):
    """把 (3,3) 或展平的 (9,) 统一成 (3,3) numpy 数组。"""
    a = np.asarray(mat, dtype=float)
    if a.size == 9:
        a = a.reshape(3, 3)
    if a.shape != (3, 3):
        raise ValueError(f"无法解析为 3x3 张量: shape={a.shape}")
    return a


def reshape_becs(becs_raw, natoms):
    """把 Born 有效电荷规整成 (N,3,3)，索引 [原子, 极化分量 i, 位移分量 j]。"""
    a = np.asarray(becs_raw, dtype=float)
    if a.shape == (natoms, 3, 3):
        return a
    if a.shape == (natoms, 9):
        return a.reshape(natoms, 3, 3)
    if a.size == natoms * 9:
        return a.reshape(natoms, 3, 3)
    raise ValueError(f"意外的 BECs 形状 {a.shape}，应为 (N,3,3) 或 (N,9)")


def coerce_hessian_to_cartesian_matrix(H_raw, natoms):
    """把模型返回的 Hessian 规整成 (3N,3N) 笛卡尔力常数矩阵。

    兼容三种布局（不同版本 `get_hessian` 返回形状可能不同）：
      - (3N, 3N)
      - (N, N, 3, 3)  -> 分块展开
      - (3N, N, 3)    -> 展平后两维
    """
    H = np.asarray(H_raw, dtype=float)
    d = 3 * natoms
    if H.shape == (d, d):
        return H
    if H.ndim == 4 and H.shape == (natoms, natoms, 3, 3):
        return np.transpose(H, (0, 2, 1, 3)).reshape(d, d)
    if H.ndim == 3 and H.shape == (d, natoms, 3):
        return H.reshape(d, natoms * 3)
    raise ValueError(f"无法解析 Hessian 形状 {H.shape}，应为 (3N,3N)、(N,N,3,3) 或 (3N,N,3)")


def assemble_Z(becs_e):
    """把 (N,3,3) 的 Born 电荷排成 (3N,3)：行=(原子,极化分量)，列=位移分量。"""
    N = becs_e.shape[0]
    Z = np.zeros((3 * N, 3))
    for k in range(N):
        Z[3 * k + 0, :] = becs_e[k][:, 0]
        Z[3 * k + 1, :] = becs_e[k][:, 1]
        Z[3 * k + 2, :] = becs_e[k][:, 2]
    return Z


def enforce_bec_charge_neutrality(becs_e):
    """强制 Born 电荷声学求和规则（每个分量对所有原子求和应为 0）。"""
    becs = np.asarray(becs_e, dtype=float)
    return becs - np.mean(becs, axis=0, keepdims=True)


def translational_projector(natoms):
    """构造投影矩阵，去掉 3 个刚性平移（声学）方向。"""
    d = 3 * natoms
    proj = np.eye(d)
    for cart in range(3):
        v = np.zeros(d)
        v[cart::3] = 1.0
        v /= np.linalg.norm(v)
        proj -= np.outer(v, v)
    return proj


def symmetric_pseudoinverse(C, tol=1e-10):
    """对称矩阵的 Moore-Penrose 伪逆（可处理半正定/奇异矩阵）。"""
    C = 0.5 * (C + C.T)
    w, V = np.linalg.eigh(C)
    wmax = np.max(np.abs(w)) if w.size else 0.0
    eps = tol * (wmax if wmax > 0 else 1.0)
    w_inv = np.where(w > eps, 1.0 / w, 0.0)
    Cinv = (V * w_inv) @ V.T
    return 0.5 * (Cinv + Cinv.T)


def dielectric_from(alpha_chi, becs_e, hessian_eV_per_A2, volume_A3):
    """由电子极化率 chi、Born 电荷、力常数矩阵、体积计算介电张量。

    参数
    ----
    alpha_chi          : (3,3) 或 (9,)  电子极化率（无量纲 chi）
    becs_e             : (N,3,3)        Born 有效电荷（单位 e）
    hessian_eV_per_A2  : (3N,3N)        Gamma 点力常数矩阵（eV/Angstrom^2）
    volume_A3          : float           晶胞体积（Angstrom^3）

    返回 (eps_inf, eps_ion, eps_static)，均为 (3,3)。
    """
    # 电子（高频）介电张量: eps_inf = I + chi
    eps_inf = np.eye(3) + reshape_sym(alpha_chi)

    N = becs_e.shape[0]
    becs = enforce_bec_charge_neutrality(becs_e)
    Z_C = assemble_Z(becs * E_CHG)                        # (3N,3) 单位 C
    C_SI = np.asarray(hessian_eV_per_A2, dtype=float) * EV_PER_A2_TO_J_PER_M2  # J/m^2
    Omega = volume_A3 * ANG ** 3                          # m^3

    C_pinv = symmetric_pseudoinverse(C_SI)
    proj = translational_projector(N)
    C_pinv = proj @ C_pinv @ proj

    chi_ion = (Z_C.T @ C_pinv @ Z_C) / (EPS0 * Omega)
    eps_ion = 0.5 * (chi_ion + chi_ion.T)

    eps_inf = 0.5 * (eps_inf + eps_inf.T)
    eps_static = eps_inf + eps_ion
    return eps_inf, eps_ion, eps_static


def scalar(eps):
    """张量 -> 标量（主对角平均）。"""
    return float(np.trace(eps) / 3.0)


# ============ 4. 主流程 ============
def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # 切到脚本目录，确保相对路径(模型文件)正确

    # ---- 检查模型文件 ----
    if not os.path.isfile(MODEL_PATH):
        raise SystemExit(
            f"\n未找到模型文件: {MODEL_PATH}\n"
            f"请先下载预训练 MACE-Field 模型：\n"
            f"  {MODEL_URL}\n"
            f"（网络慢/下载损坏时可用代理： https://gh-proxy.com/{MODEL_URL} ，\n"
            f"  或使用 16 线程下载器 C:\\Users\\Lenovo\\AppData\\Local\\Temp\\mace_dl.py ）\n"
        )

    # ---- 构建金刚石结构 Si（2 原子原胞）----
    atoms = bulk("Si", "diamond", a=A_LATT)
    print(f"构建成功！体系包含 {len(atoms)} 个原子（金刚石原胞）。")

    # ---- 加载 MACE-Field 计算器 ----
    from mace.calculators import MACECalculator

    import torch
    if DEVICE == "cuda" and not torch.cuda.is_available():
        print("未检测到 GPU，改用 CPU。")
        device = "cpu"
    else:
        device = DEVICE

    calc = MACECalculator(
        model_paths=MODEL_PATH,
        model_type="MACEField",
        head=HEAD,
        device=device,
        default_dtype=DEFAULT_DTYPE,
    )
    print(f"MACE-Field 模型加载成功！可用 head: {calc.available_heads}")
    print(f"请求 head: {HEAD} -> 实际使用 head: {calc.head}")
    if calc.head != HEAD:
        print(f"警告：请求的 head '{HEAD}' 不存在，已回退到 '{calc.head}'。")

    atoms.calc = calc

    # ---- （可选）弛豫晶胞：晶格常数 + 原子位置 ----
    if RELAX_CELL:
        from ase.filters import FrechetCellFilter
        from ase.optimize import BFGS

        print("\n开始晶胞弛豫 (FrechetCellFilter + BFGS)...")
        opt = BFGS(FrechetCellFilter(atoms))
        opt.run(fmax=FMAX)
        # 2 原子金刚石原胞 -> 常规立方晶格常数 a = (4*V)^(1/3)
        a_relaxed = (4.0 * atoms.get_volume()) ** (1.0 / 3.0)
        print(f"晶胞弛豫完成！常规立方晶格常数 a = {a_relaxed:.4f} Angstrom")

    # ---- 触发一次能量计算，填充极化率 / Born 电荷 / 极化 ----
    _ = atoms.get_potential_energy()

    chi = reshape_sym(calc.results["polarizability"])     # 电子极化率（无量纲）
    becs = reshape_becs(calc.results["becs"], len(atoms)) # Born 有效电荷 (e)
    pol = np.asarray(calc.results["polarization"])        # 极化 (3,)

    # ---- Gamma 点力常数矩阵 (Hessian)，单位 eV/Angstrom^2 ----
    hessian = coerce_hessian_to_cartesian_matrix(
        np.asarray(calc.get_hessian(atoms=atoms), dtype=float), len(atoms)
    )
    hessian = 0.5 * (hessian + hessian.T)

    # ---- 计算三种介电张量 ----
    eps_inf, eps_ion, eps_static = dielectric_from(
        chi, becs, hessian, atoms.get_volume()
    )

    # ---- 输出 ----
    print("\n" + "=" * 60)
    print("                晶体硅 (Si) 介电常数预测")
    print("=" * 60)

    print(f"\nBorn 有效电荷 Z* (单位 e，每个原子一个 3x3):")
    for k, z in enumerate(becs):
        print(f"  原子 {k}:\n{np.round(z, 3)}")

    print(f"\n电子极化率 chi (无量纲):\n{np.round(chi, 4)}")

    print("\n介电张量 (3x3):")
    print(f"  eps_inf (电子/高频):\n{np.round(eps_inf, 3)}")
    print(f"  eps_ion (离子):\n{np.round(eps_ion, 3)}")
    print(f"  eps_static (静态/总):\n{np.round(eps_static, 3)}")

    print("\n" + "-" * 60)
    print("标量介电常数 (张量主对角平均):")
    print("-" * 60)
    print(f"  电子介电常数  eps_inf    = {scalar(eps_inf):.3f}   (参考 ~13.0)")
    print(f"  离子介电常数  eps_ion    = {scalar(eps_ion):.3f}   (参考 0, 硅 Z*=0)")
    print(f"  总静态介电常数 eps_static = {scalar(eps_static):.3f}   (参考 ~13.0)")
    print("=" * 60)

    # 简单诊断：力常数矩阵最小本征值（应接近 0，3 个平移模；若明显为负说明结构未弛豫到平衡）
    wmin = np.min(np.linalg.eigvalsh(hessian))
    fmax = float(np.abs(calc.results["forces"]).max())
    print(f"\n诊断: 最大受力 {fmax:.3e} eV/Angstrom, "
          f"力常数矩阵最小本征值 {wmin:.3e} eV/Angstrom^2")
    # 3 个声学(平移)模的本征值应为 0；float32 下会有 ~1e-7 量级的数值噪声。
    if wmin < -1e-3:
        print("警告: 最小本征值明显为负，说明存在虚频/结构不稳定，请检查是否弛豫到位。")
    else:
        print("结论: 最小本征值 ≈ 0（数值噪声），结构已处于平衡，介电常数结果可靠。")


if __name__ == "__main__":
    main()
