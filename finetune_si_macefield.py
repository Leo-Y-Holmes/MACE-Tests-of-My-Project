"""
finetune_si_macefield.py
========================
对 MACE-Field 基础模型做"多头微调"，为纯 Si 增加一个介电头，用来把 ε∞ 从
~13.5 拉到 MP 参考值 ~13.0。

多头微调结构：
  - 新头 "si_dielectric"：在 si_dielectric.xyz（3 条 Si 介电数据，含极化率 chi
    和 Born 电荷）上训练，修正电子介电常数。
  - 回放头 "pt_head"：在 si_train.xyz（548 条 Si 能量/力）上做 pseudolabel 回放，
    防止微调"冲散"基础模型的能量/力精度。

基础模型：MACEField-MH-0-omat-dielectric.model（需与本脚本同目录）。
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))  # 脚本所在目录（MACE-field/）

NAME = "MACE_Si_macefield_finetuned"
FOUNDATION_MODEL = "MACEField-MH-0-omat-dielectric.model"
FOUNDATION_HEAD = "pt_head"

# 新介电头：3 条 Si 介电数据（同目录 si_dielectric.xyz）
HEADS = "{'si_dielectric': {'train_file': 'si_dielectric.xyz'}}"

# 回放数据：小胞抽样（≤8 原子，150 帧），避免 4.3GB 显存上 pseudolabel 的 BEC autograd OOM
PT_TRAIN_FILE = "si_replay_small.xyz"

DEVICE = "cuda"
BATCH_SIZE = 1              # RTX 3050 只有 4.3GB，batch 只能开小
MAX_EPOCHS = 50             # 介电头数据少，少量 epoch 即可
VALID_FRACTION = 0.1        # 回放头(548条)用它切验证；介电头已复制多份，不影响
LR = 0.001                  # 微调小学习率


def main() -> None:
    os.chdir(HERE)  # 切到脚本目录，确保模型/数据/输出目录的相对路径都正确

    cmd = [
        sys.executable, "-m", "mace.cli.run_train",
        "--name", NAME,
        "--heads", HEADS,
        "--valid_fraction", str(VALID_FRACTION),
        "--foundation_model", FOUNDATION_MODEL,
        "--foundation_head", FOUNDATION_HEAD,
        "--pt_train_file", PT_TRAIN_FILE,
        "--multiheads_finetuning", "True",
        "--pseudolabel_replay", "True",
        "--E0s", "foundation",
        "--model", "MACEField",
        "--loss", "universal_field",
        "--error_table", "PerAtomFieldRMSE",
        "--r_max", "5.0",
        "--batch_size", str(BATCH_SIZE),
        "--max_num_epochs", str(MAX_EPOCHS),
        "--device", DEVICE,
        "--default_dtype", "float32",
        "--lr", str(LR),
        "--energy_weight", "1.0",
        "--forces_weight", "100.0",
        "--polarization_weight", "1000.0",
        "--becs_weight", "100.0",
        "--polarizability_weight", "100.0",
        "--compute_forces", "True",
        "--compute_stress", "True",
        "--compute_polarization", "True",
        "--compute_becs", "True",
        "--compute_polarizability", "True",
    ]
    print("运行命令：")
    print("  " + " ".join(cmd))
    subprocess.run(cmd, check=True)

    print("\n微调完成！")
    print(f"最终模型：{NAME}.model（可能还有 {NAME}_stagetwo.model）")
    print("加载方式：")
    print(f"  MACECalculator(model_paths='{NAME}.model', model_type='MACEField', head='si_dielectric')")


if __name__ == "__main__":
    main()
