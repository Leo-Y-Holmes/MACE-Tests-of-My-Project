import subprocess
import sys

TRAIN_FILE = "si_train.xyz"
FOUNDATION_MODEL = "mh-0"              # 多头基础模型
FOUNDATION_HEAD = "mp_pbe_refit_add"   # 用 PBE 头
NAME = "MACE_Si_mh0_finetuned_2.0"   # 微调新模型时换新名字
DEVICE = "cuda"         # 用 GPU
MAX_EPOCHS = 80        # 全量训练
VALID_FRACTION = 0.1    # 10% 验证集
BATCH_SIZE = 4           # batch 太大会爆显存
LR = 0.0008              # 微调用小学习率，避免冲散预训练权重
SWA_LR = 0.0005         # Stage Two 学习率
START_SWA = 60          # 第 60 epoch 进入 Stage Two
SWA_ENERGY_WEIGHT = 100 


def main() -> None:
    cmd = [
        sys.executable, "-m", "mace.cli.run_train",
        "--name", NAME,
        "--train_file", TRAIN_FILE,
        "--valid_fraction", str(VALID_FRACTION),
        "--foundation_model", FOUNDATION_MODEL,
        "--foundation_head", FOUNDATION_HEAD,
        "--E0s", "average",
        "--model", "MACE",
        "--r_max", "5.0",
        "--batch_size", str(BATCH_SIZE),
        "--max_num_epochs", str(MAX_EPOCHS),
        "--device", DEVICE,
        "--default_dtype", "float32",
        "--lr", str(LR),
        "--swa_lr", str(SWA_LR),
        "--swa_energy_weight", str(SWA_ENERGY_WEIGHT),
        "--start_swa", str(START_SWA),
        "--swa",
    ]
    print("运行命令：")
    print("  " + " ".join(cmd))
    subprocess.run(cmd, check=True)

    print("\n微调完成！")
    print(f"最终模型（Stage Two）保存在项目根目录：{NAME}_stagetwo.model")
    print(f"加载方式：mace_mp(model='{NAME}_stagetwo.model')")


if __name__ == "__main__":
    main()
