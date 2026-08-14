bash scripts/generate_high_freq.sh --path dataset/brats2019/splits/base
python scripts/generate_low_freq.py --path dataset/brats2019/splits/base


WANDB_MODE=offline python cross_train.py dataset/brats2019/splits/base/train --run-name brats_19_base --epochs 350 -- --dataset_name brats19 --resource-monitor-interval 5 --class_type all