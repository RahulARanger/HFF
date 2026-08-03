bash scripts/download_brats2019.sh --output-dir dataset/brats2019
bash scripts/slicer.sh --dataset-root dataset/brats2019/extracted/MICCAI_BraTS_2019_Data_Training --output-dir dataset/brats2019/splits/explore  --limit 20
python scripts/generate_low_freq.py --path dataset/brats2019/splits/explore
bash scripts/generate_high_freq.sh --path dataset/brats2019/splits/explore

WANDB_MODE=offline python train.py \
  --train_list "dataset/brats2019/splits/explore/train.txt" \
  --val_list "dataset/brats2019/splits/explore/validation.txt" \
  --dataset_name brats19 \
  --class_type et \
  --num_epochs 1 \
  --batch_size 1



  WANDB_MODE=offline python train.py \
  --train_list "dataset/brats2019/splits/explore/train.txt" \
  --val_list "dataset/brats2019/splits/explore/validation.txt" \
  --dataset_name brats19 \
  --class_type et \
  --num_epochs 350 \
  --batch_size 1