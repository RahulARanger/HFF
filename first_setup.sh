bash scripts/download_brats2019.sh --output-dir dataset/brats2019
bash scripts/slicer.sh --dataset-root /Users/rahul/Documents/HFF/dataset/brats2019/extracted --output-dir /Users/rahul/Documents/HFF/dataset/brats2019/splits/explore  --limit 20
python scripts/generate_low_freq.py --path dataset/brats2019/splits/explore
bash scripts/generate_high_freq.sh --path dataset/brats2019/splits/explore