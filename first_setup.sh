bash scripts/download_brats2019.sh --output-dir dataset/brats2019
bash scripts/slicer.sh --dataset-root dataset/brats2019/extracted/MICCAI_BraTS_2019_Data_Training --output-dir dataset/brats2019/splits/explore  --limit 20
python scripts/generate_low_freq.py --path dataset/brats2019/splits/explore
bash scripts/generate_high_freq.sh --path dataset/brats2019/splits/explore

qsub \
-v HFF_CONDA_ENV=hffnet,HFF_CONDA_BASE=/apps/compilers/anaconda3 \
  -- "/Data4/me_FA0498/Data_6/HFF/scripts/submit_low_freq_cpu.pbs" \
  --path /Data4/me_FA0498/Data_6/HFF/dataset/brats2019/splits/base


qsub \
-v HFF_CONDA_ENV=hffnet,HFF_CONDA_BASE=/apps/compilers/anaconda3 \
  -- "/Data4/me_FA0498/Data_6/HFF/scripts/submit_low_freq_cpu.pbs" \
  --path /Data4/me_FA0498/Data_6/HFF/dataset/brats2019/splits/explore 

qsub -v HFF_MATLAB=/Data4/me_FA0498/Data_6/MATLAB/R2026a/bin/matlab -- \
  "/Data4/me_FA0498/Data_6/HFF/scripts/submit_high_freq_cpu.pbs" \
  --path /Data4/me_FA0498/Data_6/HFF/dataset/brats2019/splits/base

qsub -v HFF_MATLAB=/Data4/me_FA0498/Data_6/MATLAB/R2026a/bin/matlab -- \
"/Data4/me_FA0498/Data_6/HFF/scripts/submit_high_freq_cpu.pbs" \
--path /Data4/me_FA0498/Data_6/HFF/dataset/brats2019/splits/explore

 ./mpm install \
    --release=R2026a \
    --destination=MATLAB/R2026a \
    --products=MATLAB


./mpm install \
  --release=R2026a \
  --destination=/Data4/me_FA0498/Data_6/MATLAB/R2026a \
  --products=Image_Processing_Toolbox


  ./mpm install \
    --release=R2026a \
    --destination=/Data4/me_FA0498/Data_6/MATLAB/R2026a
    --products=MATLAB,Statistics_and_Machine_Learning_Toolbox,Image_Processing_Toolbox

WANDB_MODE=offline python cross_train.py dataset/brats2019/splits/explore/train --run-name explore_brats19 --epochs 1 -- --dataset_name brats19 --class_type all