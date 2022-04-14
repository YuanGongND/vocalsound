#!/bin/bash
#SBATCH -p gpu
#SBATCH -x sls-titan-[0-2]
##SBATCH -p sm
##SBATCH -x sls-1080-2,sls-sm-[1,5],sls-sm-6
#SBATCH --gres=gpu:4
#SBATCH -c 8
#SBATCH -n 1
#SBATCH --mem=48000
#SBATCH --job-name="aed-vs"
#SBATCH --output=./slurm_logs/vs-%j.txt

set -x
. /data/sls/scratch/share-201907/slstoolchainrc
base_dir=/data/sls/scratch/yuangong/vocalsound2
source /data/sls/scratch/yuangong/aed-pc/venv-new1/bin/activate
export TORCH_HOME=../pretrained_models

if [ $# -ne 2 ]
then
  effmode=effmean
  effpretrain=False
else
  effmode=$1
  effpretrain=$2
fi

lr=1e-4
bal=None
eff=0
subset=1
freqm=48
timem=192
mixup=0
att_head=1
bs=100

if [ $# -ne 1 ]
then
  aug=1.0
else
  aug=$1
fi

exp_dir=/data/sls/scratch/yuangong/vocalsound2/exp/vstestrev01-${effmode}-${eff}-${lr}--${effpretrain}-${freqm}-${timem}-${mixup}-bs$bs-r1
#if [ -d $exp_dir ]; then
#  echo 'exp exist'
#  exit
#fi
mkdir -p exp_dir

CUDA_CACHE_DISABLE=1 python ../run_vs_rev.py --lr $lr --data-train /data/sls/scratch/yuangong/vocalsound2/data/vs_processed/datafiles/tr_rev.json \
--data-val /data/sls/scratch/yuangong/vocalsound2/data/vs_processed/datafiles/val_rev.json --exp-dir $exp_dir --clean-start --train-mode \
--n-print-steps 100 --num-workers 8 --label-csv /data/sls/scratch/yuangong/vocalsound2/data/vs_processed/class_labels_indices_vs.csv --n_class 6 --n-epochs 30 --batch-size $bs \
--pretrain_mode ${effmode} --save_model True --amp False --eff_level $eff --effpretrain ${effpretrain} \
--freqm $freqm --timem $timem --mixup ${mixup} --att_head ${att_head} --bal ${bal}