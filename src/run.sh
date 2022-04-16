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
#SBATCH --output=./vs-%j.txt

set -x
. /data/sls/scratch/share-201907/slstoolchainrc
source ../venv-vs/bin/activate
export TORCH_HOME=./

model=eff_mean
model_size=0
imagenet_pretrain=False

lr=1e-4
freqm=48
timem=192
mixup=0
batch_size=100

data_dir=../data
exp_dir=../exp/vocalsound-${model}-${model_size}-im${imagenet_pretrain}-${lr}-${freqm}-${timem}-${mixup}-r1
mkdir -p exp_dir

CUDA_CACHE_DISABLE=1 python run.py --lr $lr --b $batch_size --n_class 6 --n-epochs 30 \
--freqm $freqm --timem $timem --mixup ${mixup} \
--data-train ${data_dir}/datafiles/tr.json --data-val ${data_dir}/datafiles/val.json --label-csv ${data_dir}/class_labels_indices.csv --exp-dir $exp_dir \
--model ${model} --model_size ${model_size} --imagenet_pretrain ${imagenet_pretrain} --save_model True \
--n-print-steps 100 --num-workers 8