# VocalSound: A Dataset for Improving Human Vocal Sounds Recognition
 - [Introduction](#Introduction)
 - [Citing](#Citing)  
 - [Download VocalSound](#Download-VocalSound)
 - [Dataset Details](#Dataset-Details)
 - [Baseline Experiment](#Baseline-Experiment)
 - [Contact](#Contact)
 
## Introduction  

**VocalSound** dataset consists of over *21,000* crowdsourced recordings of **laughter, sighs, coughs, throat clearing, sneezes, and sniffs** from *3,365* unique subjects. The VocalSound dataset also contains meta information such as **speaker age, gender, native language, country, and health condition**.

This repository contains the official code of the data preparation and baseline experiment in the ICASSP paper [VocalSound: A Dataset for Improving Human Vocal Sounds Recognition](https://arxiv.org/abs/dummy) (Yuan Gong, Jin Yu, and James Glass; MIT & Signify).  

## Citing  
Please cite our paper(s) if you find the VocalSound dataset and code useful. The first paper proposes introduces the VocalSound dataset and the second paper describes the training pipeline and model we used for the baseline experiment.   
```
@inproceedings{gong2022vocalsound,  
  author={Gong, Yuan and Yu, Jin and Glass, James},
  title={VOCALSOUND: A DATASET FOR IMPROVING HUMAN VOCAL SOUNDS RECOGNITION},
  booktitle={ICASSP},
  year={2022}
}
```
```  
@ARTICLE{gong_psla, 
    author={Gong, Yuan and Chung, Yu-An and Glass, James},
    title={PSLA: Improving Audio Tagging with Pretraining, Sampling, Labeling, and Aggregation}, 
    journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},  
    year={2021}, 
    doi={10.1109/TASLP.2021.3120633}
}
```  

##Download VocalSound

##Dataset Details
  
## Baseline-Experiment

Step 1. Clone or download this repository and set it as the working directory, create a virtual environment and install the dependencies.

```
cd ast/ 
python3 -m venv venvast
source venvast/bin/activate
pip install -r requirements.txt 
```
  
Step 2. Test the AST model.

```python
ASTModel(label_dim=527, \
         fstride=10, tstride=10, \
         input_fdim=128, input_tdim=1024, \
         imagenet_pretrain=True, audioset_pretrain=False, \
         model_size='base384')
```  

**Parameters:**\
`label_dim` : The number of classes (default:`527`).\
`fstride`:  The stride of patch spliting on the frequency dimension, for 16\*16 patchs, fstride=16 means no overlap, fstride=10 means overlap of 6 (used in the paper). (default:`10`)\
`tstride`:  The stride of patch spliting on the time dimension, for 16*16 patchs, tstride=16 means no overlap, tstride=10 means overlap of 6 (used in the paper). (default:`10`)\
`input_fdim`: The number of frequency bins of the input spectrogram. (default:`128`)\
`input_tdim`: The number of time frames of the input spectrogram. (default:`1024`, i.e., 10.24s)\
`imagenet_pretrain`: If `True`, use ImageNet pretrained model. (default: `True`, we recommend to set it as `True` for all tasks.)\
`audioset_pretrain`: If`True`,  use full AudioSet And ImageNet pretrained model. Currently only support `base384` model with `fstride=tstride=10`. (default: `False`, we recommend to set it as `True` for all tasks except AudioSet.)\
`model_size`: The model size of AST, should be in `[tiny224, small224, base224, base384]` (default: `base384`).

**Input:** Tensor in shape `[batch_size, temporal_frame_num, frequency_bin_num]`. Note: the input spectrogram should be normalized with dataset mean and std, see [here](https://github.com/YuanGongND/ast/blob/102f0477099f83e04f6f2b30a498464b78bbaf46/src/dataloader.py#L191). \
**Output:** Tensor of raw logits (i.e., without Sigmoid) in shape `[batch_size, label_dim]`.

``` 
cd ast/src
python
```  

```python
import os 
import torch
from models import ASTModel 
# download pretrained model in this directory
os.environ['TORCH_HOME'] = '../pretrained_models'  
# assume each input spectrogram has 100 time frames
input_tdim = 100
# assume the task has 527 classes
label_dim = 527
# create a pseudo input: a batch of 10 spectrogram, each with 100 time frames and 128 frequency bins 
test_input = torch.rand([10, input_tdim, 128]) 
# create an AST model
ast_mdl = ASTModel(label_dim=label_dim, input_tdim=input_tdim, imagenet_pretrain=True)
test_output = ast_mdl(test_input) 
# output should be in shape [10, 527], i.e., 10 samples, each with prediction of 527 classes. 
print(test_output.shape)  
```  

## ESC-50 Recipe  
The ESC-50 recipe is in `ast/egs/esc50/run_esc.sh`, the script will automatically download the ESC-50 dataset and resample it to 16kHz, then run standard 5-cross validation and report the result.
The recipe was tested on 4 GTX TITAN GPUs with 12GB memory. 
The result is saved in `ast/egs/esc50/exp/yourexpname/acc_fold.csv` (the accuracy of fold 1-5 and the averaged accuracy), you can also check details in `result.csv` and `best_result.csv` (accuracy, AUC, loss, etc of each epoch / best epoch).
We attached our log file in `ast/egs/esc50/test-esc50-f10-t10-p-b48-lr1e-5`, the model achieves `95.75%` accuracy.

To run the recipe, simply comment out `. /data/sls/scratch/share-201907/slstoolchainrc` in `ast/egs/esc50/run_esc.sh`, adjust the path if needed, and run:
``` 
cd ast/egs/esc50
(slurm user) sbatch run_esc50.sh
(local user) ./run_esc50.sh
```  

## Speechcommands V2 Recipe  
The Speechcommands recipe is in `ast/egs/speechcommands/run_sc.sh`, the script will automatically download the Speechcommands V2 dataset, train an AST model on the training set, validate it on the validation set, and evaluate it on the test set.
The recipe was tested on 4 GTX TITAN GPUs with 12GB memory. 
The result is saved in `ast/egs/speechcommands/exp/yourexpname/eval_result.csv` in format `[val_acc, val_AUC, eval_acc, eval_AUC]`, you can also check details in `result.csv` (accuracy, AUC, loss, etc of each epoch).
We attached our log file in `ast/egs/speechcommends/test-speechcommands-f10-t10-p-b128-lr2.5e-4-0.5-false`, the model achieves `98.12%` accuracy.

To run the recipe, simply comment out `. /data/sls/scratch/share-201907/slstoolchainrc` in `ast/egs/esc50/run_sc.sh`, adjust the path if needed, and run:
``` 
cd ast/egs/speechcommands
(slurm user) sbatch run_sc.sh
(local user) ./run_sc.sh
```  

## Audioset Recipe  
Audioset is a little bit more complex, you will need to prepare your data json files (i.e., `train_data.json` and `eval_data.json`) by your self.
The reason is that the raw wavefiles of Audioset is not released and you need to download them by yourself. We have put a sample json file in `ast/egs/audioset/data/datafiles`, please generate files in the same format (You can also refer to `ast/egs/esc50/prep_esc50.py` and `ast/egs/speechcommands/prep_sc.py`.). Please keep the label code consistent with `ast/egs/audioset/data/class_labels_indices.csv`.

Once you have the json files, you will need to generate the sampling weight file of your training data (please check our [PSLA paper](https://arxiv.org/abs/2102.01243) to see why it is needed).
```
cd ast/egs/audioset
python gen_weight_file.py ./data/datafiles/train_data.json
```

Then you just need to change the `tr_data` and `te_data` in `/ast/egs/audioset/run.sh` and then 
``` 
cd ast/egs/audioset
(slurm user) sbatch run.sh
(local user) ./run.sh
```  
You should get a model achieves `0.448 mAP` (without weight averaging) and `0.459` (with weight averaging). This is the best **single** model reported in the paper. 
The result of each epoch is saved in `ast/egs/audioset/exp/yourexpname/result.csv` in format `[mAP, mAUC, precision, recall, d_prime, train_loss, valid_loss, cum_mAP, cum_mAUC, lr]`
, where `cum_` results are the checkpoint ensemble results (i.e., averaging the prediction of checkpoint models of each epoch, please check our [PSLA paper](https://arxiv.org/abs/2102.01243) for details). The result of weighted averaged model is saved in `wa_result.csv` in format `[mAP, AUC, precision, recall, d-prime]`. We attached our log file in `ast/egs/audioset/test-full-f10-t10-pTrue-b12-lr1e-5/`, the model achieves `0.459` mAP.

In order to reproduce ensembe results of `0.475 mAP` and `0.485 mAP`, please train 3 models use the same setting (i.e., repeat above three times) and train 6 models with different `tstride` and `fstride`, and average the output of the models. Please refer to `ast/egs/audioset/ensemble.py`. We attached our ensemble log in `/ast/egs/audioset/exp/ensemble-s.log` and `ensemble-m.log`. You can use our pretrained models (see below) to test ensemble result.

We use `16kHz` for our experiments. Note that you might get a slightly different result with us due to the YouTube videos are being removed with the time and your downloaded version might be different with us. We provide our evaluation audio ids in `ast/egs/audioset/data/sanity_check/our_as_eval_id.csv`. And please note that in order to compre with the PSLA paper, for the **balanced training set** experiments (with results of `0.347 mAP` and `0.378 mAP`), we use the enhanced label set (i.e., a label set that is modified by an algorithm, please see the PSLA paper for detail). So if you train with the original label set for the balanced training set, you will get a slightly worse result. However, we do not use enhanced label set for **full AudioSet experiments**, i.e., for the `0.459 mAP` (single) and `0.485 mAP` (ensemble) results, we use exactly same data and label with the official release, so you should be able to reproduce that. 

## Pretrained Models
We provide full AudioSet pretrained models and Speechcommands-V2-35 pretrained model.
1. [Full AudioSet, 10 tstride, 10 fstride, with Weight Averaging (0.459 mAP)](https://www.dropbox.com/s/ca0b1v2nlxzyeb4/audioset_10_10_0.4593.pth?dl=1)
2. [Full AudioSet, 10 tstride, 10 fstride, without Weight Averaging, Model 1 (0.450 mAP)](https://www.dropbox.com/s/1tv0hovue1bxupk/audioset_10_10_0.4495.pth?dl=1)
3. [Full AudioSet, 10 tstride, 10 fstride, without Weight Averaging, Model 2  (0.448 mAP)](https://www.dropbox.com/s/6u5sikl4b9wo4u5/audioset_10_10_0.4483.pth?dl=1)
4. [Full AudioSet, 10 tstride, 10 fstride, without Weight Averaging, Model 3  (0.448 mAP)](https://www.dropbox.com/s/kt6i0v9fvfm1mbq/audioset_10_10_0.4475.pth?dl=1)
5. [Full AudioSet, 12 tstride, 12 fstride, without Weight Averaging, Model (0.447 mAP)](https://www.dropbox.com/s/snfhx3tizr4nuc8/audioset_12_12_0.4467.pth?dl=1)
6. [Full AudioSet, 14 tstride, 14 fstride, without Weight Averaging, Model (0.443 mAP)](https://www.dropbox.com/s/z18s6pemtnxm4k7/audioset_14_14_0.4431.pth?dl=1)
7. [Full AudioSet, 16 tstride, 16 fstride, without Weight Averaging, Model (0.442 mAP)](https://www.dropbox.com/s/mdsa4t1xmcimia6/audioset_16_16_0.4422.pth?dl=1)

8. [Speechcommands V2-35, 10 tstride, 10 fstride, without Weight Averaging, Model (98.12% accuracy on evaluation set)](https://www.dropbox.com/s/q0tbqpwv44pquwy/speechcommands_10_10_0.9812.pth?dl=1)

If you want to finetune AudioSet-pretrained AST model on your task, you can simply set the `audioset_pretrain=True` when you create the AST model, it will automatically download model 1 (`0.459 mAP`). In our ESC-50 recipe, AudioSet pretraining is used.

If you want to reproduce ensemble experiments, you can download these models at one click using `ast/egs/audioset/download_models.sh`. Ensemble model 2-4 achieves `0.475 mAP`, Ensemble model 2-7 achieves and `0.485 mAP`. Once you download the model, you can try `ast/egs/audioset/ensemble.py`, you need to change the `eval_data_path` and `mdl_list` to run it. We attached our ensemble log in `/ast/egs/audioset/exp/ensemble-s.log` and `ensemble-m.log`.

Please  note that we use `16kHz` audios for training and test (for all AudioSet, SpeechCommands, and ESC-50), so if you want to use the pretrained model, please prepare your data in `16kHz`.

(Note: the above links are Dropbox direct links (i.e., can be downloaded by wget) and should work for most users. For users having issue downloading with the above Dropbox links, it is recommended to use a VPN or use the [OneDrive links](https://mitprod-my.sharepoint.com/:f:/g/personal/yuangong_mit_edu/ErLKkiP-GwVMgdsCeGEjsmoBMtGvXMkX3tCj5_I0E7ikNA?e=JE9Om8) or [腾讯微云链接们](https://share.weiyun.com/xRGK6zmg), however, OneDrive and 腾讯微云 links are not direct link, please manually download the `audioset_10_10_0.4593.pth`[[OneDrive]](https://mitprod-my.sharepoint.com/:u:/g/personal/yuangong_mit_edu/EWrY3raql55CqxZNV3cVSkABaoU7pXQxAeJXudE1PTNzQg?e=gwEICj) [[腾讯微云]](https://share.weiyun.com/kcmk2KHw) and place it in `ast/pretrained_models` if you want to set `audioset_pretrain=True` because the wget link in the `ast/src/models/ast_models.py` would fail if you cannot connect to Dropbox.) 

## Use Pretrained Model For Downstream Tasks

You can use the pretrained AST model for your own dataset. There are two ways to doing so.

You can of course only take ``ast/src/models/ast_models.py``, set ``audioset_pretrain=True``, and use it with your training pipeline, the only thing need to take care of is the input normalization, we normalize our input to 0 mean and 0.5 std. To use the pretrained model, you should roughly normalize the input to this range. You can check ``ast/src/get_norm_stats.py`` to see how we compute the stats, or you can try using our AudioSet normalization ``input_spec = (input_spec + 4.26) / (4.57 * 2)``. Using your own training pipeline might be easier if you already have a good one.
Please note that AST needs smaller learning rate (we use 10 times smaller learning rate than our CNN model proposed in the [PSLA paper](https://arxiv.org/abs/2102.01243)) and converges faster, so please search the learning rate and learning rate scheduler for your task. 

If you want to use our training pipeline, you would need to modify below for your new dataset.
1. You need to create a json file, and a label index for your dataset, see ``ast/egs/audioset/data/`` for an example.
2. In ``/your_dataset/run.sh``, you need to specify the data json file path, the SpecAug parameters (``freqm`` and ``timem``, we recommend to mask 48 frequency bins out of 128, and 20% of your time frames), the mixup rate (i.e., how many samples are mixup samples), batch size, initial learning rate, etc. Please see ``ast/egs/[audioset,esc50,speechcommands]/run.sh]`` for samples.
3. In ``ast/src/run.py``, line 60-65, you need to add the normalization stats, the input frame length, and if noise augmentation is needed for your dataset. Also take a look at line 101-127 if you have a seperate validation set. For normalization stats, you need to compute the mean and std of your dataset (check ``ast/src/get_norm_stats.py``) or you can try using our AudioSet normalization ``input_spec = (input_spec + 4.26) / (4.57 * 2)``.
4. In ``ast/src/traintest.`` line 55-82, you need to specify the learning rate scheduler, metrics, warmup setting and the optimizer for your task.

To summarize, to use our training pipeline, you need to creat data files and modify the above three python scripts. You can refer to our ESC-50 and Speechcommands recipes.

Also, please note that we use `16kHz` audios for the pretrained model, so if you want to use the pretrained model, please prepare your data in `16kHz`.


 ## Contact
If you have a question, please bring up an issue (preferred) or send me an email yuangong@mit.edu.
