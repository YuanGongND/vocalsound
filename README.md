# VocalSound: A Dataset for Improving Human Vocal Sounds Recognition
 - [Introduction](#Introduction)
 - [Citing](#Citing)  
 - [Download VocalSound](#Download-VocalSound)
 - [Leaderboard](#Leaderboard)
 - [Dataset Details](#Dataset-Details)
 - [Baseline Experiment](#Baseline-Experiment)
 - [Contact](#Contact)
 
## Introduction  

<p align="center"><img src="https://raw.githubusercontent.com/YuanGongND/vocalsound/main/figure/vs_poster.png?token=GHSAT0AAAAAABHNWFG2TB6OG5DVDCY3Y7HSYTJR4RQ" alt="VocalSound Poster" width="1050"/></p>

**VocalSound** is a **free** dataset consisting of *21,024* crowdsourced recordings of **laughter, sighs, coughs, throat clearing, sneezes, and sniffs** from *3,365* unique subjects. The VocalSound dataset also contains meta information such as **speaker age, gender, native language, country, and health condition**.

This repository contains the official code of the data preparation and baseline experiment in the ICASSP paper [VocalSound: A Dataset for Improving Human Vocal Sounds Recognition](https://arxiv.org/abs/2205.03433) (Yuan Gong, Jin Yu, and James Glass; MIT & Signify). Specifically, we provide an **extremely simple one-click** *Google Colab* script [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YuanGongND/vocalsound/blob/main/colab/VocalSound.ipynb) for the baseline experiment, no GPU / local data downloading is needed.

The dataset is ideal for:

- Build vocal sound recognizer.
- Research on removing model bias on various speaker groups.
- Evaluate pretrained models (e.g., those trained with AudioSet) on vocal sound classification to check their generalization ability.
- Combine with existing large-scale general audio dataset to improve the vocal sound recognition performance.


## Citing  
Please cite our paper(s) if you find the VocalSound dataset and code useful. The first paper proposes introduces the VocalSound dataset and the second paper describes the training pipeline and model we used for the baseline experiment.   
```
@INPROCEEDINGS{gong_vocalsound,
  author={Gong, Yuan and Yu, Jin and Glass, James},
  booktitle={ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Vocalsound: A Dataset for Improving Human Vocal Sounds Recognition}, 
  year={2022},
  pages={151-155},
  doi={10.1109/ICASSP43922.2022.9746828}}
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

## Download VocalSound
The VocalSound dataset can be downloaded as a single .zip file. Click the link to download with a browser *or* use the `wget` command.

[**Sample Recordings** (Listen to it without downloading)](https://drive.google.com/drive/folders/1NGdHO34aTcBY2pFZHoHZKAcfepYV9Wnf?usp=sharing)

[**VocalSound 44.1kHz Version** (4.5 GB)](https://www.dropbox.com/s/ybgaprezl8ubcce/vs_release_44k.zip?dl=1)

```
wget -O vocalsound_44k.zip https://www.dropbox.com/s/ybgaprezl8ubcce/vs_release_44k.zip?dl=1
```

[**VocalSound 16kHz Version** (1.7 GB, used in our baseline experiment)](https://www.dropbox.com/s/c5ace70qh1vbyzb/vs_release_16k.zip?dl=1)

```
wget -O vocalsound_16k.zip https://www.dropbox.com/s/c5ace70qh1vbyzb/vs_release_16k.zip?dl=1
```
***
(Mirror Links on Tencent Weiyun) 腾讯微云下载链接: [试听24个样本](https://share.weiyun.com/31910dkK) ｜[16kHz版本](https://share.weiyun.com/JUX8OTMg) ｜[44.1kHz版本](https://share.weiyun.com/JBLbmjs6)
***
If you plan to reproduce our baseline experiments using our *Google Colab* script, you do **NOT** need to download it manually, our script will download and process the 16kHz version automatically.

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/80x15.png" /></a><br />The VocalSound dataset is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.

## Leaderboard

You can find (and upload your result to) the vocalsound dataset leaderboard at [paperwithcode](https://paperswithcode.com/dataset/vocalsound).

We also plan to maintain a non-exhaustive leaderborad for the 6-class vocalsound classification task. Please open a pull request if you want your model being included. Result should be compared when pretraining dataset and settings are similar. 

| Method   | Model                    | Pretrain Dataset                                                                                                                                   | Accuracy | Code                                        |
|----------|--------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|----------|--------------------------------------------------|
| [CLAP](https://arxiv.org/abs/2206.04769)     | CNN Audio Encoder + CLIP | 128,010 audio and text pairs from 4 datasets (36,796 pairs from FSD50k, 29,646 pairs from ClothoV2, 44,292 from AudioCaps, 17,276 pairs from MACS) | 97.95    | TBD                                              |
| [Baseline](https://arxiv.org/abs/2205.03433) | EfficientNet-B0          | None                                                                                                                                               | 90.5     | [link](https://github.com/YuanGongND/vocalsound) |



## Dataset Details

```
data
├──readme.txt
├──class_labels_indices_vs.csv # include label code and name information
├──audio_16k
│  ├──f0003_0_cough.wav # female speaker, id=0003, 0=first collection (most spks only record once, but there are exceptions), cough
│  ├──f0003_0_laughter.wav
│  ├──f0003_0_sigh.wav
│  ├──f0003_0_sneeze.wav
│  ├──f0003_0_sniff.wav
│  ├──f0003_0_throatclearing.wav
│  ├──f0004_0_cough.wav # data from another female speaker 0004
│   ... (21024 files in total)
│   
├──audio_44k
│    # same recordings with those in data/data_16k, but are no downsampled
│   ├──f0003_0_cough.wav
│    ... (21024 files in total)
│
├──datafiles  # json datafiles that we use in our baseline experiment, you can ignore it if you don't use our training pipeline
│  ├──all.json  # all data
│  ├──te.json  # test data
│  ├──tr.json  # training data
│  ├──val.json  # validation data
│  └──subtest # subset of the test set, for fine-grained evaluation
│     ├──te_age1.json  # age [18-25]
│     ├──te_age2.json  # age [26-48]
│     ├──te_age3.json  # age [49-80]
│     ├──te_female.json
│     └──te_male.json
│
└──meta  # Meta information of the speakers [spk_id, gender, age, country, native language, health condition (no=no problem)]
   ├──all_meta.json  # all data
   ├──te_meta.json  # test data
   ├──tr_meta.json  # training data
   └──val_meta.json  # validation data
```
  
## Baseline Experiment

### Option 1. One-Click Google Colab Experiment [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YuanGongND/vocalsound/blob/main/colab/VocalSound.ipynb)

We provide an extremely simple one-click *Google Colab* script for the baseline experiment.

**What you need:** 
- A free google account with Google Drive free space > 5Gb
  - A (paid) Google Colab Pro plan could speed up training, but is not necessary. Free version can run the script, just a bit slower.

**What you don't need:** 
- Download VocalSound manually (The Colab script download it to your Google Drive automatically)
- GPU or any other hardware (Google Colab provides free GPUs)
- Any enviroment setting and package installation (Google Colab provides a ready-to-use environment)
- A specific operating system (You only need a web browser, e.g., Chrome)

**Please Note**
- This script is slightly different with our local code, but the performance is not impacted.
- Free Google Colab might be slow and unstable. In our test, it takes ~5 minutes to train the model for one epoch with a free Colab account.

#### To run the baseline experiment

- *Make sure your Google Drive is mounted.* You don't need to do it by yourself, but Google Colab will ask permission to acess your Google Drive when you run the script, please allow it if you want to use Google Drive.
- *Make sure GPU is enabled for Colab.* To do so, go to the top menu > Edit > Notebook settings and select GPU as Hardware accelerator.
- *Run the script.* Just press Ctrl+F9 or go to runtime menu on top and click "run all" option. That's it.

### Option 2. Run Experiment Locally

We also provide a recipe for local experiments. 

Compared with the Google Colab online script, it has following advantages:
- It can be faster and more stable than online Google Colab (free version) if you have fast GPUs.
- It is basically the original code we used for our paper, so it should reproduce the exact numbers in the paper.

**Step 1.** Clone or download this repository and set it as the working directory, create a virtual environment and install the dependencies.

```
cd vocalsound/ 
python3 -m venv venv-vs
source venv-vs/bin/activate
pip install -r requirements.txt 
```
  
**Step 2.** Download the VocalSound dataset and process it.

```
cd data/
wget https://www.dropbox.com/s/c5ace70qh1vbyzb/vs_release_16k.zip?dl=0 -O vs_release_16k.zip
unzip vs_release_16k.zip
cd ../src
python prep_data.py

# you can provide a --data_dir augment if you download the data somewhere else
# python prep_data.py --data_dir absolute_path/data
```

**Step 3**. Run the baseline experiment

```
chmod 777 run.sh
./run.sh

# or slurm user
#sbatch run.sh
```

We test both options before this release, you should get similar accuracies.

|        Accuracy (%)       |       Colab Script [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YuanGongND/vocalsound/blob/main/colab/VocalSound.ipynb)      |    [Local Script](https://github.com/YuanGongND/vocalsound/tree/main/src)   |    ICASSP Paper   |
|:-------------------------:|:------------------------:|:-----------------:|:-----------------:|
|       Validation Set      |         91.1             |        90.2       |      90.1±0.2     |
|        All Test Set       |         91.6             |        90.6       |      90.5±0.2     |
|       Test Age 18-25      |         93.4             |        92.3       |      91.5±0.3     |
|       Test Age 26-48      |         90.8             |        90.0       |      90.1±0.2     |
|       Test Age 49-80      |         92.2             |        90.2       |      90.9±1.6     |
|         Test Male         |         89.8             |        89.6       |      89.2±0.5     |
|        Test Female        |         93.4             |        91.6       |      91.9±0.1     |
|    Model Implementation   | torchvision EfficientNet | PSLA EfficientNet | PSLA EfficientNet |
|         Batch Size        |            80            |        100        |        100        |
|            GPU            |     Google Colab Free    |      4X Titan     |      4X Titan     |
| Training Time (30 Epochs) |        ~2.5 Hours        |      ~1 Hour      |      ~1 Hour      |

## Contact
If you have a question, please bring up an issue (preferred) or send me an email yuangong@mit.edu.
