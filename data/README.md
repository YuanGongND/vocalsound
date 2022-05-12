## Download VocalSound

The VocalSound dataset can be downloaded as a single .zip file:

[**Sample Recordings** (Listen to it without downloading)](https://drive.google.com/drive/folders/1NGdHO34aTcBY2pFZHoHZKAcfepYV9Wnf?usp=sharing)

[**VocalSound 44.1kHz Version** (4.5 GB)](https://www.dropbox.com/s/ybgaprezl8ubcce/vs_release_44k.zip?dl=1)

[**VocalSound 16kHz Version** (1.7 GB, used in our baseline experiment)](https://www.dropbox.com/s/c5ace70qh1vbyzb/vs_release_16k.zip?dl=1)

(Mirror Links) 腾讯微云下载链接: [试听24个样本](https://share.weiyun.com/31910dkK) ｜[16kHz版本](https://share.weiyun.com/JUX8OTMg) ｜[44.1kHz版本](https://share.weiyun.com/JBLbmjs6)

If you plan to reproduce our baseline experiments using our *Google Colab* script, you do **NOT** need to download it manually, our script will download and process the 16kHz version automatically.

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/80x15.png" /></a><br />The VocalSound dataset is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.

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