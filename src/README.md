# Lacal Script for Baseline Experiments

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
wget https://www.dropbox.com/s/fuld3z222j9t1oy/vs_release_16k.zip?dl=0 -O vs_release_16k.zip
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