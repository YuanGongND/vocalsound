# Author: David Harwath
import argparse
import os
import pickle
import sys
from collections import OrderedDict
import time
import torch
import shutil
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
import dataloaders
import models
from traintest_vs import train, validate
import ast
from torch.utils.data import WeightedRandomSampler
import numpy as np
import random

print("I am process %s, running on %s: starting (%s)" % (
        os.getpid(), os.uname()[1], time.asctime()))

# I/O args
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data-train", type=str, default='', help="training data json")
parser.add_argument("--data-val", type=str, default='', help="validation data json")
parser.add_argument("--label-csv", type=str, default=os.path.join(basepath, 'utilities/class_labels_indices_coarse.csv'), help="csv with class labels")
parser.add_argument("--exp-dir", type=str, default="", help="directory to dump experiments")
parser.add_argument("--resume", action="store_true", dest="resume", help="load from exp_dir if True")
parser.add_argument("--train-mode", action="store_true", dest="train_mode", help="Train the models; otherwise, perform validation (default behavior)")
parser.add_argument("--clean-start", action="store_true", help="Clobber the experiment directory if it already exists")
# training and optimization args
parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('-b', '--batch-size', default=60, type=int, metavar='N', help='mini-batch size (default: 100)')
parser.add_argument('-w', '--num-workers', default=8, type=int, metavar='NW', help='# of workers for dataloading (default: 8)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--lr-decay', default=40, type=int, metavar='LRDECAY', help='Divide the learning rate by 10 every lr_decay epochs')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-7, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument("--n-epochs", type=int, default=1, help="number of maximum training epochs")
parser.add_argument("--n-print-steps", type=int, default=1, help="number of steps to print statistics")
# models args
parser.add_argument("--audio-models", type=str, default="DavenetPeakVQ", help="audio models architecture", choices=["DavenetPeakVQ"])
parser.add_argument("--image-models", type=str, default="VGG16", help="image models architecture", choices=["VGG16"])
parser.add_argument("--seed-dir", type=str, default="", help="Load image and audio models weights from a seed models. Overrides using an image models pretrained on ImageNet")
parser.add_argument("--margin", type=float, default=1.0, help="Margin paramater for triplet loss")
parser.add_argument("--embedding_dim", type=int, default=1024, help="Cross modal embedding dimension")
parser.add_argument("--n_class", type=int, default=17, help="number of classes")
parser.add_argument("--balance_class", type=str, default='none', help="path to the weight file")
parser.add_argument("--pretrained_path", type=str, default='none', help="path to the pretrained models")
parser.add_argument("--apc_rnn_layer", type=int, default=0, help="which layer of rnn to feed to succeeding layers")
parser.add_argument("--apc_trainable", type=str, default='True', help="if set APC layers as trainable")
parser.add_argument("--pretrain_mode", type=str, default='apc2', help="which pre-trained models to use")
parser.add_argument("--data_setting", type=str, default='none', help="quick setting of data used")
parser.add_argument('--save_model', help='save the models or not', type=ast.literal_eval)
parser.add_argument('--amp', help='True or False flag, input should be either "True" or "False".', type=ast.literal_eval)
parser.add_argument("--eff_level", type=int, default=0, help="which level of eff to use, the larger number, the more complex")
parser.add_argument('--esc', help='If doing an ESC exp, which will have some different behabvior', type=ast.literal_eval, default='False')
parser.add_argument('--effpretrain', help='if use pretrained imagenet efficient net', type=ast.literal_eval, default='True')
parser.add_argument('--freqm', help='frequency mask max length', type=int, default=0)
parser.add_argument('--timem', help='time mask max length', type=int, default=0)
parser.add_argument("--mixup", type=float, default=0, help="how many (0-1) samples need to be mixup during training")
parser.add_argument("--lr_patience", type=int, default=2, help="how many epoch to wait to reduce lr if mAP doesn't improve")
parser.add_argument("--att_head", type=int, default=2, help="number of attention heads")
parser.add_argument("--cont", type=str, default=None, help="start from a pretrained models, provide the path if use")
parser.add_argument("--bal", type=str, default=None, help="use balanced sampling or not")

args = parser.parse_args()
resume = args.resume

if args.resume:
    assert(bool(args.exp_dir))
    with open("%s/args.pkl" % args.exp_dir, "rb") as f:
        args = pickle.load(f)
args.resume = resume

print(args)
audio_conf = {'num_mel_bins':128, 'target_length': 512, 'freqm':args.freqm, 'timem':args.timem, 'mixup':args.mixup}

print('data')

print(args.data_train)
print(args.data_val)

pretrain_mode = args.pretrain_mode

# min efficientnet models
audio_conf = {'num_mel_bins': 128, 'target_length': 512, 'freqm': args.freqm, 'timem': args.timem,
              'mixup': args.mixup, 'mode': 'train'}
print('now train a min models')
if args.bal == 'bal':
    print('balanced sampler is being used')
    samples_weight = np.loadtxt(args.data_train[:-5]+'_weight.csv', delimiter=',')
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

    train_loader = torch.utils.data.DataLoader(
        dataloaders.VSDataset(args.data_train, label_csv=args.label_csv,
                                    audio_conf=audio_conf, raw_wav_mode=False, specaug=True),
        batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True)
else:
    print('balanced sampler is not used')
    train_loader = torch.utils.data.DataLoader(
        dataloaders.VSDataset(args.data_train, label_csv=args.label_csv,
                                    audio_conf=audio_conf, raw_wav_mode=False, specaug=True),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

val_audio_conf = {'num_mel_bins': 128, 'target_length': 512, 'mixup': 0, 'mode': 'test'}

val_loader = torch.utils.data.DataLoader(
    dataloaders.VSDataset(args.data_val, label_csv=args.label_csv,
                                audio_conf=val_audio_conf, raw_wav_mode=False),
    batch_size=200, shuffle=False, num_workers=args.num_workers, pin_memory=True)

if pretrain_mode == 'effmatt':
    audio_model = models.EffNetFullAttention(label_dim=args.n_class, level=args.eff_level, pretrain=args.effpretrain)
elif pretrain_mode == 'effsatt':
    audio_model = models.EffNetRevAttention(label_dim=args.n_class, level=args.eff_level, pretrain=args.effpretrain)
elif pretrain_mode == 'effmean':
    audio_model = models.EffNetMean(label_dim=args.n_class, level=args.eff_level, pretrain=args.effpretrain)
elif pretrain_mode == 'mbnet':
    audio_model = models.MBNet(label_dim=args.n_class, level=args.eff_level, pretrain=args.effpretrain)
else:
    raise ValueError('Model Unrecognized')

# start training
if os.path.exists(args.exp_dir):
    if args.clean_start:
        print("Deleting existing experiment directory %s" % args.exp_dir)
        shutil.rmtree(args.exp_dir)
    else:
        print("Experiment directory %s already exists; specify a different directory or run with a clean start." % args.exp_dir)
        sys.exit()
print("\nCreating experiment directory: %s" % args.exp_dir)
os.makedirs("%s/models" % args.exp_dir)
with open("%s/args.pkl" % args.exp_dir, "wb") as f:
    pickle.dump(args, f)

target_path = args.exp_dir + '/src'
os.mkdir(target_path)
os.system('cp /data/sls/scratch/yuangong/vocalsound2/src/slurm_scripts/train_vs_rev.sh ' + target_path)
os.system('cp /data/sls/scratch/yuangong/vocalsound2/src/*.py ' + target_path)
os.system('cp -r /data/sls/scratch/yuangong/vocalsound2/src/models ' + target_path)
os.system('cp /data/sls/scratch/yuangong/vocalsound2/src/dataloaders/vs_dataset.py ' + target_path)
print('Now starting training for {:d} epochs'.format(args.n_epochs))
train(audio_model, train_loader, val_loader, args)
# else:
#     validate(audio_model, val_loader, args)

# test on the test set and sub-test set, model selected on the validation set
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sd = torch.load(args.exp_dir + '/models/best_audio_model.pth', map_location=device)
audio_model = torch.nn.DataParallel(audio_model)
audio_model.load_state_dict(sd)

all_res = []

# best model on the validation set, repeat to confirm
stats, _ = validate(audio_model, val_loader, args, 'valid_set')
# note it is NOT mean of class-wise accuracy
val_acc = stats[0]['acc']
val_mAUC = np.mean([stat['auc'] for stat in stats])
print('---------------evaluate on the validation set---------------')
print("Accuracy: {:.6f}".format(val_acc))
all_res.append(val_acc)

# test the model on the evaluation set
data_eval_list = ['te_rev.json', 'subtest/te_age1_rev.json', 'subtest/te_age2_rev.json', 'subtest/te_age3_rev.json', 'subtest/te_female_rev.json', 'subtest/te_male_rev.json']
eval_name_list = ['all_test', 'age1', 'age2', 'age3', 'female', 'male']

for idx, cur_eval in enumerate(data_eval_list):
    cur_eval = '/data/sls/scratch/yuangong/vocalsound2/data/vs_processed/datafiles/' + cur_eval

    eval_loader = torch.utils.data.DataLoader(
        dataloaders.VSDataset(cur_eval, label_csv=args.label_csv, audio_conf=val_audio_conf),
        batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    stats, _ = validate(audio_model, eval_loader, args, eval_name_list[idx])
    eval_acc = stats[0]['acc']
    all_res.append(eval_acc)
    print('---------------evaluate on {:s}---------------'.format(eval_name_list[idx]))
    print("Accuracy: {:.6f}".format(eval_acc))

np.savetxt(args.exp_dir + '/all_eval_result.csv', all_res)