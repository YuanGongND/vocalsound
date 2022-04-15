# -*- coding: utf-8 -*-
# @Time    : 4/14/22 6:22 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : prep_data.py

import os
import argparse
import json

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data_dir", type=str, default=None, help="the dataset path, please use the absolute path")

def get_immediate_files(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, name))]

def change_path(json_file_path, target_path):
    with open(json_file_path, 'r') as fp:
        data_json = json.load(fp)
    data = data_json['data']

    # change the path in the json file
    for i in range(len(data)):
        ori_path = data[i]["wav"]
        new_path = target_path + '/audio_16k/' + ori_path.split('/')[-1]
        data[i]["wav"] = new_path

    with open(json_file_path, 'w') as f:
        json.dump({'data': data}, f, indent=1)

if __name__ == '__main__':
    args = parser.parse_args()
    # if no path is provided, use the default path ../data/
    if args.data_dir == None:
        cur_path = '/'.join(os.getcwd().split('/')[:-1])
        data_dir = cur_path + '/data'

    # for train, validation, test
    json_files = get_immediate_files(data_dir + '/datafiles/')
    for json_f in json_files:
        if json_f.endswith('.json'):
            print('now processing ' + data_dir + '/datafiles/' + json_f)
            change_path(data_dir + '/datafiles/' + json_f, data_dir)

    # for subtest sets
    json_files = get_immediate_files(data_dir + '/datafiles/subtest/')
    for json_f in json_files:
        if json_f.endswith('.json'):
            print('now processing ' + data_dir + '/datafiles/subtest/' + json_f)
            change_path(data_dir + '/datafiles/subtest/' + json_f, data_dir)
