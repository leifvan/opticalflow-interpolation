import argparse
import os
import json
from pprint import pprint

parser = argparse.ArgumentParser()
parser.add_argument('path')
parser.add_argument('param')

args = parser.parse_args()

folder_param = dict()

try:
    for folder in os.listdir(args.path):
        fpath = os.path.join(args.path,folder)

        parampath = fpath + '\\' + folder + '_params.json'
        with open(parampath, 'r') as paramfile:
            params = json.load(paramfile)

        folder_param[folder] = params[args.param]
except KeyError:
    print("Key not found. Possible keys are:")
    print(*[k for k in params])

pprint(folder_param)