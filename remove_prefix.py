import shutil
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('path')
parser.add_argument('prefix')

args = parser.parse_args()
c = 0

for fpath in os.listdir(args.path):
    if fpath.startswith(args.prefix) and not fpath.endswith('.json'):
        os.rename(os.path.join(args.path, fpath), os.path.join(args.path, fpath.replace(args.prefix,"")))
        c += 1

print("Renamed {} files in '{}'".format(c, args.path))
