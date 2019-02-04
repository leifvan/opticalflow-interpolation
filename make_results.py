import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from pprint import pprint
from stnwarp import warp_channels_from_vfields, unpad_image, pad_image, plot_flow, warp_from_vfields
from utils import load_image, normalize_image
from tqdm import tqdm
from skimage.util import pad

parser = argparse.ArgumentParser()
parser.add_argument('path')
parser.add_argument('-silent', action='store_true')
parser.add_argument('-n', default=5, type=int)

args = parser.parse_args()


def save_imfig(path, im):
    fig, ax = plt.subplots(1,1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(im, cmap='gray')
    plt.savefig(path, bbox_inches='tight', dpi=300)


base_path = os.path.join(args.path,os.path.split(args.path)[-1])+"_"
print(base_path)

# load all necessary files
vfields = np.load(base_path+'vfields.npy')
inv_vfields = np.load(base_path+'inv_vfields.npy')

with open(base_path+'params.json') as paramfile:
    params = json.load(paramfile)

with open(base_path+'paths.json') as pathsfile:
    meta = json.load(pathsfile)

if not args.silent:
    pprint(params)
    pprint(meta)

color_pad = ((meta['pad'],meta['pad']),(meta['pad'],meta['pad']),(0,0))

try:
    image1 = load_image(meta['path1'], meta['scale1'], astype='rgb')
    image1_gray = load_image(meta['path1'], meta['scale1'], astype='grayscale')
except FileNotFoundError:
    print("'{}' not found, recovering from local file...".format(meta['path1']))
    image1 = load_image(base_path+'w1_0_00.png', 1, astype='rgb')
    image1_gray = load_image(base_path + 'w1_0_00.png', 1, astype='grayscale')

try:
    image2 = load_image(meta['path2'], meta['scale2'], astype='rgb')
    image2_gray = load_image(meta['path2'], meta['scale2'], astype='grayscale')
except FileNotFoundError:
    print("'{}' not found, recovering from local file...".format(meta['path1']))
    image2 = load_image(base_path+'w2_0_00.png', 1, astype='rgb')
    image2_gray = load_image(base_path+'w2_0_00.png', 1, astype='grayscale')


cut = meta['cut']

image1 = image1[:cut,:cut]
image2 = image2[:cut,:cut]
image1_gray = image1_gray[:cut,:cut]
image2_gray = image2_gray[:cut,:cut]
image1 = pad(image1, color_pad, mode='edge')
image2 = pad(image2, color_pad, mode='edge')
image1_gray = pad(image1_gray, meta['pad'], mode='edge')
image2_gray = pad(image2_gray, meta['pad'], mode='edge')


result1 = warp_from_vfields(image1_gray, vfields)
result2 = warp_from_vfields(image2_gray, inv_vfields)

num_warps = args.n
skipped = 0
num_outputs = 2*len(vfields)+num_warps*2+6

with tqdm(total=num_outputs) as pbar:

    # create pretty plots
    for i, field in enumerate(vfields):
        p = base_path+'flow_{}.png'.format(i)
        if not os.path.exists(p):
            fig, ax = plt.subplots(1, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            plot_flow(ax, np.rollaxis(field, 2))
            plt.savefig(p, bbox_inches='tight', dpi=300)
        else:
            skipped += 1
        pbar.update(1)

    for i, field in enumerate(inv_vfields):
        p = base_path+'invflow_{}.png'.format(i)
        if not os.path.exists(p):
            fig, ax = plt.subplots(1, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            plot_flow(ax, np.rollaxis(field, 2))
            plt.savefig(p, bbox_inches='tight', dpi=300)
        else:
            skipped += 1
        pbar.update(1)

    alphas = np.linspace(0., 1., num_warps, endpoint=True)

    # the long way to create a simple mask
    a = np.linspace(0, 1, vfields.shape[2], endpoint=True)
    a = np.reshape(a, (1, 1, *a.shape, 1))
    a = np.repeat(a, vfields.shape[1], axis=1)
    a = np.repeat(a, vfields.shape[0], axis=0)
    ia = a[:, :, ::-1, :]

    vfields01 = a * vfields
    vfields10 = ia * vfields
    inv_vfields01 = a * inv_vfields
    inv_vfields10 = ia * inv_vfields

    for i, alph in enumerate(alphas):
        p = base_path+'w1_{}.png'.format(i)
        if not os.path.exists(p):
            w = warp_channels_from_vfields(image1, alph * vfields)
            save_imfig(p, unpad_image(w, meta['pad']))
        else:
            skipped += 1
        pbar.update(1)

        p = base_path + 'w2_{}.png'.format(i)
        if not os.path.exists(p):
            w = warp_channels_from_vfields(image2, alph * inv_vfields)
            save_imfig(p, unpad_image(w, meta['pad']))
        else:
            skipped += 1
        pbar.update(1)

    #blend = unpad_image(np.zeros_like(image1), meta['pad'])
    #iblend = unpad_image(np.zeros_like(image1), meta['pad'])

    if not os.path.exists(base_path+'blend.png'):
        smooth1_10 = warp_channels_from_vfields(image1, vfields10)
        smooth2_01 = warp_channels_from_vfields(image2, inv_vfields01)
        blend = unpad_image(a[0] * smooth1_10 + ia[0] * smooth2_01, meta['pad'])
        save_imfig(base_path+'blend.png', blend)
    else:
        skipped += 1
    pbar.update(1)

    if not os.path.exists(base_path+'iblend.png'):
        smooth1_01 = warp_channels_from_vfields(image1, vfields01)
        smooth2_10 = warp_channels_from_vfields(image2, inv_vfields10)
        iblend = unpad_image(a[0] * smooth2_10 + ia[0] * smooth1_01, meta['pad'])
        save_imfig(base_path+'iblend.png', iblend)
    else:
        skipped += 1
    pbar.update(1)

    if not os.path.exists(base_path+'lin_blend.png'):
        lin_blend = unpad_image(a[0] * image1 + ia[0] * image2, meta['pad'])
        save_imfig(base_path+'lin_blend.png', lin_blend)
    else:
        skipped += 1
    pbar.update(1)

    if not os.path.exists(base_path+'lin_iblend.png'):
        lin_iblend = unpad_image(a[0] * image2 + ia[0] * image1, meta['pad'])
        save_imfig(base_path+'lin_iblend.png', lin_iblend)
    else:
        skipped += 1
    pbar.update(1)

    replace = os.path.exists(base_path+'w1_im2.png') or os.path.exists(base_path+'w2_im1.png')

    if not os.path.exists(base_path+'w1_im2.png'):
        overlay = np.zeros((*image1_gray.shape, 3))
        overlay[:,:,0] = normalize_image(result1)
        overlay[:,:,1] = normalize_image(image2_gray)
        save_imfig(base_path+'w1_im2.png', overlay)
    else:
       skipped += 1
    pbar.update(1)

    if not os.path.exists(base_path+'w2_im1.png'):
        overlay = np.zeros((*image1_gray.shape, 3))
        overlay[:,:,0] = normalize_image(result2)
        overlay[:,:,1] = normalize_image(image1_gray)
        save_imfig(base_path+'w2_im1.png', overlay)
    else:
       skipped += 1
    pbar.update(1)
print("skipped {}/{} outputs.".format(skipped, num_outputs))
#print("replaced 2/2 red-green-overlays.")