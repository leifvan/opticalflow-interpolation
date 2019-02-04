import argparse
import numpy as np
from stnwarp import get_merged_warp, plot_flow, get_coordinate_map
import matplotlib.pyplot as plt
import os


def save_flow(path, flow):
    fig, ax = plt.subplots(1, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    plot_flow(ax, np.rollaxis(flow, 2))
    plt.savefig(path, bbox_inches='tight', dpi=300)


parser = argparse.ArgumentParser()
parser.add_argument('path')
parser.add_argument('-r', action='store_true')
args = parser.parse_args()

def export_flow_vis(path):
    vfields = np.load(path)

    if len(vfields.shape) == 3:
        vfields = np.reshape(vfields, (1, *vfields.shape))

    shape = vfields.shape[1:3]

    for i, field in enumerate(vfields):
        sflowpath = path+'.flow{}.png'.format(i)
        print("Flow saved as        '{}'".format(sflowpath))
        save_flow(sflowpath, field)

    if len(vfields) > 1:
        #coords = np.reshape(get_coordinate_map(*shape),(1,*shape,2))
        more_fields = vfields#np.concatenate([coords,vfields],axis=0)
        #merged = get_merged_warp(more_fields)
        merged = np.sum(vfields, axis=0)
        #merged -= coords[0]
        mflowpath = path+'.merged.png'
        save_flow(mflowpath, merged)
        print("Merged flow saved as '{}'".format(mflowpath))


if os.path.isdir(args.path):
    for fpath in os.listdir(args.path):
        if fpath.endswith('.npy'):
            p = os.path.join(args.path,fpath)
            try:
                export_flow_vis(p)
            except Exception as e:
                print("'{}' is not a flow\n({})\n".format(p, e))
else:
    export_flow_vis(args.path)