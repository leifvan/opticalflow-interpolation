from utils import load_image_and_preprocess
import numpy as np
from scipy.ndimage import geometric_transform
from scipy.ndimage.filters import gaussian_filter
from scipy.misc import imresize
import matplotlib.pyplot as plt


def conv2d(a, f):
    s = f.shape + tuple(np.subtract(a.shape, f.shape) + 1)
    strd = np.lib.stride_tricks.as_strided
    subM = strd(a, shape=s, strides=a.strides * 2)
    return np.einsum('ij,ijkl->kl', f, subM)


#paths = ('images/old/stone_wall_b.jpg', 'images/old/stone_wall_b_left.jpg')
#paths = ('images/old/squares2.jpg', 'images/old/hexagons2.jpg')
#paths = ('images/honeycombed_0142.jpg', 'images/grid_0007.jpg')
#paths = ('images/cracked_0155.jpg', 'images/cracked_0162.jpg')
paths = ('images/old/squares_center.jpg', 'images/old/squares_center_rot.jpg')
paths = ('images/old/squares_center.jpg', 'images/old/squares.jpg')
#paths = ('images/old/hexagons2.jpg', 'images/old/squares_center_rot.jpg')
paths = ('images/HexagonalGrid_700.gif','images/SquareGrid_800.gif')

static_img = load_image_and_preprocess(paths[0])
static_img = static_img.reshape(static_img.shape[1:])[:, :, 0]
static_img = imresize(static_img, 1.0)
static_img = static_img.astype(np.float64)
moving_img = load_image_and_preprocess(paths[1])
moving_img = moving_img.reshape(moving_img.shape[1:])[:, :, 0]
moving_img = imresize(moving_img, 1.0)
moving_img = moving_img.astype(np.float64)

image_1 = static_img.copy()
image_2 = moving_img.copy()

tx = np.zeros_like(moving_img)
ty = np.zeros_like(moving_img)

sy, sx = np.gradient(static_img)

g_alpha = 0.5
g_sigma = 0.1
g_scale = 1

fig = plt.figure(figsize=(16,4))
plt.title("alpha={}, sigma={}, scale={}".format(g_alpha, g_sigma, g_scale))
axs = plt.imshow(np.hstack((static_img, image_1, image_2, moving_img)))
sx_sy_2 = (sx ** 2 + sy ** 2)


def get_diff_demon_step(alpha, sigma, scale, tx, ty, moving_img, target_img):
    img_diff = moving_img - target_img
    my, mx = np.gradient(moving_img)

    img_diff_2 = alpha ** 2 * img_diff ** 2
    mx_my_2 = (mx ** 2 + my ** 2)
    mx_my_im_diff = (mx_my_2 + img_diff_2)
    sx_sy_im_diff = (sx_sy_2 + img_diff_2)
    ux = - np.multiply(img_diff, ((sx / sx_sy_im_diff) + (mx / mx_my_im_diff)))
    uy = - np.multiply(img_diff, ((sy / sx_sy_im_diff) + (my / mx_my_im_diff)))
    ux[np.isnan(ux)] = 0
    uy[np.isnan(uy)] = 0
    uxs = gaussian_filter(ux, sigma, mode='nearest') if sigma > 0 else ux
    uys = gaussian_filter(uy, sigma, mode='nearest') if sigma > 0 else uy
    tx = tx + scale * uxs
    ty = ty + scale * uys
    return tx, ty, img_diff, geometric_transform(static_img, lambda pos: (pos[0]-tx[pos], pos[1]+ty[pos]), mode='wrap', order=1)


for it in range(20000):
    tx, ty, img_diff, moving_img = get_diff_demon_step(g_alpha, g_sigma, g_scale, tx, ty, moving_img, image_2)

    if it % 10 == 0:
        axs.set_data(np.hstack((image_1, moving_img, np.abs(img_diff), image_2)))
        plt.pause(0.01)
        plt.show(block=False)
        print(it, np.linalg.norm(img_diff))
