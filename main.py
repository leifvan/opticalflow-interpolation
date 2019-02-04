import matplotlib

#matplotlib.use('TkAgg')

from tkinter import Tk

Tk = Tk()

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy.ndimage.filters import uniform_filter
from scipy.misc import imsave
from sklearn.feature_extraction.image import extract_patches_2d
from math import sqrt
from keras import backend as K
import os.path
from time import clock
import itertools
import random


def export_layer_vis(pack):
    y, layer_name = pack
    try:
        fig = plt.figure(figsize=(7, 7))
        # plt.axis('off')

        num = 64  # y.shape[-1]
        sqrt_num = sqrt(num)

        for i in range(num):
            ax = fig.add_subplot(sqrt_num, sqrt_num, i + 1)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            ax.imshow(y[:, :, i])

        fig.subplots_adjust(wspace=0.05, hspace=0.05)
        fig.savefig(layer_name + ".png")
        plt.close(fig)
        print("saving '{}'".format(layer_name))
    except Exception as e:
        print(e)


def load_image(path, target_size=(224, 224)):
    image = load_img(path, target_size=target_size)
    image = np.asarray(img_to_array(image), dtype=np.uint8)
    image = image.reshape((1, *image.shape))
    return image#preprocess_input(image)


def shi_tomasi_tracking_points(dx, dy, window_size=(3, 3), num=300, border=2):
    dxdy = dx * dy
    matrices = np.array([[dx * dx, dxdy],
                         [dxdy, dy * dy]])
    sum_matrices = np.zeros_like(matrices)
    uniform_filter(input=matrices, output=sum_matrices, size=(1, 1, *window_size))
    sum_matrices *= 4

    corner_vals = np.zeros_like(dx)

    tracking_points = list()
    response = list()

    for i in range(border, matrices.shape[2] - border):
        for j in range(border, matrices.shape[3] - border):
            trace = sum_matrices[0, 0, i, j] + sum_matrices[1, 1, i, j]
            det = sum_matrices[0, 0, i, j] * sum_matrices[1, 1, i, j] - sum_matrices[0, 1, i, j] * sum_matrices[
                1, 0, i, j]

            if trace ** 2 / 4 - det > 0:
                ev1 = trace / 2 + sqrt(trace ** 2 / 4 - det)
                ev2 = trace / 2 - sqrt(trace ** 2 / 4 - det)
                corner_vals[i, j] = min(ev1, ev2)

                tracking_points.append([i, j])
                response.append(corner_vals[i, j])

    order = np.argsort(-np.array(response))
    tracking_points = np.array(tracking_points)[order[0:num]]

    return np.array(tracking_points)


def lucas_kanade(dx, dy, image_a, image_b, tracking_points, window_size=3):
    assert (window_size % 2 == 1)

    d = int((window_size - 1) / 2)

    flow_vecs = list()

    for point in tracking_points:

        a_mat = np.zeros(shape=(window_size ** 2, 2))
        b_mat = np.zeros(shape=(window_size ** 2,))

        for i in range(0, window_size):
            for j in range(0, window_size):
                px = point[0] + (i - d)
                py = point[1] + (j - d)
                a_mat[i * window_size + j, 0] = dx[px, py]
                a_mat[i * window_size + j, 1] = dy[px, py]
                b_mat[i * window_size + j] = image_a[px, py] - image_b[px, py]

        flow = np.dot(np.linalg.pinv(a_mat), b_mat)
        flow_vecs.append(flow)

    return np.array(flow_vecs)


def visualize_most_active_filters(response):
    highest = get_highest_response_ix(response, num=5)

    fig = plt.figure(figsize=(4,12))

    plot_shape = (len(response)-1, 5)

    for j in range(1, len(response)):
        layer = response[j]
        ixs = highest[0,j]
        for i in range(5):
            ax = plt.subplot2grid(plot_shape, (j-1, i))
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            ax.imshow(layer[0,:,:,ixs[i]])

    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.show()


def optical_flow(image_a, image_b):
    dx_a, dy_a = np.gradient(image_a)
    #dx_b, dy_b = np.gradient(image_b)

    tps_a = shi_tomasi_tracking_points(dx_a, dy_a)
    #tps_b = shi_tomasi_tracking_points(dx_b, dy_b)

    flo_a = lucas_kanade(dx_a, dy_a, image_a, image_b, tps_a)
    #flo_b = lucas_kanade(dx_b, dy_b, image_b, image_a, tps_b)

    # check for sane flow vectors
    sane_tps = []
    sane_flo = []

    dist = 0.


    if len(tps_a) > 0:
        return np.array(tps_a), np.array(flo_a)

    # for tpa, fva in zip(tps_a, flo_a):
    #     sane_tps.append(tpa)
    #     sane_flo.append(fva)
    #     #dist += np.linalg.norm(fva)
    #     # shifted_vec = np.rint(tpa + fva)
    #     # for tpb, fvb in zip(tps_b, flo_b):
    #     #     if shifted_vec[0] == tpb[0] and shifted_vec[1] == tpb[1]:
    #     #         reverse_vec =  np.rint(tpb + fvb)
    #     #         dist += np.linalg.norm(reverse_vec - tpa)
    #     #
    #     #         sane_tps.append(tpa)
    #     #         sane_flo.append(fva)

    # if len(sane_tps) > 0:
    #     #dist /= len(sane_tps)
    #     #print('avg dist =', dist)
    #     #if dist < 10:
    #     return np.array(sane_tps), np.array(sane_flo)

    return None, None


def warp_image_based_on_response_flow(paths, iterations):
    im1 = load_image(paths[0])
    im2 = load_image(paths[1])

    for i in range(iterations):
        responses = model.predict(np.concatenate([im1, im2], axis=0))
        high_ix = get_highest_response_ix(responses, num=5)

        for j in range(1,len(responses)):
            layer = responses[j]
            layer_hix = high_ix[:,j]
            px = layer.shape[1]
            scale_factor = int(224/px)

            #print(layer_hix)

            fields = []

            for i in itertools.chain(layer_hix[0], layer_hix[1]):
                print("filter", i, "is good")
                f1 = layer[0,:,:,i]
                f2 = layer[1,:,:,i]
                tps, flow = optical_flow(f1, f2)
                if tps is not None:
                    field = get_warp_field(tps, flow, (px,px))
                    if scale_factor > 1:
                        field = np.repeat(field, scale_factor, axis=0)
                        field = np.repeat(field, scale_factor, axis=1)
                        assert(field.shape[0] == 224 and field.shape[1] == 224)
                    fields.append(field)

            if len(fields) > 0:
                field = np.mean(fields, axis=0)
                im1[0] = warp_image(im1[0], field, 0.5)
                responses = model.predict(np.concatenate([im1, im2], axis=0))
            else:
                print("no field")

        plt.imshow(im1[0])
        plt.show()


def warp_interpolate_images(im1,im2):
    tps, flow = optical_flow(im1,im2)

    if tps is not None:
        field = get_warp_field(tps, flow)
        return warp_image(im1, field, 0.5)
    else:
        #print("Failed to warp")
        return im1


def warp_all_responses(responses):
    warped = [np.zeros_like(layer) for layer in responses]

    for layer, warped_layer in zip(responses,warped):
        print("warping layer",layer.shape)
        for i in range(layer.shape[3]):
            f1 = layer[0,:,:,i]
            f2 = layer[1,:,:,i]
            warped_layer[0,:,:,i] = warp_interpolate_images(f1,f2)

    return warped


def get_warp_field(points, vecs, shape=(224,224)):
    from scipy.interpolate import griddata
    grid_x, grid_y = np.mgrid[0:shape[0],0:shape[1]]
    field = griddata(points, vecs, (grid_x, grid_y), method='nearest')
    return field


def warp_image(image, field):
    from scipy.ndimage.interpolation import geometric_transform
    #output = np.zeros_like(image)

    def warp_lookup(tup, strength=0.5):
        fv = field[tup[0],tup[1]]

        if len(tup) == 2:
            return int(tup[0] + fv[0]*strength), int(tup[1] + fv[1]*strength)
        elif len(tup) == 3:
            return int(tup[0] + fv[0] * strength), int(tup[1] + fv[1] * strength), tup[2]

    output = geometric_transform(image, warp_lookup, output_shape=image.shape, order=3)
    return output


def show_flow(im, tracking_points, flow_vecs):
    plt.imshow(im)
    ax = plt.gca()
    ax.quiver(tracking_points[:, 1], tracking_points[:, 0], flow_vecs[:, 1], flow_vecs[:, 0], color='red', width=0.005)
    plt.show()


def deprocess_image(x):
    # util function to convert a tensor into a valid image
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255

    #x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    #x = np.transpose(x, (0, 2, 1))
    #x = np.transpose(x, (1,0,2))
    #x = 255 - x
    return x


def minimize_filter_response_distance(model, target_output, start_img=None, img_size=(224,224), iterations=20, step=1, first_layer=0):
    input_img = model.input
    if start_img is None:
        input_img_data = np.random.random((1, *img_size, 3)) * 20 + 128.
    else:
        input_img_data = start_img

    input_img_data = np.asarray(input_img_data, dtype=np.float32)

    output_list = model.output

    # remove first layers
    output_list = output_list[first_layer:]
    target_output = target_output[first_layer:]

    mse_list = []

    for out, target in zip(output_list, target_output):
        o_perm = K.permute_dimensions(out[0], pattern=(2,0,1))
        o_flat = K.batch_flatten(o_perm)

        t_perm = np.transpose(target[0], axes=(2,0,1))
        t_flat = np.reshape(t_perm, (t_perm.shape[0], t_perm.shape[1]*t_perm.shape[2]))

        t_tensor = K.constant(t_flat, shape=t_flat.shape)
        mse = K.mean(K.mean(K.square(t_tensor-o_flat)))
        mse_list.append(mse)

    loss = K.mean(K.stack(mse_list))

    grads = K.gradients(loss, input_img)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # TODO try to express image as variable and run whole calculation on GPU

    iterate = K.function([input_img], [loss, grads])

    class ImageAnimator:

        def __init__(self, plot_im, start_im):
            self.plot_pos = 0
            self.plot_im = plot_im
            self.plot_image_list = [np.copy(start_im)]

        def __call__(self, *args):
            self.plot_im.set_data(self.plot_image_list[self.plot_pos])

            self.plot_pos += 1

            if self.plot_pos >= len(self.plot_image_list):
                self.plot_pos = 0

            return self.plot_im,

        def add_image(self, im):
            self.plot_image_list.append(im)

    plot_fig = plt.figure()
    plot_im = plt.imshow(deprocess_image(input_img_data[0]), animated=True)

    im_am = ImageAnimator(plot_im, deprocess_image(input_img_data[0]))

    plt_ani = animation.FuncAnimation(plot_fig, im_am, interval=200, blit=True, repeat=True)

    plot_fig.show()

    last_loss = np.inf
    raise_counter = 0

    for i in range(iterations):
        loss_value, grads_value = iterate([input_img_data])

        if loss_value < last_loss and step < 2:
            raise_counter += 1
            if raise_counter > 4:
                step += 0.1
                raise_counter = 0
        elif step > -4:
            step -= 0.4
            raise_counter = 0

        last_loss = loss_value

        input_img_data -= grads_value * (2. ** step)
        print("[{}] {} (step = {})".format(i,loss_value,step))

        Tk.update()

        if i % 5 == 0:
            img = np.copy(input_img_data[0])
            img = deprocess_image(img)
            im_am.add_image(img)

    img = input_img_data[0]
    img = deprocess_image(img)
    return img


def maximize_filter_response(model, layer_index, filter_index, img_size=(224, 224), before_relu=False, iterations=20, step=1):
    input_img = model.input
    input_img_data = np.random.random((1, *img_size, 3)) * 20 + 128.

    # build a loss function that maximizes the activation
    # of the nth filter of the layer considered
    from tensorflow import Tensor
    layer_output = model.layers[layer_index].output

    if before_relu:
        layer_output = layer_output.op.inputs[0]

    loss = K.mean(layer_output[:, :, :, filter_index])

    # compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, input_img)[0]

    # normalization trick: we normalize the gradient
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # this function returns the loss and grads given the input picture
    iterate = K.function([input_img], [loss, grads])

    for i in range(iterations):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

    img = input_img_data[0]
    img = deprocess_image(img)
    img = np.transpose(img, (0, 2, 1))
    return img


def pca(matrix, dims):
    matrix -= matrix.mean(axis=0)
    cov = np.cov(matrix)
    _, eigvecs = np.linalg.eigh(cov)
    eigvecs = eigvecs[:,::-1]
    return eigvecs[:,:dims]


from ssim import tf_ssim
from scipy.interpolate import Rbf
import tensorflow as tf

def get_ssim_placeholders(num_images):
    ssim_image1 = tf.placeholder(tf.float32, shape=(16, 16))
    ssim_image2 = tf.placeholder(tf.float32, shape=(16, 16))

    ssim_image1_expanded = tf.expand_dims(ssim_image1, 0)
    ssim_image2_expanded = tf.expand_dims(ssim_image2, 0)

    ssim_image4d_1 = tf.expand_dims(ssim_image1_expanded, -1)
    ssim_image4d_2 = tf.expand_dims(ssim_image2_expanded, -1)

    ssim_index = tf_ssim(ssim_image4d_1, ssim_image4d_2)
    return ssim_image1, ssim_image2, ssim_index

    # ssim_image1 = tf.placeholder(tf.float32, shape=(num_images, 16, 16))
    # ssim_image2 = tf.placeholder(tf.float32, shape=(16, 16))
    #
    # ssim_image2_expanded = tf.expand_dims(ssim_image2, 0)
    #
    # ssim_image2_tiled = tf.tile(ssim_image2_expanded, [num_images,1,1])
    # print(ssim_image2_tiled.shape)
    #
    # ssim_image4d_1 = tf.expand_dims(ssim_image1, -1)
    # ssim_image4d_2 = tf.expand_dims(ssim_image2_tiled, -1)
    #
    # ssim_index = tf_ssim(ssim_image4d_1, ssim_image4d_2)
    # return ssim_image1, ssim_image2, ssim_index


def patch_step(im1, im2, patch_size=(16,16)):

    # extract patches

    patches1 = extract_patches_2d(im1, patch_size)
    patches2 = extract_patches_2d(im2, patch_size)

    pairwise_ssim = np.zeros(shape=(patches1.shape[0], patches2.shape[0]))

    ssim_image1, ssim_image2, ssim_index = get_ssim_placeholders(patches1.shape[0])

    # find corresponding patches via pairwise ssim

    with tf.Session() as sess:
        #for i2 in range(patches2.shape[0]):
        #    pairwise_ssim[:,i2] = sess.run(ssim_index, feed_dict={ssim_images1:patches1, ssim_image2:patches2[i2]})
        for i1 in range(patches1.shape[0]):
            for i2 in range(i1, patches2.shape[0]):
                p1 = patches1[i1]
                p2 = patches2[i2]
                pairwise_ssim[i1,i2] = sess.run(ssim_index, feed_dict={ssim_image1:p1, ssim_image2:p2})
            print(i1,"/",patches1.shape[0])

    return patches1, patches2, pairwise_ssim


def show_patches(*images, title=""):
    if len(images) == 1:
        plt.imshow(images[0])
    elif len(images) % 2 == 0:
        plt.figure(figsize=(len(images),4))
        for i in range(len(images)):
            ax = plt.subplot2grid((2,int(len(images)/2)),(i%2,int(i/2)))
            ax.imshow(images[i])
    plt.title(title)
    plt.show()


def warp_patch(p1, p2):
    a = np.arange(0, p1.shape[0])
    xv, yv = np.meshgrid(a,a)
    xv = xv.flatten()
    yv = yv.flatten()
    a0 = np.zeros_like(xv)
    a1 = np.ones_like(xv)

    xv2 = np.tile(xv,2)
    yv2 = np.tile(yv,2)

    a = np.concatenate((a0,a1))

    p = np.concatenate((p1.flatten(),p2.flatten()))

    rbf = Rbf(xv2, yv2, a, p, function='thin_plate')

    interpolated = rbf(xv,yv,0.5*a1)
    interpolated.shape = p1.shape

    return interpolated


def get_interpolated_patches(patches1, patches2, ssim):
    inds = np.argmax(ssim, axis=0)
    patches2 = patches2[inds]
    print("interpolating patches...")
    return [warp_patch(p1, p2) for p1, p2 in zip(patches1, patches2)]


from keras.applications.vgg19 import VGG19, preprocess_input
from keras.preprocessing.image import load_img, img_to_array

from keras.models import Model

print('preparing model')

base_model = VGG19(include_top=False)

print('base model ready')

# fetch all outputs
model = Model(inputs=base_model.inputs, outputs=[layer.output for layer in base_model.layers])

print('other model also ready')

def run_model(image_paths):
    imgs = np.concatenate([load_image(p) for p in image_paths], axis=0)
    return model.predict(imgs)


def get_highest_response_ix(responses, num=10):
    batch_size = responses[0].shape[0]
    layers_highest = np.zeros(shape=(batch_size, len(responses), num), dtype=np.int)

    for i,layer_resp in enumerate(responses):
        per_filter_highest = np.mean(np.abs(layer_resp), axis=(1,2))
        sorted_ix = np.argsort(per_filter_highest, axis=1)[:,:min(num,per_filter_highest.shape[1])]
        layers_highest[:,i,:sorted_ix.shape[1]] = sorted_ix

    return layers_highest


def get_dtd_paths_from_ix(ix_list):
    base_list = os.listdir('dtd/images')
    sub_lists = [os.listdir('dtd/images/'+base_path) for base_path in base_list]

    for i in range(len(sub_lists)):
        sub_lists[i] = [p for p in sub_lists[i] if p.endswith('.jpg')]

    return tuple('dtd/images/'+base_list[i0]+'/'+sub_lists[i0][i1] for i0, i1 in ix_list)

# imgs = np.concatenate((load_image('images/squares2.jpg', target_size=(224, 224)),
#                        load_image('images/hexagons2.jpg', target_size=(224, 224))),
#                       axis=0)
# layer_outputs = model.predict(imgs)

# first_layer = 9

# for i, out in enumerate(layer_outputs[first_layer:]):
#     layer_name = base_model.layers[i + first_layer].name
#     print('analysing layer {}'.format(layer_name))
#     imgs1 = out[0, :, :, :]
#     imgs2 = out[1, :, :, :]
#
#     filters_sums = np.multiply(np.sum(imgs1, axis=(0, 1)),
#                                np.sum(imgs2, axis=(0, 1)))
#     filters_sums = np.abs(filters_sums)
#     filters_sorted = np.argsort(-filters_sums)
#
#     # get principal components from filter responses
#     dim = 3
#
#     pca1 = np.reshape(imgs1, (imgs1.shape[0] * imgs1.shape[1], imgs1.shape[2]), order='F')
#     pca2 = np.reshape(imgs2, (imgs2.shape[0] * imgs2.shape[1], imgs2.shape[2]), order='F')
#
#     pca1 = pca(pca1, dim)
#     pca2 = pca(pca2, dim)
#
#     for i in range(dim):
#         # get filters with highest activation
#         # filter_index = filters_sorted[i]
#         # im1 = imgs1[:, :, filter_index]
#         # im2 = imgs2[:, :, filter_index]
#
#         size = int(sqrt(pca1.shape[0]))
#         filter_index = "pca "+str(i)
#         im1 = np.reshape(pca1[:,i], (size,size))
#         im2 = np.reshape(pca2[:,i], (size,size))
#
#         im1 = normalize(im1)
#         im2 = normalize(im2)
#
#         ax = plt.subplot2grid((2, 3), (0, 0))
#         ax.imshow(im1)
#         ax = plt.subplot2grid((2, 3), (1, 0))
#         ax.imshow(im2)
#
#         dx, dy = np.gradient(im1)
#         tracking_points = shi_tomasi_tracking_points(dx, dy, border=2)
#         flow_vecs = lucas_kanade(dx, dy, im1, im2, tracking_points, window_size=5)
#
#         ax = plt.subplot2grid((2, 3), (0, 1), colspan=2, rowspan=2)
#         ax.imshow(im1)
#
#         if len(flow_vecs) > 0:
#             ax.quiver(tracking_points[:, 1], tracking_points[:, 0], flow_vecs[:, 1], -flow_vecs[:, 0], color='red',
#                       width=0.005)
#
#         plt.gcf().subplots_adjust(wspace=0.05, hspace=0.05)
#         plt.title("{}, filter {}".format(layer_name, filter_index))
#         plt.show()

# im1 = imgs[0][:, :, 0]
# im2 = imgs[1][:, :, 0]
# dx, dy = np.gradient(imgs[0][:, :, 0])
# tracking_points = shi_tomasi_tracking_points(dx, dy, (2, 2), border=2)
# flow_vecs = lucas_kanade(dx, dy, im1, im2, tracking_points, 5)
# ax = plt.subplot2grid((2, 3), (0, 0))
# ax.imshow(im1)
# ax = plt.subplot2grid((2,3),(1,0))
# ax.imshow(im2)
# ax = plt.subplot2grid((2, 3), (0, 1), colspan=2, rowspan=2)
# ax.imshow(im1)
# ax.quiver(tracking_points[:, 1], tracking_points[:, 0], flow_vecs[:, 1], flow_vecs[:, 0], color='red', width=0.005)
# plt.show()

# plot
# pool = Pool(processes=4)
# pool.map(export_layer_vis, [(out[0,:,:,:],base_model.layers[i+1].name) for i,out in enumerate(layer_outputs[1:-4])])


def maximize_all_filter_responses(first_layer):
    output_map = {64: (8, 8), 128: (8, 16), 256: (16, 16), 512: (16, 32)}

    input_size = (200, 200)
    padding = 1

    before_relu = False

    for i, layer in enumerate(base_model.layers):
        if i >= first_layer:
            num_filters = int(layer.output.shape[3])
            # output_w = (input_size[0] + padding) * output_map[num_filters][0] - padding
            # output_h = (input_size[1] + padding) * output_map[num_filters][1] - padding
            # output_image = np.zeros((output_w, output_h, 3))

            print("running layer {}: '{}' ({} filters)".format(i, layer.name, num_filters))

            for filter_index in range(num_filters):
                path = "responses500/max500_{}_{}{}.png".format(layer.name, filter_index, "_noReLU" if before_relu else "")
                if not os.path.exists(path):
                    start_time = clock()
                    response = maximize_filter_response(base_model, i, filter_index, input_size,
                                                        before_relu=before_relu, iterations=200, grad_limit=1e-5)
                    response_time = clock()-start_time
                    #ox = (input_size[0] + padding) * (filter_index % output_map[num_filters][0])
                    #oy = (input_size[1] + padding) * int(filter_index / output_map[num_filters][0])
                    #output_image[ox:(ox + input_size[0]), oy:(oy + input_size[1]), :] = response
                    imsave(path, response)
                    imsave_time = clock()-start_time-response_time
                    print(" -> filter {} ({:.2f}sec response, {:.2f}sec save | {:.2f}sec total)".format(filter_index,
                                                                                                        response_time,
                                                                                                        imsave_time,
                                                                                                        response_time +
                                                                                                        imsave_time))

            #imsave("max_{}.png".format(layer.name), output_image)

# #maximize_all_filter_responses(17)
# steps = [500,500,2000,2000]
# iters = [20,200,20,200,20,200]
# layer_index = 17
# filter_index = 0
#
# for step, iter in zip(steps,iters):
#     response = maximize_filter_response(base_model,layer_index, filter_index,iterations=iter, step=step)
#     imsave("tests/max{}_{}_iter{}_step{}.jpg".format(layer_index, filter_index, iter, step),response)


def filter_image_paths(paths, suffix=".jpg"):
    ret = list()
    for p in paths:
        if p.endswith(suffix):
            ret.append(p)
    return ret


def sum_layer_responses():

    max_filters = 512
    num_layers = 22
    dtd_dir = 'dtd/images'
    dtd_list = os.listdir(dtd_dir)
    num_categories = 5#len(dtd_list)

    dir_lists = [filter_image_paths(os.listdir(dtd_dir+"/"+dir)) for dir in dtd_list]
    num_img_limit = 1000
    num_imgs = [min(len(ls),num_img_limit) for ls in dir_lists]

    m = np.asscalar(np.sum(num_imgs[:num_categories],dtype=np.int))
    n = max_filters * num_layers
    vectors = np.zeros(shape=(m, n))
    labels = np.repeat(np.arange(0, num_categories), num_imgs[:num_categories])

    for dir_ix, type in enumerate(os.listdir(dtd_dir)[:num_categories]):
        print("reading from '{}'".format(type))

        ix_start = np.asscalar(np.sum(num_imgs[:dir_ix],dtype=np.int))
        reduced = np.zeros(shape=(num_imgs[dir_ix], num_layers, max_filters))

        base_path = dtd_dir+"/"+type
        paths = dir_lists[dir_ix][:num_imgs[dir_ix]]#list(image_paths_iterator(os.listdir(base_path)[:num_imgs[dir_ix]]))
        # Todo: ARTIFACTS DUE TO NON QUADRATIC IMAGES

        # check if response is already calculated
        for path_ix,p in enumerate(paths):
            if os.path.exists(base_path+"/"+p+"_response.npy"):
                vectors[ix_start + path_ix] = np.load(base_path + "/" + p + "_response.npy")
            else:
                print("-> calculating",p)
                img = load_image(base_path+"/"+p)
                output = model.predict(img)
                reduced = np.zeros(shape=(num_layers, max_filters))
                for layer_ix, layer_response in enumerate(output):
                    reduced[layer_ix, :layer_response.shape[3]] = np.abs(np.sum(layer_response, axis=(1,2)))
                    reduced[layer_ix] /= np.sum(reduced[layer_ix])  # normalize
                reduced /= np.linalg.norm(reduced)
                flattened = reduced.flatten()
                vectors[path_ix+ix_start] = flattened
                np.save(base_path + "/" + p + "_response.npy", flattened)


        # if all((os.path.exists(base_path+"/"+p+"_response.npy") for p in paths)):
        #     print("-> found precomputed responses")
        #     for path_ix,file in enumerate(paths):
        #         vectors[ix_start+path_ix] = np.load(base_path+"/"+file+"_response.npy")
        # else:
        #     print("-> calculating responses...")
        #     imgs = np.concatenate([load_image(base_path+"/"+p) for p in paths], axis=0)
        #     outputs = model.predict(imgs)
        #
        #     # sum filter responses over img
        #     for layer_ix, layer_response in enumerate(outputs):
        #         reduced[:, layer_ix, :layer_response.shape[3]] = np.abs(np.sum(layer_response, axis=(1, 2)))
        #         reduced[:, layer_ix] /= np.sum(reduced[:, layer_ix]) # normalize
        #
        #     #stacked = np.zeros(shape=(num_imgs_per_category*num_layers,max_filters))
        #
        #     # normalize
        #     for img_ix in range(num_imgs[dir_ix]):
        #         reduced[img_ix] /= np.linalg.norm(reduced[img_ix])
        #         flattened = reduced[img_ix].flatten()
        #         vectors[img_ix+ix_start] = flattened
        #         np.save(base_path+"/"+paths[img_ix]+"_response.npy", flattened)
        #         #stacked[img_ix * num_layers:(img_ix + 1) * num_layers, :] = reduced[img_ix]

        #plt.matshow(vectors[dir_ix*num_imgs[dir_ix]:(dir_ix+1)*num_imgs[dir_ix]].reshape((num_layers*num_imgs[dir_ix], max_filters)))
        #plt.title(type)
        #plt.show()
    # from sklearn.decomposition import PCA
    # from sklearn.manifold import TSNE
    # from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    # from mpl_toolkits.mplot3d import Axes3D
    #
    # print("pca...")
    # pca = PCA(n_components=200)
    # principal_comps = pca.fit_transform(vectors)
    # #
    # # comps2d = principal_comps
    #
    # # print("tsne...")
    # # tsne = TSNE(n_components=)
    # # comps3d = tsne.fit_transform(principal_comps)
    #
    # print("lda...")
    # lda = LinearDiscriminantAnalysis(n_components=3)
    # comps3d = lda.fit_transform(principal_comps, labels)
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(comps3d[:,0], comps3d[:,1], comps3d[:,2], c=labels, alpha=0.5, cmap='Dark2')
    # plt.show()

    sort_ix = np.argsort(-vectors, axis=1)

    hist = np.histogram2d(np.tile(range(sort_ix.shape[1]), sort_ix.shape[0]), sort_ix.flatten(), bins=sort_ix.shape)[0]
    plt.matshow(hist)
    plt.show()

    #for v in vectors:
    #    sort_ix = np.argsort(-v)


