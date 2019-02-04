import numpy as np
from sklearn.cluster import KMeans
from skimage.measure import compare_ssim
from collections import namedtuple, OrderedDict
from colorsys import hsv_to_rgb
from scipy.fftpack import dct
import matplotlib.pyplot as plt
from skimage.transform import warp
from scipy.ndimage.interpolation import geometric_transform

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

mask_cof = np.matrix(
    "0.390625, 0.826446, 1.000000, 0.390625, 0.173611, 0.062500, 0.038447, 0.026874; 0.694444, 0.694444, 0.510204, 0.277008, 0.147929, 0.029727, 0.027778, 0.033058; 0.510204, 0.591716, 0.390625, 0.173611, 0.062500, 0.030779, 0.021004, 0.031888; 0.510204, 0.346021, 0.206612, 0.118906, 0.038447, 0.013212, 0.015625, 0.026015; 0.308642, 0.206612, 0.073046, 0.031888, 0.021626, 0.008417, 0.009426, 0.016866; 0.173611, 0.081633, 0.033058, 0.024414, 0.015242, 0.009246, 0.007831, 0.011815; 0.041649, 0.024414, 0.016437, 0.013212, 0.009426, 0.006830, 0.006944, 0.009803; 0.019290, 0.011815, 0.011080, 0.010412, 0.007972, 0.010000, 0.009426, 0.010203",
    dtype=np.float32)
csf_cof = np.matrix(
    "1.608443, 2.339554, 2.573509, 1.608443, 1.072295, 0.643377, 0.504610, 0.421887; 2.144591, 2.144591, 1.838221, 1.354478, 0.989811, 0.443708, 0.428918, 0.467911; 1.838221, 1.979622, 1.608443, 1.072295, 0.643377, 0.451493, 0.372972, 0.459555; 1.838221, 1.513829, 1.169777, 0.887417, 0.504610, 0.295806, 0.321689, 0.415082; 1.429727, 1.169777, 0.695543, 0.459555, 0.378457, 0.236102, 0.249855, 0.334222; 1.072295, 0.735288, 0.467911, 0.402111, 0.317717, 0.247453, 0.227744, 0.279729; 0.525206, 0.402111, 0.329937, 0.295806, 0.249855, 0.212687, 0.214459, 0.254803; 0.357432, 0.279729, 0.270896, 0.262603, 0.229778, 0.257351, 0.249855, 0.259950",
    dtype=np.float32)


def extract_patches(image, offsets, size):
    patches = [image[o[0]:o[0] + size[0], o[1]:o[1] + size[1]] for o in offsets]
    return np.stack(patches, axis=0)


def sample_offsets(shape, size, num_samples):
    x = np.random.randint(0, shape[0] - size[0] + 1, size=num_samples, dtype=np.int)
    y = np.random.randint(0, shape[1] - size[1] + 1, size=num_samples, dtype=np.int)
    return np.stack([x, y], axis=1)


def get_offsets_grid(shape, size, step):
    x, y = np.meshgrid(np.arange(0, shape[0] - size[0] + 1, step[0]), np.arange(0, shape[1] - size[1] + 1, step[1]))
    return np.stack([x.ravel(), y.ravel()], axis=1)


def get_all_offsets(shape, size):
    return get_offsets_grid(shape, size, (1, 1))


def pad_image(image, padding):
    new = np.zeros((image.shape[0] + 2 * padding, image.shape[1] + 2 * padding, *image.shape[2:]))
    new[padding:image.shape[0] + padding, padding:image.shape[1] + padding] = image
    return new


def unpad_image(image, padding):
    return image[padding:image.shape[0] - padding, padding:image.shape[1] - padding]


def transfer_patch(image, patch, offset, weights=None, mode='blend'):
    if weights is None:
        weights = np.ones_like(patch)

    original = extract_patches(image, [offset], patch.shape)
    weighted_original = np.multiply(1-weights, original)
    weighted_patch = np.multiply(weights, patch)

    if mode == 'blend':
        blended = weighted_original + weighted_patch
    elif mode == 'add':
        blended = original + weighted_patch

    image[offset[0]:offset[0] + patch.shape[0], offset[1]:offset[1] + patch.shape[1]] = blended


def mse(image1, image2):
    return np.sum(np.abs(image1 - image2) ** 2)


def ssim(image1, image2, gaussian_weights=True):
    return compare_ssim(image1, image2, win_size=11, gaussian_weights=gaussian_weights)


def psnr_hvs(image1, image2):
    assert image1.shape == image2.shape
    offsets = get_offsets_grid(image1.shape, (8, 8), (8, 8))
    patches1 = extract_patches(image1, offsets, size=(8, 8))
    patches2 = extract_patches(image2, offsets, size=(8, 8))

    dct_patches1 = dct(patches1, axis=-1)
    dct_patches1 = dct(dct_patches1, axis=-2)
    dct_patches2 = dct(patches2, axis=-1)
    dct_patches2 = dct(dct_patches2, axis=-2)

    u = np.abs(dct_patches1 - dct_patches2)
    s2 = np.sum(np.multiply(u, csf_cof) ** 2, axis=None)

    if s2 == 0.:
        return 100000.
    else:
        return 10 * np.log10(1 / s2)

def get_coordinate_map(width, height):
    xx, yy = np.meshgrid(range(width), range(height))
    return np.stack((xx,yy), axis=2).astype(np.float64)

def do_geo_warp(image, field):
    def mapper(t):
        x,y = t
        tx, ty = field[x,y]
        return x+tx, y+ty
    return geometric_transform(image, mapper, mode='nearest')

# reconstruct images with warps
def get_merged_warp(fields):
    total_warp = np.copy(fields[0])#np.copy(image)
    for field in fields[1:]:
        total_warp[...,0] = do_geo_warp(total_warp[...,0], field)
        total_warp[...,1] = do_geo_warp(total_warp[...,1], field)
    return total_warp

def warp_from_vfields(image, vfields):
    more_fields = np.concatenate([np.reshape(get_coordinate_map(*image.shape),(1,*image.shape,2)),-vfields],axis=0)
    total_warp = get_merged_warp(more_fields)
    total_warp = np.transpose(total_warp, axes=[2,0,1])
    return warp(image.T, total_warp, mode='edge')

def warp_channels_from_vfields(image_channels, vfields):
    image = np.copy(image_channels)
    for channel in np.rollaxis(image, 2):
        channel[:] = warp_from_vfields(channel, vfields)
    return image

def image_style_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])


def plot_source_target(ax, source, target, width, height, color='C0'):
    ax.set_aspect('equal', 'box')
    ax.quiver(source[:, 0], source[:, 1], source[:, 0] - target[:, 0], source[:, 1] - target[:, 1],
              color=color, angles='xy', scale=1, scale_units='xy')
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)


def plot_flow_arrows(ax, flow, width, height, s=1, scale=1):
    angles = (np.arctan2(flow[0], -flow[1])+np.pi)/(2*np.pi)
    mynorm = plt.Normalize(vmin=0, vmax=1)

    ax.set_aspect('equal', 'box')
    ax.quiver(range(0, width, s), range(0, height, s), flow[1, ::s, ::s], flow[0, ::s, ::s], angles[::s,::s],
              angles='xy', scale=scale, scale_units='xy', cmap='hsv', norm=mynorm)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)


def plot_flow(ax, flow):
    angles = np.arctan2(flow[0], -flow[1])
    angles = (angles + np.pi) / (2 * np.pi)
    ax.set_aspect('equal', 'box')
    mags = np.linalg.norm(flow, axis=0)
    mags /= np.max(mags)

    angles[np.isnan(angles)] = 0.
    mags[np.isnan(mags)] = 0.

    vis = np.zeros((flow.shape[2], flow.shape[1], 3))
    for j in range(flow.shape[2]):
        for k in range(flow.shape[1]):
            vis[j, k] = hsv_to_rgb(angles[k, j], 1., mags[k, j])


    vis = np.transpose(vis, axes=[1, 0, 2])

    ax.imshow(vis)


def gaussian_kernel(size, mean, std):
    """Makes 2D gaussian Kernel for convolution."""
    d = tf.distributions.Normal(mean, std)
    vals = d.prob(tf.range(start=-size, limit=size + 1, dtype=tf.float32))
    gauss_kernel = tf.einsum('i,j->ij', vals, vals)
    gauss_kernel = gauss_kernel / tf.reduce_sum(gauss_kernel)
    return gauss_kernel[:, :, tf.newaxis, tf.newaxis]


def smoothed_image(image, size, mean, std):
    kernel = gaussian_kernel(size, mean, std)
    return tf.nn.conv2d(image, kernel, strides=[1, 1, 1, 1], padding="SAME")


def get_flat_xyval(array):
    out = np.zeros((array.size, 3))
    it = np.nditer(array, flags=['multi_index', 'f_index'])
    while not it.finished:
        out[it.index] = [*it.multi_index, it[0]]
        # pos += 1
        it.iternext()
    return out


def sample_based_on_density(density2d, n):
    flat = get_flat_xyval(density2d)
    probs = flat[:, 2] / np.sum(flat[:, 2])
    idx = np.random.choice(range(len(flat)), 20 * n, p=probs)
    cluster = KMeans(n_clusters=n)
    cluster.fit(flat[idx, :2])
    return cluster.cluster_centers_


def get_p_hvs(im1, im2, max_val):
    patches = tf.extract_image_patches(tf.concat([im1, im2], axis=0), [1, 8, 8, 1], [1, 8, 8, 1], [1., 1, 1, 1],
                                       padding='VALID')
    # (image_idx, num_rows, num_cols, patch_pxls)
    patches = tf.reshape(patches, [*patches.shape[:3], 8, 8])
    # (image_idx, num_rows, num_cols, patch_width, patch_height)

    # 2D discrete-cosine transform all patches (which is just dct(dct(x).T))
    dct_patches = tf.spectral.dct(patches, type=2)
    dct_patches = tf.spectral.dct(tf.transpose(dct_patches, [0, 1, 2, 4, 3]), type=2)
    u = tf.abs(dct_patches[0, ...] - dct_patches[1, ...])
    s2 = tf.reduce_sum(tf.multiply(csf_cof, u) ** 2, axis=None)
    p_hvs = tf.where(tf.equal(s2, 0), 100000., 10 * tf.log((max_val ** 2) / s2) / np.log(10))

    return p_hvs


def get_point_grid(num_points_per_axis, width, height):
    side_spacing = width / num_points_per_axis / 2
    linspace = np.linspace(side_spacing, width - side_spacing, num_points_per_axis, endpoint=True)
    xx, yy = np.meshgrid(linspace, linspace)
    grid = np.stack([np.ravel(xx), np.ravel(yy)], axis=1).reshape(1, num_points_per_axis ** 2, 2)
    return grid


def get_dense_flow(source_points, target_points, width, height, order, regularization):
    xx, yy = np.meshgrid(np.arange(0, width), np.arange(0, height))
    query_points = np.stack([np.ravel(xx), np.ravel(yy)], axis=1).reshape(1, width * height, 2).astype(np.float32)
    sparse_flow_points = target_points - source_points
    vector_field_flat = tf.contrib.image.interpolate_spline(source_points, sparse_flow_points, query_points,
                                                            order=order, regularization_weight=regularization)
    return tf.reshape(vector_field_flat, (1, height, width, 2))


def get_edge(im):
    return tf.reduce_sum(tf.abs(tf.image.sobel_edges(im)), axis=-1)


WarpLayer = namedtuple('WarpLayer', 'init_source_points source_points target_points vector_field inverse_vector_field out1 out2 st_loss si_loss')


class TensorFlowFlexTPSReversibleWarper:
    def __init__(self,
                 patch_shape,
                 image_shape,
                 image_value_range,
                 learning_rate,
                 num_layer_points,
                 layer_retrieval_methods,
                 layer_st_loss_weights,
                 layer_si_loss_weights,
                 metric,
                 iterations,
                 bidirectional=True,
                 lock_source_points=False,
                 pixel_loss_weight=1,
                 interpolation_order=2,
                 regularization_weight=0.0,
                 smooth_sigma=5.,
                 sobel_filter=True,
                 num_boundary_points=None,
                 init_offsets=None,
                 offset_strength=0.01):
        """Class used to estimate a warping between two images using the Adam optimizer from TensorFlow.

        Parameters
        ----------
        patch_shape : tuple of ints
            Tuple (width, height) defining the size of a patch to calculate the warping on. Has to be smaller or equal to of image_shape.
        image_shape : tuple of ints
            Tuple (width, height) defining the size of the image data. 
        image_value_range : tuple of floats
            Tuple (low, high) of the minimum and maximum value the images contain. It is used in metric normalization.
        learning_rate : float
            The initial learning rate for the Adam optimizer.
        num_layer_points : tuple of ints
            Tuple of number of mapping points per layer. The number of layers is inferred by the length of this tuple. 
        layer_retrieval_methods : tuple of ({'grid', 'grid+random', 'error'} or 2D-array)
            Tuple of values that describe how initial mapping points will be determined. Length of the tuple has to be the same as the
            length of num_layer_points. Possible values (where n is the number of mapping points for the particular layer):
                - 'grid': Create a sqrt(n) times sqrt(n) regular grid of points. Requires n to be a square number.
                - 'grid+random': Initializes the points like with 'grid', but the points will be offset by a random value.
                - 'error': Initial points will be sampled based on the per-pixel difference of intensity of the two images. Regions of higher
                    distances are therefore more likely to contain a mapping point.
                - 2D-array: If an array is given, it will perform the same sampling as in 'error', but using the given array to determine
                    the distribution to sample from.
        layer_st_loss_weights : tuple of floats
            Tuple of values weighting the (source-target-)loss between source and target mapping points per layer. It is determined as the
            mean squared distance between the source and target of each mapping point.
        layer_si_loss_weights : tuple of floats
            Tuple of value weighting the (source-initialization-)loss between source points and the initial position. Like the st-loss it
            is determined as the mean squared distance between the positions.
        metric : {'ssim', 'ssim_multiscale', 'mse', 'psnr', 'psnr_hvs'}
            Name of the metric used to calculate the pixel-wise loss between the target and the warped images.
        iterations : int
            Number of iterations for the optimizer.
        bidirectional : boolean, optional
            If true, the loss function will also include the pixel-wise loss between the inverse warping applied to image2 and image1. Otherwise,
            only the forward warping will be optimized.
        lock_source_points : boolean, optional
            If true, the source points will remain in their initial position and can not be changed by the optimizer.
        pixel_loss_weight : float, optional
            Weight for the per-pixel loss metric in the total loss.
        interpolation_order : int, optional
            Order of the spline interpolation that calculates the dense warp field.
        regularization_weight : float, optional
            Additional regularization for the spline interpolation.
        smooth_sigma : float, optional
            If > 0, a gaussian blur with the specified sigma (and 3*sigma width and height) will be applied to the images and the warping results.
            This can significantly slow down the calculation.
        sobel_filter : boolean, optional
            If true, a sobel edge detection filter will be applied to the images before warping.
        num_boundary_points : int or 'corners', optional
            Used to add border constraints for the spline interpolation, implemented as fixed mapping points. If 'corners', a boundary point will
            added in each corner of the image. If the value is an integer n > 0, n evenly spaced (centered) boundary points will be added to each
            edge of the image. Centered means that the points are evenly spaced between corners and themselves.
        init_offsets : array, optional
            If not None, the initial target points will be offset by values in this array instead of being offset randomly.
        offset_strength : float, optional
            If no init_offsets are given, this values determines how strong the random offset is applied to the initial target points. Has to be
            greater than 0 to prevent possibly unsolvable equations for the spline interpolation.
        """

        # lazily import tensorflow
        global tf
        import tensorflow as tf


        assert len(patch_shape) == 2
        assert patch_shape[0] == patch_shape[1]
        self.patch_shape = patch_shape
        self.image_shape = image_shape
        self.input_shape = (1, *patch_shape, 1)

        height, width = patch_shape

        self.init_offsets = init_offsets
        self.num_boundary_points = num_boundary_points
        self.num_layer_points = num_layer_points
        self.layer_retrieval_methods = layer_retrieval_methods
        self.iterations = iterations
        self.offset_strength = offset_strength

        self.graph = tf.Graph()

        with self.graph.as_default():  # , tf.device('/gpu:0'):

            # input variables
            self.image_off = tf.Variable(initial_value=(0,0,0,0), dtype=tf.int32, name='patch_offset')

            self.image1_t = tf.Variable(initial_value=np.zeros((1,*image_shape,1)), dtype=tf.float32, name='input1')
            self.image2_t = tf.Variable(initial_value=np.zeros((1,*image_shape,1)), dtype=tf.float32, name='input2')

            # select a patch with self.image_off as the upper left corner
            self.image1_t_patch = tf.slice(self.image1_t, self.image_off, (1,*patch_shape,1))
            self.image2_t_patch = tf.slice(self.image2_t, self.image_off, (1, *patch_shape, 1))

            if smooth_sigma is not None:
                smooth_params = (int(2 * smooth_sigma), 0., smooth_sigma)
                self.image1_t_smooth = smoothed_image(self.image1_t_patch, *smooth_params)
                self.image2_t_smooth = smoothed_image(self.image2_t_patch, *smooth_params)
            else:
                self.image1_t_smooth = self.image1_t_patch
                self.image2_t_smooth = self.image2_t_patch

            if sobel_filter:
                self.image1_t_postproc = get_edge(self.image1_t_smooth)
                self.image2_t_postproc = get_edge(self.image2_t_smooth)
            else:
                self.image1_t_postproc = self.image1_t_smooth
                self.image2_t_postproc = self.image2_t_smooth

            self.layers = []

            layer_input1 = self.image1_t_smooth
            layer_input2 = self.image2_t_smooth

            self.boundary_points = self.get_boundary_points()

            if self.init_offsets is None:
                init_offsets = [self.offset_strength*np.random.standard_normal((1, num, 2)) for num in num_layer_points]
            else:
                init_offsets = self.init_offsets

            for i, (num_points, retrieval) in enumerate(zip(self.num_layer_points, self.layer_retrieval_methods)):
                with tf.name_scope('warp_layer_'+str(i)):
                    init_sp = self.get_initial_source_points(num_points, retrieval, None)

                    sp = tf.Variable(initial_value=init_sp, dtype=tf.float32, name='source_points')
                    tp = tf.Variable(initial_value=init_sp + init_offsets[i],
                                     dtype=tf.float32, name='target_points')

                    if self.boundary_points is None:
                        sp_and_boundary = sp
                        tp_and_boundary = tp
                    else:
                        sp_and_boundary = tf.concat([sp, self.boundary_points], axis=1)
                        tp_and_boundary = tf.concat([tp, self.boundary_points], axis=1)

                    vf = get_dense_flow(sp_and_boundary, tp_and_boundary, width, height,
                                        interpolation_order, regularization_weight)
                    ivf = get_dense_flow(tp_and_boundary, sp_and_boundary, width, height,
                                         interpolation_order, regularization_weight)

                    out1 = tf.contrib.image.dense_image_warp(layer_input1, vf, name='dense1_warp')
                    out2 = tf.contrib.image.dense_image_warp(layer_input2, ivf, name='dense2_warp')
                    st_loss = layer_st_loss_weights[i] * tf.losses.mean_squared_error(sp, tp)
                    si_loss = layer_si_loss_weights[i] * tf.losses.mean_squared_error(init_sp, sp)
                    self.layers.append(WarpLayer(init_sp, sp, tp, vf, ivf, out1, out2, st_loss, si_loss))

                    layer_input1 = out1
                    layer_input2 = out2

            # self.vector_field = tf.reduce_sum([layer.vector_field for layer in layers], axis=0)
            self.source_points = tf.concat([layer.source_points for layer in self.layers], axis=1)
            self.target_points = tf.concat([layer.target_points for layer in self.layers], axis=1)
            self.vector_fields = tf.concat([layer.vector_field for layer in self.layers], axis=0)
            self.inverse_vector_fields = tf.concat([layer.inverse_vector_field for layer in self.layers], axis=0)

            # these are in fact the outputs of the last WarpLayer
            self.displaced1 = layer_input1
            self.displaced2 = layer_input2

            if smooth_sigma is not None:
                self.displaced1_smooth = smoothed_image(self.displaced1, *smooth_params)
                self.displaced2_smooth = smoothed_image(self.displaced2, *smooth_params)
            else:
                self.displaced1_smooth = self.displaced1
                self.displaced2_smooth = self.displaced2

            ssim_weights = np.array([0.0448, 0.2856, 0.3001, 0.2363])
            ssim_weights /= sum(ssim_weights)

            if sobel_filter:
                self.displaced1_postproc = get_edge(self.displaced1_smooth)
                self.displaced2_postproc = get_edge(self.displaced2_smooth)
            else:
                self.displaced1_postproc = self.displaced1_smooth
                self.displaced2_postproc = self.displaced2_smooth

            # loss functions
            cmp_1 = (self.image2_t_postproc, self.displaced1_postproc)
            cmp_2 = (self.image1_t_postproc, self.displaced2_postproc)

            with tf.name_scope('pixel_losses'):
                if metric == 'ssim':
                    self.pixel_loss1 = (1 - tf.image.ssim(*cmp_1, 1.)) / 2
                    self.pixel_loss2 = (1 - tf.image.ssim(*cmp_2, 1.)) / 2
                elif metric == 'ssim_multiscale':
                    self.pixel_loss1 = (1 - tf.image.ssim_multiscale(*cmp_1, 1., ssim_weights)) / 2
                    self.pixel_loss2 = (1 - tf.image.ssim_multiscale(*cmp_2, 1., ssim_weights)) / 2
                elif metric == 'psnr_hvs':
                    self.pixel_loss1 = -get_p_hvs(*cmp_1, image_value_range[1])
                    self.pixel_loss2 = -get_p_hvs(*cmp_2, image_value_range[1])
                elif metric == 'mse':
                    #max_distance = image_shape[0] * image_shape[1] * (image_value_range[1] - image_value_range[0])
                    self.pixel_loss1 = tf.sqrt(tf.losses.mean_squared_error(*cmp_1))
                    self.pixel_loss2 = tf.sqrt(tf.losses.mean_squared_error(*cmp_2))
                elif metric == 'psnr':
                    self.pixel_loss1 = tf.image.psnr(*cmp_1, image_value_range[1])
                    self.pixel_loss2 = tf.image.psnr(*cmp_2, image_value_range[1])

                if bidirectional:
                    self.pixel_loss = pixel_loss_weight * (self.pixel_loss1 + self.pixel_loss2)
                else:
                    self.pixel_loss = pixel_loss_weight * self.pixel_loss1
                # self.histogram_loss = histogram_loss1 + histogram_loss2

                self.st_losses = tf.reduce_sum([l.st_loss for l in self.layers]) / len(self.layers)
                self.si_losses = tf.reduce_sum([l.si_loss for l in self.layers]) / len(self.layers)

                self.loss = tf.reshape(self.pixel_loss, [1])[0] + self.st_losses + self.si_losses

            if lock_source_points:
                variables = [l.target_points for l in self.layers]
            else:
                variables = [l.source_points for l in self.layers] + [l.target_points for l in self.layers]

            with tf.name_scope('optimizer'):
                self.global_step = tf.Variable(initial_value=0, dtype=tf.int64)

                # create optimizer
                self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                self.minimize_op = self.optimizer.minimize(self.loss, var_list=variables, global_step=self.global_step)
                self.initializer = tf.variables_initializer((*self.optimizer.variables(), *variables))

                # create loss helper
                self.all_losses = tf.Variable(initial_value=np.zeros(iterations), dtype=tf.float32)
                self.save_loss = tf.scatter_update(self.all_losses, self.global_step, self.loss)

                #file_writer = tf.summary.FileWriter("tensorboard/logs",graph=self.graph)

                self.step_op = tf.group([self.minimize_op, self.save_loss])
            self.graph.finalize()

        self.tensors = dict(loss=self.loss,
                            vector_fields=self.vector_fields,
                            inverse_vector_fields=self.inverse_vector_fields,
                            source_points=self.source_points,
                            target_points=self.target_points,
                            displaced1=self.displaced1,
                            displaced2=self.displaced2,
                            displaced1_smooth=self.displaced1_smooth,
                            displaced2_smooth=self.displaced2_smooth,
                            displaced1_postproc=self.displaced1_postproc,
                            displaced2_postproc=self.displaced2_postproc,

                            pixel_loss=self.pixel_loss,
                            st_loss=self.st_losses,
                            si_loss=self.si_losses,
                            image1_t_smooth=self.image1_t_smooth,
                            image2_t_smooth=self.image2_t_smooth,
                            image1_t_postproc=self.image1_t_postproc,
                            image2_t_postproc=self.image2_t_postproc)
        self.session = tf.Session(graph=self.graph) #tf_debug.LocalCLIDebugWrapperSession(sess)

    def get_boundary_points(self):
        boundary = None

        if self.num_boundary_points == 'corners':
            boundary = np.zeros((1, 4, 2), dtype=np.float32)
            boundary[0, 0] = [0, 0]
            boundary[0, 1] = [0, self.patch_shape[1]]
            boundary[0, 2] = [self.patch_shape[0], 1]
            boundary[0, 3] = [self.patch_shape[0], self.patch_shape[1]]
        elif self.num_boundary_points > 0:
            # add n points per edge
            nbp = self.num_boundary_points
            width_spacing = self.patch_shape[0] / nbp / 2
            height_spacing = self.patch_shape[1] / nbp / 2
            boundary = np.zeros((1, 4 * nbp, 2), dtype=np.float32)
            boundary[0, :nbp, 0] = np.linspace(width_spacing, self.patch_shape[0] - width_spacing, num=nbp)
            boundary[0, :nbp, 1] = np.zeros(nbp)

            boundary[0, nbp:2 * nbp, 0] = np.linspace(width_spacing, self.patch_shape[0] - width_spacing, num=nbp)
            boundary[0, nbp:2 * nbp, 1] = np.ones(nbp) * self.patch_shape[1]

            boundary[0, 2 * nbp:3 * nbp, 0] = np.zeros(nbp)
            boundary[0, 2 * nbp:3 * nbp, 1] = np.linspace(height_spacing, self.patch_shape[1] - height_spacing, num=nbp)

            boundary[0, 3 * nbp:4 * nbp, 0] = np.ones(nbp) * self.patch_shape[0]
            boundary[0, 3 * nbp:4 * nbp, 1] = np.linspace(height_spacing, self.patch_shape[1] - height_spacing, num=nbp)

        return boundary

    def get_initial_source_points(self, num_points, retrieval, image_error):
        if retrieval == 'grid':
            num_points = int(np.sqrt(num_points))
            init_sp = get_point_grid(num_points, *self.patch_shape)
        elif retrieval == 'grid+random':
            init_sp = get_point_grid(num_points, *self.patch_shape)
            init_sp += np.random.standard_normal(init_sp.shape)
        elif retrieval == 'error':
            if image_error is None:
                num_points = int(np.sqrt(num_points))
                init_sp = get_point_grid(num_points, *self.patch_shape)
            else:
                init_sp = sample_based_on_density(image_error, n=num_points)
                init_sp = init_sp.reshape((1, *init_sp.shape))
        else:
            init_sp = sample_based_on_density(retrieval, n=num_points)
            init_sp = init_sp.reshape((1, *init_sp.shape))

        return init_sp

    def reinitialize(self, image_error):
        # TODO put everything in initializer op
        init_fetches = [self.initializer, self.all_losses.initializer,self.global_step.initializer,
                        self.image_off.initializer]
        with self.session.as_default():
            self.session.run(init_fetches)

        if self.init_offsets is None:
            init_offsets = [self.offset_strength*np.random.standard_normal((1, num, 2)) for num in self.num_layer_points]
        else:
            init_offsets = self.init_offsets

        for i, (layer, num_points, retrieval) in enumerate(
                zip(self.layers, self.num_layer_points, self.layer_retrieval_methods)):
            init_sp = self.get_initial_source_points(num_points, retrieval, image_error)

            layer.source_points.load(init_sp)
            layer.target_points.load(init_sp + init_offsets[i])

    def get_current(self, tensor_names):
        with self.session.as_default():
            try:
                return self.session.run([self.tensors[n] for n in tensor_names])
            except KeyError:
                return self.session.run(self.tensors[tensor_names])

    def load_images(self, image1, image2):
        assert image1.shape == self.image_shape and image2.shape == self.image_shape
        image1r = image1.reshape((1,*self.image_shape,1))
        image2r = image2.reshape((1,*self.image_shape,1))
        image_error = np.abs(image1 - image2) ** 2

        with self.session.as_default():
            self.reinitialize(image_error)
            self.image1_t.load(image1r)
            self.image2_t.load(image2r)

    def evaluate(self, off=None, callback=lambda *args: None):
        with self.session.as_default():
            if off is not None:
                self.reinitialize(np.zeros((1,*self.input_shape,1)))
                self.image_off.load((0, *off, 0))

            for i in range(self.iterations):
                self.session.run(self.step_op)
                callback(i)

            return self.session.run(self.all_losses)
