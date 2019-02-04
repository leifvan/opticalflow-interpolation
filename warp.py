import numpy as np
from skimage.feature import canny
from scipy.ndimage import geometric_transform
from scipy import interpolate
from pycpd import deformable_registration


def get_edge_image(image):
    # TODO improve edge detection
    fg_bg = canny(image, sigma=1, low_threshold=0.5, high_threshold=0.5, use_quantiles=True)
    fg_bg = fg_bg == 1
    bg_fg = np.invert(fg_bg)

    fg_bg_ratio = np.count_nonzero(fg_bg) / image.size

    if np.isclose(fg_bg_ratio, 0.5, atol=0.1):
        print("Warning, image has no clear foreground / background separation!")

    if fg_bg_ratio > 0.5:
        # foreground is actually background
        return bg_fg

    return fg_bg


def get_point_set(binary_image, subsample=1):
    return np.argwhere(binary_image == 1.)[::subsample]


def visualize_registration(ax, reg, show_x=True, show_y=True, show_ty=True):
    ax.clear()
    if show_x: ax.scatter(reg.X[:,0],reg.X[:,1], color='red', s=0.3)
    if show_y: ax.scatter(reg.Y[:,0],reg.Y[:,1], color='green', s=0.3)
    if show_ty: ax.scatter(reg.TY[:,0],reg.TY[:,1], color='blue', s=0.3)


def visualize_displacement(ax_u, ax_v, u, v, width, height):
    x_dense, y_dense = np.meshgrid(np.arange(0, width), np.arange(0, height))

    def plot_displacement_on_axis(ax, displacement):
        ax.pcolor(x_dense, y_dense, np.transpose(displacement), cmap='coolwarm')
        ax.scatter(x_dense, y_dense, 60, displacement, cmap='coolwarm')
        ax.set_aspect('equal')
        ax.set_ylim(height, 0)

    plot_displacement_on_axis(ax_u, u)
    plot_displacement_on_axis(ax_v, v)


def get_cpd_transformation(target_points, points_to_register, alpha, beta, tolerance=1e-6, max_iter=100,
                           callback=lambda *args: None):
    reg = deformable_registration(X=target_points , Y=points_to_register, alpha=alpha, beta=beta,
                                  tolerance=tolerance, max_iterations=max_iter)

    def cb(*args, **kwargs):
        callback(reg)

    reg.register(callback=cb)
    return reg.TY


def get_tps_displacement(points, transformed_points, width, height):
    x_dense, y_dense = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x = points[:, 0]
    y = points[:, 1]
    tx = transformed_points[:, 0]
    ty = transformed_points[:, 1]
    u_sparse = x - tx
    v_sparse = y - ty

    rbf_u = interpolate.Rbf(tx, ty, u_sparse, smooth=0, epsilon=100, function='thin_plate')
    rbf_v = interpolate.Rbf(tx, ty, v_sparse, smooth=0, epsilon=100, function='thin_plate')

    u_dense = np.transpose(rbf_u(x_dense, y_dense))
    v_dense = np.transpose(rbf_v(x_dense, y_dense))

    return u_dense, v_dense


def apply_displacement(image, u, v):
    def mapper(tup):
        x, y = tup
        return x + u[x, y], y + v[x, y]
    return geometric_transform(image, mapping=mapper, order=1, mode='nearest')


def is_binary(arr):
    return ((arr == 0) | (arr == 1)).all()


def get_displacement_from_images(target_image, image_to_displace, subsample=1, alpha=1, beta=50,
                                 tolerance=1e-6, max_iter=100, callback=lambda *args: None):
    assert target_image.shape == image_to_displace.shape
    assert target_image.dtype == image_to_displace.dtype
    height, width = target_image.shape

    if not is_binary(target_image):
        t_canny = get_edge_image(target_image)
        d_canny = get_edge_image(image_to_displace)
    else:
        t_canny = target_image
        d_canny = image_to_displace

    t_points = get_point_set(t_canny, subsample)
    d_points = get_point_set(d_canny, subsample)

    transformed = get_cpd_transformation(t_points, d_points, alpha=alpha, beta=beta,
                                         tolerance=tolerance, max_iter=max_iter, callback=callback)

    return get_tps_displacement(d_points, transformed, width, height)


