#from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from skimage.transform import rescale
import numpy as np
#import dill
from tqdm.autonotebook import tqdm


# def load_image_and_preprocess(path, target_size=(224, 224)):
#     image = load_img(path, target_size=target_size)
#     image = np.asarray(img_to_array(image), dtype=np.uint8)
#     image = image.reshape((1, *image.shape))
#     return image#preprocess_input(image)


def deprocess_vgg_image(x):
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


def get_object(path, meta):
    with open(path, 'rb') as file:
        obj = dill.load(file)
        if obj['meta'] == meta:
            return obj['data']
        print("meta does not match")
        raise ValueError("Object's meta does not match parameter.")


def store_object(path, data, meta):
    with open(path, 'wb') as file:
        obj = {'data': data, 'meta': meta}
        dill.dump(obj, file)

NUM_CHANNEL_TYPE_MAP = {1: 'grayscale', 3:'rgb', 4:'rgba'}


def get_image_type(img):
    if len(img.shape) == 3:
        num_channels = img.shape[2]
    elif len(img.shape) == 2:
        num_channels = 1
    else:
        raise ValueError("img must be 2D or 3D (channels)")

    return NUM_CHANNEL_TYPE_MAP[num_channels]


def convert_image(img, to_type=None):
    # guess image type based on channels
    cur_type = get_image_type(img)

    if cur_type == 'grayscale':
        if to_type == 'rgb':
            img = np.repeat(img.reshape(*img.shape, 1), 3, axis=2)
        elif to_type == 'rgba':
            img = np.repeat(img.reshape(*img.shape, 1), 3, axis=2)
            img = np.concatenate([img, np.ones_like(img[:,:,0])], axis=2)

    elif cur_type == 'rgb':
        if to_type == 'grayscale':
            img = np.mean(img, axis=2)
        elif to_type == 'rgba':
            img = np.concatenate([img, np.ones_like(img[:,:,0])], axis=2)

    elif cur_type == 'rgba':
            if to_type == 'grayscale':
                img = np.mean(img[:,:,:3], axis=2)
            elif to_type == 'rgb':
                img = img[:,:,:3]

    if len(img.shape) == 2:
        assert to_type == NUM_CHANNEL_TYPE_MAP[1]
    else:
        assert to_type == NUM_CHANNEL_TYPE_MAP[img.shape[2]]

    return img


def normalize_image(image):
    return (image - image.min()) / np.ptp(image)

def load_image(path, scale=1., astype=None, normalize=False):
    img = plt.imread(path)

    if astype is not None:
        img = convert_image(img, astype)

    img = rescale(img, scale)

    if normalize:
        img = normalize_image(img)

    return img


def plot_imshow_grid(images, axs=None, shape=None, figsize=None, **imshow_params):
    if shape is None:
        sqrt_len = np.sqrt(len(images))
        num_cols = int(sqrt_len)
        num_rows = int(len(images)/num_cols)
        if num_cols * num_rows < len(images):
            num_rows += 1
        shape = (num_rows, num_cols)

    if figsize is None:
        figsize = (2*shape[0], 2*shape[1])

    if axs is None:
        fig, axs = plt.subplots(*shape, figsize=figsize)
    else:
        fig = None

    for im, ax in zip(images, np.ravel(axs)):
        if im is not None:
            ax.imshow(im, **imshow_params)

    return fig, axs


class ProgressCallback:
    def __init__(self, fn=lambda *args: None, *args, **kwargs):
        self.pbar = tqdm(*args, **kwargs)
        self.callback = fn

    def __call__(self, *args, **kwargs):
        self.callback(*args, **kwargs)
        self.pbar.update(1)

    def close(self):
        self.pbar.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class CircularBuffer:
    def __init__(self, size):
        self.size = size
        self.buffer = np.zeros(self.size)
        self.buffer_pos = 0
        self.num_added = 0

    def get_relative(self, rel_index):
        abs_index = (self.buffer_pos - rel_index) % self.size
        return self.buffer[abs_index]

    def get_oldest(self):
        return self.get_relative(1)

    def get_newest(self):
        return self.get_relative(0)

    def get_mean(self):
        return self.buffer.mean()

    def get_recent_mean(self, num_elements):
        end = self.buffer_pos
        start = (end - num_elements) % self.size

        if start > end:
            mean = (self.buffer[start:].mean() + self.buffer[:end].mean()) / 2
        else:
            mean = self.buffer[start:end].mean()

        return mean

    def filled_once(self):
        return self.num_added > self.size

    def add(self, val):
        self.buffer_pos = (self.buffer_pos + 1) % self.size
        self.buffer[self.buffer_pos] = val
        self.num_added += 1


class KeepBest:
    def __init__(self, key=lambda val: val, type='min'):
        self.best_data = None
        self.best_value = None
        self.key_fn = key

        if type not in ('min', 'max'):
            raise ValueError("type parameter has to be 'min' or 'max'.")

        self.type = type

    def _is_better(self, val):
        if self.best_value is None:
            return True
        elif self.type == 'min':
            return val < self.best_value
        elif self.type == 'max':
            return val > self.best_value

    def check(self, new_data, new_value=None):
        if new_value is None:
            new_value = self.key_fn(new_data)

        self.check_lazy(lambda: new_data, new_value)

    def check_lazy(self, data_provider_fn, new_value):
        if self._is_better(new_value):
            self.best_value = new_value
            self.best_data = data_provider_fn()
