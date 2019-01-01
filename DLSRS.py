#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
U-net deep learning framework for processing hsSRS images
Right now the code is intended for denoising using 3D U-net denoising framework


Using Tensorflow and Keras as backend

@author: Haonan Lin

"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sys
import warnings
import datetime
import argparse
import json
import shutil
import os

from tifffile import imread
from keras import backend as K
from six.moves import range, zip, map, reduce#, filter
from six import string_types
from collections import namedtuple, deque
from tqdm import tqdm
from tifffile import imsave

try:
    from pathlib import Path
    Path().expanduser()
except (ImportError,AttributeError):
    from pathlib2 import Path
    
try:
    import tempfile
    tempfile.TemporaryDirectory
except (ImportError,AttributeError):
    from backports import tempfile


"""
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Utils and IO
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------ 
"""    

def plot_history(history,*keys,**kwargs):
    """Plot (Keras) training history returned by :func:`DLSRS.train`."""
    logy = kwargs.pop('logy',False)

    if all(( isinstance(k,string_types) for k in keys )):
        w, keys = 1, [keys]
    else:
        w = len(keys)

    plt.gcf()
    for i, group in enumerate(keys):
        plt.subplot(1,w,i+1)
        for k in ([group] if isinstance(group,string_types) else group):
            plt.plot(history.epoch,history.history[k],'.-',label=k,**kwargs)
            if logy:
                plt.gca().set_yscale('log', nonposy='clip')
        plt.xlabel('epoch')
        plt.legend(loc='best')
    # plt.tight_layout()
    plt.show()
    
def plot_some(*arr, **kwargs):
    """Quickly plot multiple images at once."""

    title_list = kwargs.pop('title_list',None)
    pmin = kwargs.pop('pmin',0)
    pmax = kwargs.pop('pmax',100)
    cmap = kwargs.pop('cmap','magma')
    imshow_kwargs = kwargs
    return _plot_some(arr=arr, title_list=title_list, pmin=pmin, pmax=pmax, cmap=cmap, **imshow_kwargs)

def _plot_some(arr, title_list=None, pmin=0, pmax=100, cmap='magma', **imshow_kwargs):
    """
    plots a matrix of images

    arr = [ X_1, X_2, ..., X_n]

    where each X_i is a list of images

    :param arr:
    :param title_list:
    :param pmin:
    :param pmax:
    :param imshow_kwargs:
    :return:
    """
    import matplotlib.pyplot as plt

    imshow_kwargs['cmap'] = cmap

    def color_image(a):
        return np.stack(map(to_color,a)) if 1<a.shape[-1]<=3 else np.squeeze(a)
    def max_project(a):
        return np.max(a,axis=1) if (a.ndim==4 and not 1<=a.shape[-1]<=3) else a

    arr = map(color_image,arr)
    arr = map(max_project,arr)
    arr = list(arr)

    h = len(arr)
    w = len(arr[0])
    plt.gcf()
    for i in range(h):
        for j in range(w):
            plt.subplot(h, w, i * w + j + 1)
            try:
                plt.title(title_list[i][j], fontsize=8)
            except:
                pass
            img = arr[i][j]
            if pmin!=0 or pmax!=100:
                img = normalize(img,pmin=pmin,pmax=pmax,clip=True)
            plt.imshow(np.squeeze(img),**imshow_kwargs)
            plt.axis("off")
            
def to_color(arr, pmin=1, pmax=99.8, gamma=1., colors=((0, 1, 0), (1, 0, 1), (0, 1, 1))):
    """Converts a 2D or 3D stack to a colored image (maximal 3 channels).

    Parameters
    ----------
    arr : numpy.ndarray
        2D or 3D input data
    pmin : float
        lower percentile, pass -1 if no lower normalization is required
    pmax : float
        upper percentile, pass -1 if no upper normalization is required
    gamma : float
        gamma correction
    colors : list
        list of colors (r,g,b) for each channel of the input

    Returns
    -------
    numpy.ndarray
        colored image
    """
    if not arr.ndim in (2,3):
        raise ValueError("only 2d or 3d arrays supported")

    if arr.ndim ==2:
        arr = arr[np.newaxis]

    ind_min = np.argmin(arr.shape)
    arr = np.moveaxis(arr, ind_min, 0).astype(np.float32)

    out = np.zeros(arr.shape[1:] + (3,))

    eps = 1.e-20
    if pmin>=0:
        mi = np.percentile(arr, pmin, axis=(1, 2), keepdims=True)
    else:
        mi = 0

    if pmax>=0:
        ma = np.percentile(arr, pmax, axis=(1, 2), keepdims=True)
    else:
        ma = 1.+eps

    arr_norm = (1. * arr - mi) / (ma - mi + eps)


    for i_stack, col_stack in enumerate(colors):
        if i_stack >= len(arr):
            break
        for j, c in enumerate(col_stack):
            out[..., j] += c * arr_norm[i_stack]

    return np.clip(out, 0, 1)            

def normalize(x, pmin=3, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=np.float32):
    """Percentile-based image normalization."""

    mi = np.percentile(x,pmin,axis=axis,keepdims=True)
    ma = np.percentile(x,pmax,axis=axis,keepdims=True)
    return normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)

def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
    if dtype is not None:
        x   = x.astype(dtype,copy=False)
        mi  = dtype(mi) if np.isscalar(mi) else mi.astype(dtype,copy=False)
        ma  = dtype(ma) if np.isscalar(ma) else ma.astype(dtype,copy=False)
        eps = dtype(eps)

    try:
        import numexpr
        x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
    except ImportError:
        x =                   (x - mi) / ( ma - mi + eps )

    if clip:
        x = np.clip(x,0,1)

    return x

def axes_check_and_normalize(axes,length=None,disallowed=None,return_allowed=False):
    """
    S(ample), T(ime), C(hannel), Z, Y, X
    """
    allowed = 'STCZYX'
    axes = str(axes).upper()
    consume(a in allowed or _raise(ValueError("invalid axis '%s', must be one of %s."%(a,list(allowed)))) for a in axes)
    disallowed is None or consume(a not in disallowed or _raise(ValueError("disallowed axis '%s'."%a)) for a in axes)
    consume(axes.count(a)==1 or _raise(ValueError("axis '%s' occurs more than once."%a)) for a in axes)
    length is None or len(axes)==length or _raise(ValueError('axes (%s) must be of length %d.' % (axes,length)))
    return (axes,allowed) if return_allowed else axes

def _raise(e):
    raise e

# https://docs.python.org/3/library/itertools.html#itertools-recipes
def consume(iterator):
    deque(iterator, maxlen=0)
        
def compose(*funcs):
    return lambda x: reduce(lambda f,g: g(f), funcs, x)

def move_image_axes(x, fr, to, adjust_singletons=False):
    """
    x: ndarray
    fr,to: axes string (see `axes_dict`)
    """
    fr = axes_check_and_normalize(fr, length=x.ndim)
    to = axes_check_and_normalize(to)

    fr_initial = fr
    x_shape_initial = x.shape
    adjust_singletons = bool(adjust_singletons)
    if adjust_singletons:
        # remove axes not present in 'to'
        slices = [slice(None) for _ in x.shape]
        for i,a in enumerate(fr):
            if (a not in to) and (x.shape[i]==1):
                # remove singleton axis
                slices[i] = 0
                fr = fr.replace(a,'')
        x = x[slices]
        # add dummy axes present in 'to'
        for i,a in enumerate(to):
            if (a not in fr):
                # add singleton axis
                x = np.expand_dims(x,-1)
                fr += a

    if set(fr) != set(to):
        _adjusted = '(adjusted to %s and %s) ' % (x.shape, fr) if adjust_singletons else ''
        raise ValueError(
            'image with shape %s and axes %s %snot compatible with target axes %s.'
            % (x_shape_initial, fr_initial, _adjusted, to)
        )

    ax_from, ax_to = axes_dict(fr), axes_dict(to)
    if fr == to:
        return x
    return np.moveaxis(x, [ax_from[a] for a in fr], [ax_to[a] for a in fr])

def axes_dict(axes):
    """
    from axes string to dict
    """
    axes, allowed = axes_check_and_normalize(axes,return_allowed=True)
    return { a: None if axes.find(a) == -1 else axes.find(a) for a in allowed }
    # return collections.namedtuple('Axes',list(allowed))(*[None if axes.find(a) == -1 else axes.find(a) for a in allowed ])

def limit_gpu_memory(fraction, allow_growth=False):
    """Limit GPU memory allocation for TensorFlow (TF) backend.

    Parameters
    ----------
    fraction : float
        Limit TF to use only a fraction (value between 0 and 1) of the available GPU memory.
        Reduced memory allocation can be disabled if fraction is set to ``None``.
    allow_growth : bool, optional
        If ``False`` (default), TF will allocate all designated (see `fraction`) memory all at once.
        If ``True``, TF will allocate memory as needed up to the limit imposed by `fraction`; this may
        incur a performance penalty due to memory fragmentation.

    Raises
    ------
    ValueError
        If `fraction` is not ``None`` or a float value between 0 and 1.
    NotImplementedError
        If TensorFlow is not used as the backend.
    """

    is_tf_backend() or _raise(NotImplementedError('Not using tensorflow backend.'))
    fraction is None or (np.isscalar(fraction) and 0<=fraction<=1) or _raise(ValueError('fraction must be between 0 and 1.'))

    if K.tensorflow_backend._SESSION is None:
        config = tf.ConfigProto()
        if fraction is not None:
            config.gpu_options.per_process_gpu_memory_fraction = fraction
        config.gpu_options.allow_growth = bool(allow_growth)
        session = tf.Session(config=config)
        K.tensorflow_backend.set_session(session)
        print("[tf_limit]\t setting config.gpu_options.per_process_gpu_memory_fraction to ",config.gpu_options.per_process_gpu_memory_fraction)
    else:
        warnings.warn('Too late too limit GPU memory, can only be done once and before any computation.')

def is_tf_backend():
    import keras.backend as K
    return K.backend() == 'tensorflow'

def save_tiff_imagej_compatible(file, img, axes, **imsave_kwargs):
    """Save image in ImageJ-compatible TIFF format.

    Parameters
    ----------
    file : str
        File name
    img : numpy.ndarray
        Image
    axes: str
        Axes of ``img``
    imsave_kwargs : dict, optional
        Keyword arguments for :func:`tifffile.imsave`

    """
    axes = axes_check_and_normalize(axes,img.ndim,disallowed='S')

    # convert to imagej-compatible data type
    t = img.dtype
    if   'float' in t.name: t_new = np.float32
    elif 'uint'  in t.name: t_new = np.uint16 if t.itemsize >= 2 else np.uint8
    elif 'int'   in t.name: t_new = np.int16
    else:                   t_new = t
    img = img.astype(t_new, copy=False)
    if t != t_new:
        warnings.warn("Converting data type from '%s' to ImageJ-compatible '%s'." % (t, np.dtype(t_new)))

    # move axes to correct positions for imagej
    img = move_image_axes(img, axes, 'TZCYX', True)

    imsave_kwargs['imagej'] = True
    imsave(file, img, **imsave_kwargs)



def load_training_data(file, validation_split=0, axes=None, n_images=None, verbose=False):
    """Load training data from file in ``.npz`` format.

    The data file is expected to have the keys:

    - ``X``    : Array of training input images.
    - ``Y``    : Array of corresponding target images.
    - ``axes`` : Axes of the training images.


    Parameters
    ----------
    file : str
        File name
    validation_split : float
        Fraction of images to use as validation set during training.
    axes: str, optional
        Must be provided in case the loaded data does not contain ``axes`` information.
    n_images : int, optional
        Can be used to limit the number of images loaded from data.
    verbose : bool, optional
        Can be used to display information about the loaded images.

    Returns
    -------
    tuple( tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray`), tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray`), str )
        Returns two tuples (`X_train`, `Y_train`), (`X_val`, `Y_val`) of training and validation sets
        and the axes of the input images.
        The tuple of validation data will be ``None`` if ``validation_split = 0``.

    """

    f = np.load(file)
    X, Y = f['X'], f['Y']
    if axes is None:
        axes = f['axes']
    axes = axes_check_and_normalize(axes)

    assert X.shape == Y.shape
    assert len(axes) == X.ndim
    assert 'C' in axes
    if n_images is None:
        n_images = X.shape[0]
    assert X.shape[0] == Y.shape[0]
    assert 0 < n_images <= X.shape[0]
    assert 0 <= validation_split < 1

    X, Y = X[:n_images], Y[:n_images]
    channel = axes_dict(axes)['C']

    if validation_split > 0:
        n_val   = int(round(n_images * validation_split))
        n_train = n_images - n_val
        assert 0 < n_val and 0 < n_train
        X_t, Y_t = X[-n_val:],  Y[-n_val:]
        X,   Y   = X[:n_train], Y[:n_train]
        assert X.shape[0] == n_train and X_t.shape[0] == n_val
        X_t = move_channel_for_backend(X_t,channel=channel)
        Y_t = move_channel_for_backend(Y_t,channel=channel)

    X = move_channel_for_backend(X,channel=channel)
    Y = move_channel_for_backend(Y,channel=channel)

    axes = axes.replace('C','') # remove channel
    if backend_channels_last():
        axes = axes+'C'
    else:
        axes = axes[:1]+'C'+axes[1:]

    data_val = (X_t,Y_t) if validation_split > 0 else None

    if verbose:
        ax = axes_dict(axes)
        n_train, n_val = len(X), len(X_t) if validation_split>0 else 0
        image_size = tuple( X.shape[ax[a]] for a in 'TZYX' if a in axes )
        n_dim = len(image_size)
        n_channel_in, n_channel_out = X.shape[ax['C']], Y.shape[ax['C']]

        print('number of training images:\t', n_train)
        print('number of validation images:\t', n_val)
        print('image size (%dD):\t\t'%n_dim, image_size)
        print('axes:\t\t\t\t', axes)
        print('channels in / out:\t\t', n_channel_in, '/', n_channel_out)

    return (X,Y), data_val, axes



def save_training_data(file, X, Y, axes):
    """Save training data in ``.npz`` format.

    Parameters
    ----------
    file : str
        File name
    X : :class:`numpy.ndarray`
        Array of patches extracted from source images.
    Y : :class:`numpy.ndarray`
        Array of corresponding target patches.
    axes : str
        Axes of the extracted patches.

    """
    isinstance(file,(Path,string_types)) or _raise(ValueError())
    file = Path(file).with_suffix('.npz')
    file.parent.mkdir(parents=True,exist_ok=True)

    axes = axes_check_and_normalize(axes)
    len(axes) == X.ndim or _raise(ValueError())
    np.savez(str(file), X=X, Y=Y, axes=axes)

def move_channel_for_backend(X,channel):
    if backend_channels_last():
        return np.moveaxis(X, channel, -1)
    else:
        return np.moveaxis(X, channel,  1)

def backend_channels_last():
    import keras.backend as K
    assert K.image_data_format() in ('channels_first','channels_last')
    return K.image_data_format() == 'channels_last'

def load_json(fpath):
    with open(fpath,'r') as f:
        return json.load(f)
      
def save_json(data,fpath,**kwargs):
    with open(fpath,'w') as f:
        f.write(json.dumps(data,**kwargs))
        
        
        
from keras.callbacks import Callback
        
class DLSRSTensorBoard(Callback):
    """ TODO """
    def __init__(self, log_dir='./logs',
                 freq=1,
                 compute_histograms=False,
                 n_images=3,
                 prob_out=False,
                 write_graph=False,
                 prefix_with_timestamp=True,
                 write_images=False):
        super(DLSRSTensorBoard, self).__init__()
        is_tf_backend() or _raise(RuntimeError('TensorBoard callback only works with the TensorFlow backend.'))

        self.freq = freq
        self.image_freq = freq
        self.prob_out = prob_out
        self.merged = None
        self.write_graph = write_graph
        self.write_images = write_images
        self.n_images = n_images
        self.compute_histograms = compute_histograms

        if prefix_with_timestamp:
            log_dir = os.path.join(log_dir, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%f"))

        self.log_dir = log_dir

    def set_model(self, model):
        self.model = model
        self.sess = K.get_session()
        tf_sums = []

        if self.compute_histograms and self.freq and self.merged is None:
            for layer in self.model.layers:
                for weight in layer.weights:
                    tf_sums.append(tf.summary.histogram(weight.name, weight))

                if hasattr(layer, 'output'):
                    tf_sums.append(tf.summary.histogram('{}_out'.format(layer.name),
                                                        layer.output))

        # outputs
        backend_channels_last() or _raise(NotImplementedError())

        n_channels_in = self.model.input_shape[-1]
        n_dim_in = len(self.model.input_shape)

        # FIXME: not fully baked, eg. n_dim==5 multichannel doesnt work

        if n_dim_in > 4:
            # print("tensorboard shape: %s"%str(self.model.input_shape))
            input_layer = Lambda(lambda x: K.max(K.max(x, axis=1), axis=-1, keepdims=True))(self.model.input)
        else:
            if n_channels_in > 3:
                input_layer = Lambda(lambda x: K.max(x, axis=-1, keepdims=True))(self.model.input)
            elif n_channels_in == 2:
                input_layer = Lambda(lambda x: K.concatenate([x,x[...,:1]], axis=-1))(self.model.input)
            else:
                input_layer = self.model.input

        n_channels_out = self.model.output_shape[-1]
        n_dim_out = len(self.model.output_shape)

        sep = n_channels_out
        if self.prob_out:
            # first half of output channels is mean, second half scale
            # assert n_channels_in*2 == n_channels_out
            # if n_channels_in*2 != n_channels_out:
            #     raise ValueError('prob_out: must be two output channels for every input channel')
            n_channels_out % 2 == 0 or _raise(ValueError())
            sep = sep // 2

        if n_dim_out > 4:
            output_layer = Lambda(lambda x: K.max(K.max(x[...,:sep], axis=1), axis=-1, keepdims=True))(self.model.output)
        else:
            if sep > 3:
                output_layer = Lambda(lambda x: K.max(x[...,:sep], axis=-1, keepdims=True))(self.model.output)
            elif sep == 2:
                output_layer = Lambda(lambda x: K.concatenate([x[...,:sep],x[...,:1]], axis=-1))(self.model.output)
            else:
                output_layer = Lambda(lambda x: x[...,:sep])(self.model.output)

        if self.prob_out:
            # scale images
            if n_dim_out > 4:
                scale_layer = Lambda(lambda x: K.max(K.max(x[...,sep:], axis=1), axis=-1, keepdims=True))(self.model.output)
            else:
                if sep > 3:
                    scale_layer = Lambda(lambda x: K.max(x[...,sep:], axis=-1, keepdims=True))(self.model.output)
                elif sep == 2:
                    scale_layer = Lambda(lambda x: K.concatenate([x[...,sep:],x[...,-1:]], axis=-1))(self.model.output)
                else:
                    scale_layer = Lambda(lambda x: x[...,sep:])(self.model.output)

        #
        #     n_channels = self.model.input_shape[0]
        #     if n_channels > 1:
        #         #input_layer  = Lambda(lambda x: x[n_channels // 2:n_channels // 2 + 1, :, :, :])(self.model.input)
        #         input_layer = Lambda(lambda x: K.max(x, axis = 0))(self.model.input)
        #     else:
        #         input_layer = self.model.input
        # #
        # if K.image_dim_ordering() == "tf":
        #     n_channels = self.model.input_shape[-1]
        #     if n_channels>1:
        #         input_layer = Lambda(lambda x: x[:, :, :, n_channels // 2:n_channels // 2 + 1])(self.model.input)
        #     else:
        #         input_layer = self.model.input
        # else:
        #     n_channels = self.model.input_shape[0]
        #     if n_channels > 1:
        #         #input_layer  = Lambda(lambda x: x[n_channels // 2:n_channels // 2 + 1, :, :, :])(self.model.input)
        #         input_layer = Lambda(lambda x: K.max(x, axis = 0))(self.model.input)
        #     else:
        #         input_layer = self.model.input

        tf_sums.append(tf.summary.image('input', input_layer, max_outputs=self.n_images))
        if self.prob_out:
            tf_sums.append(tf.summary.image('mean', output_layer, max_outputs=self.n_images))
            tf_sums.append(tf.summary.image('scale', scale_layer, max_outputs=self.n_images))
        else:
            tf_sums.append(tf.summary.image('output', output_layer, max_outputs=self.n_images))

        with tf.name_scope('merged'):
            self.merged = tf.summary.merge(tf_sums)
            # self.merged = tf.summary.merge([foo])

        with tf.name_scope('summary_writer'):
            if self.write_graph:
                self.writer = tf.summary.FileWriter(self.log_dir,
                                                    self.sess.graph)
            else:
                self.writer = tf.summary.FileWriter(self.log_dir)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        if self.validation_data and self.freq:
            if epoch % self.freq == 0:
                # TODO: implement batched calls to sess.run
                # (current call will likely go OOM on GPU)
                if self.model.uses_learning_phase:
                    cut_v_data = len(self.model.inputs)
                    val_data = self.validation_data + [0]
                    tensors = self.model.inputs + [K.learning_phase()]
                else:
                    val_data = list(v[:self.n_images] for v in self.validation_data)
                    tensors = self.model.inputs
                feed_dict = dict(zip(tensors, val_data))
                result = self.sess.run([self.merged], feed_dict=feed_dict)
                summary_str = result[0]

                self.writer.add_summary(summary_str, epoch)

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, epoch)
        self.writer.flush()

    def on_train_end(self, _):
        self.writer.close()
        

"""
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Data handling
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------ 
"""    
    
class RawData(namedtuple('RawData' ,('generator' ,'size' ,'description'))):
    """:func:`collections.namedtuple` with three fields: `generator`, `size`, and `description`.

    Parameters
    ----------
    generator : function
        Function without arguments that returns a generator that yields tuples `(x,y,axes,mask)`,
        where `x` is a source image (e.g., with low SNR) with `y` being the corresponding target image
        (e.g., with high SNR); `mask` can either be `None` or a boolean array that denotes which
        pixels are eligible to extracted in :func:`create_patches`. Note that `x`, `y`, and `mask`
        must all be of type :class:`numpy.ndarray` with the same shape, where the string `axes`
        indicates the order and presence of axes of all three arrays.
    size : int
        Number of tuples that the `generator` will yield.
    description : str
        Textual description of the raw data.
    """

    @staticmethod
    def from_folder(basepath, source_dirs, target_dir, axes='CZYX', pattern='*.tif*'):
        """Get pairs of corresponding TIFF images read from folders.

        Two images correspond to each other if they have the same file name, but are located in different folders.

        Parameters
        ----------
        basepath : str
            Base folder that contains sub-folders with images.
        source_dirs : list or tuple
            List of folder names relative to `basepath` that contain the source images (e.g., with low SNR).
        target_dir : str
            Folder name relative to `basepath` that contains the target images (e.g., with high SNR).
        axes : str
            Semantics of axes of loaded images (must be same for all images).
        pattern : str
            Glob-style pattern to match the desired TIFF images.

        Returns
        -------
        RawData
            :obj:`RawData` object, whose `generator` is used to yield all matching TIFF pairs.
            The generator will return a tuple `(x,y,axes,mask)`, where `x` is from
            `source_dirs` and `y` is the corresponding image from the `target_dir`;
            `mask` is set to `None`.

        Raises
        ------
        FileNotFoundError
            If an image found in `target_dir` does not exist in all `source_dirs`.
        ValueError
            If corresponding images do not have the same size (raised by returned :func:`RawData.generator`).

        Example
        --------
        >>> !tree data
        data
        ├── GT
        │   ├── imageA.tif 
        │   └── imageB.tif
        ├── source1
        │   ├── imageA.tif
        │   └── imageB.tif
        └── source2
            ├── imageA.tif
            └── imageB.tif

        >>> data = RawData.from_folder(basepath='data', source_dirs=['source1','source2'], target_dir='GT', axes='YX')
        >>> n_images = data.size
        >>> for source_x, target_y, axes, mask in data.generator():
        ...     pass

        """
        p = Path(basepath)
        image_names = [f.name for f in ( p /target_dir).glob(pattern)]
        len(image_names) > 0 or _raise(FileNotFoundError("'target_dir' doesn't exist or didn't find any images in it."))
        consume ((
            ( p / s /n).exists() or _raise(FileNotFoundError( p / s /n))
            for s in source_dirs for n in image_names
        ))
        axes = axes_check_and_normalize(axes)
        xy_name_pairs = [( p / source_dir /n, p/ target_dir / n) for source_dir in source_dirs for n in image_names]
        n_images = len(xy_name_pairs)
        description = "{p}: target='{o}', sources={s}, axes='{a}', pattern='{pt}'".format(p=basepath, s=list(source_dirs),
                                                                                        o=target_dir, a=axes, pt=pattern)

        def _gen():
            for fx, fy in xy_name_pairs:
                x, y = imread(str(fx)), imread(str(fy))
                # x,y = x[:,256:-256,256:-256],y[:,256:-256,256:-256] #tmp
                x.shape == y.shape or _raise(ValueError())
                len(axes) >= x.ndim or _raise(ValueError())
                yield x, y, axes[-x.ndim:], None

        return RawData(_gen, n_images, description)

    @staticmethod
    def from_arrays(X ,Y, axes = "CZYX"):
        """Get pairs of corresponding images from numpy arrays."""

        if X.shape != Y.shape:
            raise ValueError("X and Y should be of same shape!")

        def _gen():
            for x, y in zip(X ,Y):
                len(axes) >= x.ndim or _raise(ValueError())
                yield x, y, axes[-x.ndim:], None

        return RawData(_gen, len(X), "numpy array")


def no_background_patches(threshold=0.4, percentile=99.9):

    """Returns a patch filter to be used by :func:`create_patches` to determine for each image pair which patches
    are eligible for sampling. The purpose is to only sample patches from "interesting" regions of the raw image that
    actually contain a substantial amount of non-background signal. To that end, a maximum filter is applied to the target image
    to find the largest values in a region.

    Parameters
    ----------
    threshold : float, optional
        Scalar threshold between 0 and 1 that will be multiplied with the (outlier-robust)
        maximum of the image (see `percentile` below) to denote a lower bound.
        Only patches with a maximum value above this lower bound are eligible to be sampled.
    percentile : float, optional
        Percentile value to denote the (outlier-robust) maximum of an image, i.e. should be close 100.

    Returns
    -------
    function
        Function that takes an image pair `(y,x)` and the patch size as arguments and
        returns a binary mask of the same size as the image (to denote the locations
        eligible for sampling for :func:`create_patches`). At least one pixel of the
        binary mask must be ``True``, otherwise there are no patches to sample.

    Raises
    ------
    ValueError
        Illegal arguments.
    """

    (np.isscalar(percentile) and 0 <= percentile <= 100) or _raise(ValueError())
    (np.isscalar(threshold)  and 0 <= threshold  <=   1) or _raise(ValueError())

    from scipy.ndimage.filters import maximum_filter
    def _filter(datas, patch_size, dtype=np.float32):
        image = datas[0]
        if dtype is not None:
            image = image.astype(dtype)
        # make max filter patch_size smaller to avoid only few non-bg pixel close to image border
        patch_size = [(p//2 if p>1 else p) for p in patch_size]
        filtered = maximum_filter(image, patch_size, mode='constant')
        return filtered > threshold * np.percentile(image,percentile)
    return _filter


def sample_percentiles(pmin=(1,3), pmax=(99.5,99.9)):
    """Sample percentile values from a uniform distribution.

    Parameters
    ----------
    pmin : tuple
        Tuple of two values that denotes the interval for sampling low percentiles.
    pmax : tuple
        Tuple of two values that denotes the interval for sampling high percentiles.

    Returns
    -------
    function
        Function without arguments that returns `(pl,ph)`, where `pl` (`ph`) is a sampled low (high) percentile.

    Raises
    ------
    ValueError
        Illegal arguments.
    """
    _valid_low_high_percentiles(pmin) or _raise(ValueError(pmin))
    _valid_low_high_percentiles(pmax) or _raise(ValueError(pmax))
    pmin[1] < pmax[0] or _raise(ValueError())
    return lambda: (np.random.uniform(*pmin), np.random.uniform(*pmax))

def _valid_low_high_percentiles(ps):
    return isinstance(ps,(list,tuple,np.ndarray)) and len(ps)==2 and all(map(np.isscalar,ps)) and (0<=ps[0]<ps[1]<=100)

def norm_percentiles(percentiles=sample_percentiles(), relu_last=False):
    """Normalize extracted patches based on percentiles from corresponding raw image.

    Parameters
    ----------
    percentiles : tuple, optional
        A tuple (`pmin`, `pmax`) or a function that returns such a tuple, where the extracted patches
        are (affinely) normalized in such that a value of 0 (1) corresponds to the `pmin`-th (`pmax`-th) percentile
        of the raw image (default: :func:`sample_percentiles`).
    relu_last : bool, optional
        Flag to indicate whether the last activation of the DLSRS network is/will be using
        a ReLU activation function (default: ``False``)

    Return
    ------
    function
        Function that does percentile-based normalization to be used in :func:`create_patches`.

    Raises
    ------
    ValueError
        Illegal arguments.

    Todo
    ----
    ``relu_last`` flag problematic/inelegant.

    """
    if callable(percentiles):
        _tmp = percentiles()
        _valid_low_high_percentiles(_tmp) or _raise(ValueError(_tmp))
        get_percentiles = percentiles
    else:
        _valid_low_high_percentiles(percentiles) or _raise(ValueError(percentiles))
        get_percentiles = lambda: percentiles

    def _normalize(patches_x,patches_y, x,y,mask,channel):
        pmins, pmaxs = zip(*(get_percentiles() for _ in patches_x))
        percentile_axes = None if channel is None else tuple((d for d in range(x.ndim) if d != channel))
        _perc = lambda a,p: np.percentile(a,p,axis=percentile_axes,keepdims=True)
        patches_x_norm = normalize_mi_ma(patches_x, _perc(x,pmins), _perc(x,pmaxs))
        if relu_last:
            pmins = np.zeros_like(pmins)
        patches_y_norm = normalize_mi_ma(patches_y, _perc(y,pmins), _perc(y,pmaxs))
        return patches_x_norm, patches_y_norm

    return _normalize

## Sample patches

def sample_patches_from_multiple_stacks(datas, patch_size, n_samples, datas_mask=None, patch_filter=None, verbose=False):
    """ sample matching patches of size `patch_size` from all arrays in `datas` """

    # TODO: some of these checks are already required in 'create_patches'
    len(patch_size)==datas[0].ndim or _raise(ValueError())

    if not all(( a.shape == datas[0].shape for a in datas )):
        raise ValueError("all input shapes must be the same: %s" % (" / ".join(str(a.shape) for a in datas)))

    if not all(( 0 < s <= d for s,d in zip(patch_size,datas[0].shape) )):
        raise ValueError("patch_size %s negative or larger than data shape %s along some dimensions" % (str(patch_size), str(datas[0].shape)))

    if patch_filter is None:
        patch_mask = np.ones(datas[0].shape,dtype=np.bool)
    else:
        patch_mask = patch_filter(datas, patch_size)

    if datas_mask is not None:
        # TODO: Test this
        warnings.warn('Using pixel masks for raw/transformed images not tested.')
        datas_mask.shape == datas[0].shape or _raise(ValueError())
        datas_mask.dtype == np.bool or _raise(ValueError())
        from scipy.ndimage.filters import minimum_filter
        patch_mask &= minimum_filter(datas_mask, patch_size, mode='constant', cval=False)

    # get the valid indices

    border_slices = tuple([slice(s // 2, d - s + s // 2 + 1) for s, d in zip(patch_size, datas[0].shape)])
    valid_inds = np.where(patch_mask[border_slices])

    if len(valid_inds[0]) == 0:
        raise ValueError("'patch_filter' didn't return any region to sample from")

    valid_inds = [v + s.start for s, v in zip(border_slices, valid_inds)]

    # sample
    sample_inds = np.random.choice(len(valid_inds[0]), n_samples, replace=len(valid_inds[0])<n_samples)

    rand_inds = [v[sample_inds] for v in valid_inds]

    # res = [np.stack([data[r[0] - patch_size[0] // 2:r[0] + patch_size[0] - patch_size[0] // 2,
    #                  r[1] - patch_size[1] // 2:r[1] + patch_size[1] - patch_size[1] // 2,
    #                  r[2] - patch_size[2] // 2:r[2] + patch_size[2] - patch_size[2] // 2,
    #                  ] for r in zip(*rand_inds)]) for data in datas]

    res = [np.stack([data[tuple(slice(_r-(_p//2),_r+_p-(_p//2)) for _r,_p in zip(r,patch_size))] for r in zip(*rand_inds)]) for data in datas]

    return res

def create_patches(
        raw_data,
        patch_size,
        n_patches_per_image,
        patch_axes    = None,
        save_file     = None,
        transforms    = None,
        patch_filter  = no_background_patches(),
        normalization = norm_percentiles(),
        shuffle       = True,
        verbose       = True,
    ):
    """Create normalized training data to be used for neural network training.

    Parameters
    ----------
    raw_data : :class:`RawData`
        Object that yields matching pairs of raw images.
    patch_size : tuple
        Shape of the patches to be extraced from raw images.
        Must be compatible with the number of dimensions and axes of the raw images.
        As a general rule, use a power of two along all XYZT axes, or at least divisible by 8.
    n_patches_per_image : int
        Number of patches to be sampled/extracted from each raw image pair (after transformations, see below).
    patch_axes : str or None
        Axes of the extracted patches. If ``None``, will assume to be equal to that of transformed raw data.
    save_file : str or None
        File name to save training data to disk in ``.npz`` format (see :func:`csbdeep.io.save_training_data`).
        If ``None``, data will not be saved.
    transforms : list or tuple, optional
        List of :class:`Transform` objects that apply additional transformations to the raw images.
        This can be used to augment the set of raw images (e.g., by including rotations).
        Set to ``None`` to disable. Default: ``None``.
    patch_filter : function, optional
        Function to determine for each image pair which patches are eligible to be extracted
        (default: :func:`no_background_patches`). Set to ``None`` to disable.
    normalization : function, optional
        Function that takes arguments `(patches_x, patches_y, x, y, mask, channel)`, whose purpose is to
        normalize the patches (`patches_x`, `patches_y`) extracted from the associated raw images
        (`x`, `y`, with `mask`; see :class:`RawData`). Default: :func:`norm_percentiles`.
    shuffle : bool, optional
        Randomly shuffle all extracted patches.
    verbose : bool, optional
        Display overview of images, transforms, etc.

    Returns
    -------
    tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray`, str)
        Returns a tuple (`X`, `Y`, `axes`) with the normalized extracted patches from all (transformed) raw images
        and their axes.
        `X` is the array of patches extracted from source images with `Y` being the array of corresponding target patches.
        The shape of `X` and `Y` is as follows: `(n_total_patches, n_channels, ...)`.
        For single-channel images, `n_channels` will be 1.

    Raises
    ------
    ValueError
        Various reasons.

    Example
    -------
    >>> raw_data = RawData.from_folder(basepath='data', source_dirs=['source1','source2'], target_dir='GT', axes='ZYX')
    >>> X, Y, XY_axes = create_patches(raw_data, patch_size=(32,128,128), n_patches_per_image=16)

    Todo
    ----
    - Save created patches directly to disk using :class:`numpy.memmap` or similar?
      Would allow to work with large data that doesn't fit in memory.

    """
    ## images and transforms
    if transforms is None:
        transforms = []
    transforms = list(transforms)
    if patch_axes is not None:
        transforms.append(permute_axes(patch_axes))
    if len(transforms) == 0:
        transforms.append(Transform.identity())


    image_pairs, n_raw_images = raw_data.generator(), raw_data.size
    tf = Transform(*zip(*transforms)) # convert list of Transforms into Transform of lists
    image_pairs = compose(*tf.generator)(image_pairs) # combine all transformations with raw images as input
    n_transforms = np.prod(tf.size)
    n_images = n_raw_images * n_transforms
    n_patches = n_images * n_patches_per_image
    n_required_memory_bytes = 2 * n_patches*np.prod(patch_size) * 4

    ## memory check
    _memory_check(n_required_memory_bytes)

    ## summary
    if verbose:
        print('='*66)
        print('%5d raw images x %4d transformations   = %5d images' % (n_raw_images,n_transforms,n_images))
        print('%5d images     x %4d patches per image = %5d patches in total' % (n_images,n_patches_per_image,n_patches))
        print('='*66)
        print('Input data:')
        print(raw_data.description)
        print('='*66)
        print('Transformations:')
        for t in transforms:
            print('{t.size} x {t.name}'.format(t=t))
        print('='*66)
        print('Patch size:')
        print(" x ".join(str(p) for p in patch_size))
        print('=' * 66)

    sys.stdout.flush()

    ## sample patches from each pair of transformed raw images
    X = np.empty((n_patches,)+tuple(patch_size),dtype=np.float32)
    Y = np.empty_like(X)

    for i, (x,y,_axes,mask) in tqdm(enumerate(image_pairs),total=n_images):
        if i==0:
            axes = axes_check_and_normalize(_axes,len(patch_size))
            channel = axes_dict(axes)['C']
        # checks
        # len(axes) >= x.ndim or _raise(ValueError())
        axes == axes_check_and_normalize(_axes) or _raise(ValueError('not all images have the same axes.'))
        x.shape == y.shape or _raise(ValueError())
        mask is None or mask.shape == x.shape or _raise(ValueError())
        (channel is None or (isinstance(channel,int) and 0<=channel<x.ndim)) or _raise(ValueError())
        channel is None or patch_size[channel]==x.shape[channel] or _raise(ValueError('extracted patches must contain all channels.'))

        _Y,_X = sample_patches_from_multiple_stacks((y,x), patch_size, n_patches_per_image, mask, patch_filter)

        s = slice(i*n_patches_per_image,(i+1)*n_patches_per_image)
        X[s], Y[s] = normalization(_X,_Y, x,y,mask,channel)

    if shuffle:
        shuffle_inplace(X,Y)

    axes = 'SC'+axes.replace('C','')
    if channel is None:
        X = np.expand_dims(X,1)
        Y = np.expand_dims(Y,1)
    else:
        X = np.moveaxis(X, 1+channel, 1)
        Y = np.moveaxis(Y, 1+channel, 1)

    if save_file is not None:
        print('Saving data to %s.' % str(Path(save_file)))
        save_training_data(save_file, X, Y, axes)

    return X,Y,axes

def shuffle_inplace(*arrs):
    rng = np.random.RandomState()
    state = rng.get_state()
    for a in arrs:
        rng.set_state(state)
        rng.shuffle(a)

def _memory_check(n_required_memory_bytes, thresh_free_frac=0.5, thresh_abs_bytes=1024*1024**2):
    try:
        # raise ImportError
        import psutil
        mem = psutil.virtual_memory()
        mem_frac = n_required_memory_bytes / mem.available
        if mem_frac > 1:
            raise MemoryError('Not enough available memory.')
        elif mem_frac > thresh_free_frac:
            print('Warning: will use at least %.0f MB (%.1f%%) of available memory.\n' % (n_required_memory_bytes/1024**2,100*mem_frac), file=sys.stderr)
            sys.stderr.flush()
    except ImportError:
        if n_required_memory_bytes > thresh_abs_bytes:
            print('Warning: will use at least %.0f MB of memory.\n' % (n_required_memory_bytes/1024**2), file=sys.stderr)
            sys.stderr.flush()

class Transform(namedtuple('Transform',('name','generator','size'))):
    """Extension of :func:`collections.namedtuple` with three fields: `name`, `generator`, and `size`.

    Parameters
    ----------
    name : str
        Name of the applied transformation.
    generator : function
        Function that takes a generator as input and itself returns a generator; input and returned
        generator have the same structure as that of :class:`RawData`.
        The purpose of the returned generator is to augment the images provided by the input generator
        through additional transformations.
        It is important that the returned generator also includes every input tuple unchanged.
    size : int
        Number of transformations applied to every image (obtained from the input generator).
    """

    @staticmethod
    def identity():
        """
        Returns
        -------
        Transform
            Identity transformation that passes every input through unchanged.
        """
        def _gen(inputs):
            for d in inputs:
                yield d
        return Transform('Identity', _gen, 1)

    # def flip(axis):
    #     """TODO"""
    #     def _gen(inputs):
    #         for x,y,m_in in inputs:
    #             axis < x.ndim or _raise(ValueError())
    #             yield x, y, m_in
    #             yield np.flip(x,axis), np.flip(y,axis), None if m_in is None else np.flip(m_in,axis)
    #     return Transform('Flip (axis=%d)'%axis, _gen, 2)

def permute_axes(axes):
    """Transformation to permute images axes.

    Parameters
    ----------
    axes : str
        Target axes, to which the input images will be permuted.

    Returns
    -------
    Transform
        Returns a :class:`Transform` object whose `generator` will
        perform the axes permutation of `x`, `y`, and `mask`.

    """
    axes = axes_check_and_normalize(axes)
    def _generator(inputs):
        for x, y, axes_in, mask in inputs:
            axes_in = axes_check_and_normalize(axes_in)
            if axes_in != axes:
                # print('permuting axes from %s to %s' % (axes_in,axes))
                x = move_image_axes(x, axes_in, axes, True)
                y = move_image_axes(y, axes_in, axes, True)
                if mask is not None:
                    mask = move_image_axes(mask, axes_in, axes)
            yield x, y, axes, mask

    return Transform('Permute axes to %s' % axes, _generator, 1)


def export_SavedModel(model, outpath, meta={}, format='zip'):
    """Export Keras model in TensorFlow's SavedModel_ format.

    See `Your Model in Fiji`_ to learn how to use the exported model with our CSBDeep Fiji plugins.

    .. _SavedModel: https://www.tensorflow.org/programmers_guide/saved_model#structure_of_a_savedmodel_directory
    .. _`Your Model in Fiji`: https://github.com/CSBDeep/CSBDeep_website/wiki/Your-Model-in-Fiji

    Parameters
    ----------
    model : :class:`keras.models.Model`
        Keras model to be exported.
    outpath : str
        Path of the file/folder that the model will exported to.
    meta : dict, optional
        Metadata to be saved in an additional ``meta.json`` file.
    format : str, optional
        Can be 'dir' to export as a directory or 'zip' (default) to export as a ZIP file.

    Raises
    ------
    ValueError
        Illegal arguments.

    """

    def export_to_dir(dirname):
        if len(model.inputs) > 1 or len(model.outputs) > 1:
            warnings.warn('Not tested with multiple input or output layers.')
        builder = tf.saved_model.builder.SavedModelBuilder(dirname)
        signature = tf.saved_model.signature_def_utils.predict_signature_def(
            inputs  = {'input':  model.input},
            outputs = {'output': model.output}
        )
        signature_def_map = { tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature }
        builder.add_meta_graph_and_variables(K.get_session(),
                                             [tf.saved_model.tag_constants.SERVING],
                                             signature_def_map=signature_def_map)
        builder.save()
        if meta is not None and len(meta) > 0:
            save_json(meta, os.path.join(dirname,'meta.json'))



    ## checks
    isinstance(model,Model) or _raise(ValueError("'model' must be a Keras model."))
    # supported_formats = tuple(['dir']+[name for name,description in shutil.get_archive_formats()])
    supported_formats = 'dir','zip'
    format in supported_formats or _raise(ValueError("Unsupported format '%s', must be one of %s." % (format,str(supported_formats))))

    # remove '.zip' file name extension if necessary
    if format == 'zip' and outpath.endswith('.zip'):
        outpath = os.path.splitext(outpath)[0]

    if format == 'dir':
        export_to_dir(outpath)
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpsubdir = os.path.join(tmpdir,'model')
            export_to_dir(tmpsubdir)
            shutil.make_archive(outpath, format, tmpsubdir)



from six import add_metaclass
from abc import ABCMeta, abstractmethod, abstractproperty
@add_metaclass(ABCMeta)
class Normalizer():
    """Abstract base class for normalization methods."""

    @abstractmethod
    def before(self, img, axes):
        """Normalization of the raw input image (method stub).

        Parameters
        ----------
        img : :class:`numpy.ndarray`
            Raw input image.
        axes : str
            Axes of ``img``.

        Returns
        -------
        :class:`numpy.ndarray`
            Normalized input image with suitable values for neural network input.
        """
        pass

    @abstractmethod
    def after(self, mean, scale):
        """Possible adjustment of predicted restored image (method stub).

        It is assumed that the image axes are the same as in the call to :func:`before`.

        Parameters
        ----------
        mean : :class:`numpy.ndarray`
            Predicted restored image or per-pixel ``mean`` of Laplace distributions
            for probabilistic model.
        scale: :class:`numpy.ndarray` or None
            Per-pixel ``scale`` of Laplace distributions for probabilistic model (``None`` otherwise.)

        Returns
        -------
        :class:`numpy.ndarray`
            Adjusted restored image.
        """
        pass

    @abstractproperty
    def do_after(self):
        """bool : Flag to indicate whether :func:`after` should be called."""
        pass


class NoNormalizer(Normalizer):
    """No normalization.

    Parameters
    ----------
    do_after : bool
        Flag to indicate whether to undo normalization.

    Raises
    ------
    ValueError
        If :func:`after` is called, but parameter `do_after` was set to ``False`` in the constructor.
    """

    def __init__(self, do_after=False):
        """foo"""
        self._do_after = do_after

    def before(self, img, axes):
        return img

    def after(self, mean, scale):
        self.do_after or _raise(ValueError())
        return mean, scale

    @property
    def do_after(self):
        return self._do_after


class PercentileNormalizer(Normalizer):
    """Percentile-based image normalization.

    Parameters
    ----------
    pmin : float
        Low percentile.
    pmax : float
        High percentile.
    do_after : bool
        Flag to indicate whether to undo normalization (original data type will not be restored).
    dtype : type
        Data type after normalization.
    kwargs : dict
        Keyword arguments for :func:`csbdeep.utils.normalize_mi_ma`.
    """

    def __init__(self, pmin=2, pmax=99.8, do_after=True, dtype=np.float32, **kwargs):
        """TODO."""
        (np.isscalar(pmin) and np.isscalar(pmax) and 0 <= pmin < pmax <= 100) or _raise(ValueError())
        self.pmin = pmin
        self.pmax = pmax
        self._do_after = do_after
        self.dtype = dtype
        self.kwargs = kwargs

    def before(self, img, axes):
        """Percentile-based normalization of raw input image.

        See :func:`csbdeep.predict.Normalizer.before` for parameter descriptions.
        Note that percentiles are computed individually for each channel (if present in `axes`).
        """
        len(axes) == img.ndim or _raise(ValueError())
        channel = axes_dict(axes)['C']
        axes = None if channel is None else tuple((d for d in range(img.ndim) if d != channel))
        self.mi = np.percentile(img,self.pmin,axis=axes,keepdims=True).astype(self.dtype,copy=False)
        self.ma = np.percentile(img,self.pmax,axis=axes,keepdims=True).astype(self.dtype,copy=False)
        return normalize_mi_ma(img, self.mi, self.ma, dtype=self.dtype, **self.kwargs)

    def after(self, mean, scale):
        """Undo percentile-based normalization to map restored image to similar range as input image.

        See :func:`csbdeep.predict.Normalizer.before` for parameter descriptions.

        Raises
        ------
        ValueError
            If parameter `do_after` was set to ``False`` in the constructor.

        """
        self.do_after or _raise(ValueError())
        alpha = self.ma - self.mi
        beta  = self.mi
        return (
            ( alpha*mean+beta ).astype(self.dtype,copy=False),
            ( alpha*scale     ).astype(self.dtype,copy=False) if scale is not None else None
        )

    @property
    def do_after(self):
        """``do_after`` parameter from constructor."""
        return self._do_after

@add_metaclass(ABCMeta)
class Resizer():
    """Abstract base class for resizing methods."""

    @abstractmethod
    def before(self, x, div_n, exclude):
        """Resizing of the raw input image (method stub).

        Parameters
        ----------
        x : :class:`numpy.ndarray`
            Raw input image.
        div_n : int
            Resized image must be evenly divisible by this value.
        exclude : int or list(int) or None
            Indicates axis indices to exclude (can be ``None``),
            e.g. channel dimension.

        Returns
        -------
        :class:`numpy.ndarray`
            Resized input image.
        """
        pass

    @abstractmethod
    def after(self, x, exclude):
        """Resizing of the restored image (method stub).

        Parameters
        ----------
        x : :class:`numpy.ndarray`
            Raw input image.
        exclude : int or list(int) or None
            Indicates axis indices to exclude (can be ``None``),
            e.g. channel dimension.
            Afert ignoring the excludes axis indices,
            note that the shape of x must be same as in :func:`before`.

        Returns
        -------
        :class:`numpy.ndarray`
            Resized restored image image.
        """
        pass


    def _normalize_exclude(self, exclude, n_dim):
        """Return normalized list of excluded axes."""
        if exclude is None:
            return []
        exclude_list = [exclude] if np.isscalar(exclude) else list(exclude)
        exclude_list = [d%n_dim for d in exclude_list]
        len(exclude_list) == len(np.unique(exclude_list)) or _raise(ValueError())
        all(( isinstance(d,int) and 0<=d<n_dim for d in exclude_list )) or _raise(ValueError())
        return exclude_list

class PadAndCropResizer(Resizer):
    """Resize image by padding and cropping.

    If necessary, input image is padded before prediction
    and restored image is cropped back to size of input image
    after prediction.

    Parameters
    ----------
    mode : str
        Parameter ``mode`` of :func:`numpy.pad` that
        controls how the image is padded.
    kwargs : dict
        Keyword arguments for :func:`numpy.pad`.
    """

    def __init__(self, mode='reflect', **kwargs):
        """TODO."""
        self.mode = mode
        self.kwargs = kwargs

    def before(self, x, div_n, exclude):
        """Pad input image.

        See :func:`csbdeep.predict.Resizer.before` for parameter descriptions.
        """
        def _split(v):
            a = v // 2
            return a, v-a
        exclude = self._normalize_exclude(exclude, x.ndim)
        self.pad = [_split((div_n-s%div_n)%div_n) if (i not in exclude) else (0,0) for i,s in enumerate(x.shape)]
        # print(self.pad)
        x_pad = np.pad(x, self.pad, mode=self.mode, **self.kwargs)
        for i in exclude:
            del self.pad[i]
        return x_pad

    def after(self, x, exclude):
        """Crop restored image to retain size of input image.

        See :func:`csbdeep.predict.Resizer.after` for parameter descriptions.
        """
        crop = [slice(p[0], -p[1] if p[1]>0 else None) for p in self.pad]
        for i in self._normalize_exclude(exclude, x.ndim):
            crop.insert(i,slice(None))
        len(crop) == x.ndim or _raise(ValueError())
        return x[crop]

def tile_iterator(x,axis,n_tiles,block_size,n_block_overlap):
    """Tile iterator for one dimension of array x.

    Parameters
    ----------
    x : numpy.ndarray
        Input array
    axis : int
        Axis which sould be tiled, all other axis not tiled.
    n_tiles : int
        Number of tiles for axis of x
    block_size : int
        axis of x is assumed to be ebenly divisible by block_size
        all tiles are aligned with the block_size
    n_block_overlap : int
        tiles will overlap a this many blocks

    """
    n = x.shape[axis]

    n % block_size == 0 or _raise(ValueError("'x' must be evenly divisible by 'block_size' along 'axis'"))
    n_blocks = n // block_size

    n_tiles_valid = int(np.clip(n_tiles,1,n_blocks))
    if n_tiles != n_tiles_valid:
        warnings.warn("invalid value (%d) for 'n_tiles', changing to %d" % (n_tiles,n_tiles_valid))
        n_tiles = n_tiles_valid

    s = n_blocks // n_tiles # tile size
    r = n_blocks %  n_tiles # blocks remainder
    assert n_tiles * s + r == n_blocks

    # list of sizes for each tile
    tile_sizes = s*np.ones(n_tiles,int)
    # distribute remaning blocks to tiles at beginning and end
    if r > 0:
        tile_sizes[:r//2]      += 1
        tile_sizes[-(r-r//2):] += 1

    # n_block_overlap = int(np.ceil(92 / block_size))
    # n_block_overlap -= 1
    # print(n_block_overlap)

    # (pre,post) offsets for each tile
    off = [(n_block_overlap if i > 0 else 0, n_block_overlap if i < n_tiles-1 else 0) for i in range(n_tiles)]

    # tile_starts = np.concatenate(([0],np.cumsum(tile_sizes[:-1])))
    # print([(_st-_pre,_st+_sz+_post) for (_st,_sz,(_pre,_post)) in zip(tile_starts,tile_sizes,off)])

    def to_slice(t):
        sl = [slice(None) for _ in x.shape]
        sl[axis] = slice(
            t[0]*block_size,
            t[1]*block_size if t[1]!=0 else None)
        return tuple(sl)

    start = 0
    for i in range(n_tiles):
        off_pre, off_post = off[i]

        # tile starts before block 0 -> adjust off_pre
        if start-off_pre < 0:
            off_pre = start
        # tile end after last block -> adjust off_post
        if start+tile_sizes[i]+off_post > n_blocks:
            off_post = n_blocks-start-tile_sizes[i]

        tile_in   = (start-off_pre,start+tile_sizes[i]+off_post)  # src in input image     / tile
        tile_out  = (start,start+tile_sizes[i])                   # dst in output image    / s_dst
        tile_crop = (off_pre,-off_post)                           # crop of src for output / s_src

        yield x[to_slice(tile_in)], to_slice(tile_crop), to_slice(tile_out)
        start += tile_sizes[i]

   
        
"""
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Internal functions using Keras to build network
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------ 
"""    
from keras.layers import Input, Conv2D, Conv3D, Activation, Lambda, BatchNormalization
from keras.layers import MaxPooling2D, UpSampling2D, MaxPooling3D, UpSampling3D, Dropout
from keras.models import Model
from keras.layers.merge import Add, Concatenate
from keras.callbacks import Callback, TerminateOnNaN
from keras.utils import Sequence


def common_unet(n_dim=2, n_depth=1, kern_size=3, n_first=16, n_channel_out=1, residual=True, prob_out=False, last_activation='linear'):
    """Construct a common DLSRS neural net based on U-Net [1]_ and residual learning [2]_ to be used for image restoration/enhancement.

    Parameters
    ----------
    n_dim : int
        number of image dimensions (2 or 3)
    n_depth : int
        number of resolution levels of U-Net architecture
    kern_size : int
        size of convolution filter in all image dimensions
    n_first : int
        number of convolution filters for first U-Net resolution level (value is doubled after each downsampling operation)
    n_channel_out : int
        number of channels of the predicted output image
    residual : bool
        if True, model will internally predict the residual w.r.t. the input (typically better)
        requires number of input and output image channels to be equal
    prob_out : bool
        standard regression (False) or probabilistic prediction (True)
        if True, model will predict two values for each input pixel (mean and positive scale value)
    last_activation : str
        name of activation function for the final output layer

    Returns
    -------
    function
        Function to construct the network, which takes as argument the shape of the input image

    Example
    -------
    >>> model = common_unet(2, 1,3,16, 1, True, False)(input_shape)

    References
    ----------
    .. [1] Olaf Ronneberger, Philipp Fischer, Thomas Brox, *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI 2015
    .. [2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. *Deep Residual Learning for Image Recognition*, CVPR 2016
    """
    def _build_this(input_shape):
        return custom_unet(input_shape, last_activation, n_depth, n_first, (kern_size,)*n_dim, pool_size=(2,)*n_dim, n_channel_out=n_channel_out, residual=residual, prob_out=prob_out)
    return _build_this

def custom_unet(input_shape,
                last_activation,
                n_depth=2,
                n_filter_base=16,
                kernel_size=(3,3,3),
                n_conv_per_depth=2,
                activation="relu",
                batch_norm=False,
                dropout=0.0,
                pool_size=(2,2,2),
                n_channel_out=1,
                residual=False,
                prob_out=False,
                eps_scale=1e-3):
    """ TODO """

    if last_activation is None:
        raise ValueError("last activation has to be given (e.g. 'sigmoid', 'relu')!")

    all((s % 2 == 1 for s in kernel_size)) or _raise(ValueError('kernel size should be odd in all dimensions.'))

    channel_axis = -1 if backend_channels_last() else 1

    n_dim = len(kernel_size)
    conv = Conv2D if n_dim==2 else Conv3D

    input = Input(input_shape, name = "input")
    unet = unet_block(n_depth, n_filter_base, kernel_size,
                      activation=activation, dropout=dropout, batch_norm=batch_norm,
                      n_conv_per_depth=n_conv_per_depth, pool=pool_size)(input)

    final = conv(n_channel_out, (1,)*n_dim, activation='linear')(unet)
    if residual:
        if not (n_channel_out == input_shape[-1] if backend_channels_last() else n_channel_out == input_shape[0]):
            raise ValueError("number of input and output channels must be the same for a residual net.")
        final = Add()([final, input])
    final = Activation(activation=last_activation)(final)

    if prob_out:
        scale = conv(n_channel_out, (1,)*n_dim, activation='softplus')(unet)
        scale = Lambda(lambda x: x+np.float32(eps_scale))(scale)
        final = Concatenate(axis=channel_axis)([final,scale])

    return Model(inputs=input, outputs=final)

def unet_block(n_depth=2, n_filter_base=16, kernel_size=(3,3), n_conv_per_depth=2,
               activation="relu",
               batch_norm=False,
               dropout=0.0,
               last_activation=None,
               pool=(2,2),
               prefix=''):

    if len(pool) != len(kernel_size):
        raise ValueError('kernel and pool sizes must match.')
    n_dim = len(kernel_size)
    if n_dim not in (2,3):
        raise ValueError('unet_block only 2d or 3d.')

    conv_block = conv_block2  if n_dim == 2 else conv_block3
    pooling    = MaxPooling2D if n_dim == 2 else MaxPooling3D
    upsampling = UpSampling2D if n_dim == 2 else UpSampling3D

    if last_activation is None:
        last_activation = activation

    channel_axis = -1 if backend_channels_last() else 1

    def _name(s):
        return prefix+s

    def _func(input):
        skip_layers = []
        layer = input

        # down ...
        for n in range(n_depth):
            for i in range(n_conv_per_depth):
                layer = conv_block(n_filter_base * 2 ** n, *kernel_size,
                                   dropout=dropout,
                                   activation=activation,
                                   batch_norm=batch_norm, name=_name("down_level_%s_no_%s" % (n, i)))(layer)
            skip_layers.append(layer)
            layer = pooling(pool, name=_name("max_%s" % n))(layer)

        # middle
        for i in range(n_conv_per_depth - 1):
            layer = conv_block(n_filter_base * 2 ** n_depth, *kernel_size,
                               dropout=dropout,
                               activation=activation,
                               batch_norm=batch_norm, name=_name("middle_%s" % i))(layer)

        layer = conv_block(n_filter_base * 2 ** max(0, n_depth - 1), *kernel_size,
                           dropout=dropout,
                           activation=activation,
                           batch_norm=batch_norm, name=_name("middle_%s" % n_conv_per_depth))(layer)

        # ...and up with skip layers
        for n in reversed(range(n_depth)):
            layer = Concatenate(axis=channel_axis)([upsampling(pool)(layer), skip_layers[n]])
            for i in range(n_conv_per_depth - 1):
                layer = conv_block(n_filter_base * 2 ** n, *kernel_size,
                                   dropout=dropout,
                                   activation=activation,
                                   batch_norm=batch_norm, name=_name("up_level_%s_no_%s" % (n, i)))(layer)

            layer = conv_block(n_filter_base * 2 ** max(0, n - 1), *kernel_size,
                               dropout=dropout,
                               activation=activation if n > 0 else last_activation,
                               batch_norm=batch_norm, name=_name("up_level_%s_no_%s" % (n, n_conv_per_depth)))(layer)

        return layer

    return _func


def conv_block2(n_filter, n1, n2,
                activation="relu",
                border_mode="same",
                dropout=0.0,
                batch_norm=False,
                init="glorot_uniform",
                **kwargs):

    def _func(lay):
        if batch_norm:
            s = Conv2D(n_filter, (n1, n2), padding=border_mode, kernel_initializer=init, **kwargs)(lay)
            s = BatchNormalization()(s)
            s = Activation(activation)(s)
        else:
            s = Conv2D(n_filter, (n1, n2), padding=border_mode, kernel_initializer=init, activation=activation, **kwargs)(lay)
        if dropout is not None and dropout > 0:
            s = Dropout(dropout)(s)
        return s

    return _func



def conv_block3(n_filter, n1, n2, n3,
                activation="relu",
                border_mode="same",
                dropout=0.0,
                batch_norm=False,
                init="glorot_uniform",
                **kwargs):

    def _func(lay):
        if batch_norm:
            s = Conv3D(n_filter, (n1, n2, n3), padding=border_mode, kernel_initializer=init, **kwargs)(lay)
            s = BatchNormalization()(s)
            s = Activation(activation)(s)
        else:
            s = Conv3D(n_filter, (n1, n2, n3), padding=border_mode, kernel_initializer=init, activation=activation, **kwargs)(lay)
        if dropout is not None and dropout > 0:
            s = Dropout(dropout)(s)
        return s

    return _func

# Define different losses 
    
def _mean_or_not(mean):
    # return (lambda x: K.mean(x,axis=(-1 if backend_channels_last() else 1))) if mean else (lambda x: x)
    # Keras also only averages over axis=-1, see https://github.com/keras-team/keras/blob/master/keras/losses.py
    return (lambda x: K.mean(x,axis=-1)) if mean else (lambda x: x)

def loss_laplace(mean=True):
    R = _mean_or_not(mean)
    C = np.log(2.0)
    if backend_channels_last():
        def nll(y_true, y_pred):
            n     = K.shape(y_true)[-1]
            mu    = y_pred[...,:n]
            sigma = y_pred[...,n:]
            return R(K.abs((mu-y_true)/sigma) + K.log(sigma) + C)
        return nll
    else:
        def nll(y_true, y_pred):
            n     = K.shape(y_true)[1]
            mu    = y_pred[:,:n,...]
            sigma = y_pred[:,n:,...]
            return R(K.abs((mu-y_true)/sigma) + K.log(sigma) + C)
        return nll

def loss_mae(mean=True):
    R = _mean_or_not(mean)
    if backend_channels_last():
        def mae(y_true, y_pred):
            n = K.shape(y_true)[-1]
            return R(K.abs(y_pred[...,:n] - y_true))
        return mae
    else:
        def mae(y_true, y_pred):
            n = K.shape(y_true)[1]
            return R(K.abs(y_pred[:,:n,...] - y_true))
        return mae

def loss_mse(mean=True):
    R = _mean_or_not(mean)
    if backend_channels_last():
        def mse(y_true, y_pred):
            n = K.shape(y_true)[-1]
            return R(K.square(y_pred[...,:n] - y_true))
        return mse
    else:
        def mse(y_true, y_pred):
            n = K.shape(y_true)[1]
            return R(K.square(y_pred[:,:n,...] - y_true))
        return mse

    
def prepare_model(model, optimizer, loss, metrics=('mse','mae'),
                  loss_bg_thresh=0, loss_bg_decay=0.06, Y=None):
    """ TODO """

    from keras.optimizers import Optimizer
    isinstance(optimizer,Optimizer) or _raise(ValueError())


    loss_standard   = eval('loss_%s()'%loss)
    _metrics        = [eval('loss_%s()'%m) for m in metrics]
    callbacks       = [TerminateOnNaN()]

    # checks
    assert 0 <= loss_bg_thresh <= 1
    assert loss_bg_thresh == 0 or Y is not None
    if loss == 'laplace':
        assert K.image_data_format() == "channels_last", "TODO"
        assert model.output.shape.as_list()[-1] >= 2 and model.output.shape.as_list()[-1] % 2 == 0

    # loss
    if loss_bg_thresh == 0:
        _loss = loss_standard
    else:
        freq = np.mean(Y > loss_bg_thresh)
        # print("class frequency:", freq)
        alpha = K.variable(1.0)
        loss_per_pixel = eval('loss_{loss}(mean=False)'.format(loss=loss))
        _loss = loss_thresh_weighted_decay(loss_per_pixel, loss_bg_thresh,
                                           0.5 / (0.1 + (1 - freq)),
                                           0.5 / (0.1 +      freq),
                                           alpha)
        callbacks.append(ParameterDecayCallback(alpha, loss_bg_decay, name='alpha'))
        if not loss in metrics:
            _metrics.append(loss_standard)


    # compile model
    model.compile(optimizer=optimizer, loss=_loss, metrics=_metrics)

    return callbacks

def loss_thresh_weighted_decay(loss_per_pixel, thresh, w1, w2, alpha):
    def _loss(y_true, y_pred):
        val = loss_per_pixel(y_true, y_pred)
        k1 = alpha * w1 + (1 - alpha)
        k2 = alpha * w2 + (1 - alpha)
        return K.mean(K.tf.where(K.tf.less_equal(y_true, thresh), k1 * val, k2 * val),
                      axis=(-1 if backend_channels_last() else 1))
    return _loss



class ParameterDecayCallback(Callback):
    """ TODO """
    def __init__(self, parameter, decay, name=None, verbose=0):
        self.parameter = parameter
        self.decay = decay
        self.name = name
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        old_val = K.get_value(self.parameter)
        if self.name:
            logs = logs or {}
            logs[self.name] = old_val
        new_val = old_val * (1. / (1. + self.decay * (epoch + 1)))
        K.set_value(self.parameter, new_val)
        if self.verbose:
            print("\n[ParameterDecayCallback] new %s: %s\n" % (self.name if self.name else 'parameter', new_val))

class DataWrapper(Sequence):

    def __init__(self, X, Y, batch_size):
        self.X, self.Y = X, Y
        self.batch_size = batch_size
        self.perm = np.random.permutation(len(self.X))

    def __len__(self):
        return int(np.ceil(len(self.X) / float(self.batch_size)))

    def on_epoch_end(self):
        self.perm = np.random.permutation(len(self.X))

    def __getitem__(self, i):
        idx = slice(i*self.batch_size,(i+1)*self.batch_size)
        idx = self.perm[idx]
        return self.X[idx], self.Y[idx]

def tile_overlap(n_depth, kern_size):
    rf = {(1, 3):   9, (1, 5):  17, (1, 7): 25,
          (2, 3):  22, (2, 5):  43, (2, 7): 62,
          (3, 3):  46, (3, 5):  92, (3, 7): 138,
          (4, 3):  94, (4, 5): 188, (4, 7): 282,
          (5, 3): 190, (5, 5): 380, (5, 7): 570}
    try:
        return rf[n_depth, kern_size]
    except KeyError:
        raise ValueError('tile_overlap value for n_depth=%d and kern_size=%d not available.' % (n_depth, kern_size))

from scipy.stats import laplace
class ProbabilisticPrediction(object):
    """Laplace distribution (independently per pixel)."""

    def __init__(self, loc, scale):
        loc.shape == scale.shape or _raise(ValueError())
        #
        self._loc     = loc
        self._scale   = scale
        # expose methods from laplace object
        _laplace      = laplace(loc=self._loc,scale=self._scale)
        self.rvs      = _laplace.rvs
        self.pdf      = _laplace.pdf
        self.logpdf   = _laplace.logpdf
        self.cdf      = _laplace.cdf
        self.logcdf   = _laplace.logcdf
        self.sf       = _laplace.sf
        self.logsf    = _laplace.logsf
        self.ppf      = _laplace.ppf
        self.isf      = _laplace.isf
        self.moment   = _laplace.moment
        self.stats    = _laplace.stats
        self.entropy  = _laplace.entropy
        self.expect   = _laplace.expect
        self.median   = _laplace.median
        self.mean     = _laplace.mean
        self.var      = _laplace.var
        self.std      = _laplace.std
        self.interval = _laplace.interval

    def __getitem__(self, indices):
        return ProbabilisticPrediction(loc=self._loc[indices],scale=self._scale[indices])

    def __len__(self):
        return len(self._loc)

    @property
    def shape(self):
        return self._loc.shape

    @property
    def ndim(self):
        return self._loc.ndim

    @property
    def size(self):
        return self._loc.size

    def scale(self):
        return self._scale

    def sampling_generator(self,n=None):
        if n is None:
            while True:
                yield self.rvs()
        else:
            for i in range(n):
                yield self.rvs()

def to_tensor(x,channel=None,single_sample=True):
    if single_sample:
        x = x[np.newaxis]
        if channel is not None and channel >= 0:
            channel += 1
    if channel is None:
        x, channel = np.expand_dims(x,-1), -1
    return move_channel_for_backend(x,channel)



def from_tensor(x,channel=0,single_sample=True):
    return np.moveaxis((x[0] if single_sample else x), (-1 if backend_channels_last() else 1), channel)
def predict_direct(keras_model,x,channel_in=None,channel_out=0,single_sample=True,**kwargs):
    """TODO."""
    return from_tensor(keras_model.predict(to_tensor(x,channel=channel_in,single_sample=single_sample),**kwargs),
                       channel=channel_out,single_sample=single_sample)


def predict_tiled(keras_model,x,n_tiles,block_size,tile_overlap,channel_in=None,channel_out=0,**kwargs):
    """TODO."""

    # TODO: better check, write an axis normalization function that converts negative indices to positive ones
    channel_in  = (channel_in  + x.ndim) % x.ndim
    channel_out = (channel_out + x.ndim) % x.ndim

    def _remove_and_insert(x,a):
        # remove element at channel_in and insert a at channel_out
        lst = list(x)
        if channel_in is not None:
            del lst[channel_in]
        lst.insert(channel_out,a)
        return tuple(lst)

    # largest axis (that is not channel_in)
    axis = [i for i in np.argsort(x.shape) if i != channel_in][-1]

    if isinstance(n_tiles,(list,tuple)):
        0 < len(n_tiles) <= x.ndim-(0 if channel_in is None else 1) or _raise(ValueError())
        n_tiles, n_tiles_remaining = n_tiles[0], n_tiles[1:]
    else:
        n_tiles_remaining = []

    n_block_overlap = int(np.ceil(tile_overlap / block_size))
    # n_block_overlap += -1
    # n_block_overlap = 10
    # print(tile_overlap,n_block_overlap)

    dst = None
    for tile, s_src, s_dst in tile_iterator(x,axis=axis,n_tiles=n_tiles,block_size=block_size,n_block_overlap=n_block_overlap):

        if len(n_tiles_remaining) == 0:
            pred = predict_direct(keras_model,tile,channel_in=channel_in,channel_out=channel_out,**kwargs)
        else:
            pred = predict_tiled(keras_model,tile,n_tiles_remaining,block_size,tile_overlap,channel_in=channel_in,channel_out=channel_out,**kwargs)

        if dst is None:
            dst_shape = _remove_and_insert(x.shape, pred.shape[channel_out])
            dst = np.empty(dst_shape, dtype=x.dtype)

        s_src = _remove_and_insert(s_src, slice(None))
        s_dst = _remove_and_insert(s_dst, slice(None))

        dst[s_dst] = pred[s_src]

    return dst

class NoResizer(Resizer):
    """No resizing.

    Raises
    ------
    ValueError
        In :func:`before`, if image resizing is necessary.
    """

    def before(self, x, div_n, exclude):
        exclude = self._normalize_exclude(exclude, x.ndim)
        consume ((
            (s%div_n==0) or _raise(ValueError('%d (axis %d) is not divisible by %d.' % (s,i,div_n)))
            for i,s in enumerate(x.shape) if (i not in exclude)
        ))
        return x

    def after(self, x, exclude):
        return x




"""
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Model config and training
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------ 
"""    


class Config(argparse.Namespace):
    """Default configuration for a DLSRS model.

    This configuration is meant to be used with :class:`DLSRS`
    and related models (e.g., :class:`IsotropicDLSRS`).

    Parameters
    ----------
    axes : str
        Axes of the neural network (channel axis optional).
    n_channel_in : int
        Number of channels of given input image.
    n_channel_out : int
        Number of channels of predicted output image.
    probabilistic : bool
        Probabilistic prediction of per-pixel Laplace distributions or
        typical regression of per-pixel scalar values.
    kwargs : dict
        Overwrite (or add) configuration attributes (see below).

    Example
    -------
    >>> config = Config('YX', probabilistic=True, unet_n_depth=3)

    Attributes
    ----------
    n_dim : int
        Dimensionality of input images (2 or 3).
    unet_residual : bool
        Parameter `residual` of :func:`csbdeep.nets.common_unet`. Default: ``n_channel_in == n_channel_out``
    unet_n_depth : int
        Parameter `n_depth` of :func:`csbdeep.nets.common_unet`. Default: ``2``
    unet_kern_size : int
        Parameter `kern_size` of :func:`csbdeep.nets.common_unet`. Default: ``5 if n_dim==2 else 3``
    unet_n_first : int
        Parameter `n_first` of :func:`csbdeep.nets.common_unet`. Default: ``32``
    unet_last_activation : str
        Parameter `last_activation` of :func:`csbdeep.nets.common_unet`. Default: ``linear``
    train_loss : str
        Name of training loss. Default: ``'laplace' if probabilistic else 'mae'``
    train_epochs : int
        Number of training epochs. Default: ``100``
    train_steps_per_epoch : int
        Number of parameter update steps per epoch. Default: ``400``
    train_learning_rate : float
        Learning rate for training. Default: ``0.0004``
    train_batch_size : int
        Batch size for training. Default: ``16``
    train_tensorboard : bool
        Enable TensorBoard for monitoring training progress. Default: ``True``
    train_checkpoint : str
        Name of checkpoint file for model weights (only best are saved); set to ``None`` to disable. Default: ``weights_best.h5``
    train_reduce_lr : dict
        Parameter :class:`dict` of ReduceLROnPlateau_ callback; set to ``None`` to disable. Default: ``{'factor': 0.5, 'patience': 10}``

        .. _ReduceLROnPlateau: https://keras.io/callbacks/#reducelronplateau
    """

    def __init__(self, axes, n_channel_in=1, n_channel_out=1, probabilistic=False, **kwargs):
        """See class docstring."""

        # parse and check axes
        axes = axes_check_and_normalize(axes)
        ax = axes_dict(axes)
        ax = {a: (ax[a] is not None) for a in ax}

        (ax['X'] and ax['Y']) or _raise(ValueError('lateral axes X and Y must be present.'))
        not (ax['Z'] and ax['T']) or _raise(ValueError('using Z and T axes together not supported.'))

        axes.startswith('S') or (not ax['S']) or _raise(ValueError('sample axis S must be first.'))
        axes = axes.replace('S','') # remove sample axis if it exists

        n_dim = 3 if (ax['Z'] or ax['T']) else 2

        # TODO: Config not independent of backend. Problem?
        # could move things around during train/predict as an alternative... good idea?
        # otherwise, users can choose axes of input image anyhow, so doesn't matter if model is fixed to something else
        if backend_channels_last():
            if ax['C']:
                axes[-1] == 'C' or _raise(ValueError('channel axis must be last for backend (%s).' % K.backend()))
            else:
                axes += 'C'
        else:
            if ax['C']:
                axes[0] == 'C' or _raise(ValueError('channel axis must be first for backend (%s).' % K.backend()))
            else:
                axes = 'C'+axes

        # directly set by parameters
        self.n_dim                 = n_dim
        self.axes                  = axes
        self.n_channel_in          = int(n_channel_in)
        self.n_channel_out         = int(n_channel_out)
        self.probabilistic         = bool(probabilistic)

        # default config (can be overwritten by kwargs below)
        self.unet_residual         = self.n_channel_in == self.n_channel_out
        self.unet_n_depth          = 2
        self.unet_kern_size        = 5 if self.n_dim==2 else 3
        self.unet_n_first          = 32
        self.unet_last_activation  = 'linear'
        if backend_channels_last():
            self.unet_input_shape  = self.n_dim*(None,) + (self.n_channel_in,)
        else:
            self.unet_input_shape  = (self.n_channel_in,) + self.n_dim*(None,)

        self.train_loss            = 'laplace' if self.probabilistic else 'mae'
        self.train_epochs          = 100
        self.train_steps_per_epoch = 400
        self.train_learning_rate   = 0.0004
        self.train_batch_size      = 16
        self.train_tensorboard     = True
        self.train_checkpoint      = 'weights_best.h5'
        self.train_reduce_lr       = {'factor': 0.5, 'patience': 10}

        # disallow setting 'n_dim' manually
        try:
            del kwargs['n_dim']
            # warnings.warn("ignoring parameter 'n_dim'")
        except:
            pass

        for k in kwargs:
            setattr(self, k, kwargs[k])


    def is_valid(self, return_invalid=False):
        """Check if configuration is valid.

        Returns
        -------
        bool
            Flag that indicates whether the current configuration values are valid.
        """
        def _is_int(v,low=None,high=None):
            return (
                isinstance(v,int) and
                (True if low is None else low <= v) and
                (True if high is None else v <= high)
            )

        ok = {}
        ok['n_dim'] = self.n_dim in (2,3)
        try:
            axes_check_and_normalize(self.axes,self.n_dim+1,disallowed='S')
            ok['axes'] = True
        except:
            ok['axes'] = False
        ok['n_channel_in']  = _is_int(self.n_channel_in,1)
        ok['n_channel_out'] = _is_int(self.n_channel_out,1)
        ok['probabilistic'] = isinstance(self.probabilistic,bool)

        ok['unet_residual'] = (
            isinstance(self.unet_residual,bool) and
            (not self.unet_residual or (self.n_channel_in==self.n_channel_out))
        )
        ok['unet_n_depth']         = _is_int(self.unet_n_depth,1)
        ok['unet_kern_size']       = _is_int(self.unet_kern_size,1)
        ok['unet_n_first']         = _is_int(self.unet_n_first,1)
        ok['unet_last_activation'] = self.unet_last_activation in ('linear','relu')
        ok['unet_input_shape'] = (
            isinstance(self.unet_input_shape,(list,tuple)) and
            len(self.unet_input_shape) == self.n_dim+1 and
            self.unet_input_shape[-1] == self.n_channel_in and
            all((d is None or (_is_int(d) and d%(2**self.unet_n_depth)==0) for d in self.unet_input_shape[:-1]))
        )
        ok['train_loss'] = (
            (    self.probabilistic and self.train_loss == 'laplace'   ) or
            (not self.probabilistic and self.train_loss in ('mse','mae'))
        )
        ok['train_epochs']          = _is_int(self.train_epochs,1)
        ok['train_steps_per_epoch'] = _is_int(self.train_steps_per_epoch,1)
        ok['train_learning_rate']   = np.isscalar(self.train_learning_rate) and self.train_learning_rate > 0
        ok['train_batch_size']      = _is_int(self.train_batch_size,1)
        ok['train_tensorboard']     = isinstance(self.train_tensorboard,bool)
        ok['train_checkpoint']      = self.train_checkpoint is None or isinstance(self.train_checkpoint,string_types)
        ok['train_reduce_lr']       = self.train_reduce_lr  is None or isinstance(self.train_reduce_lr,dict)

        if return_invalid:
            return all(ok.values()), tuple(k for (k,v) in ok.items() if not v)
        else:
            return all(ok.values())



class DLSRS(object):
    """Standard DLSRS network for image restoration and enhancement.

    Uses a convolutional neural network created by :func:`common_unet`.

    Parameters
    ----------
    config : :class:`Config` or None
        Valid configuration of DLSRS network (see :func:`Config.is_valid`).
        Will be saved to disk as JSON (``config.json``).
        If set to ``None``, will be loaded from disk (must exist).
    name : str or None
        Model name. Uses a timestamp if set to ``None`` (default).
    basedir : str
        Directory that contains (or will contain) a folder with the given model name.

    Raises
    ------
    FileNotFoundError
        If ``config=None`` and config cannot be loaded from disk.
    ValueError
        Illegal arguments, including invalid configuration.

    Example
    -------
    >>> model = DLSRS(config, 'my_model')

    Attributes
    ----------
    config : :class:`Config`
        Configuration of DLSRS network, as provided during instantiation.
    keras_model : `Keras model <https://keras.io/getting-started/functional-api-guide/>`_
        Keras neural network model.
    name : str
        Model name.
    logdir : :class:`pathlib.Path`
        Path to model folder (which stores configuration, weights, etc.)
    """

    def __init__(self, config, name=None, basedir='.'):
        """See class docstring."""

        config is None or isinstance(config,Config) or _raise(ValueError('Invalid configuration: %s' % str(config)))
        if config is not None and not config.is_valid():
            invalid_attr = config.is_valid(True)[1]
            raise ValueError('Invalid configuration attributes: ' + ', '.join(invalid_attr))

        name is None or isinstance(name,string_types) or _raise(ValueError())
        isinstance(basedir,(string_types,Path)) or _raise(ValueError())
        self.config = config
        self.basedir = Path(basedir)
        self.name = name
        self._set_logdir()
        self._model_prepared = False
        self.keras_model = self._build()
        if config is None:
            self._find_and_load_weights()


    def _set_logdir(self):
        if self.name is None:
            self.name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%f")
        self.logdir = self.basedir / self.name

        config_file =  self.logdir / 'config.json'
        if self.config is None:
            if config_file.exists():
                config_dict = load_json(str(config_file))
                self.config = Config(**config_dict)
                if not self.config.is_valid():
                    invalid_attr = self.config.is_valid(True)[1]
                    raise ValueError('Invalid attributes in loaded config: ' + ', '.join(invalid_attr))
            else:
                raise FileNotFoundError("config file doesn't exist: %s" % str(config_file.resolve()))
        else:
            if self.logdir.exists():
                warnings.warn('output path for model already exists, files may be overwritten: %s' % str(self.logdir.resolve()))
            self.logdir.mkdir(parents=True, exist_ok=True)
            save_json(vars(self.config), str(config_file))


    def _find_and_load_weights(self,prefer='best'):
        from itertools import chain
        # get all weight files and sort by modification time descending (newest first)
        weights_ext   = ('*.h5','*.hdf5')
        weights_files = chain(*(self.logdir.glob(ext) for ext in weights_ext))
        weights_files = reversed(sorted(weights_files, key=lambda f: f.stat().st_mtime))
        weights_files = list(weights_files)
        if len(weights_files) == 0:
            warnings.warn("Couldn't find any network weights (%s) to load." % ', '.join(weights_ext))
            return
        weights_preferred = list(filter(lambda f: prefer in f.name, weights_files))
        weights_chosen = weights_preferred[0] if len(weights_preferred)>0 else weights_files[0]
        print("Loading network weights from '%s'." % weights_chosen.name)
        self.load_weights(weights_chosen.name)


    def _build(self):
        return common_unet(
            n_dim           = self.config.n_dim,
            n_channel_out   = self.config.n_channel_out,
            prob_out        = self.config.probabilistic,
            residual        = self.config.unet_residual,
            n_depth         = self.config.unet_n_depth,
            kern_size       = self.config.unet_kern_size,
            n_first         = self.config.unet_n_first,
            last_activation = self.config.unet_last_activation,
        )(self.config.unet_input_shape)


    def load_weights(self, name='weights_best.h5'):
        """Load neural network weights from model folder.

        Parameters
        ----------
        name : str
            Name of HDF5 weight file (as saved during or after training).
        """
        self.keras_model.load_weights(str(self.logdir/name))


    def prepare_for_training(self, optimizer=None, **kwargs):
        """Prepare for neural network training.

        Calls :func:`csbdeep.internals.train.prepare_model` and creates
        `Keras Callbacks <https://keras.io/callbacks/>`_ to be used for training.

        Note that this method will be implicitly called once by :func:`train`
        (with default arguments) if not done so explicitly beforehand.

        Parameters
        ----------
        optimizer : obj or None
            Instance of a `Keras Optimizer <https://keras.io/optimizers/>`_ to be used for training.
            If ``None`` (default), uses ``Adam`` with the learning rate specified in ``config``.
        kwargs : dict
            Additional arguments for :func:`prepare_model`.

        """
        if optimizer is None:
            from keras.optimizers import Adam
            optimizer = Adam(lr=self.config.train_learning_rate)
        self.callbacks = prepare_model(self.keras_model, optimizer, self.config.train_loss, **kwargs)

        if self.config.train_checkpoint is not None:
            from keras.callbacks import ModelCheckpoint
            self.callbacks.append(ModelCheckpoint(str(self.logdir / self.config.train_checkpoint), save_best_only=True, save_weights_only=True))

        if self.config.train_tensorboard:
            #from ..utils.tf import DLSRSTensorBoard
            self.callbacks.append(DLSRSTensorBoard(log_dir=str(self.logdir), prefix_with_timestamp=False, n_images=3, write_images=True, prob_out=self.config.probabilistic))

        if self.config.train_reduce_lr is not None:
            from keras.callbacks import ReduceLROnPlateau
            rlrop_params = self.config.train_reduce_lr
            if 'verbose' not in rlrop_params:
                rlrop_params['verbose'] = True
            self.callbacks.append(ReduceLROnPlateau(**rlrop_params))

        self._model_prepared = True


    def train(self, X,Y, validation_data, epochs=None, steps_per_epoch=None):
        """Train the neural network with the given data.

        Parameters
        ----------
        X : :class:`numpy.ndarray`
            Array of source images.
        Y : :class:`numpy.ndarray`
            Array of target images.
        validation_data : tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray`)
            Tuple of arrays for source and target validation images.
        epochs : int
            Optional argument to use instead of the value from ``config``.
        steps_per_epoch : int
            Optional argument to use instead of the value from ``config``.

        Returns
        -------
        ``History`` object
            See `Keras training history <https://keras.io/models/model/#fit>`_.

        """

        ((isinstance(validation_data,(list,tuple)) and len(validation_data)==2)
            or _raise(ValueError('validation_data must be a pair of numpy arrays')))

        n_train, n_val = len(X), len(validation_data[0])
        frac_val = (1.0 * n_val) / (n_train + n_val)
        frac_warn = 0.05
        if frac_val < frac_warn:
            warnings.warn("small number of validation images (only %.1f%% of all images)" % (100*frac_val))
        axes = axes_check_and_normalize('S'+self.config.axes,X.ndim)
        ax = axes_dict(axes)
        div_by = 2**self.config.unet_n_depth
        axes_relevant = ''.join(a for a in 'XYZT' if a in axes)
        for a in axes_relevant:
            n = X.shape[ax[a]]
            if n % div_by != 0:
                raise ValueError(
                    "training images must be evenly divisible by %d along axes %s"
                    " (axis %s has incompatible size %d)" % (div_by,axes_relevant,a,n)
                )

        if epochs is None:
            epochs = self.config.train_epochs
        if steps_per_epoch is None:
            steps_per_epoch = self.config.train_steps_per_epoch

        if not self._model_prepared:
            self.prepare_for_training()

        training_data = DataWrapper(X, Y, self.config.train_batch_size)

        history = self.keras_model.fit_generator(generator=training_data, validation_data=validation_data,
                                                 epochs=epochs, steps_per_epoch=steps_per_epoch,
                                                 callbacks=self.callbacks, verbose=1)

        self.keras_model.save_weights(str(self.logdir / 'weights_last.h5'))

        if self.config.train_checkpoint is not None:
            self.load_weights(self.config.train_checkpoint)

        return history


    def export_TF(self):
        """Export neural network via :func:`export_SavedModel`."""
        fout = self.logdir / 'TF_SavedModel.zip'
        meta = {
            'type':          self.__class__.__name__,
            #'version':       package_version,
            'probabilistic': self.config.probabilistic,
            'axes':          self.config.axes,
            'axes_div_by':   [(2**self.config.unet_n_depth if a in 'XYZT' else 1) for a in self.config.axes],
            'tile_overlap':  tile_overlap(self.config.unet_n_depth, self.config.unet_kern_size),
        }
        export_SavedModel(self.keras_model, str(fout), meta=meta)
        print("\nModel exported in TensorFlow's SavedModel format:\n%s" % str(fout.resolve()))


    def predict(self, img, axes, normalizer=PercentileNormalizer(), resizer=PadAndCropResizer(), n_tiles=1):
        """Apply neural network to raw image to predict restored image.

        Parameters
        ----------
        img : :class:`numpy.ndarray`
            Raw input image
        axes : str
            Axes of the input ``img``.
        normalizer : :class:`Normalizer` or None
            Normalization of input image before prediction and (potentially) transformation back after prediction.
        resizer : :class:`csbdeep.data.Resizer` or None
            If necessary, input image is resized to enable neural network prediction and result is (possibly)
            resized to yield original image size.
        n_tiles : int
            Out of memory (OOM) errors can occur if the input image is too large.
            To avoid this problem, the input image is broken up into (overlapping) tiles
            that can then be processed independently and re-assembled to yield the restored image.
            This parameter denotes the number of tiles. Note that if the number of tiles is too low,
            it is adaptively increased until OOM errors are avoided, albeit at the expense of runtime.

        Returns
        -------
        :class:`numpy.ndarray`
            Returns the restored image. If the model is probabilistic, this denotes the `mean` parameter of
            the predicted per-pixel Laplace distributions (i.e., the expected restored image).
            Axes semantics are the same as in the input image. Only if the output is multi-channel and
            the input image didn't have a channel axis, then output channels are appended at the end.

        """
        return self._predict_mean_and_scale(img, axes, normalizer, resizer, n_tiles)[0]


    def predict_probabilistic(self, img, axes, normalizer=PercentileNormalizer(), resizer=PadAndCropResizer(), n_tiles=1):
        """Apply neural network to raw image to predict probability distribution for restored image.

        See :func:`predict` for parameter explanations.

        Returns
        -------
        :class:`ProbabilisticPrediction`
            Returns the probability distribution of the restored image.

        Raises
        ------
        ValueError
            If this is not a probabilistic model.

        """
        self.config.probabilistic or _raise(ValueError('This is not a probabilistic model.'))
        mean, scale = self._predict_mean_and_scale(img, axes, normalizer, resizer, n_tiles)
        return ProbabilisticPrediction(mean, scale)


    def _predict_mean_and_scale(self, img, axes, normalizer, resizer, n_tiles=1):
        """Apply neural network to raw image to predict restored image.

        See :func:`predict` for parameter explanations.

        Returns
        -------
        tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray` or None)
            If model is probabilistic, returns a tuple `(mean, scale)` that defines the parameters
            of per-pixel Laplace distributions. Otherwise, returns the restored image via a tuple `(restored,None)`

        """
        normalizer, resizer = self._check_normalizer_resizer(normalizer, resizer)
        axes = axes_check_and_normalize(axes,img.ndim)
        _permute_axes = self._make_permute_axes(axes, self.config.axes)

        x = _permute_axes(img)
        channel = axes_dict(self.config.axes)['C']

        self.config.n_channel_in == x.shape[channel] or _raise(ValueError())

        # normalize
        x = normalizer.before(x,self.config.axes)
        # resize: make divisible by power of 2 to allow downsampling steps in unet
        div_n = 2 ** self.config.unet_n_depth
        x = resizer.before(x,div_n,exclude=channel)

        done = False
        while not done:
            try:
                if n_tiles == 1:
                    x = predict_direct(self.keras_model,x,channel_in=channel,channel_out=channel)
                else:
                    overlap = tile_overlap(self.config.unet_n_depth, self.config.unet_kern_size)
                    x = predict_tiled(self.keras_model,x,channel_in=channel,channel_out=channel,
                                      n_tiles=n_tiles,block_size=div_n,tile_overlap=overlap)
                done = True
            except tf.errors.ResourceExhaustedError:
                n_tiles = max(4, 2*n_tiles)
                print('Out of memory, retrying with n_tiles = %d' % n_tiles)

        n_channel_predicted = self.config.n_channel_out * (2 if self.config.probabilistic else 1)
        x.shape[channel] == n_channel_predicted or _raise(ValueError())

        x = resizer.after(x,exclude=channel)

        mean, scale = self._mean_and_scale_from_prediction(x,axis=channel)

        if normalizer.do_after and self.config.n_channel_in==self.config.n_channel_out:
            mean, scale = normalizer.after(mean, scale)

        mean, scale = _permute_axes(mean,undo=True), _permute_axes(scale,undo=True)

        return mean, scale


    def _mean_and_scale_from_prediction(self,x,axis=-1):
        # separate mean and scale
        if self.config.probabilistic:
            _n = self.config.n_channel_out
            assert x.shape[axis] == 2*_n
            slices = [slice(None) for _ in x.shape]
            slices[axis] = slice(None,_n)
            mean = x[slices]
            slices[axis] = slice(_n,None)
            scale = x[slices]
        else:
            mean, scale = x, None
        return mean, scale

    def _make_permute_axes(self,axes_in,axes_out=None):
        if axes_out is None:
            axes_out = self.config.axes
        channel_in  = axes_dict(axes_in) ['C']
        channel_out = axes_dict(axes_out)['C']
        assert channel_out is not None

        def _permute_axes(data,undo=False):
            if data is None:
                return None
            if undo:
                if channel_in is not None:
                    return move_image_axes(data, axes_out, axes_in, True)
                else:
                    # input is single-channel and has no channel axis
                    data = move_image_axes(data, axes_out, axes_in+'C', True)
                    # output is single-channel -> remove channel axis
                    if data.shape[-1] == 1:
                        data = data[...,0]
                    return data
            else:
                return move_image_axes(data, axes_in, axes_out, True)
        return _permute_axes

    def _check_normalizer_resizer(self, normalizer, resizer):
        if normalizer is None:
            normalizer = NoNormalizer()
        if resizer is None:
            resizer = NoResizer()
        isinstance(resizer,Resizer) or _raise(ValueError())
        isinstance(normalizer,Normalizer) or _raise(ValueError())
        if normalizer.do_after:
            if self.config.n_channel_in != self.config.n_channel_out:
                warnings.warn('skipping normalization step after prediction because ' +
                              'number of input and output channels differ.')

        return normalizer, resizer




















