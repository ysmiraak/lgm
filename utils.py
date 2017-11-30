import numpy as np
import tensorflow as tf

seed = 25
np.random.seed(seed)
tf.set_random_seed(seed)


uniform_prob = tf.random_uniform_initializer()
uniform_binary = lambda *args, **kwargs: tf.round(uniform_prob(*args, **kwargs))
binary_variable = lambda name, shape, dtype: tf.get_variable(
    name= name
    , shape= shape
    , dtype= dtype
    , validate_shape= False
    , initializer= uniform_binary)


def mnist(batch_size, ds= 'train', with_labels= True, binary= False):
    from tensorflow.examples.tutorials.mnist.input_data import read_data_sets
    ds = getattr(read_data_sets("/tmp/tensorflow/mnist/input_data", validation_size=0), ds)
    if binary: ds._images = np.round(ds.images)
    while True: yield ds.next_batch(batch_size) if with_labels else ds.next_batch(batch_size)[0]


def binary(x, transform= tf.sigmoid, threshold= tf.random_uniform):
    """returns a binary tensor with the same shape and type as `x`.
    `threshold` must be a function that takes a `shape` and a `dtype`
    as the arguments.

    """
    if transform: x = transform(x)
    if threshold:
        return tf.cast(threshold(shape= tf.shape(x), dtype= x.dtype) <= x, dtype= x.dtype)
    else:
        return tf.round(x)


def tile(images, cols= None, width= 28, channels= 1):
    if cols is None:
        rows = len(images)
        for cols in range(int(rows ** 0.5), rows + 1):
            if not (rows % cols):
                break
    return np.concatenate(
        [imgs.reshape(1, -1, width, channels) for imgs in images.reshape(cols, -1)]
        , axis= 2)


def plot_fn(name, plot= tf.placeholder(dtype= tf.float32, shape= (1, None, None, 1))):
    summary = tf.summary.image(name= name, tensor= plot)
    return lambda sess, wtr, v, step: wtr.add_summary(
        sess.run(summary, feed_dict= {plot: tile(v)})
        , global_step= step)


class Record(object):

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        return repr(self.__dict__)

    def __str__(self):
        return str(self.__dict__)

    def __bool__(self):
        return bool(self.__dict__)

    def __call__(self, attr):
        return getattr(self, attr)

    def __contains__(self, item):
        return item in self.__dict__

    def __getitem__(self, key):
        return self.__dict__[key]

    def get(self, k, d= None):
        return self.__dict__.get(k, d)

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __delitem__(self, key):
        del self.__dict__[key]

    def __iter__(self):
        return iter(self.__dict__.items())

    def keys(self):
        return self.__dict__.keys()
