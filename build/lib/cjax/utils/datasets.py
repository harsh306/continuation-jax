# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
@software{jax2018github,
  author = {James Bradbury and Roy Frostig and Peter Hawkins and Matthew James Johnson and Chris Leary and Dougal Maclaurin and George Necula and Adam Paszke and Jake Vander{P}las and Skye Wanderman-{M}ilne and Qiao Zhang},
  title = {{JAX}: composable transformations of {P}ython+{N}um{P}y programs},
  url = {http://github.com/google/jax},
  version = {0.2.5},
  year = {2018},
}
Datasets used in examples.
"""


import array
import gzip
import os
from os import path
import struct
import urllib.request
import cv2
import numpy as np

import numpy.random as npr

_DATA = "/opt/ml/tmp/jax_example_data/"


def center_data(X):
    mean_x = np.mean(X, axis=0, keepdims=True)
    reduced_mean = np.subtract(X, mean_x)
    reduced_mean = reduced_mean.astype(np.float32)
    return reduced_mean


def synth_batches(input_shape):
    while True:
        images = np.random.rand(*input_shape).astype("float32")
        yield images


def _download(url, filename):
    """Download a url to a file in the JAX data temp directory."""
    if not path.exists(_DATA):
        os.makedirs(_DATA)
    out_file = path.join(_DATA, filename)
    if not path.isfile(out_file):
        urllib.request.urlretrieve(url, out_file)
        print("downloaded {} to {}".format(url, _DATA))


def _partial_flatten(x):
    """Flatten all but the first dimension of an ndarray."""
    return np.reshape(x, (x.shape[0], -1))


def _one_hot(x, k, dtype=np.float32):
    """Create a one-hot encoding of x of size k."""
    return np.array(x[:, None] == np.arange(k), dtype)


def mnist_raw():
    """Download and parse the raw MNIST dataset."""
    # CVDF mirror of http://yann.lecun.com/exdb/mnist/
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"

    def parse_labels(filename):
        with gzip.open(filename, "rb") as fh:
            _ = struct.unpack(">II", fh.read(8))
            return np.array(array.array("B", fh.read()), dtype=np.uint8)

    def parse_images(filename):
        with gzip.open(filename, "rb") as fh:
            _, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
            return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(
                num_data, rows, cols
            )

    for filename in [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]:
        _download(base_url + filename, filename)

    train_images = parse_images(path.join(_DATA, "train-images-idx3-ubyte.gz"))
    train_labels = parse_labels(path.join(_DATA, "train-labels-idx1-ubyte.gz"))
    test_images = parse_images(path.join(_DATA, "t10k-images-idx3-ubyte.gz"))
    test_labels = parse_labels(path.join(_DATA, "t10k-labels-idx1-ubyte.gz"))

    return train_images, train_labels, test_images, test_labels


def img_resize(train_images):
    train_data = []
    for img in train_images:
        resized_img = cv2.resize(img, (6, 6))
        train_data.append(resized_img)
    return np.asarray(train_data)

def img_resize_noise(train_images):
    train_data = []
    for img in train_images:
        #resized_img = cv2.resize(img, (6, 6))
        dst = cv2.GaussianBlur(img, (5, 5), 0.0, cv2.BORDER_DEFAULT)
        #train_data.append(resized_img)
    return dst #np.asarray(train_data)


def mnist(permute_train=False, resize=False, filter=False):
    """Download, parse and process MNIST data to unit scale and one-hot labels."""

    train_images, train_labels, test_images, test_labels = mnist_raw()
    if filter:
        train_filter = np.where(train_labels == 1)
        train_images = train_images[train_filter]
        train_labels = train_labels[train_filter]

        test_filter = np.where(test_labels == 1)
        test_images = test_images[test_filter]
        test_labels = test_labels[test_filter]

    if resize:
        train_images = img_resize(train_images)
        test_images = img_resize(test_images)
    train_images = _partial_flatten(train_images) / np.float32(255.0)
    test_images = _partial_flatten(test_images) / np.float32(255.0)
    train_labels = _one_hot(train_labels, 10)
    test_labels = _one_hot(test_labels, 10)
    if permute_train:
        perm = np.random.RandomState(0).permutation(train_images.shape[0])
        train_images = train_images[perm]
        train_labels = train_labels[perm]
    return train_images, train_labels, test_images, test_labels


def meta_mnist(batch_size, filter=False):
    train_len = 60000
    test_len = 10000
    if filter:
        train_len= 6000
        test_len = 1000
    num_complete_batches, leftover = divmod(train_len, batch_size)
    num_batches = num_complete_batches + bool(leftover)
    return locals()


def get_mnist_data(batch_size, resize, filter=False):
    train_images, train_labels, test_images, test_labels = mnist_raw()
    if resize:
        train_images = img_resize(train_images)
        #test_images = img_resize(test_images)
    train_images = _partial_flatten(train_images) / np.float32(255.0)
    #test_images = _partial_flatten(test_images) / np.float32(255.0)
    if filter:
        train_filter = np.where(train_labels == 1)
        train_images = train_images[train_filter]
        train_labels = train_labels[train_filter]
    train_labels = _one_hot(train_labels, 10)
    #test_labels = _one_hot(test_labels, 10)
    train_images = center_data(train_images)
    total_data_len = train_images.shape[0]
    num_complete_batches, leftover = divmod(total_data_len, batch_size)
    num_batches = num_complete_batches + bool(leftover)

    rng = npr.RandomState(0)
    while True:
        perm = rng.permutation(total_data_len)
        for i in range(num_batches):
            batch_idx = perm[i * batch_size : (i + 1) * batch_size]
            yield train_images[batch_idx], train_labels[batch_idx]


def get_preload_mnist_data(train_images, train_labels, test_images, test_labels, batch_size, resize, filter=False):
    if resize:
        train_images = img_resize(train_images)
        #test_images = img_resize(test_images)
    train_images = _partial_flatten(train_images) / np.float32(255.0)
    #test_images = _partial_flatten(test_images) / np.float32(255.0)
    if filter:
        train_filter = np.where(train_labels == 1)
        train_images = train_images[train_filter]
        train_labels = train_labels[train_filter]
    train_labels = _one_hot(train_labels, 10)
    #test_labels = _one_hot(test_labels, 10)
    train_images = center_data(train_images)
    total_data_len = train_images.shape[0]
    num_complete_batches, leftover = divmod(total_data_len, batch_size)
    num_batches = num_complete_batches + bool(leftover)

    rng = npr.RandomState(0)
    while True:
        perm = rng.permutation(total_data_len)
        for i in range(num_batches):
            batch_idx = perm[i * batch_size : (i + 1) * batch_size]
            yield train_images[batch_idx], train_labels[batch_idx]


def mnist_preprocess_cont(resize, filter, center=True):
    train_images, train_labels, test_images, test_labels = mnist(
        permute_train=False, resize=resize, filter=filter)
    if center:
        train_images = center_data(train_images)
        test_images = center_data(test_images)
    return train_images, train_labels, test_images, test_labels


def get_mnist_batch_alter(train_images, train_labels, test_images,
                    test_labels, alter: list, batch_size, resize, filter=False):
    alter = 1.0 - alter[0]
    if alter<0.1:
        alter = 0.1
    # alter batch len
    batch_size = int(batch_size/alter)
    print("batch_size: ",batch_size)
    total_data_len = train_images.shape[0]
    num_complete_batches, leftover = divmod(total_data_len, batch_size)
    num_batches = num_complete_batches + bool(leftover)
    rng = npr.RandomState(0)
    while True:
        perm = rng.permutation(total_data_len)
        for i in range(num_batches):
            batch_idx = perm[i * batch_size : (i + 1) * batch_size]
            yield train_images[batch_idx], train_labels[batch_idx]



if __name__ == "__main__":
    train_images, _, _, _ = mnist_raw()
    #train_images = _partial_flatten(train_images) / np.float32(255.0)
    for src in train_images:
        #src = train_images[0]#/np.float32(255.0)
        #print(src)
        #print(src/np.float32(255.0))
        #print(src.shape)
        dst = adjust_gamma(src, 0.5)
        #dst = cv2.medianBlur(src, 3)
        #src =cv2.resize(src, (6,6))
        #dst = cv2.resize(dst, (6, 6))
        print(dst.shape)
    # cv2.imshow("Gaussian Smoothing", np.hstack((src, dst)))
    # cv2.waitKey(0)  # waits until a key is pressed
    # cv2.destroyAllWindows()
    # print(train_images.shape[0])
    # print(test_images.shape[0])
    # # z = meta_mnist(5000)
    # print(meta_mnist(6742, True)["num_batches"])
