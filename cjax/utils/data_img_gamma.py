from cjax.utils.datasets import mnist_raw, center_data, img_resize, _partial_flatten, _one_hot
import numpy as np
import cv2
import numpy.random as npr

def mnist_gamma(resize=True, filter=False, center=True):

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

    train_dict = {}
    alters = np.linspace(start=0.05, stop=0.95, num=9)
    for k in range(9):
        train_images_tmp = []
        for img in train_images:
            dst = adjust_gamma(img, alters[k])
            train_images_tmp.append(dst)
        train_images_tmp = np.asarray(train_images_tmp)
        train_images_tmp = _partial_flatten(train_images_tmp) / np.float32(255.0)
        if center:
            train_images_tmp = center_data(train_images_tmp)
        train_dict.update({k: train_images_tmp})
    train_images = _partial_flatten(train_images) / np.float32(255.0)
    if center:
        train_images = center_data(train_images)
        test_images = center_data(test_images)
    train_dict.update({9: train_images})

    test_images = _partial_flatten(test_images) / np.float32(255.0)
    train_labels = _one_hot(train_labels, 10)
    test_labels = _one_hot(test_labels, 10)
    del train_images
    return train_dict, train_labels, test_images, test_labels

def adjust_gamma(image, gamma=1.0):
    if (gamma<=0.05):
        gamma = 0.05
    invGamma = 1.0/gamma
    table = np.array([
      ((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)])
    return cv2.LUT(image.astype(np.uint8), table.astype(np.uint8))


def get_mnist_batch_alter(train_images_dict: dict, train_labels, test_images,
                    test_labels, alter: list, batch_size, resize, filter=False):
    # alter gamma in image
    alter = alter[0]
    k = int(alter*10)
    if k>9:
        k=9
    train_images = train_images_dict[k]
    print("batch_size: ",batch_size, train_images[0].shape)
    total_data_len = train_images.shape[0]
    num_complete_batches, leftover = divmod(total_data_len, batch_size)
    num_batches = num_complete_batches + bool(leftover)
    rng = npr.RandomState(0)
    while True:
        perm = rng.permutation(total_data_len)
        for i in range(num_batches):
            batch_idx = perm[i * batch_size : (i + 1) * batch_size]
            yield train_images[batch_idx], train_labels[batch_idx]


