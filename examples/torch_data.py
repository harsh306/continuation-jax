"""

@article{sinha2020curriculum,
  title={Curriculum By Smoothing},
  author={Sinha, Samarth and Garg, Animesh and Larochelle, Hugo},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
"""
import os
import scipy.io
import numpy as np
import jax.numpy as jnp
import random

import torch
import torch.utils.data as data
from torchvision import transforms, datasets

# import torch.multiprocessing as multiprocessing
# multiprocessing.set_start_method('spawn')
_DATA = "/opt/ml/tmp/jax_example_data/"


class JaxDataWrapper(torch.utils.data.Dataset):
    def __init__(self, train, args, seed=0):
        super().__init__()
        if args["dataset"] == "mnist":
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
            )
            self._data = datasets.MNIST(
                root=args["data"], train=train, download=True, transform=transform
            )
        elif args["dataset"] == "cifar10":
            transform = transforms.Compose(
                [
                    transforms.Scale(32),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (
                            0.5,
                            0.5,
                            0.5,
                        ),
                        (
                            0.5,
                            0.5,
                            0.5,
                        ),
                    ),
                ]
            )
            self._data = datasets.CIFAR10(
                root=args["data"], train=train, download=True, transform=transform
            )
        elif args["dataset"] == "cifar100":
            transform = transforms.Compose(
                [
                    transforms.Scale(32),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (
                            0.5,
                            0.5,
                            0.5,
                        ),
                        (
                            0.5,
                            0.5,
                            0.5,
                        ),
                    ),
                ]
            )
            self._data = datasets.CIFAR100(
                root=args["data"], train=train, download=True, transform=transform
            )
        elif args["dataset"] == "imagenet":
            transform = transforms.Compose(
                [
                    transforms.Scale(64),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (
                            0.5,
                            0.5,
                            0.5,
                        ),
                        (
                            0.5,
                            0.5,
                            0.5,
                        ),
                    ),
                ]
            )
            if train:
                self._data = datasets.ImageFolder(
                    os.path.join(args["data"], "tiny-imagenet-200", "train"),
                    transform=transform,
                )
            else:
                self._data = datasets.ImageFolder(
                    os.path.join(args["data"], "tiny-imagenet-200", "val"),
                    transform=transform,
                )
        else:
            self._data = datasets.MNIST(root=args["data"], train=False, download=True)

        self._data_len = len(self._data)

    def __getitem__(self, index):
        img, label = self._data[index]
        return jnp.asarray(np.asarray(img)), label

    def __len__(self):
        return self._data_len


def jax_collate_fn(batch):
    if isinstance(batch[0], jnp.ndarray):
        return jnp.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        return type(batch[0])(jax_collate_fn(samples) for samples in zip(*batch))
    else:
        return jnp.asarray(batch)


def get_data(**args):
    args.update({"data": _DATA})
    train_data = JaxDataWrapper(
        train=True,
        args=args,
    )
    test_data = JaxDataWrapper(
        train=False,
        args=args,
    )

    train_loader = data.DataLoader(
        train_data,
        batch_size=args["batch_size"],
        pin_memory=True,
        collate_fn=jax_collate_fn,
        num_workers=args["num_workers"],
        shuffle=True,
        drop_last=True,
    )

    test_loader = data.DataLoader(
        test_data,
        batch_size=args["batch_size"],
        pin_memory=True,
        collate_fn=jax_collate_fn,
        num_workers=args["num_workers"],
        shuffle=True,
        drop_last=False,
    )
    if args["train_only"]:
        return train_loader
    elif args["test_only"]:
        return test_loader
    else:
        return train_loader, test_loader


if __name__ == "__main__":
    train_loader = get_data(
        dataset="mnist",
        batch_size=2,
        data=_DATA,
        num_workers=2,
        train_only=True,
        test_only=False,
    )
