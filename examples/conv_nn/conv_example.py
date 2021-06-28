from jax.experimental import stax
from jax.experimental.stax import *
import jax.numpy as np
from jax import random, jit, grad
from jax.experimental.optimizers import l2_norm
from cjax.optimizer.optimizer import OptimizerCreator
from cjax.utils.datasets import get_mnist_data, mnist, meta_mnist

# init_fun, conv_net = stax.serial(Conv(16, (3, 3), (2, 2), padding="SAME"),
#                                  BatchNorm(), Relu,
#                                  # Conv(32, (3, 3), (2, 2), padding="SAME"),
#                                  # BatchNorm(), Relu,
#                                  # Conv(10, (3, 3), (2, 2), padding="SAME"),
#                                  # BatchNorm(), Relu,
#                                  #Conv(10, (3, 3), (2, 2), padding="SAME"), Relu,
#                                  Flatten,
#                                  Dense(10),
#                                  LogSoftmax)


def objective(params, batch):
    train_batch, targets = batch
    print(train_batch.shape)
    #train_batch = np.moveaxis(train_batch, 1, -1)
    train_batch = np.moveaxis(train_batch, -1, 0)
    train_batch = np.moveaxis(train_batch, -1, 0)
    train_batch = np.moveaxis(train_batch, -2, -1)
    print(train_batch.shape)
    logits = conv_net(params, train_batch)

    # logits = predict_fun(params, train_batch)
    loss = -np.sum(logits * targets)
    #loss += l2_norm(params) + l2_norm(bparam)
    return loss


def accuracy(params, batch):
    train_batch, targets = batch
    #x = np.reshape(x, (x.shape[0], -1)) # 32, 28, 28, 1
    # 28, 28, 1, 32
    #train_batch = np.moveaxis(train_batch, 1, -1)
    train_batch = np.moveaxis(train_batch, -1, 0)
    train_batch = np.moveaxis(train_batch, -1, 0)
    train_batch = np.moveaxis(train_batch, -2, -1)

    target_class = np.argmax(targets, axis=-1)
    #predicted_class = conv_net(params, train_batch)
    predicted_class = np.argmax(conv_net(params, train_batch), axis=-1)
    #predicted_class = np.argmax(predict_fun(params, train_batch, bparam=bparam[0], activation_func=relu), axis=-1)
    return np.mean(predicted_class == target_class)

def ConvBlock(kernel_size, filters, strides=(2, 2)):
  ks = kernel_size
  filters1, filters2, filters3 = filters
  Main = stax.serial(
      Conv(filters1, (1, 1), strides), BatchNorm(), Relu,
      Conv(filters2, (ks, ks), padding='SAME'), BatchNorm(), Relu,
      Conv(filters3, (1, 1)), BatchNorm())
  Shortcut = stax.serial(Conv(filters3, (1, 1), strides), BatchNorm())
  return stax.serial(FanOut(2), stax.parallel(Main, Shortcut), FanInSum, Relu)


def IdentityBlock(kernel_size, filters):
  ks = kernel_size
  filters1, filters2 = filters
  def make_main(input_shape):
    # the number of output channels depends on the number of input channels
    return stax.serial(
        Conv(filters1, (1, 1)), BatchNorm(), Relu,
        Conv(filters2, (ks, ks), padding='SAME'), BatchNorm(), Relu,
        Conv(input_shape[0], (1, 1)), BatchNorm())
  Main = stax.shape_dependent(make_main)
  return stax.serial(FanOut(2), stax.parallel(Main, Identity), FanInSum, Relu)

def ResNet50(num_classes):
    return stax.serial(
      GeneralConv(('HWCN', 'OIHW', 'NHWC'), 64, (7, 7), (2, 2), 'SAME'),
      BatchNorm(), Relu, MaxPool((3, 3), strides=(2, 2)),
      ConvBlock(3, [64, 64, 256], strides=(1, 1)),
      IdentityBlock(3, [64, 64]),
      IdentityBlock(3, [64, 64]),
      ConvBlock(3, [128, 128, 512]),
      IdentityBlock(3, [128, 128]),
      IdentityBlock(3, [128, 128]),
      IdentityBlock(3, [128, 128]),
      ConvBlock(3, [256, 256, 1024]),
      IdentityBlock(3, [256, 256]),
      IdentityBlock(3, [256, 256]),
      IdentityBlock(3, [256, 256]),
      IdentityBlock(3, [256, 256]),
      IdentityBlock(3, [256, 256]),
      ConvBlock(3, [512, 512, 2048]),
      IdentityBlock(3, [512, 512]),
      IdentityBlock(3, [512, 512]),
      AvgPool((7, 7)), Flatten, Dense(num_classes), LogSoftmax)

init_fun, conv_net = ResNet50(10)
  # _, init_params = init_fun(rng_key, input_shape)

if __name__ == '__main__':
    img_size = 28
    channels = 1
    key = random.PRNGKey(0)
    # resnet_model_def.apply()
    input_shape = (img_size, img_size, channels, 1)
    print(input_shape)
    _, params = init_fun(key, input_shape)

    data_loader = iter(get_mnist_data(batch_size=32, resize=False, filter=False))
    num_batches = meta_mnist(batch_size=32, filter=False)["num_batches"]
    print(f"num of bathces: {num_batches}")


    compute_grad_fn = jit(grad(objective, [0]))
    opt = OptimizerCreator("adam", learning_rate=0.05).get_optimizer()
    ma_loss = []
    _, _, test_images, test_labels = mnist(
        permute_train=False, resize=False, filter=False
    )
    for epoch in range(100):
        for b_j in range(num_batches):
            batch = next(data_loader)
            ae_grads = compute_grad_fn(params, batch)

            params = opt.update_params(params, ae_grads[0], step_index=epoch)
            # bparam = opt.update_params(bparam, b_grads, step_index=epoch)
            loss = objective(params, batch)
            ma_loss.append(loss)
        print(f"loss:{loss}  norm:{l2_norm(ae_grads)}")


    #del _
    print(test_labels.shape)
    val_acc =accuracy(params, (test_images, test_labels))
    print(f"val acc: {val_acc}")


