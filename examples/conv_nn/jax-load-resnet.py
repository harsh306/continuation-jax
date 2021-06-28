import jax.numpy as np
from jax_resnet.resnet import ResNet18

from jax import random, jit, grad
from jax.experimental.optimizers import l2_norm
from cjax.optimizer.optimizer import OptimizerCreator
from cjax.utils.datasets import get_mnist_data, mnist, meta_mnist

model = ResNet18(n_classes=10)
key = random.PRNGKey(0)

print(type(model))
def objective(variables, batch):
    train_batch, targets = batch
    #print(train_batch.shape)
    train_batch = np.moveaxis(train_batch, 1, -1)
    # train_batch = np.moveaxis(train_batch, -1, 0)
    # train_batch = np.moveaxis(train_batch, -1, 0)
    # train_batch = np.moveaxis(train_batch, -2, -1)
    #print(train_batch.shape)

    logits = model.apply(variables,
                      train_batch,  # ImageNet sized inputs.
                      mutable=False)  # Ensure `batch_stats` aren't updated.

    #logits = conv_net(params, train_batch)

    # logits = predict_fun(params, train_batch)
    loss = -np.sum(logits * targets)
    #loss += l2_norm(params) + l2_norm(bparam)
    return loss


def accuracy(variables, batch):
    train_batch, targets = batch
    #x = np.reshape(x, (x.shape[0], -1)) # 32, 28, 28, 1
    # 28, 28, 1, 32
    train_batch = np.moveaxis(train_batch, 1, -1)
    # train_batch = np.moveaxis(train_batch, -1, 0)
    # train_batch = np.moveaxis(train_batch, -1, 0)
    # train_batch = np.moveaxis(train_batch, -2, -1)

    target_class = np.argmax(targets, axis=-1)
    #predicted_class = conv_net(params, train_batch)

    predicted_class = np.argmax(model.apply(variables,
                      train_batch,  # ImageNet sized inputs.
                      mutable=False) , axis=-1) # Ensure `batch_stats` aren't updated.
    #predicted_class = np.argmax(conv_net(params, train_batch), axis=-1)
    #predicted_class = np.argmax(predict_fun(params, train_batch, bparam=bparam[0], activation_func=relu), axis=-1)
    return np.mean(predicted_class == target_class)



if __name__ == '__main__':
    img_size = 28
    channels = 1
    #key = random.PRNGKey(0)
    # resnet_model_def.apply()
    input_shape = (img_size, img_size, channels, 1)
    # print(input_shape)
    # _, params = init_fun(key, input_shape)
    variables = model.init(key, np.ones((1, 28, 28, 1)))

    data_loader = iter(get_mnist_data(batch_size=32, resize=False, filter=False))
    num_batches = meta_mnist(batch_size=32, filter=False)["num_batches"]
    print(f"num of bathces: {num_batches}")

    compute_grad_fn = jit(grad(objective, [0]))
    opt = OptimizerCreator("adam", learning_rate=0.05).get_optimizer()
    ma_loss = []
    _, _, test_images, test_labels = mnist(
        permute_train=False, resize=False, filter=False
    )
    for epoch in range(1):
        for b_j in range(num_batches):
            batch = next(data_loader)
            ae_grads = compute_grad_fn(variables, batch)
            #params = variables["params"]
            variables = opt.update_params(variables, ae_grads[0], step_index=epoch)
            # bparam = opt.update_params(bparam, b_grads, step_index=epoch)
            loss = objective(variables, batch)
            ma_loss.append(loss)
            print(f"loss:{loss}  norm:{l2_norm(ae_grads)}")

            break
    #del _
    print(test_labels.shape)
    val_acc =accuracy(variables, (test_images, test_labels))
    print(f"val acc: {val_acc}")
