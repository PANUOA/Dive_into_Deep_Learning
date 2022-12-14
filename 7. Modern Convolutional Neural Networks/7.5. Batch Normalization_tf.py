import matplotlib.pyplot as plt

import tensorflow as tf
from d2l import tensorflow as d2l


def batch_norm(X, gamma, beta, moving_mean, moving_var, eps):
    # Compute reciprocal of square root of the moving variance elementwise
    inv = tf.cast(tf.math.rsqrt(moving_var + eps), X.dtype)
    # Scale and shift
    inv *= gamma
    Y = X * inv + (beta - moving_mean * inv)
    return Y

class BatchNorm(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BatchNorm, self).__init__(**kwargs)

    def build(self, input_shape):
        weight_shape = [input_shape[-1], ]
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        self.gamma = self.add_weight(name='gamma', shape=weight_shape,
            initializer=tf.initializers.ones, trainable=True)
        self.beta = self.add_weight(name='beta', shape=weight_shape,
            initializer=tf.initializers.zeros, trainable=True)
        # The variables that are not model parameters are initialized to 0
        self.moving_mean = self.add_weight(name='moving_mean',
            shape=weight_shape, initializer=tf.initializers.zeros,
            trainable=False)
        self.moving_variance = self.add_weight(name='moving_variance',
            shape=weight_shape, initializer=tf.initializers.ones,
            trainable=False)
        super(BatchNorm, self).build(input_shape)

    def assign_moving_average(self, variable, value):
        momentum = 0.9
        delta = variable * momentum + value * (1 - momentum)
        return variable.assign(delta)

    @tf.function
    def call(self, inputs, training):
        if training:
            axes = list(range(len(inputs.shape) - 1))
            batch_mean = tf.reduce_mean(inputs, axes, keepdims=True)
            batch_variance = tf.reduce_mean(tf.math.squared_difference(
                inputs, tf.stop_gradient(batch_mean)), axes, keepdims=True)
            batch_mean = tf.squeeze(batch_mean, axes)
            batch_variance = tf.squeeze(batch_variance, axes)
            mean_update = self.assign_moving_average(
                self.moving_mean, batch_mean)
            variance_update = self.assign_moving_average(
                self.moving_variance, batch_variance)
            self.add_update(mean_update)
            self.add_update(variance_update)
            mean, variance = batch_mean, batch_variance
        else:
            mean, variance = self.moving_mean, self.moving_variance
        output = batch_norm(inputs, moving_mean=mean, moving_var=variance,
            beta=self.beta, gamma=self.gamma, eps=1e-5)
        return output

# # Recall that this has to be a function that will be passed to `d2l.train_ch6`
# # so that model building or compiling need to be within `strategy.scope()` in
# # order to utilize the CPU/GPU devices that we have
# def net():
#     return tf.keras.models.Sequential([
#         tf.keras.layers.Conv2D(filters=6, kernel_size=5,
#                                input_shape=(28, 28, 1)),
#         BatchNorm(),
#         tf.keras.layers.Activation('sigmoid'),
#         tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
#         tf.keras.layers.Conv2D(filters=16, kernel_size=5),
#         BatchNorm(),
#         tf.keras.layers.Activation('sigmoid'),
#         tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(120),
#         BatchNorm(),
#         tf.keras.layers.Activation('sigmoid'),
#         tf.keras.layers.Dense(84),
#         BatchNorm(),
#         tf.keras.layers.Activation('sigmoid'),
#         tf.keras.layers.Dense(10)]
#     )
#
lr, num_epochs, batch_size = 1.0, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# net = d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
#
# plt.show()
#
# print(tf.reshape(net.layers[1].gamma, (-1,)), tf.reshape(net.layers[1].beta, (-1,)))

def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=5,
                               input_shape=(28, 28, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.Dense(84),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.Dense(10),
    ])

d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

plt.show()