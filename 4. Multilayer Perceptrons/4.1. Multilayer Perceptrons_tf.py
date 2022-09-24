import tensorflow as tf
from d2l import tensorflow as d2l
import matplotlib.pyplot as plt

x = tf.Variable(tf.range(-8.0, 8.0, 0.1), dtype=tf.float32)
y = tf.nn.relu(x)
# d2l.plot(x.numpy(), y.numpy(), 'x', 'relu(x)', figsize=(5, 2.5))
with tf.GradientTape() as t:
    y = tf.nn.relu(x)
# d2l.plot(x.numpy(), t.gradient(y, x).numpy(), 'x', 'grad of relu',
#          figsize=(5, 2.5))

y = tf.nn.sigmoid(x)
# d2l.plot(x.numpy(), y.numpy(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
with tf.GradientTape() as t:
    y = tf.nn.sigmoid(x)
d2l.plot(x.numpy(), t.gradient(y, x).numpy(), 'x', 'grad of sigmoid',
         figsize=(5, 2.5))

y = tf.nn.tanh(x)
# d2l.plot(x.numpy(), y.numpy(), 'x', 'tanh(x)', figsize=(5, 2.5))
with tf.GradientTape() as t:
    y = tf.nn.tanh(x)
d2l.plot(x.numpy(), t.gradient(y, x).numpy(), 'x', 'grad of tanh',
         figsize=(5, 2.5))

plt.show()