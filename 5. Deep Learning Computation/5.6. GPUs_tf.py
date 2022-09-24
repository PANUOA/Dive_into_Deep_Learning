import tensorflow as tf

# print(tf.device('/CPU:0'), tf.device('/GPU:0'), tf.device('/GPU:1'))

# print(len(tf.config.experimental.list_physical_devices('GPU')))

def try_gpu(i=0):  #@save
    """Return gpu(i) if exists, otherwise return cpu()."""
    if len(tf.config.experimental.list_physical_devices('GPU')) >= i + 1:
        return tf.device(f'/GPU:{i}')
    return tf.device('/CPU:0')

def try_all_gpus():  #@save
    """Return all available GPUs, or [cpu(),] if no GPU exists."""
    num_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
    devices = [tf.device(f'/GPU:{i}') for i in range(num_gpus)]
    return devices if devices else [tf.device('/CPU:0')]

# print(try_gpu(), try_gpu(10), try_all_gpus())

x = tf.constant([1, 2, 3])
# print(x.device)

with try_gpu():
    X = tf.ones((2, 3))
# print(X)

with try_gpu(1):
    Y = tf.random.uniform((2, 3))
# print(Y)

with try_gpu(1):
    Z = X
# print(X)
# print(Z)

with try_gpu(1):
    Z2 = Z
# print(Z2 is Z)

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    net = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1)])

print(net(X))

print(net.layers[0].weights[0].device, net.layers[0].weights[1].device)