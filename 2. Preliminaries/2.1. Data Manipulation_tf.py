import tensorflow as tf

x = tf.range(12, dtype=tf.float32)
# print(x)
# print(x.shape)
# print(tf.size(x))
X = tf.reshape(x, (3, 4))
# print(X)
# print(tf.zeros((2, 3, 4)))
# print(tf.ones((2, 3, 4)))
# print(tf.random.normal(shape=[3, 4]))
# print(tf.constant([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]]))

x = tf.constant([1.0, 2, 4, 8])
y = tf.constant([2.0, 2, 2, 2])
# print(x + y, x - y, x * y, x / y, x ** y)  # The ** operator is exponentiation
# print(tf.exp(x))
X = tf.reshape(tf.range(12, dtype=tf.float32), (3, 4))
Y = tf.constant([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
# print(tf.concat([X, Y], axis=0), tf.concat([X, Y], axis=1))
# print(X == Y)
# print(tf.reduce_sum(X))

a = tf.reshape(tf.range(3), (3, 1))
b = tf.reshape(tf.range(2), (1, 2))
# print(a, b)
# print(a + b)
# print(X[-1], X[1:3])
X_var = tf.Variable(X)
X_var[1, 2].assign(9)
# print(X_var)
X_var = tf.Variable(X)
X_var[0:2, :].assign(tf.ones(X_var[0:2,:].shape, dtype = tf.float32) * 12)
# print(X_var)

before = id(Y)
Y = Y + X
# print(id(Y) == before)
Z = tf.Variable(tf.zeros_like(Y))
# print('id(Z):', id(Z))
Z.assign(X + Y)
# print('id(Z):', id(Z))
@tf.function
def computation(X, Y):
    Z = tf.zeros_like(Y)  # This unused value will be pruned out
    A = X + Y  # Allocations will be re-used when no longer needed
    B = A + Y
    C = B + Y
    return C + Y

# print(computation(X, Y))

A = X.numpy()
B = tf.constant(A)
# print(type(A), type(B))
a = tf.constant([3.5]).numpy()
print(a, a.item(), float(a), int(a))