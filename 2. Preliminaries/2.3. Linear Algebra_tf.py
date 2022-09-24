import tensorflow as tf

x = tf.constant(3.0)
y = tf.constant(2.0)

# print(x + y, x * y, x / y, x**y)
x = tf.range(4)
# print(x)
# print(x[3])
# print(len(x))
# print(x.shape)

A = tf.reshape(tf.range(20), (5, 4))
# print(A)
# print(tf.transpose(A))
B = tf.constant([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
# print(B)
# print(B == tf.transpose(B))

X = tf.reshape(tf.range(24), (2, 3, 4))
# print(X)

A = tf.reshape(tf.range(20, dtype=tf.float32), (5, 4))
B = A  # No cloning of `A` to `B` by allocating new memory
# print(A, A + B)
# print(A * B)
a = 2
X = tf.reshape(tf.range(24), (2, 3, 4))
# print(a + X, (a * X).shape)

x = tf.range(4, dtype=tf.float32)
# print(x, tf.reduce_sum(x))
# print(A.shape, tf.reduce_sum(A))
A_sum_axis0 = tf.reduce_sum(A, axis=0)
# print(A_sum_axis0, A_sum_axis0.shape)
A_sum_axis1 = tf.reduce_sum(A, axis=1)
# print(A_sum_axis1, A_sum_axis1.shape)
# print(tf.reduce_sum(A, axis=[0, 1]))  # Same as `tf.reduce_sum(A)`
# print(tf.reduce_mean(A), tf.reduce_sum(A) / tf.size(A).numpy())
# print(tf.reduce_mean(A, axis=0), tf.reduce_sum(A, axis=0) / A.shape[0])

sum_A = tf.reduce_sum(A, axis=1, keepdims=True)
# print(sum_A)
# print(A / sum_A)
# print(tf.cumsum(A, axis=0))

y = tf.ones(4, dtype=tf.float32)
# print(x, y, tf.tensordot(x, y, axes=1))
# print(tf.reduce_sum(x * y))

# print(A.shape, x.shape, tf.linalg.matvec(A, x))

B = tf.ones((4, 3), tf.float32)
# print(tf.matmul(A, B))

u = tf.constant([3.0, -4.0])
# print(tf.norm(u))

# print(tf.reduce_sum(tf.abs(u)))

print(tf.norm(tf.ones((4, 9))))