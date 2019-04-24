import tensorflow as tf
import numpy as np

dw_1 = tf.get_variable(name="dw_1", shape=[2, 2, 2, 1, 1], initializer=tf.initializers.random_normal(stddev=0.02),
                       trainable=True, dtype=tf.float64)
db_1 = tf.get_variable(name="db_1", shape=[256, 128, 128, 1, 1], initializer=tf.initializers.random_normal(stddev=0.02),
                       trainable=True, dtype=tf.float64)
dw_2 = tf.get_variable(name="dw_2", shape=[2, 2, 2, 1, 1], initializer=tf.initializers.random_normal(stddev=0.02),
                       trainable=True, dtype=tf.float64)
db_2 = tf.get_variable(name="db_2", shape=[256, 26, 26, 1, 1], initializer=tf.initializers.random_normal(stddev=0.02),
                       trainable=True, dtype=tf.float64)
dw_3 = tf.get_variable(name="dw_3", shape=[2, 2, 2, 1, 1], initializer=tf.initializers.random_normal(stddev=0.02),
                       trainable=True, dtype=tf.float64)
db_3 = tf.get_variable(name="db_3", shape=[256, 13, 13, 1, 1], initializer=tf.initializers.random_normal(stddev=0.02),
                       trainable=True, dtype=tf.float64)
dw_4 = tf.get_variable(name="dw_4", shape=[43264, 208], initializer=tf.initializers.random_normal(stddev=0.02),
                       # start shape changing via matmul
                       trainable=True, dtype=tf.float64)
db_4 = tf.get_variable(name="db_4", shape=[208], initializer=tf.initializers.random_normal(stddev=0.02),
                       # elementwise scalar addition
                       trainable=True, dtype=tf.float64)
dw_5 = tf.get_variable(name="dw_5", shape=[1, 208], initializer=tf.initializers.random_normal(stddev=0.02),
                       trainable=True, dtype=tf.float64)
db_5 = tf.get_variable(name="db_5", shape=[1], initializer=tf.initializers.random_normal(stddev=0.02),
                       trainable=True, dtype=tf.float64)

gw_1 = tf.get_variable(name="gw_1", shape=[1, 1, 1, 1, 1], initializer=tf.initializers.random_normal(stddev=0.02),
                       trainable=True, dtype=tf.float64)
gb_1 = tf.get_variable(name="gb_1", shape=[8, 8, 8, 1, 1], initializer=tf.initializers.random_normal(stddev=0.02),
                       trainable=True, dtype=tf.float64)
gw_2 = tf.get_variable(name="gw_2", shape=[8, 8, 8, 1, 1], initializer=tf.initializers.random_normal(stddev=0.02),
                       trainable=True, dtype=tf.float64)
gb_2 = tf.get_variable(name="gb_2", shape=[32, 32, 32, 1, 1], initializer=tf.initializers.random_normal(stddev=0.02),
                       trainable=True, dtype=tf.float64)
gw_3 = tf.get_variable(name="gw_3", shape=[32, 32, 32, 1, 1], initializer=tf.initializers.random_normal(stddev=0.02),
                       trainable=True, dtype=tf.float64)
gb_3 = tf.get_variable(name="gb_3", shape=[128, 128, 128, 1, 1], initializer=tf.initializers.random_normal(stddev=0.02),
                       trainable=True, dtype=tf.float64)
gw_4 = tf.get_variable(name="gw_4", shape=[128, 128, 128, 1, 1], initializer=tf.initializers.random_normal(stddev=0.02),
                       trainable=True, dtype=tf.float64)
gb_4 = tf.get_variable(name="gb_4", shape=[256, 256, 256, 1, 1], initializer=tf.initializers.random_normal(stddev=0.02),
                       trainable=True, dtype=tf.float64)
gw_5 = tf.get_variable(name="gw_5", shape=[256, 256, 256, 1, 1], initializer=tf.initializers.random_normal(stddev=0.02),
                       trainable=True, dtype=tf.float64)
gb_5 = tf.get_variable(name="gb_5", shape=[256, 256, 256, 1, 1], initializer=tf.initializers.random_normal(stddev=0.02),
                       trainable=True, dtype=tf.float64)

def discriminator(data):
    # do calculations
    d = tf.nn.conv3d(data, filter=dw_1, strides=[2, 2, 2, 1, 1], padding="SAME")
    d =  d + db_1
    d = tf.nn.relu(d)
    d = tf.nn.conv3d(d, filter=dw_2, strides=[5, 5, 5, 1, 1], padding="SAME") + db_2
    d = tf.nn.relu(d)
    d = tf.nn.conv3d(d, filter=dw_3, strides=[2, 2, 2, 1, 1], padding="SAME") + db_3
    d = tf.nn.relu(d)
    d = tf.reshape(d, shape=[1, 43264])
    d = tf.matmul(d, dw_4)
    d =   tf.add(d, db_4)
    d = tf.nn.relu(d)
    d = tf.reshape(d, shape=[208, 1])
    d = tf.matmul(dw_5, d)
    print(d.shape)
    d = d + db_5

    return d


def generator(label):
    np.random.seed(int(label))
    label = np.random.normal(size=[4, 4, 4, 1, 1])
    # do calculations
    g = tf.nn.conv3d_transpose(label, gw_1, [8, 8, 8, 1, 1], [2, 2, 2, 1, 1], padding="SAME") + gb_1
    g = tf.nn.leaky_relu(g)
    g = tf.nn.conv3d_transpose(g, gw_2, [32, 32, 32, 1, 1], strides=[4, 4, 4, 1, 1], padding="SAME") + gb_2
    g = tf.nn.leaky_relu(g)
    g = tf.nn.conv3d_transpose(g, gw_3, [128, 128, 128, 1, 1], strides=[4, 4, 4, 1, 1], padding="SAME") + gb_3
    g = tf.nn.relu(g)
    g = tf.nn.conv3d_transpose(g, gw_4, [256, 256, 256, 1, 1], strides=[2, 2, 2, 1, 1], padding="SAME") + gb_4
    g = tf.nn.relu(g)
    g = tf.matmul(g, gw_5) + gb_5


    return g


