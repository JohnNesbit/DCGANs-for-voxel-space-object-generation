import tensorflow as tf
import numpy as np



def discriminator(data):

    # initalize discriminator variables

    dw_1 = tf.get_variable(name="dw_1", shape=[], initializer=tf.initializers.random_normal(stddev=0.02), trainable=True)
    db_1 = tf.get_variable(name="db_1", shape=[], initializer=tf.initializers.random_normal(stddev=0.02), trainable=True)
    dw_2 = tf.get_variable(name="dw_2", shape=[], initializer=tf.initializers.random_normal(stddev=0.02), trainable=True)
    db_2 = tf.get_variable(name="db_2", shape=[], initializer=tf.initializers.random_normal(stddev=0.02), trainable=True)
    dw_3 = tf.get_variable(name="dw_3", shape=[], initializer=tf.initializers.random_normal(stddev=0.02), trainable=True)
    db_3 = tf.get_variable(name="db_3", shape=[], initializer=tf.initializers.random_normal(stddev=0.02), trainable=True)
    dw_4 = tf.get_variable(name="dw_4", shape=[], initializer=tf.initializers.random_normal(stddev=0.02), trainable=True)
    db_4 = tf.get_variable(name="db_4", shape=[], initializer=tf.initializers.random_normal(stddev=0.02), trainable=True)

    d = tf.nn.conv3d(data, dw_1, db_1, strides=[])
    d = tf.nn.batch_normalization(d)
    d = tf.nn.conv3d(d, dw_2, db_2, strides=[])
    d = tf.nn.batch_normalization(d)
    d = tf.nn.conv3d(d, dw_3, db_3, strides=[])
    d = tf.nn.relu(d)
    d = tf.matmul(d, dw_4) + db_4

    return d


def generator(label):
    
    # initalize generator vars
    
    gw_1 = tf.get_variable(name="gw_1", shape=[], initializer=tf.initializers.random_normal(stddev=0.02), trainable=True)
    gb_1 = tf.get_variable(name="gb_1", shape=[], initializer=tf.initializers.random_normal(stddev=0.02), trainable=True)
    gw_2 = tf.get_variable(name="gw_2", shape=[], initializer=tf.initializers.random_normal(stddev=0.02), trainable=True)
    gb_2 = tf.get_variable(name="gb_2", shape=[], initializer=tf.initializers.random_normal(stddev=0.02), trainable=True)
    gw_3 = tf.get_variable(name="gw_3", shape=[], initializer=tf.initializers.random_normal(stddev=0.02), trainable=True)
    gb_3 = tf.get_variable(name="gb_3", shape=[], initializer=tf.initializers.random_normal(stddev=0.02), trainable=True)
    gw_4 = tf.get_variable(name="gw_4", shape=[], initializer=tf.initializers.random_normal(stddev=0.02), trainable=True)
    gb_4 = tf.get_variable(name="gb_4", shape=[], initializer=tf.initializers.random_normal(stddev=0.02), trainable=True)
    gw_5 = tf.get_variable(name="gw_5", shape=[], initializer=tf.initializers.random_normal(stddev=0.02), trainable=True)
    gb_5 = tf.get_variable(name="gb_5", shape=[], initializer=tf.initializers.random_normal(stddev=0.02), trainable=True)

    g = tf.nn.conv3d_transpose(label, gw_1, gb_1, strides=[])
    g = tf.nn.leaky_relu(g)
    g = tf.nn.conv3d_transpose(g, gw_2, gb_2, strides=[])
    g = tf.nn.leaky_relu(g)
    g = tf.nn.conv3d_transpose(g, gw_3, gb_3, strides=[])
    g = tf.nn.relu(g)
    g = tf.nn.conv3d_transpose(g, gw_4, gb_4, strides=[])
    g = tf.nn.relu(g)
    g = tf.matmul(g, gw_5) + gb_5

    return g

