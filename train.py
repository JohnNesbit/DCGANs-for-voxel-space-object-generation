
# data files: "XChair.pickle", "YChair.pickle", "Xtable", "Ytable" generator and descriminator file: model.py
import pickle
import matplotlib
matplotlib.use("tkagg")
import binvox_rw
import numpy as np
import tensorflow as tf

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

dw_1 = tf.get_variable(name="dw_1", shape=[1, 2, 2, 1, 1], initializer=tf.initializers.random_normal(stddev=0.02),
                       trainable=True, dtype=tf.float64)
db_1 = tf.get_variable(name="db_1", shape=[1, 128, 128, 128, 1], initializer=tf.initializers.random_normal(stddev=0.02),
                       trainable=True, dtype=tf.float64)
dw_2 = tf.get_variable(name="dw_2", shape=[1, 2, 2, 1, 1], initializer=tf.initializers.random_normal(stddev=0.02),
                       trainable=True, dtype=tf.float64)
db_2 = tf.get_variable(name="db_2", shape=[1, 26, 26, 26, 1], initializer=tf.initializers.random_normal(stddev=0.02),
                       trainable=True, dtype=tf.float64)
dw_3 = tf.get_variable(name="dw_3", shape=[1, 2, 2, 1, 1], initializer=tf.initializers.random_normal(stddev=0.02),
                       trainable=True, dtype=tf.float64)
db_3 = tf.get_variable(name="db_3", shape=[1, 13, 13, 13, 1], initializer=tf.initializers.random_normal(stddev=0.02),
                       trainable=True, dtype=tf.float64)
dw_4 = tf.get_variable(name="dw_4", shape=[562432, 1], initializer=tf.initializers.random_normal(stddev=0.02),
                       # start shape changing via matmul
                       trainable=True, dtype=tf.float64)
db_4 = tf.get_variable(name="db_4", shape=[1], initializer=tf.initializers.random_normal(stddev=0.02),
                       trainable=True, dtype=tf.float64)
"""
dw_5 = tf.get_variable(name="dw_5", shape=[1, 2197], initializer=tf.initializers.random_normal(stddev=0.02),
                       trainable=True, dtype=tf.float64)
db_5 = tf.get_variable(name="db_5", shape=[1], initializer=tf.initializers.random_normal(stddev=0.02),
                       trainable=True, dtype=tf.float64)
                       """

gw_1 = tf.get_variable(name="gw_1", shape=[1, 2, 2, 1, 1], initializer=tf.initializers.random_normal(stddev=0.02),
                       trainable=True, dtype=tf.float64)
gb_1 = tf.get_variable(name="gb_1", shape=[1, 8, 8, 8, 1], initializer=tf.initializers.random_normal(stddev=0.02),
                       trainable=True, dtype=tf.float64)
gw_2 = tf.get_variable(name="gw_2", shape=[1, 4, 4, 1, 1], initializer=tf.initializers.random_normal(stddev=0.02),
                       trainable=True, dtype=tf.float64)
gb_2 = tf.get_variable(name="gb_2", shape=[1, 32, 32, 32, 1], initializer=tf.initializers.random_normal(stddev=0.02),
                       trainable=True, dtype=tf.float64)
gw_3 = tf.get_variable(name="gw_3", shape=[1, 4, 4, 1, 1], initializer=tf.initializers.random_normal(stddev=0.02),
                       trainable=True, dtype=tf.float64)
gb_3 = tf.get_variable(name="gb_3", shape=[1, 128, 128, 128, 1], initializer=tf.initializers.random_normal(stddev=0.02),
                       trainable=True, dtype=tf.float64)
gw_4 = tf.get_variable(name="gw_4", shape=[1, 2, 2, 1, 1], initializer=tf.initializers.random_normal(stddev=0.02),
                       trainable=True, dtype=tf.float64)
gb_4 = tf.get_variable(name="gb_4", shape=[1, 256, 256, 256, 1], initializer=tf.initializers.random_normal(stddev=0.02),
                       trainable=True, dtype=tf.float64)
gw_5 = tf.get_variable(name="gw_5", shape=[256, 256, 256, 1, 1], initializer=tf.initializers.random_normal(stddev=0.02),
                       trainable=True, dtype=tf.float64)
gb_5 = tf.get_variable(name="gb_5", shape=[256, 256, 256, 1, 1], initializer=tf.initializers.random_normal(stddev=0.02),
                       trainable=True, dtype=tf.float64)


def discriminator(data):
    # do calculations
    print("descriminator stars here:")
    d = tf.nn.conv3d(data, filter=dw_1, strides=[1, 2, 2, 2, 1], padding="SAME")
    d =  d + db_1
    d = tf.nn.relu(d)
    print(d.shape)
    d = tf.nn.conv3d(d, filter=dw_2, strides=[1, 5, 5, 5, 1], padding="SAME") + db_2
    d = tf.nn.relu(d)
    print(d.shape)
    d = tf.nn.conv3d(d, filter=dw_3, strides=[1, 2, 2, 2, 1], padding="SAME")  + db_3
    d = tf.nn.relu(d)
    print(d.shape)
    d = tf.reshape(d, shape=[1, 562432])
    d = tf.matmul(d, dw_4)
    d =  tf.add(d, db_4)
    d = tf.nn.relu(d)
    print(d.shape)
    """
    d = tf.matmul(dw_5, d)
    print(d.shape)
    d = d + db_5
    """
    print("d end")

    return d


def generator(label):
    np.random.seed(int(label))
    label = np.random.normal(size=[1, 4, 4, 4, 1])
    # do calculations
    # problem  Incompatible shapes between op input and calculated input gradient not with biases
    print("gen prints start:")
    g = tf.nn.conv3d_transpose(label, gw_1,  [1, 8, 8, 8, 1], strides=[1, 2, 2, 2, 1], padding="SAME", name="conv3d1")+ gb_1
    g = tf.nn.leaky_relu(g)
    g = tf.nn.conv3d_transpose(g, gw_2, [1, 32, 32, 32, 1] , strides=[1, 4, 4, 4, 1], padding="SAME")+ gb_2
    g = tf.nn.leaky_relu(g)
    g = tf.nn.conv3d_transpose(g, gw_3,  [1, 128, 128, 128, 1] , strides=[1, 4, 4, 4, 1], data_format="NDHWC", padding="SAME")+ gb_3
    g = tf.nn.relu(g)
    print(g.shape)
    g = tf.nn.conv3d_transpose(g, gw_4,[1, 256, 256, 256, 1] , strides=[1, 2, 2, 2, 1], padding="SAME")+ gb_4
    g = tf.nn.relu(g)
    print(g.shape)
    g = tf.reshape(g, [256, 256, 256, 1, 1])
    g = tf.matmul(g, gw_5) + gb_5
    g = tf.nn.softmax(g)
    print(g.shape)
    print("g end")

    return g


# temparary data values that we can swap easily.
xtrain = np.array(pickle.load(open("XChair.pickle", "rb")))
ylabels = np.array(pickle.load(open("YChair.pickle", "rb")))
#txtrain = np.array(pickle.load(open("Xtable.pickle", "rb")))
#tylabels = np.array(pickle.load(open("Ytable.pickle", "rb")))
#xtrain = np.append(xtrain, txtrain)
#ylabels = np.append(ylabels, tylabels)

"""np.random.seed(5)
np.random.shuffle(xtrain)
np.random.seed(5)
np.random.shuffle(ylabels)
"""


sess = tf.Session()
x_placeholder = tf.placeholder(tf.float64, shape= [256, 256, 256, 1, 1], name='x_placeholder')

ylabels = ylabels.reshape([21])
xtrain = xtrain.reshape([21, 256, 256, 256, 1, 1])
print(xtrain.shape)


Gz = generator(ylabels[1])

Dx = discriminator(xtrain[1])

Dg = discriminator(Gz)


# defines loss functions for models
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.constant([1], shape=[1, 1], dtype=tf.float64), logits=Dg))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.constant([1], shape=[1, 1], dtype=tf.float64),logits=Dg))
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.constant([0], shape=[1, 1], dtype=tf.float64),logits=Dx))
d_loss = d_loss_real + d_loss_real

t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if 'd' and "_" in var.name]
g_vars = [var for var in t_vars if 'g' and "_" in var.name]

with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE) as scope:

    d_trainer_fake = tf.train.AdamOptimizer(0.0001).minimize(d_loss_fake, var_list=d_vars)
    d_trainer_real = tf.train.AdamOptimizer(0.0001).minimize(d_loss_real, var_list=d_vars)


    g_trainer = tf.train.AdamOptimizer(0.0001).minimize(g_loss, var_list=g_vars, colocate_gradients_with_ops=True)

d_real_count_ph = tf.placeholder(tf.float64)
d_fake_count_ph = tf.placeholder(tf.float64)
g_count_ph = tf.placeholder(tf.float64)

d_on_generated = tf.reduce_mean(discriminator(generator(ylabels[0])))
d_on_real = tf.reduce_mean(discriminator(x_placeholder))

# initializes all variables with tensorflow
merged = tf.summary.merge_all()
sess.run(tf.global_variables_initializer())


saver = tf.train.Saver()

d_fake_count = 0
g_count = 0
d_real_count = 0
# define loss vars
gLoss = 0
dLossFake, dLossReal = 1, 1

# training loop
for i in range(21):
    real_image_batch = xtrain[i]

    # Train discriminator on generated images
    _, dLossReal, dLossFake, gLoss = sess.run([d_trainer_fake, d_loss_real, d_loss_fake, g_loss],
                                                {x_placeholder: real_image_batch})
    d_fake_count += 1


    # Train the generator
    sess.run([g_trainer, d_loss_real, d_loss_fake, g_loss],
                                                {x_placeholder: real_image_batch})
    g_count += 1


    # train d on real images
    sess.run([d_trainer_real, d_loss_real, d_loss_fake, g_loss],
                                                {x_placeholder: real_image_batch})
    d_real_count += 1


    """summary = sess.run(merged, {x_placeholder: real_image_batch, d_real_count_ph: d_real_count,
                                d_fake_count_ph: d_fake_count, g_count_ph: g_count})"""
    d_real_count, d_fake_count, g_count = 0, 0, 0

    if i % 5 == 0:

        images = sess.run(generator(ylabels[i]))
        d_result = sess.run(discriminator(x_placeholder), {x_placeholder: images})
        print("TRAINING STEP", i)
        print("Descriminator_loss:" + str(dLossReal))
        generated_model = binvox_rw.Voxels(images, dims=[256, 256, 256], translate=[-12.75, -9.37502, -26,75], scale=53.3
                                           , axis_order="xzy")
        binvox_rw.write(generated_model, fp=open("generated_file_no_" + str(i) + ".binvox", "wb"))

    if i % 20 == 0:
        save_path = saver.save(sess, "models/pretrained_3ddcgan.ckpt", global_step=i)
        print("saved to %s" % save_path)

