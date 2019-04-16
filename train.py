# data files: "X.pickle", "Y.pickle" generator and descriminator file: model.py
import pickle
import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt
from model import generator, discriminator
import numpy as np
import tensorflow as tf

# temparary data values that we can swap easily.
xtrain = np.array(pickle.load(open("X.pickle", "r")))
ylabels = np.array(pickle.load(open("Y.pickle", "r")))

shape = xtrain.shape

sess = tf.Session()
x_placeholder = tf.placeholder("float", shape=shape, name='x_placeholder')

Gz = generator(ylabels[1])

Dx = discriminator(x_placeholder)

Dg = discriminator(Gz)

# defines loss functions for models
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.ones_like(Dg)))
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dx, labels=tf.fill([1, 1], 0.9)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.zeros_like(Dg)))
d_loss = d_loss_real + d_loss_fake

tvars = tf.trainable_variables()

d_vars = [var for var in tvars if 'd_' in var.name]
g_vars = [var for var in tvars if 'g_' in var.name]

with tf.variable_scope(tf.get_variable_scope(), reuse=False) as scope:

    d_trainer_fake = tf.train.AdamOptimizer(0.0001).minimize(d_loss_fake, var_list=d_vars)
    d_trainer_real = tf.train.AdamOptimizer(0.0001).minimize(d_loss_real, var_list=d_vars)


    g_trainer = tf.train.AdamOptimizer(0.0001).minimize(g_loss, var_list=g_vars)


d_real_count_ph = tf.placeholder(tf.float32)
d_fake_count_ph = tf.placeholder(tf.float32)
g_count_ph = tf.placeholder(tf.float32)

d_on_generated = tf.reduce_mean(discriminator(generator(ylabels[0])))
d_on_real = tf.reduce_mean(discriminator(x_placeholder))

# initializes all variables with tensorflow
merged = tf.summary.merge_all()
sess.run(tf.global_variables_initializer())

# makes saver
saver = tf.train.Saver()

d_fake_count = 0
g_count = 0
d_real_count = 0
# define loss vars
gLoss = 0
dLossFake, dLossReal = 1, 1

# training loop
for i in range(5001):
    real_image_batch = xtrain[i]

    # Train discriminator on generated images
    _, dLossReal, dLossFake, gLoss = sess.run([d_trainer_fake, d_loss_real, d_loss_fake, g_loss],
                                                {x_placeholder: real_image_batch})
    d_fake_count += 1


    # Train the generator
    _, dLossReal, dLossFake, gLoss = sess.run([g_trainer, d_loss_real, d_loss_fake, g_loss],
                                              {x_placeholder: real_image_batch})
    print(gLoss)
    g_count += 1


    # train d on real images
    sess.run([d_trainer_real, d_loss_real, d_loss_fake, g_loss],
                                                {x_placeholder: real_image_batch})
    d_real_count += 1


    summary = sess.run(merged, {x_placeholder: real_image_batch, d_real_count_ph: d_real_count,
                                d_fake_count_ph: d_fake_count, g_count_ph: g_count})
    d_real_count, d_fake_count, g_count = 0, 0, 0
   
    
    if i % 1000 == 0:
        images = sess.run(generator(ylabels[i]))
        d_result = sess.run(discriminator(x_placeholder), {x_placeholder: images})
        print("TRAINING STEP", i)
        plt.imshow(images, cmap='Greys')
        plt.show()

    if i % 5000 == 0:
        save_path = saver.save(sess, "models/pretrained_dcgan.ckpt", global_step=i)
        print("saved to %s" % save_path)
