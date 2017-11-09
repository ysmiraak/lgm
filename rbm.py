import numpy as np
import tensorflow as tf

seed = 25
np.random.seed(seed)
tf.set_random_seed(seed)


def shuffle(x):
    idxs = np.arange(len(x))
    np.random.shuffle(idxs)
    return x[idxs]


def batches(x, bat):
    o = len(x)
    assert bat <= o
    x = shuffle(x)
    a = 0
    while True:
        b = a + bat
        if b <= o:
            yield x[a:b]
            a = b
        else:
            x = shuffle(x)
            a = 0


def rbm(dv, dh, bat, learning_rate):
    # initalizers
    iw = tf.random_uniform_initializer(minval= -0.01, maxval= 0.01)
    ib = tf.zeros_initializer()
    # visible input
    v = tf.placeholder(name= 'v', dtype= tf.float32, shape= (None, dv))
    # params
    w  = tf.get_variable(name= 'w', shape= (dv, dh), initializer= iw)
    bh = tf.get_variable(name= 'bh', shape= (1, dh), initializer= ib)
    bv = tf.get_variable(name= 'bv', shape= (1, dv), initializer= ib)
    with tf.name_scope('hgv'):
        hgv = v @ w + bh
    with tf.name_scope('h'):
        h = tf.cast(
            tf.greater_equal(
                tf.sigmoid(hgv)
                , tf.random_uniform(shape= (bat, dh), minval= 0.0, maxval= 1.0))
            , tf.float32)
    with tf.name_scope('vgh'):
        vgh = tf.matmul(h, w, transpose_b= True) + bv
    with tf.name_scope('recons'):
        recons = tf.cast(
            tf.greater_equal(
                tf.sigmoid(vgh)
                , tf.random_uniform(shape= (bat, dv), minval= 0.0, maxval= 1.0))
            , tf.float32)
    with tf.name_scope('free_energy'):
        fe = tf.reduce_mean(
            - tf.matmul(v, bv, transpose_b= True)
            - tf.reduce_sum(tf.log1p(tf.exp(hgv)), axis= 1))
    with tf.name_scope('pos'):
        pos = tf.train.GradientDescentOptimizer(learning_rate).minimize(fe)
    with tf.name_scope('neg'):
        neg = tf.train.GradientDescentOptimizer(learning_rate).minimize(- fe)
    return {'bat': bat, 'v': v, 'w': w, 'bv': bv, 'bh': bh
            , 'hgv': hgv, 'h': h, 'vgh': vgh, 'recons': recons
            , 'fe': fe, 'pos': pos, 'neg': neg}


def fit_summary(log, sess, g, ds, cd_steps, pcd_steps, step_plot, k):
    bat = g['bat']
    with tf.summary.FileWriter(log) as wtr:
        pos = g['pos'], tf.summary.scalar(name= 'loss_pos', tensor= g['fe'])
        neg = g['neg'], tf.summary.scalar(name= 'loss_neg', tensor= g['fe'])
        recons = g['recons']
        plot = tf.summary.image(
            name= 'plot'
            , tensor= tf.reshape(g['v'], (bat, 28, 28, 1))
            , max_outputs= bat)
        v = g['v']
        fpos = {v: None}
        fneg = {v: None}
        it = batches(ds['x'], bat= bat)
        for step in range(cd_steps + pcd_steps):
            # positive
            fpos[v] = next(it)
            _, summ = sess.run(pos, feed_dict= fpos)
            wtr.add_summary(summ, step)
            # gibbs update
            if step <= cd_steps: fneg[v] = next(it)
            for _ in range(k): fneg[v] = sess.run(recons, feed_dict= fneg)
            if not (step % step_plot): wtr.add_summary(sess.run(plot, feed_dict= fneg), step)
            # negative
            _, summ = sess.run(neg, feed_dict= fneg)
            wtr.add_summary(summ, step)


if False:
    ds = np.load("../../idl/mnist.npy").item()
    ds['x'] = ds['x'] >= 128
    ds['x1'] = ds['x1'] >= 128


    sess.close()
    tf.reset_default_graph()
    # rm -r log

    g = rbm(dv= 784, dh= 512, bat= 10, learning_rate= 0.01)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    tf.summary.FileWriter('log/graph', sess.graph).close()
    fit_summary(
        'log/train', sess, g, ds
        , cd_steps= 6000
        , pcd_steps= 2*6000
        , step_plot= 1800
        , k= 3)


    # pick 10 examples from test set
    v = {}
    for x, y in zip(ds['x1'], ds['y1']):
        if y not in v:
            v[y] = x
            if 10 == len(v): break
    del x
    del y
    v = np.stack([v[y] for y in range(10)])

    s = v
    with tf.summary.FileWriter("log/gen1") as wtr:
        gen = tf.summary.image(
            name= 'gen1',
            tensor= tf.reshape(g['v'], (10, 28, 28, 1)),
            max_outputs= 10)
        for step in range(101):
            if not (step % 10):
                wtr.add_summary(sess.run(gen, feed_dict= {g['v']: s}), step)
            else:
                s = sess.run(g['recons'], feed_dict= {g['v']: s})

    s = v
    with tf.summary.FileWriter("log/gen2") as wtr:
        gen = tf.summary.image(
            name= 'gen2',
            tensor= tf.reshape(g['v'], (10, 28, 28, 1)),
            max_outputs= 10)
        for step in range(1001):
            if not (step % 100):
                wtr.add_summary(sess.run(gen, feed_dict= {g['v']: s}), step)
            else:
                s = sess.run(g['recons'], feed_dict= {g['v']: s})

    s = v
    with tf.summary.FileWriter("log/gen3") as wtr:
        gen = tf.summary.image(
            name= 'gen3',
            tensor= tf.reshape(g['v'], (10, 28, 28, 1)),
            max_outputs= 10)
        for step in range(1001):
            if not (step % 100):
                wtr.add_summary(sess.run(gen, feed_dict= {g['v']: s}), step)
            else:
                s = sess.run(g['recons'], feed_dict= {g['v']: s})
