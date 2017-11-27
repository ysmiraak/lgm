from functools import reduce, partial
from itertools import product
from rbm import Rbm
from utils import np, tf, tile


def plot_fn(rbm, *rbms):
    return lambda sess, wtr, v, step= None: wtr.add_summary(
        sess.run(rbm.summ_recons, feed_dict= {
            rbm.recons_: tile(
                reduce(lambda h, rbm: rbm.activate_v(sess, h), rbms, v))})
        , rbm.step if step is None else step)


if False:
    from utils import mnist
    batchit1 = mnist(batch_size= 100, ds= 'train', with_labels= False, binary= True)

    dims = 784, 210, 56, 15, 4
    rbm1 = Rbm(dims[0], dims[1], bh= False, scope= 'rbm1')
    rbm2 = Rbm(dims[1], dims[2], bh= False, scope= 'rbm2')
    rbm3 = Rbm(dims[2], dims[3], bh= False, scope= 'rbm3')
    rbm4 = Rbm(dims[3], dims[4], scope= 'rbm4')

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # rm -r log
    # tf.summary.FileWriter("log/dbn", sess.graph).close()
    # tf.reset_default_graph()
    # sess.close()

    ##############
    # train rbm1 #
    ##############
    wtr = tf.summary.FileWriter("log/dbn/train")
    v = next(batchit1)
    v = rbm1.pcd(sess, wtr, batchit1, steps= 60000, step_plot= 10000, lr= 0.01, v= v)
    v = rbm1.pcd(sess, wtr, batchit1, steps= 12000, step_plot= 3000, lr= 0.001, v= v)
    wtr.close()

    with tf.summary.FileWriter("log/dbn/gen1k1") as wtr:
        for step, v in zip(range(10), rbm1.gen_v(sess, k= 1000, v= v)):
            rbm1.plot(sess, wtr, v, step)
    ##############
    # train rbm2 #
    ##############
    # batchit2 = rbm1.gen_h(sess, v= v)
    batchit2 = map(partial(rbm1.activate_h, sess), batchit1)
    rbm2.plot = plot_fn(rbm2, rbm1)

    wtr = tf.summary.FileWriter("log/dbn/train")
    v = next(batchit2)
    v = rbm2.pcd(sess, wtr, batchit2, steps= 60000, step_plot= 10000, lr= 0.01, v= v)
    v = rbm2.pcd(sess, wtr, batchit2, steps= 12000, step_plot= 3000, lr= 0.001, v= v)
    wtr.close()

    with tf.summary.FileWriter("log/dbn/gen1k2") as wtr:
        for step, v in zip(range(10), rbm2.gen_v(sess, k= 1000, v= v)):
            rbm2.plot(sess, wtr, v, step)
    ##############
    # train rbm3 #
    ##############
    # batchit3 = rbm2.gen_h(sess, v= v)
    batchit3 = map(partial(rbm2.activate_h, sess), batchit2)
    rbm3.plot = plot_fn(rbm3, rbm2, rbm1)

    wtr = tf.summary.FileWriter("log/dbn/train")
    v = next(batchit3)
    v = rbm3.pcd(sess, wtr, batchit3, steps= 60000, step_plot= 10000, lr= 0.01, v= v)
    v = rbm3.pcd(sess, wtr, batchit3, steps= 12000, step_plot= 3000, lr= 0.001, v= v)
    wtr.close()

    with tf.summary.FileWriter("log/dbn/gen1k3") as wtr:
        for step, v in zip(range(10), rbm3.gen_v(sess, k= 1000, v= v)):
            rbm3.plot(sess, wtr, v, step)
    ##############
    # train rbm4 #
    ##############
    # batchit4 = rbm3.gen_h(sess, v= v)
    batchit4 = map(partial(rbm3.activate_h, sess), batchit3)
    rbm4.plot = plot_fn(rbm4, rbm3, rbm2, rbm1)

    wtr = tf.summary.FileWriter("log/dbn/train")
    v = next(batchit4)
    v = rbm4.pcd(sess, wtr, batchit4, steps= 60000, step_plot= 10000, lr= 0.01, v= v)
    v = rbm4.pcd(sess, wtr, batchit4, steps= 12000, step_plot= 3000, lr= 0.001, v= v)
    wtr.close()

    with tf.summary.FileWriter("log/dbn/gen1k4") as wtr:
        for step, v in zip(range(10), rbm4.gen_v(sess, k= 1000, v= v)):
            rbm4.plot(sess, wtr, v, step)
    ##########
    # finish #
    ##########
    tf.train.Saver().save(sess, "./models/dbn")

    plot = plot_fn(rbm4, rbm4, rbm3, rbm2, rbm1)
    b = 0, 1
    q = np.array(list(product(b, b, b, b)), dtype= np.bool)
    for n, h in enumerate((np.tile(q, (100, 1)) for q in q)):
        with tf.summary.FileWriter("log/dbn/res{:02d}".format(n)) as wtr:
            plot(sess, wtr, h)
