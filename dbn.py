from itertools import product
from rbm import Rbm
from utils import np, tf, Record, binary, plot_fn


class Dbn(Record):

    def __init__(self, dim, samples
                 , init_w= tf.random_uniform_initializer(minval= -0.01, maxval= 0.01)
                 , ftype= tf.float32, scope= 'dbn'):
        self.dim, self.ftype = dim, ftype
        with tf.variable_scope(scope):
            self.rbm = tuple(
                Rbm(scope= "rbm{}".format(i)
                    , dim_v= dim_v
                    , dim_h= dim_h
                    , samples= samples
                    , init_w= init_w
                    , ftype= self.ftype)
                for i, (dim_v, dim_h) in enumerate(zip(dim, dim[1:]), 1))
            self.w = tuple(rbm.w for rbm in self.rbm)
            # wake
            self.v_ = self.rbm[0].v_
            with tf.name_scope('wake'):
                wake = [self.v_]
                for w in self.w:
                    wake.append(tf.sigmoid(tf.matmul(wake[-1], w)))
                self.wake = tuple(wake)
            with tf.name_scope('pos'):
                bs = tf.cast(tf.shape(self.wake[0])[0], dtype= self.ftype)
                self.pos = tuple(
                    (tf.matmul(d, a, transpose_a= True) / bs)
                    for d, a in zip(self.wake, self.wake[1:]))
            # sleep
            top = self.rbm[-1]
            self.k_ = top.k_
            with tf.name_scope('sleep'):
                sleep = list(top.gibbs[::-1])
                for w in self.w[-2::-1]:
                    sleep.append(binary(tf.matmul(sleep[-1], w, transpose_b= True)))
                self.sleep = tuple(sleep[::-1])
            self.v, self.a = self.sleep[0], self.sleep[-1]
            with tf.name_scope('neg'):
                bs = tf.cast(tf.shape(self.sleep[-1])[0], dtype= self.ftype)
                self.neg = tuple(
                    (tf.matmul(d, a, transpose_a= True) / bs)
                    for d, a in zip(self.sleep, self.sleep[1:]))
            # parameter update
            self.lr_ = tf.placeholder(name= 'lr_', dtype= self.ftype, shape= ())
            with tf.name_scope('up'):
                self.up = tuple(
                    w.assign_add((pos - neg) * self.lr_).op
                    for w, pos, neg in zip(self.w, self.pos, self.neg))
            self.step = 0

    def fit(self, sess, wtr, batchit, k= 4, lr= 0.01, steps= 0, step_plot= 0, plot= plot_fn('recons')):
        if not (plot and step_plot): step_plot = 1 + steps
        for step in range(1, 1 + steps):
            self.step += 1
            # todo summarise loss
            sess.run(self.up, feed_dict= {self.v_: next(batchit), self.k_: k, self.lr_: lr})
            if not (step % step_plot):
                plot(sess, wtr, sess.run(self.v, feed_dict= {self.k_: k}), self.step)

    def ans(self, sess, a):
        return sess.run(self.v, feed_dict= {self.a: a, self.k_: 1})

    def gen(self, sess, k= 4):
        while True: yield sess.run(self.v, feed_dict= {self.k_: k})

    def pre(self, sess, wtr, batchit, k= 4, lr= 0.01, steps= 0, step_plot= 0):
        h2v = lambda x: x
        for rbm in self.rbm:
            # plot function from this rbm down to the bottom
            rbm.plot = plot_fn(rbm.scope)
            plot = lambda sess, wtr, v, step= None, rbm= rbm: rbm.plot(
                sess, wtr, step= rbm.step if step is None else step
                , v= h2v(v))
            # train this rbm
            rbm.pcd(sess, wtr, batchit, k= k, lr= lr, steps= steps, step_plot= step_plot, plot= plot)
            # downward closure of this rbm, to be used by the next plot function
            rbm.h2v = binary(tf.matmul(rbm.h, rbm.w, transpose_b= True))
            h2v = lambda h, rbm= rbm, h2v= h2v: h2v(sess.run(rbm.h2v, feed_dict= {rbm.h: h}))
            # # generate hidden states from this rbm
            # batchit = rbm.gen(sess, k= k, ret_v= False, ret_h= True)
            # upward closure of this rbm, translating visibles to hiddens
            rbm.v2h = binary(rbm.hgv, transform= False, threshold= False)
            v2h = lambda v, rbm= rbm: sess.run(rbm.v2h, feed_dict= {rbm.v_: v})
            batchit = map(v2h, batchit)


if False:
    from utils import mnist
    batchit = mnist(batch_size= 100, ds= 'train', with_labels= False, binary= True)

    dbn = Dbn((784, 210, 56, 15, 4), samples= 100)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # rm -r log
    # tf.summary.FileWriter("log/dbn", sess.graph).close()
    # tf.reset_default_graph()
    # sess.close()

    wtr = tf.summary.FileWriter("log/dbn/train")
    dbn.pre(sess, wtr, batchit, k= 4, lr= 0.01, steps= 60000, step_plot= 6000)
    dbn.fit(sess, wtr, batchit, k= 4, lr= 0.01, steps= 60000, step_plot= 6000)
    wtr.close()

    plot = plot_fn('gen')
    b = 0, 1
    q = np.array(list(product(b, b, b, b)), dtype= np.bool)
    for n, a in enumerate((np.tile(q, (100, 1)) for q in q)):
        with tf.summary.FileWriter("log/dbn/res{:02d}".format(n)) as wtr:
            plot(sess, wtr, dbn.ans(sess, a), dbn.step)

    tf.train.Saver().save(sess, "./models/dbn")
