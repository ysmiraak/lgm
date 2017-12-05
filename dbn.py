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
            self.w = tuple(rbm.w for rbm in self.rbm[::-1])
            self.wg = tuple(tf.transpose(w) for w in self.w)
            self.wr = tuple(
                tf.get_variable(name= "wr{}".format(i), shape= (dim_d, dim_a), initializer= init_w)
                for i, (dim_d, dim_a) in enumerate(zip(self.dim, self.dim[1:]), 1))
            self.lr_ = tf.placeholder(name= 'lr_', dtype= self.ftype, shape= ())
            # wake
            self.v_ = self.rbm[0].v_
            with tf.name_scope('wake'):
                recogn = [self.v_]
                for w in self.wr: recogn.append(binary(tf.matmul(recogn[-1], w)))
                self.recogn = tuple(recogn)
                recogn = recogn[::-1]
                eps = self.lr_ / tf.cast(tf.shape(self.v_)[0], dtype= self.ftype)
                self.wake = tuple(
                    w.assign_add(tf.matmul((sj - pj), sk, transpose_a= True) * eps).op
                    for w, sk, sj, pj in zip(
                            self.w, recogn, recogn[1:]
                            , (tf.sigmoid(tf.matmul(s, w))
                               for w, s in zip(self.wg, recogn))))
            # sleep
            top = self.rbm[-1]
            self.k_, (self.v, self.a) = top.k_, top.gibbs
            with tf.name_scope('sleep'):
                recons = [self.a, self.v]
                for w in self.wg[1::]: recons.append(binary(tf.matmul(recons[-1], w)))
                self.recons = tuple(recons)
                recons = recons[::-1]
                eps = self.lr_ / tf.cast(tf.shape(self.a)[0], dtype= self.ftype)
                self.sleep = tuple(
                    w.assign_add(tf.matmul(sj, (sk - qk), transpose_a= True) * eps).op
                    for w, sj, sk, qk in zip(
                            self.wr, recons, recons[1:]
                            , (tf.sigmoid(tf.matmul(s, w))
                               for w, s in zip(self.wr, recons))))
            # the waking world is the amnesia of dream.
            self.v = self.recons[-1]
            with tf.name_scope('ances'):
                self.a = self.rbm[-1].h
                ances = [self.a]
                for w in self.wg: ances.append(binary(tf.matmul(ances[-1], w)))
                self.ances = ances[-1]
            self.step = 0

    def fit(self, sess, wtr, batchit, k= 4, lr= 0.01, steps= 0, step_plot= 0, plot= plot_fn('recons')):
        if not (plot and step_plot): step_plot = 1 + steps
        for step in range(1, 1 + steps):
            self.step += 1
            # todo summarise loss
            sess.run(self.wake, feed_dict= {self.v_: next(batchit), self.lr_: lr})
            sess.run(self.sleep, feed_dict= {self.k_: k, self.lr_: lr})
            if not (step % step_plot):
                plot(sess, wtr, sess.run(self.v, feed_dict= {self.k_: k}), self.step)

    def ans(self, sess, a):
        return sess.run(self.ances, feed_dict= {self.a: a})

    def gen(self, sess, k= 4):
        while True: yield sess.run(self.v, feed_dict= {self.k_: k})

    def pre(self, sess, wtr, batchit, k= 4, lr= 0.01, steps= 0, step_plot= 0, sleep= 0):
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
        for _ in range(sleep): sess.run(self.sleep, feed_dict= {self.k_: k, self.lr_: lr})


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
    dbn.pre(sess, wtr, batchit, k= 4, lr= 0.01, steps= 60000, step_plot= 6000, sleep= 6000)
    dbn.fit(sess, wtr, batchit, k= 4, lr= 0.01, steps= 360000, step_plot= 36000)
    wtr.close()

    plot = plot_fn('gen')
    b = 0, 1
    q = np.array(list(product(b, b, b, b)), dtype= np.bool)
    for n, a in enumerate((np.tile(q, (100, 1)) for q in q)):
        with tf.summary.FileWriter("log/dbn/res{:02d}".format(n)) as wtr:
            plot(sess, wtr, dbn.ans(sess, a), dbn.step)

    tf.train.Saver().save(sess, "./models/dbn")
