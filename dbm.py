from dbn import plot_fn
from functools import reduce, partial
from itertools import product
from rbm import Rbm
from utils import np, tf, tile, Record


class Dbm(Record):

    def __init__(self, dim, scope= 'dbm', dtype= 'float32'):
        self.dim, self.dtype = dim, dtype
        two = np.array(2, dtype= self.dtype)
        rbm = [Rbm(scope= 'rbm1', dv= dim[0], dh= dim[1], bv= False, bh= False, sh= two)]
        for i, (dv, dh) in enumerate(zip(dim[1:-2], dim[2:-1]), 2):
            rbm.append(Rbm(scope= "rbm{}".format(i), dv= dv, dh= dh, bv= False, bh= False, sv= two, sh= two))
            rbm[-1].plot = plot_fn(*rbm[::-1])
        rbm.append(Rbm(scope= 'rbm0', dv= dim[-2], dh= dim[-1], bv= False, bh= False, sv= two))
        rbm[-1].plot = plot_fn(*rbm[::-1])
        self.rbm = tuple(rbm)
        self.w = tuple(rbm.w for rbm in self.rbm)
        with tf.variable_scope(scope):
            self.lr_ = tf.placeholder(name= 'lr_', dtype= self.dtype, shape= ())
            self.bs_ = tf.placeholder(name= 'bs_', dtype= tf.int32, shape= ())
            im = tf.random_uniform_initializer(dtype= self.dtype)
            self.m = tuple(
                tf.get_variable(
                    name= "m{}".format(i)
                    , shape= (1, dim)
                    , validate_shape= False
                    , initializer= im)
                for i, dim in enumerate(self.dim[1:]))
            ih = lambda *args, **kwargs: tf.round(im(*args, **kwargs))
            self.h = tuple(
                tf.get_variable(
                    name= "h{}".format(i)
                    , shape= (1, dim)
                    , validate_shape= False
                    , initializer= ih)
                for i, dim in enumerate(self.dim[1:]))
            self.v_ = self.rbm[0].v_
            self.k_meanf_ = tf.placeholder(name= 'k_meanf_', dtype= tf.int32, shape= ())
            self.k_gibbs_ = tf.placeholder(name= 'k_gibbs_', dtype= tf.int32, shape= ())
            self.v = tf.get_variable(name= 'v', shape= (1, dim[0]), validate_shape= False, initializer= ih)
            # variational inference
            meanf, m0 = [], self.v_
            for m2, w0, w2 in zip(self.m[1:], self.w, self.w[1:]):
                m1 = tf.sigmoid(tf.matmul(m0, w0) + tf.matmul(m2, w2, transpose_b= True))
                meanf.append(m1)
                m0 = m1
            meanf.append(tf.sigmoid(tf.matmul(meanf[-1], self.w[-1])))
            self.meanf = tuple(tf.assign(m, mf, validate_shape= False) for m, mf in zip(self.m, tf.while_loop(
                name= 'meanf'
                , cond= lambda _, k: 0 < k
                , body= lambda m, k: (tuple(meanf), k - 1)
                , loop_vars= (self.m, self.k_meanf_))[0]))
            # stochastic approximation
            gibbs, h0 = [], self.v
            for h2, w0, w2, dim in zip(self.h[1:], self.w, self.w[1:], self.dim[1:]):
                h1 = tf.cast(
                    tf.random_uniform(shape= (self.bs_, dim))
                    <= tf.sigmoid(tf.matmul(h0, w0) + tf.matmul(h2, w2, transpose_b= True))
                    , dtype= self.dtype)
                gibbs.append(h1)
                h0 = h1
            gibbs.append(tf.cast(
                tf.random_uniform(shape= (self.bs_, self.dim[-1]))
                <= tf.sigmoid(tf.matmul(gibbs[-1], self.w[-1]))
                , dtype= self.dtype))
            gibbs.append(tf.cast(
                tf.random_uniform(shape= (self.bs_, self.dim[0]))
                <= tf.sigmoid(tf.matmul(gibbs[0], self.w[0], transpose_b= True))
                , dtype= self.dtype))
            hv = self.h + (self.v, )
            self.gibbs = tuple(tf.assign(x, xg, validate_shape= False) for x, xg in zip(hv, tf.while_loop(
                name= 'gibbs'
                , cond= lambda _, k: 0 < k
                , body= lambda h, k: (tuple(gibbs), k - 1)
                , loop_vars= (hv, self.k_gibbs_))[0]))
            # parameter update
            vm = (self.v_,) + self.meanf
            vh = self.gibbs[-1:] + self.gibbs[:-1]
            self.up = tuple(
                w.assign_add(
                    (tf.matmul(m1, m2, transpose_a= True) - tf.matmul(x1, x2, transpose_a= True))
                    * (self.lr_ / tf.cast(self.bs_, dtype= self.dtype))).op
                for w, m1, m2, x1, x2 in zip(self.w, vm, vm[1:], vh, vh[1:]))
            self.step = 0
            self.recons_ = tf.placeholder(name= 'recons_', dtype= tf.float32, shape= (1, None, None, 1))
            self.summ_recons = tf.summary.image(name= 'recons', tensor= self.recons_)

    def pretrain(self, sess, wtr, batchit, steps, step_plot, lr= 0.01):
        for rbm in self.rbm:
            v = next(batchit)
            v = rbm.pcd(sess, wtr, batchit, steps= steps, step_plot= step_plot, lr= lr, v= v)
            # batchit = rbm.gen_h(sess, v= v)
            batchit = map(partial(rbm.activate_h, sess), batchit)

    def plot(self, sess, wtr, v, step= None):
        wtr.add_summary(
            sess.run(self.summ_recons, feed_dict= {self.recons_: tile(v)})
            , self.step if step is None else step)

    def pcd(self, sess, wtr, batchit, steps, step_plot, k_meanf= 4, k_gibbs= 4, lr= 0.01):
        for step in range(1, 1 + steps):
            self.step += 1
            v = next(batchit)
            sess.run(self.up, feed_dict= {
                self.v_: v
                , self.bs_: len(v)
                , self.lr_: lr
                , self.k_meanf_: k_meanf
                , self.k_gibbs_: k_gibbs})
            if not (step % step_plot):
                self.plot(sess, wtr, sess.run(self.v))

    def gen(self, sess, k_gibbs= 4):
        return sess.run(self.gibbs[-1], feed_dict= {self.k_gibbs_: k_gibbs})


if False:
    from utils import mnist
    batchit = mnist(batch_size= 100, ds= 'train', with_labels= False, binary= True)

    dbm = Dbm((784, 256, 256, 784))

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # rm -r log
    # tf.summary.FileWriter("log/dbm", sess.graph).close()
    # tf.reset_default_graph()
    # sess.close()

    # with tf.summary.FileWriter("log/dbm/pretrain") as wtr:
    #     dbm.pretrain(sess, wtr, batchit, steps= 10000, step_plot= 10000)

    with tf.summary.FileWriter("log/dbm/train") as wtr:
        dbm.pcd(sess, wtr, batchit, steps= 6000, step_plot= 600)

    with tf.summary.FileWriter("log/dbm/train2") as wtr:
        dbm.pcd(sess, wtr, batchit, steps= 60000, step_plot= 6000)

    # tf.train.Saver().save(sess, "./models/dbm")
