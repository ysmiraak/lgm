from rbm import Rbm
from utils import np, tf, tile, Record, binary, binary_variable, plot_fn


class Dbm(Record):

    def __init__(self, dim, chains
                 , init_w= tf.random_uniform_initializer(minval= -0.01, maxval= 0.01)
                 , ftype= tf.float32, scope= 'dbm'):
        self.dim, self.ftype = dim, ftype
        # todo pretraining
        with tf.variable_scope(scope):
            self.rbm = tuple(
                Rbm(scope= "rbm{}".format(i)
                    , dim_v= dim_v
                    , dim_h= dim_h
                    , chains= chains
                    , init_w= init_w
                    , ftype= self.ftype)
                for i, (dim_v, dim_h) in enumerate(zip(dim, dim[1:]), 1))
            self.w = tuple(rbm.w for rbm in self.rbm)
            # positive stage: variational inference
            self.m = tuple(rbm.h for rbm in self.rbm)
            self.v_ = self.rbm[0].v_
            self.k_meanf_ = tf.placeholder(name= 'k_meanf_', dtype= tf.int32, shape= ())

            def meanf(m):
                mf, ml = [], self.v_
                for wl, wr, mr in zip(self.w, self.w[1:], m[1:]):
                    mf.append(tf.sigmoid(tf.matmul(ml, wl) + tf.matmul(mr, wr, transpose_b= True)))
                    ml = mf[-1]
                mf.append(tf.sigmoid(tf.matmul(ml, wr)))
                return tuple(mf)

            with tf.name_scope('meanf'):
                self.meanf = tuple(
                    tf.assign(m, mf, validate_shape= False) for m, mf in zip(
                        self.m, tf.while_loop(
                            loop_vars= (self.k_meanf_, self.m)
                            , cond= lambda k, _: (0 < k)
                            , body= lambda k, m: (k - 1, meanf(m)))[1]))

            with tf.name_scope('pos'):
                bs = tf.cast(tf.shape(self.v_)[0], dtype= self.ftype)
                vm = (self.v_,) + self.meanf
                self.pos = tuple((tf.matmul(ml, mr, transpose_a= True) / bs) for ml, mr in zip(vm, vm[1:]))
            # negative stage: stochastic approximation
            self.x = tuple(rbm.v for rbm in self.rbm)
            self.x += (binary_variable(name= 'x', shape= (chains, self.dim[-1]), dtype= self.ftype),)
            self.v = self.x[0]
            self.k_gibbs_ = tf.placeholder(name= 'k_gibbs_', dtype= tf.int32, shape= ())

            def gibbs(x):
                x = list(x)
                # update odd layers
                for i, (xl, xr, wl, wr) in enumerate(zip(x[::2], x[2::2], self.w, self.w[1:])):
                    x[1+(2*i)] = binary(tf.matmul(xl, wl) + tf.matmul(xr, wr, transpose_b= True))
                # update first layer
                x[0] = binary(tf.matmul(x[1], self.w[0], transpose_b= True))
                # update even layers
                for i, (xl, xr, wl, wr) in enumerate(zip(x[1::2], x[3::2], self.w[1:], self.w[2:])):
                    x[2+(2*i)] = binary(tf.matmul(xl, wl) + tf.matmul(xr, wr, transpose_b= True))
                # update last layer
                x[-1] = binary(tf.matmul(x[-2], self.w[-1]))
                return tuple(x)

            with tf.name_scope('gibbs'):
                x = self.gibbs = tuple(
                    tf.assign(x, xg, validate_shape= False) for x, xg in zip(
                        self.x, tf.while_loop(
                            loop_vars= (self.k_gibbs_, self.x)
                            , cond= lambda k, x: (0 < k)
                            , body= lambda k, x: (k - 1, gibbs(x)))[1]))

            with tf.name_scope('neg'):
                bs = tf.cast(tf.shape(self.v)[0], dtype= self.ftype)
                self.neg = tuple((tf.matmul(xl, xr, transpose_a= True) / bs) for xl, xr in zip(x, x[1:]))
            # parameter update
            self.lr_ = tf.placeholder(name= 'lr_', dtype= self.ftype, shape= ())
            with tf.name_scope('up'):
                self.up = tuple(
                    w.assign_add((pos - neg) * self.lr_).op
                    for w, pos, neg in zip(self.w, self.pos, self.neg))
            self.step = 0

    def pcd(self, sess, wtr, batchit, k_meanf= 9, k_gibbs= 9, lr= 0.01, steps= 0, step_plot= 0, plot= plot_fn('recons')):
        if not (plot and step_plot): step_plot = 1 + steps
        for step in range(1, 1 + steps):
            self.step += 1
            # todo summarise loss
            sess.run(self.up, feed_dict= {
                self.v_: next(batchit)
                , self.k_meanf_: k_meanf
                , self.k_gibbs_: k_gibbs
                , self.lr_: lr})
            if not (step % step_plot):
                plot(sess, wtr, sess.run(self.v), self.step)

    def gen(self, sess, k_gibbs= 4):
        while True: yield sess.run(self.gibbs[0], feed_dict= {self.k_gibbs_: k_gibbs})


if False:
    from utils import mnist
    batchit = mnist(batch_size= 100, ds= 'train', with_labels= False, binary= True)

    dbm = Dbm((784, 256, 256, 784), chains= 100)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # rm -r log
    # tf.summary.FileWriter("log/dbm", sess.graph).close()
    # tf.reset_default_graph()
    # sess.close()

    # with tf.summary.FileWriter("log/dbm/pretrain") as wtr:
    #     dbm.pretrain(sess, wtr, batchit, steps= 10000, step_plot= 10000)

    with tf.summary.FileWriter("log/dbm") as wtr:
        dbm.pcd(sess, wtr, batchit, steps= 60000, step_plot= 6000)

    plot = plot_fn('gen1k')
    with tf.summary.FileWriter("log/dbm") as wtr:
        for step, v in zip(range(10), dbm.gen(sess, k_gibbs= 1000)):
            plot(sess, wtr, v, step)

    tf.train.Saver().save(sess, "./models/dbm")
