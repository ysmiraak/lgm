from utils import np, tf, Record, binary, binary_variable, plot_fn


class Rbm(Record):

    def __init__(self, dim_v, dim_h, chains
                 , init_w= tf.random_uniform_initializer(minval= -0.01, maxval= 0.01)
                 , ftype= tf.float32
                 , scope= 'rbm'):
        self.dim_v, self.dim_h, self.ftype = dim_v, dim_h, ftype
        with tf.variable_scope(scope):
            self.w = tf.get_variable(name= 'w', shape= (self.dim_v, self.dim_h), initializer= init_w)
            # positive stage: inference
            self.v_ = tf.placeholder(name= 'v_', dtype= self.ftype, shape= (None, self.dim_v))
            with tf.name_scope('hgv'):
                self.hgv = tf.sigmoid(tf.matmul(self.v_, self.w))
            # self.act_h = binary(self.hgv, transform= False, threshold= None)
            # self.h_ = tf.placeholder(name= 'h_', dtype= self.ftype, shape= (None, self.dim_h))
            # self.vgh = tf.matmul(self.h_, self.w, transpose_b= True)
            # self.act_v = binary(self.vgh, transform= False, threshold= None)

            with tf.name_scope('pos'):
                self.pos = tf.matmul(self.v_, self.hgv, transpose_a= True)
                self.pos /= tf.cast(tf.shape(self.v_)[0], dtype= self.ftype)
            # negative stage: stochastic approximation
            self.v = binary_variable(name= 'v', shape= (chains, self.dim_v), dtype= self.ftype)
            self.h = binary_variable(name= 'h', shape= (chains, self.dim_h), dtype= self.ftype)
            self.k_ = tf.placeholder(name= 'k_', dtype= tf.int32, shape= ())

            def gibbs(v, _h):
                h = binary(tf.matmul(v, self.w))
                v = binary(tf.matmul(h, self.w, transpose_b= True))
                # todo try real
                # v = tf.matmul(h, self.w, transpose_b= True)
                return v, h

            with tf.name_scope('gibbs'):
                vh = self.v, self.h
                v, h = self.gibbs = tuple(
                    tf.assign(x, x2, validate_shape= False) for x, x2 in zip(
                        vh, tf.while_loop(
                            loop_vars= (self.k_, vh)
                            , cond= lambda k, vh: (0 < k)
                            , body= lambda k, vh: (k - 1, gibbs(*vh)))[1]))

            with tf.name_scope('neg'):
                self.neg = tf.matmul(v, h, transpose_a= True)
                self.neg /= tf.cast(tf.shape(v)[0], dtype= self.ftype)
            self.lr_ = tf.placeholder(name= 'lr_', dtype= self.ftype, shape= ())
            with tf.name_scope('up'):
                self.up = self.w.assign_add((self.pos - self.neg) * self.lr_).op
            self.step = 0

    def pcd(self, sess, wtr, batchit, k= 4, lr= 0.01, steps= 0, step_plot= 0, plot= plot_fn('recons')):
        if not (plot and step_plot): step_plot = 1 + steps
        for step in range(1, 1 + steps):
            self.step += 1
            # todo summarise loss
            sess.run(self.up, feed_dict= {self.v_: next(batchit), self.k_: k, self.lr_: lr})
            if not (step % step_plot):
                plot(sess, wtr, sess.run(self.v), self.step)

    def gen(self, sess, k= 4, v= None, ret_v= True, ret_h= False):
        if v is not None: sess.run(tf.assign(self.v, v, validate_shape= False))
        if ret_v and ret_h:
            ret = self.gibbs
        elif ret_v:
            ret = self.gibbs[0]
        elif ret_h:
            ret = self.gibbs[1]
        else:
            raise StopIteration("not ret_v and not ret_h")
        while True: yield sess.run(ret, feed_dict= {self.k_: k})


if False:
    from utils import mnist
    batchit = mnist(batch_size= 100, ds= 'train', with_labels= False, binary= True)

    rbm = Rbm(28*28, 512, chains= 100)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # rm -r log
    # tf.summary.FileWriter("log/rbm", sess.graph).close()
    # tf.reset_default_graph()
    # sess.close()

    wtr = tf.summary.FileWriter("log/rbm")
    rbm.pcd(sess, wtr, batchit, k= 4, lr= 0.01, steps= 60000, step_plot= 10000)
    rbm.pcd(sess, wtr, batchit, k= 4, lr= 0.001, steps= 12000, step_plot= 3000)
    wtr.close()

    plot = plot_fn('gen1k')
    with tf.summary.FileWriter("log/rbm") as wtr:
        for step, v in zip(range(10), rbm.gen(sess, k= 1000, v= next(batchit))):
            plot(sess, wtr, v, step)

    tf.train.Saver().save(sess, "./models/rbm")
