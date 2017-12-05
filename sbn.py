from itertools import product
from utils import np, tf, Record, binary, plot_fn


class Sbn(Record):

    def __init__(self, dim, samples
                 , init_w= tf.random_uniform_initializer(minval= -0.01, maxval= 0.01)
                 , ftype= tf.float32, scope= 'sbn'):
        self.dim, self.ftype, self.scope = dim, ftype, scope
        with tf.variable_scope(scope):
            self.wr = tuple(
                tf.get_variable(name= "wr{}".format(i), shape= (dim_d, dim_a), initializer= init_w)
                for i, (dim_d, dim_a) in enumerate(zip(self.dim, self.dim[1:]), 1))
            self.wg = tuple(
                tf.get_variable(name= "wg{}".format(i), shape= (dim_a, dim_d), initializer= init_w)
                for i, (dim_d, dim_a) in enumerate(zip(self.dim, self.dim[1:]), 1))[::-1]
            self.lr_ = tf.placeholder(name= 'lr_', dtype= self.ftype, shape= ())
            # wake
            self.v_ = tf.placeholder(name= 'v_', dtype= self.ftype, shape= (None, self.dim[0]))
            with tf.name_scope('wake'):
                recogn = [self.v_]
                for w in self.wr: recogn.append(binary(tf.matmul(recogn[-1], w)))
                self.recogn = tuple(recogn)
                recogn = recogn[::-1]
                eps = self.lr_ / tf.cast(tf.shape(self.v_)[0], dtype= self.ftype)
                self.wake = tuple(
                    w.assign_add(tf.matmul(sk, (sj - pj), transpose_a= True) * eps).op
                    for w, sk, sj, pj in zip(
                            self.wg, recogn, recogn[1:]
                            , (tf.sigmoid(tf.matmul(s, w))
                               for w, s in zip(self.wg, recogn))))
            # sleep
            with tf.name_scope('a'):
                self.a = tf.round(tf.random_uniform(shape= (samples, self.dim[-1])))
            with tf.name_scope('sleep'):
                recons = [self.a]
                for w in self.wg: recons.append(binary(tf.matmul(recons[-1], w)))
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
            self.step = 0

    def fit(self, sess, wtr, batchit, lr= 0.01, steps= 0, step_plot= 0, plot= plot_fn('recons')):
        if not (plot and step_plot): step_plot = 1 + steps
        for step in range(1, 1 + steps):
            self.step += 1
            # todo summarise loss
            sess.run(self.wake, feed_dict= {self.v_: next(batchit), self.lr_: lr})
            sess.run(self.sleep, feed_dict= {self.lr_: lr})
            if not (step % step_plot):
                plot(sess, wtr, sess.run(self.v), self.step)

    def ans(self, sess, a):
        return sess.run(self.v, feed_dict= {self.a: a})

    def gen(self, sess):
        while True: yield sess.run(self.v)


if False:
    from utils import mnist
    batchit = mnist(batch_size= 100, ds= 'train', with_labels= False, binary= True)

    sbn = Sbn((784, 210, 56, 15, 4), samples= 100)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # rm -r log/sbn
    # tf.summary.FileWriter("log/sbn", sess.graph).close()
    # tf.reset_default_graph()
    # sess.close()

    with tf.summary.FileWriter("log/sbn") as wtr:
        sbn.fit(sess, wtr, batchit, lr= 0.01, steps= 600000, step_plot= 60000)

    plot = plot_fn('gen')
    b = 0, 1
    q = np.array(list(product(b, b, b, b)), dtype= np.bool)
    for n, a in enumerate((np.tile(q, (100, 1)) for q in q)):
        with tf.summary.FileWriter("log/sbn/res{:02d}".format(n)) as wtr:
            plot(sess, wtr, sbn.ans(sess, a), sbn.step)

    tf.train.Saver().save(sess, "./models/sbn")
