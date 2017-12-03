from itertools import product
from utils import np, tf, Record, binary, plot_fn


class Sbn(Record):

    def __init__(self, dim, samples
                 , init_w= tf.random_uniform_initializer(minval= -0.01, maxval= 0.01)
                 , ftype= tf.float32, scope= 'sbn'):
        self.dim, self.ftype = dim, ftype
        with tf.variable_scope(scope):
            self.w = tuple(
                tf.get_variable(name= "w{}".format(i), shape= (dim_d, dim_a), initializer= init_w)
                for i, (dim_d, dim_a) in enumerate(zip(self.dim, self.dim[1:]), 1))
            # wake
            self.v_ = tf.placeholder(name= 'v_', dtype= self.ftype, shape= (None, self.dim[0]))
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
            with tf.name_scope('sleep'):
                sleep = [tf.round(tf.random_uniform(shape= (samples, self.dim[-1])))]
                for w in self.w[::-1]:
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

    def fit(self, sess, wtr, batchit, lr= 0.01, steps= 0, step_plot= 0, plot= plot_fn('recons')):
        if not (plot and step_plot): step_plot = 1 + steps
        for step in range(1, 1 + steps):
            self.step += 1
            # todo summarise loss
            sess.run(self.up, feed_dict= {self.v_: next(batchit), self.lr_: lr})
            if not (step % step_plot):
                plot(sess, wtr, sess.run(self.v), self.step)

    def ans(self, sess, a):
        return sess.run(self.v, feed_dict= {self.a: a})

    def gen(self, sess):
        while True: yield sess.run(self.v)


if False:
    from utils import mnist
    batchit = mnist(batch_size= 100, ds= 'train', with_labels= False, binary= True)

    # dim = 4, 8, 16, 32, 64, 128, 256, 512, 784
    dim = 784, 784, 784, 784, 784, 784, 784, 784
    sbn = Sbn(dim[::-1], samples= 100)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # rm -r log/sbn
    # tf.summary.FileWriter("log/sbn", sess.graph).close()
    # tf.reset_default_graph()
    # sess.close()

    with tf.summary.FileWriter("log/sbn") as wtr:
        sbn.fit(sess, wtr, batchit, lr= 0.01, steps= 60000, step_plot= 6000)

    # plot = plot_fn('gen')
    # b = 0, 1
    # q = np.array(list(product(b, b, b, b)), dtype= np.bool)
    # for n, a in enumerate((np.tile(q, (100, 1)) for q in q)):
    #     with tf.summary.FileWriter("log/sbn/res{:02d}".format(n)) as wtr:
    #         plot(sess, wtr, sbn.ans(sess, a), sbn.step)

    tf.train.Saver().save(sess, "./models/sbn")
