from utils import np, tf, tile, Record


class Rbm(Record):

    def __init__(self, dv, dh
                 , bv= True, bh= True
                 , sv= None, sh= None
                 , iw= tf.random_uniform_initializer(minval= -0.01, maxval= 0.01)
                 , ib= tf.zeros_initializer()
                 , scope= 'rbm'
                 , dtype= 'float32'):
        self.dv, self.dh = dv, dh
        with tf.variable_scope(scope):
            self.r_ = tf.placeholder(name= 'r_', dtype= tf.float32)
            self.w = tf.get_variable(name= 'w', shape= (dv, dh), initializer= iw)
            if bh: self.bh = tf.get_variable(name= 'bh', shape= (1, dh), initializer= ib)
            if bv: self.bv = tf.get_variable(name= 'bv', shape= (1, dv), initializer= ib)
            if sh: self.sh = tf.placeholder_with_default(name= 'sh', input= sh, shape= ())
            if sv: self.sv = tf.placeholder_with_default(name= 'sv', input= sv, shape= ())
            self.v_ = tf.placeholder(name= 'v_', dtype= tf.float32, shape= (None, dv))
            with tf.name_scope('v2h'):
                self.v2h = self.v_ @ self.w
                if bh: self.v2h += self.bh
                if sh: self.v2h *= self.sh
            with tf.name_scope('h'):
                self.h = tf.cast(tf.less_equal(self.r_, tf.sigmoid(self.v2h)), tf.float32)
            self.h_ = tf.placeholder(name= 'h_', dtype= tf.float32, shape= (None, dh))
            with tf.name_scope('h2v'):
                self.h2v = tf.matmul(self.h_, self.w, transpose_b= True)
                if bv: self.h2v += self.bv
                if sv: self.h2v *= self.sv
            with tf.name_scope('v'):
                self.v = tf.cast(tf.less_equal(self.r_, tf.sigmoid(self.h2v)), tf.float32)
            with tf.name_scope('fe'):
                self.fe = - tf.reduce_sum(tf.log1p(tf.exp(self.v2h)), axis= 1)
                if bv: self.fe -= tf.matmul(self.v_, self.bv, transpose_b= True)
                self.fe = tf.reduce_mean(self.fe)
            self.lr_ = tf.placeholder(name= 'lr_', dtype= tf.float32, shape= ())
            with tf.name_scope('pos'):
                self.pos = tf.train.GradientDescentOptimizer(self.lr_).minimize(self.fe)
            with tf.name_scope('neg'):
                self.neg = tf.train.GradientDescentOptimizer(self.lr_).minimize(- self.fe)
            self.summ_pos = tf.summary.scalar(name= "loss_pos".format(scope), tensor= self.fe)
            self.summ_neg = tf.summary.scalar(name= "loss_neg".format(scope), tensor= self.fe)
            self.recons_ = tf.placeholder(name= 'recons_', dtype= tf.float32, shape= (1, None, None, 1))
            self.summ_recons = tf.summary.image(name= 'recons', tensor= self.recons_)
            self.step = 0

    def activate_h(self, sess, v):
        return sess.run(self.h, feed_dict= {
            self.r_: np.random.rand(len(v), self.dh)
            , self.v_: v})

    def activate_v(self, sess, h):
        return sess.run(self.v, feed_dict= {
            self.r_: np.random.rand(len(h), self.dv)
            , self.h_: h})

    def gibbs_update(self, sess, v):
        return self.activate_v(sess, self.activate_h(sess, v))

    def fit_pos(self, sess, v, lr= 0.01):
        return sess.run((self.summ_pos, self.pos), feed_dict= {self.v_: v, self.lr_: lr})[0]

    def fit_neg(self, sess, v, lr= 0.01):
        return sess.run((self.summ_neg, self.neg), feed_dict= {self.v_: v, self.lr_: lr})[0]

    def plot(self, sess, wtr, v, step= None):
        wtr.add_summary(
            sess.run(self.summ_recons, feed_dict= {self.recons_: tile(v)})
            , self.step if step is None else step)

    def cd(self, sess, wtr, batchit, steps, step_plot, k= 1, lr= 0.01):
        for step in range(1, 1 + steps):
            self.step += 1
            wtr.add_summary(self.fit_pos(sess, next(batchit), lr), self.step)
            v = next(batchit)
            for _ in range(k): v = self.gibbs_update(sess, v)
            if not (step % step_plot):
                self.plot(sess, wtr, v)
            wtr.add_summary(self.fit_neg(sess, v, lr), self.step)
        return v

    def pcd(self, sess, wtr, batchit, steps, step_plot, k= 1, lr= 0.01, v= None):
        if v is None: v = next(batchit)
        for step in range(1, 1 + steps):
            self.step += 1
            wtr.add_summary(self.fit_pos(sess, next(batchit), lr), self.step)
            for _ in range(k): v = self.gibbs_update(sess, v)
            if not (step % step_plot):
                self.plot(sess, wtr, v)
            wtr.add_summary(self.fit_neg(sess, v, lr), self.step)
        return v

    def gen(self, sess, k= 1, v= None, n= None):
        """either `v` or `n` must be provided."""
        if v is None: v = np.random.randint(2, size= (n, self.dv), dtype= np.bool)
        while True:
            for _ in range(k):
                h = self.activate_h(sess, v)
                v = self.activate_v(sess, h)
            yield h, v

    def gen_v(self, sess, k= 1, v= None, n= None):
        return map(lambda hv: hv[1], self.gen(sess, k, v, n))

    def gen_h(self, sess, k= 1, v= None, n= None):
        return map(lambda hv: hv[0], self.gen(sess, k, v, n))

if False:
    from utils import mnist
    batchit = mnist(batch_size= 100, ds= 'train', with_labels= False, binary= True)

    rbm = Rbm(28*28, 210)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # rm -r log
    # tf.summary.FileWriter("log/rbm", sess.graph).close()
    # tf.reset_default_graph()
    # sess.close()

    wtr = tf.summary.FileWriter("log/rbm/train")
    v = next(batchit1)
    v = rbm1.pcd(sess, wtr, batchit1, steps= 60000, step_plot= 10000, lr= 0.01, v= v)
    v = rbm1.pcd(sess, wtr, batchit1, steps= 12000, step_plot= 3000, lr= 0.001, v= v)
    wtr.close()

    with tf.summary.FileWriter("log/rbm/gen1k") as wtr:
        for step, v in zip(range(10), rbm.gen_v(sess, k= 1000, v= v)):
            rbm.plot(sess, wtr, v, step)

    tf.train.Saver().save(sess, "./models/rbm")
