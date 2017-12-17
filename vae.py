from utils import np, tf, Record, mnist, plot_fn


class Vae(Record):

    def __init__(self, dat, dim_rec, dim_z, dim_gen, scope= 'vae'):
        assert 2 == dat.ndim
        assert isinstance(dim_rec, tuple)
        assert isinstance(dim_z, int)
        assert isinstance(dim_gen, tuple)
        init_w = tf.variance_scaling_initializer(scale= 2.0, mode= 'fan_in', distribution= 'uniform')
        init_b = tf.constant_initializer(0.01)
        init_z = tf.zeros_initializer()
        with tf.variable_scope(scope):
            dat = self.dat = tf.constant(name= 'dat', value= dat)
            bs_ = self.bs_ = tf.placeholder(name= 'bs_', dtype= tf.int32, shape= ())
            bat = self.bat = tf.random_uniform(name= 'bat', shape= (bs_,), minval= 0, maxval= dat.shape[0], dtype= tf.int32)
            h = x = self.x = tf.nn.embedding_lookup(name= 'x', params= dat, ids= bat)
            for i, dim in enumerate(dim_rec, 1):
                name = "hr{}".format(i)
                h = tf.layers.dense(
                    name= name
                    , inputs= h
                    , units= dim
                    , activation= tf.nn.relu
                    , kernel_initializer= init_w
                    , bias_initializer= init_b)
                setattr(self, name, h)
            mu = self.mu = tf.layers.dense(
                name= 'mu'
                , inputs= h
                , units= dim_z
                , kernel_initializer= init_w
                , bias_initializer= init_z)
            lv = self.lv = tf.layers.dense(
                name= 'lv'
                , inputs= h
                , units= dim_z
                , kernel_initializer= init_w
                , bias_initializer= init_z)
            with tf.name_scope('z'):
                h = z = self.z = mu + tf.exp(0.5 * lv) * tf.random_normal(shape= tf.shape(lv))
            for i, dim in enumerate(dim_gen, 1):
                name = "hg{}".format(i)
                h = tf.layers.dense(
                    name= name
                    , inputs= h
                    , units= dim
                    , activation= tf.nn.relu
                    , kernel_initializer= init_w
                    , bias_initializer= init_b)
                setattr(self, name, h)
            logits = tf.layers.dense(
                name= 'logits'
                , inputs= h
                , units= dat.shape[1]
                # , activation= tf.nn.sigmoid
                , kernel_initializer= init_w
                , bias_initializer= init_z)
            g = self.g = tf.sigmoid(logits)
            with tf.name_scope('loss_recons'):
                # loss_recons = self.loss_recons = tf.reduce_mean(
                #     tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels= x, logits= logits), axis= 1))
                loss_recons = self.loss_recons = tf.reduce_mean(
                    tf.reduce_sum(tf.square(x - g), axis= 1))
            with tf.name_scope('loss_relent'):
                # loss_relent = self.loss_relent = tf.reduce_mean(
                #     0.5 * tf.reduce_sum((- 1.0 - lv + tf.exp(lv) + tf.square(mu)), axis= 1))
                loss_relent = self.loss_relent = tf.reduce_mean(
                    tf.reduce_sum((- 1.0 - lv + tf.exp(lv) + tf.square(mu)), axis= 1))
            with tf.name_scope('loss'):
                loss = self.loss = loss_relent + loss_recons
            up = self.up = tf.train.AdamOptimizer().minimize(loss)
            self.step = 0


if False:
    dat = next(mnist(60000, with_labels= False, binary= False))

    vae = Vae(dat, dim_rec= (128, 128), dim_z= 128, dim_gen= (128, 128))
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # rm -r log
    # tf.summary.FileWriter('log/vae', sess.graph).close()
    # tf.reset_default_graph()
    # sess.close()

    loss = tf.summary.merge(
        (tf.summary.scalar(name= 'loss', tensor= vae.loss)
         , tf.summary.scalar(name= 'loss_recons', tensor= vae.loss_recons)
         , tf.summary.scalar(name= 'loss_relent', tensor= vae.loss_relent)))
    plot = plot_fn("recons")
    fd = {vae.bs_: 100}

    with tf.summary.FileWriter('log/vae') as wtr:
        for step in range(60000):
            vae.step += 1
            if vae.step % 60:
                sess.run(vae.up, feed_dict= fd)
            else:
                summ, _ = sess.run((loss, vae.up), feed_dict= fd)
                wtr.add_summary(summ, vae.step)
            if not (vae.step % 6000):
                plot(sess, wtr, sess.run(vae.g, feed_dict= fd), vae.step / 6000)

    with tf.summary.FileWriter('log/vae/gen') as wtr:
        plot(sess, wtr, sess.run(vae.g, feed_dict= {vae.z: np.random.normal(size= (100, 128))}), vae.step)
