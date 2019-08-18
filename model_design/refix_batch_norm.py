import tensorflow as tf

from tensorflow.python.training.moving_averages import assign_moving_average


def my_batch_norm(inputs,
               is_training=True,
               epsilon=1e-8,
               decay=0.9):
    with tf.variable_scope("batch-normalization"):
        pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False, name="pop-mean")  # [depth]
        pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False, name="pop-var")

        def mean_and_var_update():
            axes = list(range(len(inputs.get_shape()) - 1))
            batch_mean, batch_var = tf.nn.moments(inputs, axes, name="moments")  # [depth]

            with tf.control_dependencies([assign_moving_average(pop_mean, batch_mean, decay),
                                          assign_moving_average(pop_var, batch_var, decay)]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        is_training = tf.cast(is_training, dtype=tf.bool)
        mean, variance = tf.cond(is_training, mean_and_var_update, lambda: (pop_mean, pop_var))

        beta = tf.Variable(initial_value=tf.zeros(inputs.get_shape()[-1]), name="shift")
        gamma = tf.Variable(initial_value=tf.ones(inputs.get_shape()[-1]), name="scale")
        return tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, epsilon)


class BatchNormalization(tf.keras.layers.BatchNormalization):
    """
    Make trainable=False freeze BN for real (the og version is sad)
    """

    def call(self, x, training=False):
        if training is None:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)




