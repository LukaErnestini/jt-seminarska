import tensorflow as tf

class MLP(tf.keras.Model):
  def __init__(self, rnn_units):
    super().__init__(self)
    self.dense = tf.keras.layers.Dense(rnn_units)
    self.dropout = tf.keras.layers.Dropout(0.5)
    self.dense2 = tf.keras.layers.Dense(rnn_units)

  def call(self, inputs, states=None, return_state=False, training=False):
    x = inputs
    x = self.dense(x, training=training)
    x = self.dropout(x, training=training)
    x = self.dense2(x, training=training)
    return x