"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

utils.py

(1) train_test_divide: Divide train and test data for both original and synthetic data.
(2) extract_time: Returns Maximum sequence length and each sequence length.
(3) rnn_cell: Basic RNN Cell.
(4) random_generator: random vector generator
(5) batch_generator: mini-batch generator
"""

## Necessary Packages
import os
import numpy as np
# TF 2.x port: legacy RNN cells need Keras 2 (tf-keras); must be set before TensorFlow is imported.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
import tensorflow.compat.v1 as tf


def train_test_divide (data_x, data_x_hat, data_t, data_t_hat, train_rate = 0.8):
  """Divide train and test data for both original and synthetic data.
  
  Args:
    - data_x: original data
    - data_x_hat: generated data
    - data_t: original time
    - data_t_hat: generated time
    - train_rate: ratio of training data from the original data
  """
  # Divide train/test index (original data)
  no = len(data_x)
  idx = np.random.permutation(no)
  train_idx = idx[:int(no*train_rate)]
  test_idx = idx[int(no*train_rate):]
    
  train_x = [data_x[i] for i in train_idx]
  test_x = [data_x[i] for i in test_idx]
  train_t = [data_t[i] for i in train_idx]
  test_t = [data_t[i] for i in test_idx]      
    
  # Divide train/test index (synthetic data)
  no = len(data_x_hat)
  idx = np.random.permutation(no)
  train_idx = idx[:int(no*train_rate)]
  test_idx = idx[int(no*train_rate):]
  
  train_x_hat = [data_x_hat[i] for i in train_idx]
  test_x_hat = [data_x_hat[i] for i in test_idx]
  train_t_hat = [data_t_hat[i] for i in train_idx]
  test_t_hat = [data_t_hat[i] for i in test_idx]
  
  return train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat


def extract_time (data):
  """Returns Maximum sequence length and each sequence length.
  
  Args:
    - data: original data
    
  Returns:
    - time: extracted time information
    - max_seq_len: maximum sequence length
  """
  time = []
  max_seq_len = 0
  for i in range(len(data)):
    max_seq_len = max(max_seq_len, len(data[i][:,0]))
    time.append(len(data[i][:,0]))
    
  return time, max_seq_len


def rnn_cell(module_name, hidden_dim):
  """Basic RNN Cell.
    
  Args:
    - module_name: gru, lstm, or lstmLN
    
  Returns:
    - rnn_cell: RNN Cell
  """
  assert module_name in ['gru','lstm','lstmLN']
  
  # GRU
  if (module_name == 'gru'):
    rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_dim, activation=tf.nn.tanh)
  # LSTM
  elif (module_name == 'lstm'):
    rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_dim, activation=tf.nn.tanh)
  # LSTM Layer Normalization
  elif (module_name == 'lstmLN'):
    rnn_cell = LayerNormBasicLSTMCell(num_units=hidden_dim, activation=tf.nn.tanh)
  return rnn_cell


class LayerNormBasicLSTMCell(tf.nn.rnn_cell.RNNCell):
  """Port of TF 1.15 tf.contrib.rnn.LayerNormBasicLSTMCell (removed in TF 2.x), defaults as used here.

  LSTM with layer normalization on the i/j/f/o gate pre-activations and on the new cell state
  (Ba et al., 2016). No bias in the projection (layer-norm beta plays that role), forget_bias 1.0,
  gamma init 1.0, beta init 0.0, epsilon 1e-12 as in contrib layer_norm. Recurrent dropout is not ported
  (TimeGAN uses keep_prob 1.0).
  """

  def __init__(self, num_units, forget_bias=1.0, activation=tf.nn.tanh, reuse=None):
    super(LayerNormBasicLSTMCell, self).__init__(_reuse=reuse)
    self._num_units = num_units
    self._forget_bias = forget_bias
    self._activation = activation

  @property
  def state_size(self):
    return tf.nn.rnn_cell.LSTMStateTuple(self._num_units, self._num_units)

  @property
  def output_size(self):
    return self._num_units

  def build(self, inputs_shape):
    proj_size = int(inputs_shape[-1]) + self._num_units
    self._kernel = self.add_weight("kernel", shape=[proj_size, 4 * self._num_units])
    self._norm_params = {}
    for scope in ["input", "transform", "forget", "output", "state"]:
      self._norm_params[scope] = (
          self.add_weight(scope + "/gamma", shape=[self._num_units], initializer=tf.constant_initializer(1.0)),
          self.add_weight(scope + "/beta", shape=[self._num_units], initializer=tf.constant_initializer(0.0)))
    self.built = True

  def _norm(self, inp, scope):
    gamma, beta = self._norm_params[scope]
    mean, variance = tf.nn.moments(inp, [1], keep_dims=True)
    return tf.nn.batch_normalization(inp, mean, variance, beta, gamma, 1e-12)

  def call(self, inputs, state):
    c, h = state
    concat = tf.matmul(tf.concat([inputs, h], 1), self._kernel)
    i, j, f, o = tf.split(value=concat, num_or_size_splits=4, axis=1)
    i = self._norm(i, "input")
    j = self._norm(j, "transform")
    f = self._norm(f, "forget")
    o = self._norm(o, "output")
    g = self._activation(j)
    new_c = c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) * g
    new_c = self._norm(new_c, "state")
    new_h = self._activation(new_c) * tf.sigmoid(o)
    return new_h, tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h)


def random_generator (batch_size, z_dim, T_mb, max_seq_len):
  """Random vector generation.
  
  Args:
    - batch_size: size of the random vector
    - z_dim: dimension of random vector
    - T_mb: time information for the random vector
    - max_seq_len: maximum sequence length
    
  Returns:
    - Z_mb: generated random vector
  """
  Z_mb = []
  for i in range(batch_size):
    temp = np.zeros([max_seq_len, z_dim])
    temp_Z = np.random.uniform(0., 1, [T_mb[i], z_dim])
    temp[:T_mb[i],:] = temp_Z
    Z_mb.append(temp_Z)
  return Z_mb


def batch_generator(data, time, batch_size):
  """Mini-batch generator.
  
  Args:
    - data: time-series data
    - time: time information
    - batch_size: the number of samples in each batch
    
  Returns:
    - X_mb: time-series data in each batch
    - T_mb: time information in each batch
  """
  no = len(data)
  idx = np.random.permutation(no)
  train_idx = idx[:batch_size]     
            
  X_mb = [data[i] for i in train_idx]
  T_mb = [time[i] for i in train_idx]
  
  return X_mb, T_mb