import tensorflow as tf
from tensorflow import math
import math


def create_masks(input_size, hidden_size, n_hidden, input_degrees=None):
  """
  Creates masks for MADE connections based on input degrees.

  Args:
    input_size: Size of the input layer.
    hidden_size: Size of the hidden layers.
    n_hidden: Number of hidden layers.
    input_order: Order of connections ('sequential' or 'random').
    input_degrees: Optional pre-defined degrees for connections.

  Returns:
    masks: List of masks for MADE connections.
    input_degree: Degree of the input layer.
  """

  # Initialize degrees
  degrees = []

  if input_size > 1: #never happens
    
      degrees += [tf.range(input_size)] if input_degrees is None else [input_degrees]
      for _ in range(n_hidden + 1):
          degrees += [tf.range(hidden_size) % (input_size - 1)]
      degrees += [tf.range(input_size) % input_size - 1] if input_degrees is None else [input_degrees % input_size - 1]

  else:
    
      degrees += [tf.zeros([1], dtype=tf.int32)]
      for _ in range(n_hidden + 1):
          degrees += [tf.zeros([hidden_size], dtype=tf.int32)]
      degrees += [tf.zeros([input_size], dtype=tf.int32)]

  # Construct masks
  masks = []
  for (d0, d1) in zip(degrees[:-1], degrees[1:]):
    masks += [tf.cast((tf.expand_dims(d1, -1) >= tf.expand_dims(d0, 0)), dtype=tf.float32)]

  return masks, degrees[0]




class MaskedLinear(tf.keras.layers.Layer):
    """ MADE building block layer """
    #adjust initializations
    def __init__(self, input_size, n_outputs, mask, cond_label_size=None, **kwargs):
        super(MaskedLinear, self).__init__(**kwargs)
        
        trainable = kwargs.get('trainable', True)  # Default to True if not provided

        self.input_size = input_size
        self.n_outputs = n_outputs
        self.mask = tf.Variable(mask, trainable=False, name='mask')

        self.cond_label_size = cond_label_size
        

    def build(self, input_shape):

        stdv = 1. / math.sqrt(self.input_size)

        self.kernel = self.add_weight(
            shape=(self.n_outputs, self.input_size),
            initializer=tf.keras.initializers.RandomUniform(-stdv, stdv),
            name='kernel' #trainable default is true
        )

        self.bias = self.add_weight(
            shape=(self.n_outputs,),
            initializer=tf.keras.initializers.RandomUniform(-stdv, stdv),
            name='bias'  #trainable default is true
        )
        
        if self.cond_label_size is not None:

            stdv =  1. / math.sqrt(self.cond_label_size)

            self.cond_weight = self.add_weight(
                shape=(self.cond_label_size, self.n_outputs),
                initializer=tf.keras.initializers.RandomUniform(0., stdv),
                name='cond_weight'  #trainable default is true
            )

        super().build(input_shape)


    def call(self, x, y=None):
        
        out = tf.matmul(x, tf.transpose(self.kernel * self.mask)) + self.bias
        if y is not None:
            out = out + tf.matmul(y, tf.transpose(self.cond_weight))

        return out



import tensorflow_probability as tfp
class MADE(tf.keras.layers.Layer):
  
    def __init__(self, input_size, hidden_size, n_hidden, cond_label_size=None, activation='relu', input_degrees=None):
        super(MADE, self).__init__()
        

        if activation == 'relu':
            self.activation_fn = tf.keras.layers.ReLU()
        elif activation == 'tanh':
            self.activation_fn = tf.keras.layers.Activation('tanh')
        else:
            raise ValueError('Check activation function.')

        
        masks, self.input_degrees = create_masks(input_size, hidden_size, n_hidden, input_degrees)
              
        
        self.base_dist_mean =  tf.Variable(tf.zeros(shape=input_size), trainable=False, name='base_dist_mean')
        self.base_dist_var = tf.Variable(tf.ones(shape=input_size), trainable=False, name='base_dist_var')
        
        self.net_input = MaskedLinear(input_size, hidden_size, masks[0], cond_label_size)
        
        modules = []

        for m in masks[1:-1]:
            modules += [self.activation_fn, MaskedLinear(hidden_size, hidden_size, m)]
        
        modules += [self.activation_fn, MaskedLinear(hidden_size, 2 * input_size, tf.tile(masks[-1], [2,1]))]
        #2 variables (mu, sigma) for each input entity, this 2 are used only to calculate new u
        #so u is conditioned to input y
        self.net = modules


    @property
    def base_dist(self):
        return tfp.distributions.Normal(loc=self.base_dist_mean, scale=self.base_dist_var)

    def call(self, x, y=None):

        out = self.net_input(x, y)
        for layer in self.net:
            out = layer(out)

        m, loga = tf.split(out, 2, axis=1)
        #goal of MADE: fing mu and sigma (m, loga)
        u = (x - m) * tf.exp(-loga) #reparametrization trick
        log_abs_det_jacobian = -loga #loga = log of covariance matrix
        return u, log_abs_det_jacobian


class FlowSequential(tf.keras.Model):
    
    """ Container for layers of a normalizing flow """

    def __init__(self, layers) -> None:
        super().__init__()
        self.modules = layers
    
    def call(self, x, y):
        sum_log_abs_det_jacobians = 0.0
        for layer in self.modules:
            x, log_abs_det_jacobian = layer(x, y)
            sum_log_abs_det_jacobians += log_abs_det_jacobian
        return x, sum_log_abs_det_jacobians


class MAF(tf.keras.Model):
    """
    MADE-based Anomaly Detector.

    Args:
        n_blocks: Number of MADE blocks to stack.
        n_sensor: Number of sensor channels in the data.
        input_size: Dimensionality of the input data point.
        hidden_size: Dimensionality of the hidden layers in MADE.
        n_hidden: Number of hidden layers in each MADE block.
        cond_label_size (optional): Dimensionality of conditional labels (if applicable).
        activation: Activation function for MADE hidden layers ('relu' or 'tanh').
        input_order: Ordering strategy for connections in MADE ('sequential' or 'random').
        batch_norm (bool): Whether to use batch normalization after each MADE block.
        mode: Initialization mode for the base distribution ('zero' or 'rand').
    """

    def __init__(self, n_blocks, n_sensor, input_size, hidden_size, n_hidden, cond_label_size=None, activation='relu', batch_norm=True, mode='rand'):
        super(MAF, self).__init__()

        # Base distribution parameters
        if mode == 'zero':
            self.base_dist_mean = tf.Variable(tf.zeros(shape=(n_sensor, 1)), trainable=False, name='base_dist_mean')
            self.base_dist_var = tf.Variable(tf.ones(shape=(n_sensor, 1)), trainable=False, name='base_dist_var')
        elif mode == 'rand':
            self.base_dist_mean = tf.Variable(tf.random.normal(shape=(n_sensor, 1)), trainable=False, name='base_dist_mean')
            self.base_dist_var = tf.Variable(tf.ones(shape=(n_sensor, 1)), trainable=False, name='base_dist_var')
        else:
            raise ValueError('Invalid mode for base distribution initialization.')

        # Model construction
        self.input_size = input_size #
        self.input_degrees = None
        
        modules = []
        self.input_size = input_size
        self.input_degrees = None
        for i in range(n_blocks):
            modules += [MADE(input_size, hidden_size, n_hidden, cond_label_size, activation, self.input_degrees)]
            self.input_degrees = tf.reverse(modules[-1].input_degrees, axis=[0])

        self.net = FlowSequential(modules)

    
    def base_dist(self, z, k, window_size):

        N = tf.shape(z)[0] // k // window_size #batch size
    
        # Repeat base_dist_mean
        mean_repeated = tf.tile(tf.repeat(self.base_dist_mean, window_size, axis=0), [N, 1]) 
        # Calculate log probability
        logp = -0.5 * tf.square(z - mean_repeated)

        return logp


    def call(self, x, y=None):
        return self.net(x, y)
        
    def log_prob(self, x, k, window_size, y=None):
        """Calculates log-probability of the data under the model."""
        u, sum_log_abs_det_jacobians = self.call(x, y)
        C = u.shape[1]
        #u (z) is the Normalized score, output of affine transformations: in normal samples is distributed like a N(0,1)
        
        gconst = -0.9189385332046727 # -ln(sqrt(2*pi))
        return tf.reduce_sum(self.base_dist(u, k, window_size) + sum_log_abs_det_jacobians, axis=1) + C * gconst  # Assuming _GCONST_ is a constant







