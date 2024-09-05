import tensorflow as tf
from tensorflow import math
from models.NF import MAF

class GNN(tf.keras.Model):
    """
    The GNN module applied in GANF
    """
    def __init__(self, hidden_size):
        super(GNN, self).__init__()
        self.lin_n = tf.keras.layers.Dense(hidden_size)
        self.lin_r = tf.keras.layers.Dense(hidden_size, use_bias=False)
        self.lin_2 = tf.keras.layers.Dense(hidden_size)

    def call(self, h, A):
        """
        Forward pass of the GNN layer.

        Args:
          h: Input tensor of shape (batch_size, num_nodes, feature_dim, layer_dim->num_entities in graph)
          A: Adjacency matrix of shape (num_nodes, num_nodes)

        Returns:
          Updated hidden representation of shape (batch_size, num_nodes, feature_dim, layer_dim)
        """
        #h.shape = N, K, L, H
        #A.shape = N, K x K -> attention normalized along first K (num_sensors)

        # Einsum operation for efficient matrix multiplication: n (batch_dim) is unchanged 

        # n is done in parallel as it appears in both A and h and in result
        # k is summed over
        # other (l,d,j) are preserved
        # in substance
        # h_n is result of attention applied to hidden_feature of LSTM
        # so for each batch separately (n)
        # it is calculating attention over sensors (k)
        # for each time in window (l)
        # such that the 2 remaining dimension j (as k sensor) and l

        # result of tf.einsum
        # N x K x L x H

        # "graph conv" x W1
        h_n = self.lin_n(tf.einsum('nkld,nkj->njld', h, A)) 
        
        # "temporal info" x W2
        h_r = self.lin_r(h[:, :, :-1])  

        #addition of hidden state t-1
        h_n = tf.concat([h_n[:, :, :1], h_n[:, :, 1:] + h_r], axis=2)

        return self.lin_2(tf.nn.relu(h_n))

        

class ScaleDotProductAttention(tf.keras.Model):
    """
    Compute scaled dot-product attention.

    Args:
      d_model: Dimensionality of the model.

    """
    def __init__(self, d_model):
        super(ScaleDotProductAttention, self).__init__()
        self.w_q = tf.keras.layers.Dense(d_model) #attention over features
        self.w_k = tf.keras.layers.Dense(d_model)
        self.softmax = tf.keras.layers.Softmax(axis=1)
        self.dropout = tf.keras.layers.Dropout(0.2)

    def call(self, x, training=True):
        """
        Forward pass of the attention layer.

        Args:
          x: Input tensor of shape (batch_size, head, length, d_model)
          
        Returns:
          attention_weights: Attention weights after applying softmax (shape: (batch_size, head, length, length))
          values: Weighted sum of values (shape: (batch_size, head, length, d_model))
        """

        shape = x.shape
        x_shape = tf.reshape(x, (shape[0], shape[1], -1)) #batch, num_sensors, ws
        batch_size, length, c = x_shape.get_shape().as_list() #c is length x d_model

        # Project queries, keys, and values
        q = self.w_q(x_shape)
        k = self.w_k(x_shape)
        # Transpose for efficient dot product, ignoring batch 
        #k_transpose = tf.transpose(k, perm = [0, 2, 1])
        k_transpose = tf.reshape(k, (batch_size, c, length))
        
        # Scaled dot product attention
        scores = tf.matmul(q, k_transpose) / math.sqrt(tf.cast(c, tf.float32))
        # Apply softmax for attention weights
        scores = self.dropout(self.softmax(scores), training=training)

        #attention scores are an num_sensorsxnum_sensors matrix
        return scores, k



class MTGFLOW(tf.keras.Model):

    def __init__(self, n_blocks, input_size, hidden_size, n_hidden, window_size, n_sensor, dropout=0.1, batch_norm=True):
        super(MTGFLOW, self).__init__()

        self.rnn = tf.keras.layers.LSTM(hidden_size, return_sequences=True, dropout=dropout, input_shape=(None, input_size))
        
        self.gcn = GNN(hidden_size=hidden_size)
        
        self.nf = MAF(n_blocks, n_sensor, input_size, hidden_size, n_hidden, cond_label_size=hidden_size, batch_norm=batch_norm, activation='tanh')
            
        self.attention = ScaleDotProductAttention(window_size*input_size)
        

    def call(self, x, training=True):
        return tf.reduce_mean(self.test(x, training=training)) #loss is 0 with <0 llikehood

    def test(self, x, training=True):
        # x: N X K X L X D = N_BATCH x K_SENSORS x L_WINDOW x MODEL_DIM (1)
        full_shape = x.shape
        # Get graph using self attention on input features
        graph, _ = self.attention(x, training=training)
        self.graph = graph
        # Reshape for RNN: batch and sensors are treated independently such that temporal information extracted are relative to only one sensor
        x = tf.reshape(x, (x.shape[0]*x.shape[1], x.shape[2], x.shape[3]))  # N*K, L, D
        
        h = self.rnn(x, training = training) 
        
        # Reshape for GCN: re-extract info for each sensor
        h = tf.reshape(h, (full_shape[0], full_shape[1], h.shape[1], h.shape[2]))  # N, K, L, H = N_BATCH x K_SENSORS x L_WINDOW x N_HIDDEN (num of entitites hyp)
        
        h = self.gcn(h, self.graph)
        # Reshape for MAF
        h = tf.reshape(h, (-1, h.shape[3]))  # N*K*L, H
        x = tf.reshape(x, (-1, full_shape[3]))  # N*K*L, D
        
        # Calculate log prob with MAF
        # h are the so called spatio-temporal conditions
        # our objective is to calculate log(P(x))
        # but to do so we need to pass through a NF:
        # P(x|C) where C = h = spatio temporal condition extracted through LSTM and combined with GNN
        log_prob = self.nf.log_prob(x, full_shape[1], full_shape[2], h)
        log_prob = tf.reshape(log_prob, (full_shape[0], -1))

        return tf.reduce_mean(log_prob, axis=1)  # Average log prob to obtain window log_prob





    