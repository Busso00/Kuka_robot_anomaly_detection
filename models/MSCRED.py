import numpy as np
import tensorflow as tf
from tqdm import tqdm
import math


class MSCRED(tf.keras.Model):
    def __init__(self, opt, matrixes_train, matrixes_test):
        super(MSCRED, self).__init__()
        # Initialize parameters
        self.batch_size = opt['batch_size']
        self.learning_rate = opt['learning_rate']
        self.training_iters = opt['training_iters']
        self.step_max = opt['step_max']

        self.matrixes_train = matrixes_train
        self.matrixes_test = matrixes_test
        self.sensor_n = matrixes_train.shape[2]
        self.win_size = matrixes_train.shape[1]

        self.value_colnames = ['total_count', 'error_count', 'error_rate']
        self.scale_n = len(self.value_colnames)

        # Define CNN encoder layers
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='selu')
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', strides=(2, 2), activation='selu')
        self.conv3 = tf.keras.layers.Conv2D(128, (2, 2), padding='same', strides=(2, 2), activation='selu')
        self.conv4 = tf.keras.layers.Conv2D(256, (2, 2), padding='same', strides=(2, 2), activation='selu')

        self.conv1_lstm = tf.keras.layers.ConvLSTM2D(32, (2, 2), padding='same', return_sequences=True, activation='selu')
        self.conv2_lstm = tf.keras.layers.ConvLSTM2D(64, (2, 2), padding='same', return_sequences=True, activation='selu')
        self.conv3_lstm = tf.keras.layers.ConvLSTM2D(128, (2, 2), padding='same', return_sequences=True, activation='selu')
        self.conv4_lstm = tf.keras.layers.ConvLSTM2D(256, (2, 2), padding='same', return_sequences=True, activation='selu')

        self.deconv4 = tf.keras.layers.Conv2DTranspose(128, (2, 2), padding='same', strides=(2, 2), activation='selu')
        self.deconv3 = tf.keras.layers.Conv2DTranspose(64, (2, 2), padding='same', strides=(2, 2), activation='selu')
        self.deconv2 = tf.keras.layers.Conv2DTranspose(32, (3, 3), padding='same', strides=(2, 2), activation='selu')
        self.deconv1 = tf.keras.layers.Conv2DTranspose(self.scale_n, (3, 3), padding='same', activation='selu')

    def create_sequences(self,data):
        step_max = self.step_max
        num_sequences = data.shape[0] - step_max + 1
        sequences = np.zeros((num_sequences, step_max, self.sensor_n, self.sensor_n, self.scale_n))
        for i in range(num_sequences):
            sequences[i] = data[i:i + step_max]
        return sequences
    def attention_layer(self, inputs):
        batch_size = tf.shape(inputs)[0]
        step_max = tf.shape(inputs)[1]
        height = tf.shape(inputs)[2]
        width = tf.shape(inputs)[3]
        channels = tf.shape(inputs)[4]

        # Estrai l'ultimo passo temporale
        last_output = inputs[:, -1]  # Dimensione: (batch_size, height, width, channels)

        # Funzione per calcolare i punteggi di attenzione
        def compute_attention_score(t):
            step_output = inputs[:, t]  # Dimensione: (batch_size, height, width, channels)
            score = tf.reduce_sum(tf.multiply(step_output, last_output), axis=[1, 2, 3])
            return score

        # Calcola i punteggi di attenzione per ogni passo temporale
        attention_scores = tf.map_fn(compute_attention_score, tf.range(step_max), dtype=tf.float32)
        attention_scores = tf.transpose(attention_scores, [1, 0])  # Dimensione: (batch_size, step_max)

        attention_weights = tf.nn.softmax(attention_scores, axis=1)  # Normalizza i punteggi

        # Applica i pesi di attenzione
        reshaped_inputs = tf.reshape(inputs, [batch_size, step_max, -1])  # Reshape per la moltiplicazione
        context_vector = tf.matmul(attention_weights[:, tf.newaxis, :], reshaped_inputs)  # Dimensione: (batch_size, 1, altezza * larghezza * canali)
        context_vector = tf.reshape(context_vector, [batch_size, height, width, channels])

        return context_vector

    def call(self, inputs, training=False):

        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        step_max = input_shape[1]
        height = input_shape[2]
        width = input_shape[3]
        channels = input_shape[4]

        # Reshape the input to merge batch and sequence dimensions
        reshaped_inputs = tf.reshape(inputs, (batch_size * step_max, height, width, channels))
        # Encoder
        conv1 = self.conv1(reshaped_inputs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)


        # Reshape for ConvLSTM layers
        conv1 = tf.reshape(conv1, [-1, self.step_max, self.sensor_n, self.sensor_n, 32])
        conv2 = tf.reshape(conv2, [-1, self.step_max, int(math.ceil(float(self.sensor_n)/2)), int(math.ceil(float(self.sensor_n)/2)), 64])
        conv3 = tf.reshape(conv3, [-1, self.step_max, int(math.ceil(float(self.sensor_n)/4)), int(math.ceil(float(self.sensor_n)/4)), 128])
        conv4 = tf.reshape(conv4, [-1, self.step_max, int(math.ceil(float(self.sensor_n)/8)), int(math.ceil(float(self.sensor_n)/8)), 256])

        # ConvLSTM layers with attention
        conv1_lstm_out = self.conv1_lstm(conv1)
        conv1_lstm_out_attention= self.attention_layer(conv1_lstm_out)

        conv2_lstm_out = self.conv2_lstm(conv2)
        conv2_lstm_out_attention= self.attention_layer(conv2_lstm_out)

        conv3_lstm_out = self.conv3_lstm(conv3)
        conv3_lstm_out_attention= self.attention_layer(conv3_lstm_out)

        conv4_lstm_out = self.conv4_lstm(conv4)
        conv4_lstm_out_attention= self.attention_layer(conv4_lstm_out)


        # Decoder
        deconv4 = self.deconv4(conv4_lstm_out_attention)
        deconv4 = tf.image.resize(deconv4, [int(math.ceil(float(self.sensor_n)/4)), int(math.ceil(float(self.sensor_n)/4))], method='bilinear')
        deconv4 = tf.concat([deconv4, conv3_lstm_out_attention], axis=-1)

        deconv3 = self.deconv3(deconv4)
        deconv3 = tf.image.resize(deconv3, [int(math.ceil(float(self.sensor_n)/2)), int(math.ceil(float(self.sensor_n)/2))], method='bilinear')
        deconv3 = tf.concat([deconv3, conv2_lstm_out_attention], axis=-1)


        deconv2 = self.deconv2(deconv3)
        deconv2 = tf.image.resize(deconv2, [self.sensor_n, self.sensor_n], method='bilinear')
        deconv2 = tf.concat([deconv2, conv1_lstm_out_attention], axis=-1)

        deconv1 = self.deconv1(deconv2)

        return deconv1

    def compile(self):
        # Define a custom loss function if needed
        def custom_loss(y_true, y_pred):
            y_true_reshaped = y_true[:, -1]
            return tf.reduce_mean(tf.square(y_true_reshaped - y_pred))

        super(MSCRED, self).compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),loss = custom_loss)

    def train(self):
        train_data = np.expand_dims(self.matrixes_train, axis=-1)
        train_data = np.tile(train_data, (1, 1, 1, 3))

        dataset = tf.data.Dataset.from_tensor_slices(train_data)
        dataset = dataset.window(self.step_max, shift=1, drop_remainder=True)
        dataset = dataset.flat_map(lambda window: window.batch(self.step_max))

        dataset = dataset.map(lambda window: (window, window))
        dataset = dataset.batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE)


        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=50,
            verbose=1,
            restore_best_weights=True
        )

        self.fit(dataset, epochs=self.training_iters, callbacks=[early_stopping])

    def test(self):
        test_data = np.expand_dims(self.matrixes_test, axis=-1)
        test_data = np.tile(test_data, (1, 1, 1, 3))

        dataset = tf.data.Dataset.from_tensor_slices(test_data)
        dataset = dataset.window(self.step_max, shift=1, drop_remainder=True)
        dataset = dataset.flat_map(lambda window: window.batch(self.step_max))

        dataset = dataset.batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        reconstructed_matrices = []
        for batch in tqdm(dataset, desc="Testing", unit="batch"):
              reconstructed_matrix = self(batch, training=False)
              reconstructed_matrix_np = reconstructed_matrix.numpy()
              reconstructed_matrices.append(reconstructed_matrix_np)

        return np.concatenate(reconstructed_matrices)