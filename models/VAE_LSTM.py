import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import tensorflow as tf
from tensorflow import keras, data
import tensorflow_probability as tfp
from tensorflow.keras import layers, regularizers, activations
from tensorflow.keras import backend as K

class Sampling(layers.Layer):
    def __init__(self, z_dim, name='sampling_z'):
        super(Sampling, self).__init__(name=name)
        self.z_dim = z_dim

    def call(self, inputs):
        mu, logvar = inputs
        print('mu: ', mu)
        sigma = K.exp(logvar * 0.5)
        epsilon = K.random_normal(shape=(mu.shape[0], self.z_dim), mean=0.0, stddev=1.0)
        return mu + epsilon * sigma
    
    def get_config(self):
        config = super(Sampling, self).get_config()
        config.update({'name': self.name})
        return config

class Encoder(layers.Layer):
    def __init__(self, time_step, x_dim, lstm_h_dim, z_dim, name='encoder', **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)

        self.encoder_inputs = keras.Input(shape=(time_step, x_dim))
        self.encoder_lstm = layers.LSTM(lstm_h_dim, activation='softplus', name='encoder_lstm', stateful=True)
        self.z_mean = layers.Dense(z_dim, name='z_mean')
        self.z_logvar = layers.Dense(z_dim, name='z_log_var')
        self.z_sample = Sampling(z_dim)
    
    def call(self, inputs):
        self.encoder_inputs = inputs
        hidden = self.encoder_lstm(self.encoder_inputs)
        mu_z = self.z_mean(hidden)
        logvar_z = self.z_logvar(hidden)
        z = self.z_sample((mu_z, logvar_z))
        return mu_z, logvar_z, z
    
    def get_config(self):
        config = super(Encoder, self).get_config()
        config.update({
            'name': self.name,
            'z_sample': self.z_sample.get_config()
        })
        return config

class Decoder(layers.Layer):
    def __init__(self, time_step, x_dim, lstm_h_dim, z_dim, name='decoder', **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)

        self.z_inputs = layers.RepeatVector(time_step, name='repeat_vector')
        self.decoder_lstm_hidden = layers.LSTM(lstm_h_dim, activation='softplus', return_sequences=True, name='decoder_lstm')
        self.x_mean = layers.Dense(x_dim, name='x_mean')
        self.x_sigma = layers.Dense(x_dim, name='x_sigma', activation='tanh')
    
    def call(self, inputs):
        z = self.z_inputs(inputs)
        hidden = self.decoder_lstm_hidden(z)
        mu_x = self.x_mean(hidden)
        sigma_x = self.x_sigma(hidden)
        return mu_x, sigma_x
    
    def get_config(self):
        config = super(Decoder, self).get_config()
        config.update({
            'name': self.name
        })
        return config

loss_metric = keras.metrics.Mean(name='loss')
likelihood_metric = keras.metrics.Mean(name='log likelihood')

class LSTM_VAE(keras.Model):
    def __init__(self, time_step, x_dim, lstm_h_dim, z_dim, name='lstm_vae', **kwargs):
        super(LSTM_VAE, self).__init__(name=name, **kwargs)

        self.encoder = Encoder(time_step, x_dim, lstm_h_dim, z_dim, **kwargs)
        self.decoder = Decoder(time_step, x_dim, lstm_h_dim, z_dim, **kwargs)
    
    def call(self, inputs):
        mu_z, logvar_z, z = self.encoder(inputs)
        mu_x, sigma_x = self.decoder(z)

        var_z = K.exp(logvar_z)
        kl_loss = K.mean(-0.5 * K.sum(var_z - logvar_z + tf.square(1 - mu_z), axis=1), axis=0)
        self.add_loss(kl_loss)

        dist = tfp.distributions.Normal(loc=mu_x, scale=tf.abs(sigma_x))
        log_px = -dist.log_prob(inputs)

        return mu_x, sigma_x, log_px
    
    def get_config(self):
        config = {
            'encoder': self.encoder.get_config(),
            'decoder': self.decoder.get_config(),
            'name': self.name
        }
        return config
    
    def reconstruct_loss(self, x, mu_x, sigma_x):
        var_x = K.square(sigma_x)
        reconst_loss = -0.5 * K.sum(K.log(var_x), axis=2) + K.sum(K.square(x - mu_x) / var_x, axis=2)
        reconst_loss = K.reshape(reconst_loss, shape=(x.shape[0], 1))
        return K.mean(reconst_loss, axis=0)

    def mean_log_likelihood(self, log_px):
        log_px = K.reshape(log_px, shape=(log_px.shape[0], log_px.shape[2]))
        mean_log_px = K.mean(log_px, axis=1)
        return K.mean(mean_log_px, axis=0)

    def train_step(self, data):
        if isinstance(data, tuple):
            x = data[0]
        else:
            x = data
        
        with tf.GradientTape() as tape:
            mu_x, sigma_x, log_px = self(x, training=True)
            loss = self.reconstruct_loss(x, mu_x, sigma_x)
            loss += sum(self.losses)
            #trainable_variables = self.model.trainable_variables
            #gradients.
            mean_log_px = self.mean_log_likelihood(log_px)
            
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        loss_metric.update_state(loss)
        likelihood_metric.update_state(mean_log_px)
        return {'loss': loss_metric.result(), 'log_likelihood': likelihood_metric.result()}
    
    def save_model(self, model_dir):
        with open(model_dir + 'lstm_vae.json', 'w') as f:
            f.write(self.to_json())
        self.save_weights(model_dir + 'lstm_vae_ckpt' + ".weights.h5")

    def load_model(model_dir):
        lstm_vae_obj = {'Encoder': Encoder, 'Decoder': Decoder, 'Sampling': Sampling}
        with keras.utils.custom_object_scope(lstm_vae_obj):
            with open(model_dir + 'lstm_vae.json', 'r'):
                model = keras.models.model_from_json(model_dir + 'lstm_vae.json')
            model.load_weights(model_dir + 'lstem_vae_ckpt')
        return model