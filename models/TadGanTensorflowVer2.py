import numpy as np
import tensorflow as tf
from scipy.spatial.distance import euclidean
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Bidirectional, LSTM, Flatten, Dense, Reshape, UpSampling1D, TimeDistributed
from tensorflow.keras.layers import Activation, Conv1D, LeakyReLU, Dropout, Input, Layer
from tensorflow.keras.optimizers import Adam
import os
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.neighbors import KernelDensity
import wandb
from scipy.integrate import simps
from fastdtw import fastdtw
from scipy.fft import fft
from tensorflow.keras import backend as K


# Custom layer for random weighted average
class RandomWeightedAverage(Layer):
    def _merge_function(self, inputs):
        alpha = tf.random.uniform((64, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


class TadGAN:
    def __init__(self, window_size=150, latent_dim=30, learning_rate=0.002, batch_size=32, n_critics=5, epochs=350,feature_num=51):
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_critics = n_critics
        self.epochs = epochs
        self.feature_num =feature_num

        self.encoder = self.build_encoder_layer()
        self.generator = self.build_generator_layer()
        self.critic_x = self.build_critic_x_layer()
        self.critic_z = self.build_critic_z_layer()

        self.encoder_optimizer = Adam(self.learning_rate)
        self.generator_optimizer = Adam(self.learning_rate)
        self.critic_x_optimizer = Adam(self.learning_rate)
        self.critic_z_optimizer = Adam(self.learning_rate)


    def get_batch_size(self):
        return self.batch_size

    def get_learning_rate(self):
        return self.learning_rate

    def get_latent_dim(self):
        return self.latent_dim

    def get_epochs(self):
        return self.epochs

    def build_encoder_layer(self):
        input_shape = (self.window_size, self.feature_num)
        encoder_reshape_shape = (self.latent_dim, self.feature_num)

        print(input_shape)

        input_layer = Input(shape=input_shape)
        x = Bidirectional(LSTM(units=100, return_sequences=True))(input_layer)
        x = Flatten()(x)
        x = Dense(self.latent_dim*self.feature_num)(x)
        x = Reshape(target_shape=encoder_reshape_shape)(x)
        model = Model(input_layer, x, name='encoder')
        return model

    def build_generator_layer(self):
        input_shape = (self.latent_dim, self.feature_num)
        #generator_reshape_shape = (self.window_size//2, self.feature_num)
        generator_reshape_shape = (self.window_size//2 , self.feature_num)
        input_layer = Input(shape=input_shape)
        x = Flatten()(input_layer)
        #dense_units = (self.window_size // 2) * self.feature_num
        dense_units = (self.window_size//2) * self.feature_num
        x = Dense(dense_units)(x)
        x = Reshape(target_shape=generator_reshape_shape)(x)
        x = Bidirectional(LSTM(units=64, return_sequences=True), merge_mode='concat')(x)
        x = UpSampling1D(size=2)(x)
        x = Bidirectional(LSTM(units=64, return_sequences=True), merge_mode='concat')(x)
        x = TimeDistributed(Dense(self.feature_num))(x)
        x = Activation(activation='tanh')(x)
        model = Model(input_layer, x, name='generator')
        return model

    def build_critic_x_layer(self):
        input_shape = (self.window_size, self.feature_num)

        input_layer = layers.Input(shape=input_shape)
        x = Conv1D(filters=64, kernel_size=5)(input_layer)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(rate=0.25)(x)
        x = Conv1D(filters=64, kernel_size=5)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(rate=0.25)(x)
        x = Conv1D(filters=64, kernel_size=5)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(rate=0.25)(x)
        x = Conv1D(filters=64, kernel_size=5)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(rate=0.25)(x)
        x = Flatten()(x)
        x = Dense(units=1)(x)
        model = Model(input_layer, x, name='critic_x')
        return model

    def build_critic_z_layer(self):
        input_shape = (self.latent_dim, self.feature_num)

        input_layer = Input(shape=input_shape)
        x = Flatten()(input_layer)
        x = Dense(units=100)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(rate=0.2)(x)
        x = Dense(units=100)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(rate=0.2)(x)
        x = Dense(units=1)(x)
        model = Model(input_layer, x, name='critic_z')
        return model

    @tf.function
    def critic_x_train_on_batch(self, x, z):
        with tf.GradientTape() as tape:
            valid_x = self.critic_x(x)
            x_ = self.generator(z)
            fake_x = self.critic_x(x_)
            alpha = tf.random.uniform([self.batch_size, 1, 1], 0.0, 1.0)
            interpolated = alpha * x + (1 - alpha) * x_
            with tf.GradientTape() as gp_tape:
                gp_tape.watch(interpolated)
                pred = self.critic_x(interpolated)
            grads = gp_tape.gradient(pred, interpolated)
            grad_norm = tf.norm(tf.reshape(grads, (self.batch_size, -1)), axis=1)
            gp_loss = 10.0 * tf.reduce_mean(tf.square(grad_norm - 1.0))
            loss1 = wasserstein_loss(-tf.ones_like(valid_x), valid_x)
            loss2 = wasserstein_loss(tf.ones_like(fake_x), fake_x)
            loss = loss1 + loss2 + gp_loss
        gradients = tape.gradient(loss, self.critic_x.trainable_weights)
        self.critic_x_optimizer.apply_gradients(zip(gradients, self.critic_x.trainable_weights))
        return loss

    @tf.function
    def critic_z_train_on_batch(self, x, z):
        with tf.GradientTape() as tape:
            z_ = self.encoder(x)
            valid_z = self.critic_z(z)
            fake_z = self.critic_z(z_)
            alpha = tf.random.uniform([self.batch_size, 1, 1], 0.0, 1.0)
            interpolated = alpha * z + (1 - alpha) * z_
            with tf.GradientTape() as gp_tape:
                gp_tape.watch(interpolated)
                pred = self.critic_z(interpolated)
            grads = gp_tape.gradient(pred, interpolated)
            grad_norm = tf.norm(tf.reshape(grads, (self.batch_size, -1)), axis=1)
            gp_loss = 10.0 * tf.reduce_mean(tf.square(grad_norm - 1.0))
            loss1 = wasserstein_loss(-tf.ones_like(valid_z), valid_z)
            loss2 = wasserstein_loss(tf.ones_like(fake_z), fake_z)
            loss = loss1 + loss2 + gp_loss
        gradients = tape.gradient(loss, self.critic_z.trainable_weights)
        self.critic_z_optimizer.apply_gradients(zip(gradients, self.critic_z.trainable_weights))
        return loss

    @tf.function
    def enc_gen_train_on_batch(self, x, z):
        with tf.GradientTape() as enc_tape:
            z_gen_ = self.encoder(x, training=True)
            x_gen_ = self.generator(z, training=False)
            x_gen_rec = self.generator(z_gen_, training=False)
            fake_gen_x = self.critic_x(x_gen_, training=False)
            fake_gen_z = self.critic_z(z_gen_, training=False)
            loss1 = wasserstein_loss(fake_gen_x, -tf.ones_like(fake_gen_x))
            loss2 = wasserstein_loss(fake_gen_z, -tf.ones_like(fake_gen_z))
            loss3 = 10.0 * tf.reduce_mean(tf.keras.losses.MeanSquaredError()(x, x_gen_rec))
            enc_loss = loss1 + loss2 + loss3
        gradients_encoder = enc_tape.gradient(enc_loss, self.encoder.trainable_weights)
        self.encoder_optimizer.apply_gradients(zip(gradients_encoder, self.encoder.trainable_weights))

        with tf.GradientTape() as gen_tape:
            z_gen_ = self.encoder(x, training=False)
            x_gen_ = self.generator(z, training=True)
            x_gen_rec = self.generator(z_gen_, training=True)
            fake_gen_x = self.critic_x(x_gen_, training=False)
            fake_gen_z = self.critic_z(z_gen_, training=False)
            loss1 = wasserstein_loss(fake_gen_x, -tf.ones_like(fake_gen_x))
            loss2 = wasserstein_loss(fake_gen_z, -tf.ones_like(fake_gen_z))
            loss3 = 10.0 * tf.reduce_mean(tf.keras.losses.MeanSquaredError()(x, x_gen_rec))
            gen_loss = loss1 + loss2 + loss3
        gradients_generator = gen_tape.gradient(gen_loss, self.generator.trainable_weights)
        self.generator_optimizer.apply_gradients(zip(gradients_generator, self.generator.trainable_weights))
        return enc_loss, gen_loss

    def predict(self, X_test):
        X_test = np.array(X_test)
        print("Initial X_test shape:", X_test.shape)  # Debug statement

        total_elements = X_test.size
        print("Initial X_test size:", total_elements)  # Debug statement



        reconstructions = []
        critic_scores = []
        area_reconstruction_errors=[]

        for i in range(0, X_test.shape[0], self.batch_size):
            x_batch = X_test[i:i + self.batch_size]
            z_gen = self.encoder.predict(x_batch)
            x_reconstructed = self.generator.predict(z_gen)
            reconstructions.append(x_reconstructed)

            critic_score_batch = self.critic_x.predict(x_batch, batch_size=self.batch_size)
            critic_scores.append(critic_score_batch)

        # Combine the reconstructed batches
        reconstructions = np.vstack(reconstructions)
        print("Reconstructed shape:", reconstructions.shape)  # Debug statement

        critic_scores = np.vstack(critic_scores).flatten()
        print("Critic scores shape:", critic_scores.shape)

        kde = KernelDensity(kernel='gaussian', bandwidth=1.0).fit(critic_scores.reshape(-1, 1))
        log_density_scores = kde.score_samples(critic_scores.reshape(-1, 1))
        smoothed_critic_scores = np.exp(log_density_scores)

        #point-wise


        point_wise_reconstruction_errors = np.mean(np.square(X_test - reconstructions), axis=(1, 2))

        for i in range(X_test.shape[0]):
            area_diff = simps(np.abs(X_test[i] - reconstructions[i]), dx=1)
            area_reconstruction_errors.append(area_diff)
        area_reconstruction_errors = np.array(area_reconstruction_errors)


        dtw_reconstruction_errors = []
        for i in range(X_test.shape[0]):
            dtw_distance, _ = fastdtw(X_test[i], reconstructions[i], dist=euclidean)
            dtw_reconstruction_errors.append(dtw_distance)
        dtw_reconstruction_errors = np.array(dtw_reconstruction_errors)



        return smoothed_critic_scores,point_wise_reconstruction_errors,area_reconstruction_errors,dtw_reconstruction_errors

    def train(self, X, wandb):
        checkpoint_dir = './checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_callback = ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'epoch-{epoch:03d}-loss-{loss:.4f}.h5'),
            save_weights_only=False,
            monitor='loss',
            mode='min',
            save_best_only=True,
            save_freq='epoch'
        )

        X_ = np.copy(X)

        epoch_e_loss = []
        epoch_g_loss = []
        epoch_cx_loss = []
        epoch_cz_loss = []

        for epoch in range(1, self.epochs + 1):
            np.random.shuffle(X_)
            minibatches_size = self.batch_size * self.n_critics
            num_minibatches = int(X_.shape[0] // minibatches_size)

            self.encoder.trainable = False
            self.generator.trainable = False

            for i in range(num_minibatches):
                minibatch = X_[i * minibatches_size: (i + 1) * minibatches_size]
                for j in range(self.n_critics):
                    x = minibatch[j * self.batch_size: (j + 1) * self.batch_size]
                    x = x.astype(np.float32)
                    z = tf.random.normal(shape=(self.batch_size, self.latent_dim, self.feature_num), mean=0.0, stddev=1.0)
                    self.critic_x.trainable = True
                    self.critic_z.trainable = False
                    #discriminator Cx
                    epoch_cx_loss.append(self.critic_x_train_on_batch(x, z))
                    self.critic_x.trainable = False
                    self.critic_z.trainable = True
                    #discriminator Cz
                    epoch_cz_loss.append(self.critic_z_train_on_batch(x, z))

                self.critic_z.trainable = False
                self.critic_x.trainable = False
                self.encoder.trainable = True
                self.generator.trainable = True

                enc_loss, gen_loss = self.enc_gen_train_on_batch(x, z)
                epoch_e_loss.append(enc_loss)
                epoch_g_loss.append(gen_loss)

            cx_loss = np.mean(np.array(epoch_cx_loss), axis=0)
            cz_loss = np.mean(np.array(epoch_cz_loss), axis=0)
            e_loss = np.mean(np.array(epoch_e_loss), axis=0)
            g_loss = np.mean(np.array(epoch_g_loss), axis=0)

            print(
                f'Epoch: {epoch}/{self.epochs}, [Dx loss: {cx_loss}] [Dz loss: {cz_loss}] [E loss: {e_loss}] [G loss: {g_loss}]')

            #wandb.log({
                #'critic_x_loss': cx_loss,
                #'critic_z_loss': cz_loss,
                #'encoder_loss': e_loss,
                #'generator_loss': g_loss
            #})



    def save_models(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        self.encoder.save(os.path.join(save_dir, 'encoder.h5'))
        self.generator.save(os.path.join(save_dir, 'generator.h5'))
        self.critic_x.save(os.path.join(save_dir, 'critic_x.h5'))
        self.critic_z.save(os.path.join(save_dir, 'critic_z.h5'))

    def load_models(self, save_dir):
        self.encoder = tf.keras.models.load_model(os.path.join(save_dir, 'encoder.h5'), compile=False)
        self.generator = tf.keras.models.load_model(os.path.join(save_dir, 'generator.h5'), compile=False)
        self.critic_x = tf.keras.models.load_model(os.path.join(save_dir, 'critic_x.h5'), compile=False)
        self.critic_z = tf.keras.models.load_model(os.path.join(save_dir, 'critic_z.h5'), compile=False)



