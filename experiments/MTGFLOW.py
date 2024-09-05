from load_data import *
#from experiments.MSCRED import MSCREDExperiment
#from experiments.TADGAN import TADGANExperiment
#from experiments.VAE_LSTM import VAELSTMExperiment
#from experiments.MTGFLOW import MTGFLOWExperiment
import tensorflow as tf
#import wandb
from sklearn.metrics import *
import numpy as np
from metrics import *

import wandb


class MTGFLOWExperiment:

    
    def __init__(self, opt, n):

        self.opt = opt
        self.n = n
        self.version = self.opt['tf']
        if self.version:
            from models.MTGFLOW import MTGFLOW

            self.model = MTGFLOW(
              self.opt['n_blocks'], 
              1, #input_size=1
              self.opt['hidden_size'], 
              1, #n_hidden=1 
              self.opt['window_size'], 
              self.opt['num_features'],
              self.opt['dropout'], 
              batch_norm = False
            )
            
            #since there are reshapes in intermediate modules --> need a predefined input shape
            #tf.compat.v1.enable_eager_execution()
            self.model.build((self.opt['batch_size'], self.opt['num_features'], self.opt['window_size'], 1))

            # Get the trainable variables
            trainable_vars = self.model.trainable_variables

            # Convert to dictionary with variable names as keys
            trainable_vars_dict = {var.name: var for var in trainable_vars}
                        
            # Print the dictionary
            for name, var in trainable_vars_dict.items():
                print(f"Variable name: {name}, Variable shape: {var.shape}")
            
            
        

    @tf.function
    def run_opt(self, x):
        
        with tf.GradientTape() as g:
            
            loss = -self.model(x)
            trainable_variables = self.model.trainable_variables
            gradients = g.gradient(loss, trainable_variables)
            # Clip gradients by value, for example, between -1 and 1
            clipped_grads = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients]
            # Apply gradients to update the model
            self.optimizer.apply_gradients(zip(clipped_grads, trainable_variables))

        return loss
        
    #NOTE: saving is possible only in pytorch

    def save_model(self,epoch=0, overwrite=True):
        
        if self.version:
            return 
        
    def load_model(self,epoch=0):
        
        if self.version:
            return 



    def train(self, X_train, time):
        
        loss_best = 1e5
        X_ = np.copy(X_train)
        num_minibatches = int(X_.shape[0] // self.opt['batch_size'])
      
        if self.version:
            
            import tensorflow as tf
           
            num_minibatches = int(X_.shape[0] // self.opt['batch_size'])
            
            epochs_per_step = 10
            gamma = 0.9

            boundaries = []
            values = []

            for i in range(self.opt['epochs']):

                if i%epochs_per_step==epochs_per_step-1:
                    boundaries.append(i*num_minibatches)
                if i%epochs_per_step==0:
                    values.append(self.opt['lr']*(gamma**(i/epochs_per_step)))

            values.append(self.opt['lr']*(gamma**float(self.opt['epochs']/10)))

            print(values)
            self.lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                    boundaries=boundaries,  # Adjust based on your epoch size
                    values=values)

            # Create an optimizer with the learning rate scheduler
            #step LR of Pytorch
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule, epsilon=1e-8)
                        
            for epoch in range(self.opt['epochs']): #training here
                
                print(f"epoch: {epoch}/{self.opt['epochs']}")
                avg_loss = 0.0
                
                np.random.shuffle(X_)
                
                for i in range(num_minibatches):
                    
                    batch = X_[i * self.opt['batch_size']: (i + 1) * self.opt['batch_size']] 
                    batch = tf.transpose(batch,(0,2,1))
                    batch = tf.expand_dims(batch, axis=3)
                    l=self.run_opt(batch)
                    avg_loss += l/num_minibatches

                wandb.log({
                    'train_loss': avg_loss
                })

                if isinstance(self.optimizer.learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule):
                    current_lr = self.optimizer.learning_rate(self.optimizer.iterations).numpy()
                else:
                    current_lr = self.optimizer.learning_rate.numpy()
                
                print(f"Epoch {epoch+1} train loss: {avg_loss:.4f}, lr: {current_lr}") 
            
                if avg_loss < loss_best: #best approximator decided based on val set
                    self.save_model()
                    loss_best = avg_loss
                    print("saved")

            print("Training complete!")
            print(f"Kept loss: {loss_best}")
                
        
        
        
    def detect_anomalies(self, X_test, collision_times):
        # MOdel is based on estimating log(p(X_test_window))
        # after learning p[X_window] on not anomalous data (X_window_train)
        # then comparing with t, if log(p(X_test_window)) < t ---> is anomaly
        # to uniform the procedure with other metods I compare with t a scaled 
        # version in (0,1) of -log_prob
        
        #pass through model
       
        if self.version:

            X_test = tf.convert_to_tensor(X_test)
            X_test = tf.cast(X_test, tf.float32)  # Add this line before the transpose operation
            X_test = tf.transpose(X_test,(0,2,1))
            X_test = tf.expand_dims(X_test, axis=3)
            log_probs = []
              
            for i in range(X_test.shape[0]//self.opt['batch_size']):
                
                batch = X_test[i*self.opt['batch_size']:(i+1)*self.opt['batch_size']]
                batch = tf.cast(batch, tf.float64)
                log_prob = - self.model.test(batch, training=False)
                
                log_probs.append(np.array(log_prob))

        
        scores = np.concatenate(log_probs)

        print("scores examples")
        print(scores[::self.opt['window_size']])

        scores = compute_average_score(scores,self.opt)
        recallsp, precisionsp, fprsp, _, f1sp, _, _ = metrics_by_point_vectorized(scores, collision_times, full=True)
        
        # Create tables with accumulated data
        tableROC_p = wandb.Table(columns=["fpr", "recall"])
        tablePR_p = wandb.Table(columns=["recall","precision"])
        
        for i,(fpr, recall, precision) in enumerate(zip(fprsp, recallsp, precisionsp)):
            tableROC_p.add_data(fpr, recall) #ROC
            tablePR_p.add_data(recall, precision) #PRC
        
        
        graphROC_p = wandb.plot.line(
            tableROC_p,  # Pass the table to the line plot function
            x="fpr",
            y="recall",
            title="ROC",
        )
        graphPRC_p = wandb.plot.line(
            tablePR_p,
            x="recall",
            y="precision",
            title="PRC"
        )

        wandb.log({
            'best F1 P': max(f1sp),
            'AUROC P': calculate_auc(fprsp, recallsp),
            'AUPRC P': calculate_auc(recallsp, precisionsp),
            'ROC P': graphROC_p,
            'PRC P': graphPRC_p,
        })        
        
        
        
        

