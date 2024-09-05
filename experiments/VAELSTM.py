from sklearn.preprocessing import MinMaxScaler
from models.VAE_LSTM import LSTM_VAE
import numpy as np
from load_data import get_windowed_data
import pandas as pd
import matplotlib.pyplot as plt
import random
from plot_data import plot_windowed_data
import wandb
from datetime import datetime
from scipy.stats import zscore
from sklearn.metrics import roc_curve, auc
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from tensorflow import keras, data
from metrics import *
import seaborn as sns

class VAELSTMExperiment:
    def __init__(self, opt):
        self.scaler = MinMaxScaler()
        self.opt = opt
        self.time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.batch_size = self.opt["batch_size"]
        self.epoch_num = self.opt["epoch_num"]
        self.lr = self.opt['lr']
        self.epsilon = self.opt['epsilon']
        self.gradient_clipvalue = self.opt['gradient_clipvalue']
        self.window_size = self.opt['window_size']
        self.mode = "test" if self.opt["test"] else "train"
        self.model_dir = "MLinApps-Proj-FP01/saved_models/"
        self.model = LSTM_VAE(self.opt['time_step'], self.opt['x_dim'], self.opt['lstm_h_dim'], self.opt['z_dim'], dtype='float32')

        # wandb initialization
        wandb.init(
        entity="marcodalmo",
        project="mlinapp-FP01-2024-08",
        tags=["VaeLstm", self.opt['experiment']],
        name=f"{opt['experiment']}_{self.time}"
        )

        # initialize wandb config, add hyperparameters
        config = wandb.config
        config.backbone = opt['experiment']
        config.experiment = opt['experiment']
        config.max_iterations = opt['max_iterations']
        config.epochs = opt["epoch_num"]
        config.batch_size = opt["batch_size"]
        config.learning_rate = opt['lr']

    def fit(self, train_data):
        opt = keras.optimizers.Adam(learning_rate=self.lr, epsilon=self.epsilon, amsgrad=False, clipvalue=self.gradient_clipvalue)
        self.model.compile(optimizer=opt, loss=LSTM_VAE.reconstruct_loss)
        train_data = data.Dataset.from_tensor_slices(train_data)
        train_data = train_data.shuffle(buffer_size=1024).batch(self.batch_size, drop_remainder=True)
        history = self.model.fit(train_data, epochs=self.epoch_num, shuffle=True).history
        self.model.summary()
        self.plot_loss_moment(history)
        self.model.save_model(self.model_dir)

    def plot_loss_moment(self, history, save=True, image_dir='MLinApps-Proj-FP01/images/'):
        _, ax = plt.subplots(figsize=(14, 6), dpi=80)
        ax.plot(history['loss'], 'blue', label='Loss', linewidth=1)
        ax.plot(history['log_likelihood'], 'red', label='Log likelihood', linewidth=1)
        ax.set_title('Loss and log likelihood over epochs')
        ax.set_ylabel('Loss and log likelihood')
        ax.set_xlabel('Epoch')
        ax.legend(loc='upper right')
        if save:
            plt.savefig(image_dir + 'loss_lstm_vae_' + self.mode + '.png')

    def plot_log_likelihood(self, df_log_px, image_dir='MLinApps-Proj-FP01/images/'):
        plt.figure(figsize=(14, 6), dpi=80)
        plt.title("Log likelihood")
        sns.set_color_codes()
        sns.distplot(df_log_px, bins=40, kde=True, rug=True, color='blue')
        plt.savefig(image_dir + 'log_likelihood_' + self.mode + '.png')

    def detect_anomalies(self, X_test, collision_time):
        zeros_to_add = np.zeros((5, X_test.shape[1], X_test.shape[2]))
        X_test = np.concatenate((X_test, zeros_to_add), axis=0)
        X_test = data.Dataset.from_tensor_slices(X_test)
        X_test = X_test.batch(self.batch_size)
        _, _, anomaly_scores = self.model.predict(X_test)
        anomaly_scores.reshape(anomaly_scores.shape[0], anomaly_scores.shape[2])
        anomaly_scores = np.mean(anomaly_scores, axis=1)
        anomaly_scores = np.mean(anomaly_scores, axis=1)
        anomaly_scores = anomaly_scores[:-5]

        recallsp, precisionsp, fprsp, _, f1sp, _, _ = metrics_by_point_vectorized(anomaly_scores, collision_time, full=True)
        
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


    def f1_analysis(self, anomaly_score, collision_times_validation_init, collision_times_validation_end,
                    collision_time, note):
        best_precision_f1 = -1
        best_precision = -1
        best_recall = -1
        max_threshold = anomaly_score.max()
        min_threshold = anomaly_score.min()
        thresholds = np.linspace(min_threshold, max_threshold, num=300)
        position = 0
        count_tp = 0
        count_fp = 0
        count_fn = 0
        count_tn = 0
        tpr_list = []
        fpr_list = []
        best_f1 = -1
        f1_list = []
        tp_score = []
        fp_score = []
        fn_score = []
        tn_score = []
        best_tp_score = []
        best_fp_score = []
        best_fn_score = []
        best_tn_score = []
        accuracy = -1
        best_accuracy = -1



        for th in thresholds:
            for x in range(len(anomaly_score)- self.opt["window_size"]):
                act_th = anomaly_score[x]
                if act_th >= th:
                    tp = self.check_tp(position, collision_time, collision_times_validation_init,
                                       collision_times_validation_end)
                    if tp:
                        count_tp += 1
                        tp_score.append(act_th)

                    else:
                        count_fp += 1
                        fp_score.append(act_th)
                else:
                    tp = self.check_tp(position, collision_time, collision_times_validation_init,
                                       collision_times_validation_end)
                    if tp:
                        count_fn += 1
                        fn_score.append(act_th)
                    else:
                        count_tn += 1
                        tn_score.append(act_th)
                position = position + 1

            tpr = count_tp / (count_tp + count_fn) if (count_tp + count_fn) != 0 else 0
            fpr = count_fp / (count_fp + count_tn) if (count_fp + count_tn) != 0 else 0
            tpr_list.append(tpr)
            fpr_list.append(fpr)
            precision = count_tp / (count_tp + count_fp) if (count_tp + count_fp) != 0 else 0
            recall = tpr
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
            f1_list.append(f1_score)
            accuracy = (count_tp + count_tn) / (count_tp + count_tn + count_fn + count_fp)

            if precision > best_precision:
                best_precision = precision
                best_precision_threshold = th
                count_tp_prc = count_tp
                count_fp_prc = count_fp
            count_fn = 0
            count_tp = 0
            count_fp = 0
            count_tn = 0
            position = 0
            max_f1 = max(f1_list)
            if max_f1 > best_f1:
                best_f1 = max_f1
                best_threshold = th
                best_tpr_list = tpr_list
                best_fpr_list = fpr_list
                best_precision_f1 = precision
                best_recall = recall
                best_tn_score = tn_score
                best_fn_score = fn_score
                best_tp_score = tp_score
                best_fp_score = fp_score

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold_acc = th
                best_acc_f1 = f1_score
                best_tn_score_acc = tn_score
                best_fn_score_acc = fn_score
                best_tp_score_acc = tp_score
                best_fp_score_acc = fp_score

            tp_score = []
            fp_score = []
            fn_score = []
            tn_score = []
            best_tp_score = np.asarray(best_tp_score)
            best_fp_score = np.asarray(best_fp_score)
            best_fn_score = np.asarray(best_fn_score)
            best_tn_score = np.asarray(best_tn_score)
            best_tp_score_acc = np.asarray(best_tp_score_acc)
            best_fp_score_acc = np.asarray(best_fp_score_acc)
            best_fn_score_acc = np.asarray(best_fn_score_acc)
            best_tn_score_acc = np.asarray(best_tn_score_acc)

        print(note + " best f1 score : " + str(best_f1))
        print(note + " best precision f1 : " + str(best_precision_f1))
        print(note + " best recall : " + str(best_recall))
        print(note + "tp_score mean: " + str(best_tp_score.mean()) + " count: " + str(best_tp_score.shape))
        print(note + "fp_score mean: " + str(best_fp_score.mean()) + " count: " + str(best_fp_score.shape))
        print(note + "tn_scores mean: " + str(best_tn_score.mean()) + " count: " + str(best_tn_score.shape))
        print(note + "fn_score mean: " + str(best_fn_score.mean()) + " count: " + str(best_tn_score.shape))
        print(note + "best threshold: " + str(best_threshold))

        print(note + "best acc:" + str(best_accuracy))
        print(note + "best acc th: " + str(best_threshold_acc))
        print(note + "best acc f1: " + str(best_acc_f1))
        print(note + "tp_score_acc mean: " + str(best_tp_score_acc.mean()) + " count: " + str(best_tp_score_acc.shape))
        print(note + "fp_score_acc mean: " + str(best_fp_score_acc.mean()) + " count: " + str(best_fp_score_acc.shape))
        print(note + "tn_scores_acc mean: " + str(best_tn_score_acc.mean()) + " count: " + str(best_tn_score_acc.shape))
        print(note + "fn_score_acc mean: " + str(best_fn_score_acc.mean()) + " count: " + str(best_tn_score_acc.shape))

        print(note + " best precision  : " + str(best_precision))
        print(note + " best precision threshold : " + str(best_precision_threshold))
        print(note + " best precision count tp : " + str(count_tp_prc))
        print(note + " best precision count fp : " + str(count_fp_prc))

        def plot_gaussian(scores, label, color):
            if len(scores) > 0:
                mu, std = norm.fit(scores)
                xmin, xmax = scores.min(), scores.max()
                x = np.linspace(xmin, xmax, 100)
                p = norm.pdf(x, mu, std)
                plt.plot(x, p, label=f'{label} (mean: {mu:.2f}, std: {std:.2f})', color=color)
                plt.fill_between(x, p, alpha=0.2, color=color)

        plt.figure(figsize=(10, 6))

        plot_gaussian(best_tp_score, 'True Positives', 'blue')
        plot_gaussian(best_fp_score, 'False Positives', 'red')
        plot_gaussian(best_fn_score, 'False Negatives', 'green')
        plot_gaussian(best_tn_score, 'True Negatives', 'purple')

        plt.title(f'{note} Best Scores Gaussian Distribution')
        plt.xlabel('Score')
        plt.ylabel('Density')
        plt.legend()
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 2)
        plt.plot(fpr_list, tpr_list, color='blue', lw=2, label=f'ROC curve )')
        plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

    def multiple_of_3_over_40(self, number):
        if number < 40:
            return 0
        else:
            return number // 40

    def check_tp(self, position, collision_time, collision_time_validation_init, collision_time_validation_end):
        collision_time_start = collision_time[position][0]
        collision_time_end = collision_time[position][self.window_size - 1]
        collision_time_start = collision_time_start[0]
        collision_time_end = collision_time_end[0]
        for time in collision_time[position]:
            time_str = time[0]

            for i in range(len(collision_time_validation_init)):
                if collision_time_validation_init[i] <= time_str <= collision_time_validation_end[i]:
                    return True
                else:
                    continue

        return False

    def anomaly_pruning(self, anomaly_scores, threshold=0.1):
        """
        Prune anomalies based on the decrease percentage method.

        Parameters:
        - anomaly_scores: List or numpy array of anomaly scores (one per sequence).
        - threshold: Decrease percentage threshold for pruning.

        Returns:
        - pruned_scores: Numpy array with pruned anomaly scores.
        """
        # Ensure anomaly scores are a numpy array
        anomaly_scores = np.array(anomaly_scores)

        # Sort the scores in descending order
        sorted_indices = np.argsort(anomaly_scores)[::-1]
        sorted_scores = anomaly_scores[sorted_indices]

        # Compute decrease percentages
        decrease_percentages = []
        for i in range(1, len(sorted_scores)):
            decrease_percent = (sorted_scores[i - 1] - sorted_scores[i]) / sorted_scores[i - 1]
            decrease_percentages.append(decrease_percent)

        # Find the first index where the decrease percentage exceeds the threshold
        reclassify_index = len(decrease_percentages)  # Default to reclassify none
        for i, decrease_percent in enumerate(decrease_percentages):
            if decrease_percent <= threshold:
                reclassify_index = i + 1
                break

        # Reclassify subsequent sequences as normal
        pruned_scores = np.zeros_like(anomaly_scores)
        for i in range(reclassify_index):
            pruned_scores[sorted_indices[i]] = sorted_scores[i]

        return pruned_scores

    def plot_random_frame_anomalies(self, anomalies_index):
        random_index = random.randint(0, len(anomalies_index) - 1)
        random_frame_index = anomalies_index[random_index]
        random_frame_index = str(random_frame_index)
        part1 = int(random_frame_index[:-2])
        part2 = int(random_frame_index[-2:])
        if part2 > self.opt['window_size'] / 2:
            random_frame_index = part1 + 1
            left = True
        else:
            random_frame_index = part1
            left = False
            part2 = self.opt['window_size'] - part2
        plot_windowed_data(root_path='', period='0.1', window_size=self.opt['window_size'],
                           stride=self.opt['window_size'], train=False,
                           frame_num=random_frame_index, left=left, anomaly_frame_num=part2)
