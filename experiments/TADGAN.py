from sklearn.preprocessing import MinMaxScaler
from models.TadGanTensorflowVer2 import TadGAN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb
from datetime import datetime
from scipy.stats import zscore
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import os
import tensorflow as tf
import time

ROOTDIR_DATASET_ANOMALY = "collisions"


def pruning_for_convexpw100(score):
    for i in range(len(score)):
        if 1.70 <= score[i] <= -1.71:
            score[i] = -0.57
        if score[i] > 0.52:
            score[i] = -2.17

    return score


class TADGANExperiment:
    def __init__(self, opt):
        self.scaler = MinMaxScaler()
        self.opt = opt
        self.time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model = TadGAN(window_size=self.opt['window_size'], feature_num=self.opt['feature_num'])
        self.window_size = self.opt['window_size']

        # wandb login
        # wandb.login(key="0a7ade235c4e8c59cafa897ca7e9e8c105f820b0")

        # wandb initialization
        # wandb.init(
        # entity="marcodalmo",
        # project="mlinapp-FP01-2024-08",
        # tags=["TadGan", self.opt['experiment']],
        # name=f"{opt['experiment']}_{self.time}"
        # )

        # initialize wandb config, add hyperparameters
        # config = wandb.config
        # config.backbone = "TadGan"
        # config.experiment = opt['experiment']
        # config.max_iterations = opt['max_iterations']
        # config.epochs = self.model.get_epochs()
        # config.batch_size = self.model.get_batch_size()
        # config.learning_rate = self.model.get_learning_rate()
        # config.learning_rate = opt['lr']

    def fit(self, X_train):
        if self.opt['train']:
            self.model.train(X_train, wandb)
            self.model.save_models("saved_models")
        else:
            self.model.load_models('saved_models/')

    def measure_latency(self, num_runs=1000):
        x = tf.random.normal(shape=(32, 100, 51))
        z = tf.random.normal(shape=(32, 30, self.opt['feature_num']), mean=0.0, stddev=1.0)

        start_time = time.time()
        for _ in range(num_runs):
            _ = self.model.critic_x_train_on_batch(x, z)
        end_time = time.time()

        avg_latency = (end_time - start_time) / num_runs
        return avg_latency


    def detect_anomalies(self, X_test, threshold=None, collision_times_validation_init=None,
                         collision_times_validation_end=None, collision_time=None):

        # Encode test data
        critic_score, pw_rs, area_rs, dtw_rs = self.model.predict(X_test)

        X_test = np.array(X_test)
        alpha = 0.3

        z_pw_rs = zscore(pw_rs)
        z_dtw_rs = zscore(dtw_rs)
        z_area_rs = zscore(area_rs)
        z_area_rs_mean = np.mean(z_area_rs, axis=1)

        z_critic_scores = zscore(critic_score)

        convex_anomaly_scores_pw = alpha * z_pw_rs + (1 - alpha) * z_critic_scores
        convex_anomaly_scores_dtw = alpha * z_dtw_rs + (1 - alpha) * z_critic_scores
        convex_anomaly_scores_area=alpha * z_area_rs_mean + (1 - alpha)*z_critic_scores



        product_anomaly_scores_pw = z_pw_rs * z_critic_scores
        product_anomaly_scores_dtw = z_dtw_rs * z_critic_scores


        window_size = 100

        convex_anomaly_scores_pw_pruned = pruning_for_convexpw100(convex_anomaly_scores_pw)
        convex_anomaly_scores_dtw_pruned= pruning_for_convexpw100(convex_anomaly_scores_dtw)

        adjusted_anomalies_pw=self.adjusted_anomalies(pruning_for_convexpw100(convex_anomaly_scores_pw_pruned))
        adjusted_anomalies_dtw=self.adjusted_anomalies(convex_anomaly_scores_dtw)
        adjusted_anomalies_area=self.adjusted_anomalies(convex_anomaly_scores_area)
        product_anomaly_scores_pw_adjusted=self.adjusted_anomalies(product_anomaly_scores_pw)
        adjusted_anomaly_scores_critc=self.adjusted_anomalies(z_critic_scores)



        converted_time = self.convert(collision_time)
        self.metrics_by_point_vectorized(adjusted_anomaly_scores_critc, converted_time, full=False, method='CRITIC')
        self.metrics_by_point_vectorized(adjusted_anomalies_pw,converted_time,full=False,method='MSE')
        #self.metrics_by_point_print_best(adjusted_anomalies_pw,converted_time,full=False,method='MSE')
        self.metrics_by_point_vectorized(adjusted_anomalies_area, converted_time, full=False, method='AREA')
        self.metrics_by_point_vectorized(adjusted_anomalies_dtw,converted_time,full=False,method='DTW')
        self.metrics_by_point_vectorized(product_anomaly_scores_pw_adjusted, converted_time, full=False,
                                         method='MSE product')

        self.metrics_by_window_vectorized_1(z_critic_scores, converted_time)
        #self.metrics_by_window_vectorized_1(convex_anomaly_scores_pw_pruned, converted_time)
        self.metrics_by_window_vectorized_1(convex_anomaly_scores_pw_pruned, converted_time)
        self.metrics_by_window_vectorized_1(convex_anomaly_scores_dtw,converted_time)
        self.metrics_by_window_vectorized_1(convex_anomaly_scores_area, converted_time)
        self.metrics_by_window_vectorized_1(product_anomaly_scores_pw, converted_time)



        return convex_anomaly_scores_pw, product_anomaly_scores_pw  # , convex_anomaly_scores_dtw

    def convert(self, times_windowed):
        data = []
        for time_window in times_windowed:
            start = pd.Timestamp(str(time_window[0][0]))
            end = pd.Timestamp(str(time_window[-1][0]))
            data.append({'start': start, 'end': end})
        df = pd.DataFrame(data)
        return df

    def adjusted_anomalies(self,scores):
        window_size=100
        anomaly_scores = scores.flatten()
        adjusted_anomalies = np.zeros(len(anomaly_scores))

        for i in range(len(anomaly_scores)-window_size):
            adjusted_anomalies[i:i + 100] += anomaly_scores[i]

        # Media dei punteggi di anomalia per i punti che appartengono a più finestre
        counts = np.zeros(len(anomaly_scores))
        for i in range(len(anomaly_scores)):
            counts[i:i + 100] += 1

        adjusted_anomalies = adjusted_anomalies / counts
        return adjusted_anomalies

    def metrics_by_window_vectorized_1(self, score, time_collision):
        # works both with set of th (linspace) and scores

        # thresholds = np.sort(score)
        thresholds = np.linspace(score.min(), score.max(), num=1000)

        collisions = pd.read_excel(os.path.join(ROOTDIR_DATASET_ANOMALY, "20220811_collisions_timestamp.xlsx"))
        collisions_init = collisions[collisions['Inizio/fine'] == "i"].Timestamp - pd.to_timedelta(
            [2] * len(collisions[collisions['Inizio/fine'] == "i"].Timestamp), 'h')
        collision_end = collisions[collisions['Inizio/fine'] == "f"].Timestamp - pd.to_timedelta(
            [2] * len(collisions[collisions['Inizio/fine'] == "f"].Timestamp), 'h')

        time_collision = time_collision[:len(score)]
        assert len(score) == len(time_collision), "unmatching score/thresholds/timestamp"
        print(f"--- LOADED {len(collisions_init)} COLLISIONS ---")

        # Convert timestamps to numpy arrays
        start_times = time_collision['start'].to_numpy().astype('datetime64[ns]')
        end_times = time_collision['end'].to_numpy().astype('datetime64[ns]')

        collisions_init_np = collisions_init.to_numpy().astype('datetime64[ns]')
        collisions_end_np = collision_end.to_numpy().astype('datetime64[ns]')

        # Create a mask for each threshold
        threshold_masks = score[:, np.newaxis] >= thresholds

        n_samples = len(score)
        n_coll = len(collisions_init)

        # Calculate metrics for each threshold
        results = []
        for threshold_mask in threshold_masks.T:
            pos_pred = np.sum(threshold_mask)
            neg_pred = n_samples - pos_pred

            # Check if there are collisions in window
            # 1) window include start of anomaly
            # 2) window include end of anomaly
            # 3) window is included in anomaly
            collision_in_window = (((collisions_init_np[:, np.newaxis] <= start_times[threshold_mask]) & \
                                    (start_times[threshold_mask] <= collisions_end_np[:, np.newaxis])) | \
                                   ((collisions_init_np[:, np.newaxis] <= end_times[threshold_mask]) & \
                                    (end_times[threshold_mask] <= collisions_end_np[:, np.newaxis]))) | \
                                  ((start_times[threshold_mask] <= collisions_init_np[:, np.newaxis]) & \
                                   (end_times[threshold_mask] >= collisions_end_np[:, np.newaxis]))
            '''collision_in_window = ((start_times[threshold_mask] >= collisions_init_np[:, np.newaxis]) & (
                    start_times[threshold_mask] < collisions_end_np[:, np.newaxis]))'''

            tp = np.sum(collision_in_window)  # overall sum is necessary

            fp = pos_pred - tp

            not_threshold_mask = np.where(threshold_mask, False, True)
            # gt says there are collisions in windows but window is predicted as not anomaly: same as before but with mask of not_anomaly prediction
            false_not_collision_in_window = (((collisions_init_np[:, np.newaxis] <= start_times[not_threshold_mask]) & \
                                              (start_times[not_threshold_mask] <= collisions_end_np[:, np.newaxis])) | \
                                             ((collisions_init_np[:, np.newaxis] <= end_times[not_threshold_mask]) & \
                                              (end_times[not_threshold_mask] <= collisions_end_np[:, np.newaxis]))) | \
                                            ((start_times[not_threshold_mask] <= collisions_init_np[:, np.newaxis]) & \
                                             (end_times[not_threshold_mask] >= collisions_end_np[:, np.newaxis]))
            '''false_not_collision_in_window = ((start_times[not_threshold_mask] >= collisions_init_np[:, np.newaxis]) &
                                             (start_times[not_threshold_mask] < collisions_end_np[:, np.newaxis]))'''

            fn = np.sum(false_not_collision_in_window)

            tn = neg_pred - fn

            anomaly_indices = np.where(threshold_mask)[0][np.any(collision_in_window, axis=0)]

            cm_anomaly = np.array([[tn, fp], [fn, tp]])

            precision = tp / (tp + fp) if tp + fp != 0 else 0
            recall = tp / (tp + fn) if tp + fn != 0 else 0
            fpr = fp / (fp + tn) if fp + tn != 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn != 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0

            results.append((recall, precision, fpr, accuracy, f1, cm_anomaly, anomaly_indices))

        recalls, precisions, fprs, accuracies, f1s, cms, anomaly_indices_list = zip(*results)

        all_anomaly_indices = np.concatenate(anomaly_indices_list)

        # Crea y_true in base alla presenza negli indici delle anomalie
        y_true = np.array([1 if i in all_anomaly_indices else 0 for i in range(n_samples)])

        fpr, tpr, _ = roc_curve(y_true, score)
        roc_auc = auc(fpr, tpr)

        precision, recall, _ = precision_recall_curve(y_true, score)
        avg_precision = average_precision_score(y_true, score)

        print("MAX f1 : " + str(max(f1s)))

        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

        plt.figure()
        plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.show()

        return recalls, precisions, fprs, accuracies, f1s, cms, anomaly_indices_list

    def metrics_by_point_vectorized(self,scores, time_collision, full=False,method=""):
        # score is attributed to last point in window

        if full:
            thresholds = np.sort(scores)
        else:
            thresholds = np.linspace(scores.min(), scores.max(), num=300)


        collisions = pd.read_excel(os.path.join(ROOTDIR_DATASET_ANOMALY, "20220811_collisions_timestamp.xlsx"))
        collisions_init = collisions[collisions['Inizio/fine'] == "i"].Timestamp - pd.to_timedelta(
            [2] * len(collisions[collisions['Inizio/fine'] == "i"].Timestamp), 'h')
        collision_end = collisions[collisions['Inizio/fine'] == "f"].Timestamp - pd.to_timedelta(
            [2] * len(collisions[collisions['Inizio/fine'] == "f"].Timestamp), 'h')

        time_collision = time_collision[:len(scores)]
        print(f"--- LOADED {len(collisions_init)} COLLISIONS ---")

        # Convert timestamps to numpy arrays
        start_times = time_collision['start'].to_numpy().astype('datetime64[ns]')
        end_times = time_collision['end'].to_numpy().astype('datetime64[ns]')

        collisions_init_np = collisions_init.to_numpy().astype('datetime64[ns]')
        collisions_end_np = collision_end.to_numpy().astype('datetime64[ns]')

        # Create a mask for each threshold
        threshold_masks = scores[:, np.newaxis] >= thresholds

        n_samples = len(scores)

        # Calculate metrics for each threshold
        results = []
        for threshold_mask in threshold_masks.T:
            pos_pred = np.sum(threshold_mask)
            neg_pred = n_samples - pos_pred

            # Check if there are collisions in window
            # 1) window include start of anomaly
            # 2) window include end of anomaly
            # 3) window is included in anomaly
            collision_in_window = ((start_times[threshold_mask] >= collisions_init_np[:, np.newaxis]) & (
                        start_times[threshold_mask] < collisions_end_np[:, np.newaxis]))

            tp = np.sum(collision_in_window)  # overall sum is necessary

            fp = pos_pred - tp

            not_threshold_mask = np.where(threshold_mask, False, True)
            # gt says there are collisions in windows but window is predicted as not anomaly: same as before but with
            # mask of not_anomaly prediction
            false_not_collision_in_window = ((start_times[not_threshold_mask] >= collisions_init_np[:, np.newaxis]) &
                                             (start_times[not_threshold_mask] < collisions_end_np[:, np.newaxis]))

            fn = np.sum(false_not_collision_in_window)

            tn = neg_pred - fn

            anomaly_indices = np.where(threshold_mask)[0][np.any(collision_in_window, axis=0)]

            cm_anomaly = np.array([[tn, fp], [fn, tp]])

            precision = tp / (tp + fp) if tp + fp != 0 else 0
            recall = tp / (tp + fn) if tp + fn != 0 else 0
            fpr = fp / (fp + tn) if fp + tn != 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn != 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0

            results.append((recall, precision, fpr, accuracy, f1, cm_anomaly, anomaly_indices))

        recalls, precisions, fprs, accuracies, f1s, cms, anomaly_indices_list = zip(*results)
        all_anomaly_indices = np.concatenate(anomaly_indices_list)

        # Crea y_true in base alla presenza negli indici delle anomalie
        y_true = np.array([1 if i in all_anomaly_indices else 0 for i in range(n_samples)])

        fpr, tpr, roc_thresholds = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)

        precision, recall, _ = precision_recall_curve(y_true, scores)
        avg_precision = average_precision_score(y_true, scores)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})'+method)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

        plt.figure()
        plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (AP = {avg_precision:.2f})'+method)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.show()

        print('AUC '+ method+ ' : %.4f' % roc_auc )

        print("MAX f1 "+method +" : " + str(max(f1s)))

        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = roc_thresholds[optimal_idx]
        print(f"Best threshold to maximize TPR and minimize FPR: {optimal_threshold:.4f}")

        return recalls, precisions, fprs, accuracies, f1s, cms, anomaly_indices_list


    def metrics_by_point_print_best(self,scores, time_collision, full=False,method=""):
        # score is attributed to last point in window

        thresholds = -0.7730
        thresholds=np.asarray(thresholds)
        collisions = pd.read_excel(os.path.join(ROOTDIR_DATASET_ANOMALY, "20220811_collisions_timestamp.xlsx"))
        collisions_init = collisions[collisions['Inizio/fine'] == "i"].Timestamp - pd.to_timedelta(
            [2] * len(collisions[collisions['Inizio/fine'] == "i"].Timestamp), 'h')
        collision_end = collisions[collisions['Inizio/fine'] == "f"].Timestamp - pd.to_timedelta(
            [2] * len(collisions[collisions['Inizio/fine'] == "f"].Timestamp), 'h')

        time_collision = time_collision[:len(scores)]
        print(f"--- LOADED {len(collisions_init)} COLLISIONS ---")

        # Convert timestamps to numpy arrays
        start_times = time_collision['start'].to_numpy().astype('datetime64[ns]')
        end_times = time_collision['end'].to_numpy().astype('datetime64[ns]')

        collisions_init_np = collisions_init.to_numpy().astype('datetime64[ns]')
        collisions_end_np = collision_end.to_numpy().astype('datetime64[ns]')

        # Create a mask for each threshold
        threshold_masks = scores[:, np.newaxis] >= thresholds

        n_samples = len(scores)

        # Calculate metrics for each threshold
        results = []
        for threshold_mask in threshold_masks.T:
            pos_pred = np.sum(threshold_mask)
            neg_pred = n_samples - pos_pred

            # Check if there are collisions in window
            # 1) window include start of anomaly
            # 2) window include end of anomaly
            # 3) window is included in anomaly
            collision_in_window = ((start_times[threshold_mask] >= collisions_init_np[:, np.newaxis]) & (
                        start_times[threshold_mask] < collisions_end_np[:, np.newaxis]))

            tp = np.sum(collision_in_window)  # overall sum is necessary

            fp = pos_pred - tp

            not_threshold_mask = np.where(threshold_mask, False, True)
            # gt says there are collisions in windows but window is predicted as not anomaly: same as before but with
            # mask of not_anomaly prediction
            false_not_collision_in_window = ((start_times[not_threshold_mask] >= collisions_init_np[:, np.newaxis]) &
                                             (start_times[not_threshold_mask] < collisions_end_np[:, np.newaxis]))

            fn = np.sum(false_not_collision_in_window)

            tn = neg_pred - fn

            anomaly_indices = np.where(threshold_mask)[0][np.any(collision_in_window, axis=0)]

            cm_anomaly = np.array([[tn, fp], [fn, tp]])

            precision = tp / (tp + fp) if tp + fp != 0 else 0
            recall = tp / (tp + fn) if tp + fn != 0 else 0
            fpr = fp / (fp + tn) if fp + tn != 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn != 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0

            results.append((recall, precision, fpr, accuracy, f1, cm_anomaly, anomaly_indices))

        recalls, precisions, fprs, accuracies, f1s, cms, anomaly_indices_list = zip(*results)
        all_anomaly_indices = np.concatenate(anomaly_indices_list)



        return recalls, precisions, fprs, accuracies, f1s, cms, anomaly_indices_list