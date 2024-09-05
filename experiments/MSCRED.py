'''
Project: MLinApps-Proj
Author:MLinApps-Proj group 2024/08
Members:
Haochen He S307771
Federico Bussolino S317641
Marco D'almo  S301199
Youness Bouchari S323624
'''
import numpy as np
import pandas as pd
import os

ROOTDIR_DATASET_ANOMALY = "/content/drive/MyDrive/Kuka_v1/collisions"
filepath_csv_anomaly = [ROOTDIR_DATASET_ANOMALY + f"rec{r}_collision_20220811_rbtc_0.1s.csv" for r in [1, 5]]
filepath_meta_anomaly = [ROOTDIR_DATASET_ANOMALY + f"rec{r}_collision_20220811_rbtc_0.1s.metadata" for r in [1, 5]]


def signature_matrix_generator(windows):
    sensor_n = windows.shape[2]
    win_size = windows.shape[1]

    # Generazione delle signature matrices
    matrix_all = []
    print("Generating signature matrices...")

    for window in windows:
        matrix_t = np.zeros((sensor_n, sensor_n))
        for i in range(sensor_n):
            for j in range(i, sensor_n):
                matrix_t[i][j] = np.inner(window[:, i], window[:, j]) / win_size  # rescale by win_size
                matrix_t[j][i] = matrix_t[i][j]
        matrix_all.append(matrix_t)

    matrix_all = np.array(matrix_all)
    return matrix_all

class MSCREDExperiment:
    def __init__(self, opt, windows_train, windows_test):
        matrices_train = signature_matrix_generator(windows_train)
        matrices_test = signature_matrix_generator(windows_test)
        from models.MSCRED import MSCRED
        self.model = MSCRED(opt,matrices_train, matrices_test)
        self.matrices_test = matrices_test

    def score(self,reconstructed_matrices,opt):
        matrices_test= self.matrices_test[opt['step_max']-1:]
        reconstructed_matrix_temp = np.transpose(reconstructed_matrices, [0, 3, 1, 2])
        matrixes_test_array = np.array(matrices_test)
        select_matrix_error = np.square(matrixes_test_array - reconstructed_matrix_temp[:, 0, :, :])
        scores = np.max(select_matrix_error, axis=(1, 2))
        return scores
    
    def evaluate(self, scores, time_collision, full=False, convert=False):
        #score is attributed to last point in window

        if full:
            thresholds = np.sort(scores)
        else:
            thresholds = np.linspace(scores.min(), scores.max(), num=300)

        if convert:
            time_collision = convert(time_collision) #TADGANLOADER conversion

        collisions =pd.read_excel(os.path.join(ROOTDIR_DATASET_ANOMALY, "20220811_collisions_timestamp.xlsx"),sheet_name=None)
        collisions = pd.concat(collisions.values(), ignore_index=True)
        collisions_init = collisions[collisions['Inizio/fine'] == "i"].Timestamp - pd.to_timedelta([2] * len(collisions[collisions['Inizio/fine'] == "i"].Timestamp), 'h')
        collision_end = collisions[collisions['Inizio/fine'] == "f"].Timestamp - pd.to_timedelta([2] * len(collisions[collisions['Inizio/fine'] == "f"].Timestamp), 'h')
        
        time_collision = time_collision[:len(scores)]
        assert len(scores) == len(time_collision), "unmatching score/thresholds/timestamp"
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

            pos_pred  = np.sum(threshold_mask)
            neg_pred = n_samples - pos_pred

            # count anomaly timestamps included in an anomaly window -> tp
            collision_in_window = ((start_times[threshold_mask] >= collisions_init_np[:, np.newaxis]) & \
                                (start_times[threshold_mask] < collisions_end_np[:, np.newaxis]))
            
            tp = np.sum(collision_in_window) #overall sum is necessary
            fp = pos_pred - tp

            not_threshold_mask = np.where(threshold_mask, False, True)
        
            # count non anomaly timestamps included in an anomaly window -> fn
            false_not_collision_in_window = ((start_times[not_threshold_mask] >= collisions_init_np[:, np.newaxis]) & \
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

        return recalls, precisions, fprs, accuracies, f1s, cms, anomaly_indices_list