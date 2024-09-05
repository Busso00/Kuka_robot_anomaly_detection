import numpy as np
import pandas as pd
import os

ROOTDIR_DATASET_ANOMALY = "collisions/"
filepath_csv_anomaly = [ROOTDIR_DATASET_ANOMALY + f"rec{r}_collision_20220811_rbtc_0.1s.csv" for r in [1, 5]]
filepath_meta_anomaly = [ROOTDIR_DATASET_ANOMALY + f"rec{r}_collision_20220811_rbtc_0.1s.metadata" for r in [1, 5]]

def convert(times_windowed):
    # Lista per memorizzare le righe
    rows = []
    
    for time_window in times_windowed:
        # Estrai i timestamp di inizio e fine dalla finestra temporale
        start = time_window[0][0]
        end = time_window[-1][0]
        # Aggiungi la riga alla lista
        rows.append({'start': start, 'end': end})
    
    # Crea il DataFrame una sola volta usando la lista di righe
    df = pd.DataFrame(rows, columns=['start', 'end'])
    df['start'] = pd.to_datetime(df['start'])
    df['end'] = pd.to_datetime(df['end'])
    
    return df

def calculate_auc(x, y):
    x=np.array(x)
    y=np.array(y)
    # Ensure TPR and FPR arrays are sorted together by FPR (ascending)
    sorted_idx = np.argsort(x)
    x = x[sorted_idx]
    y = y[sorted_idx]

    # Calculate trapezoidal areas for each interval
    auc = np.trapz(y, x)

    return auc

def compute_average_score(scores, opt):
    window_size = opt['window_size']
    adjusted_anomalies = np.zeros(len(scores))

    for i in range(len(scores)-window_size):
        adjusted_anomalies[i:i + window_size] += scores[i]

    counts = np.zeros(len(scores))
    for i in range(len(scores)-window_size):
        counts[i:i + window_size] += 1

    adjusted_anomalies = adjusted_anomalies / counts

    return adjusted_anomalies


def metrics_by_point_vectorized(scores, time_collision, full=False, convert=False):
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



def metrics_by_window(score, thresholds, time_nonan):

    collisions = pd.read_excel("collisions/20220811_collisions_timestamp.xlsx")
    collisions_init = collisions[collisions['Inizio/fine'] == "i"].Timestamp - pd.to_timedelta([2] * len(collisions[collisions['Inizio/fine'] == "i"].Timestamp), 'h')

    len(score) == len(time_nonan), "unmatching score/thresholds/timestamp"
    print(f"--- LOADED {len(collisions_init)} COLLISIONS ---")

    recalls = []
    precisions = []
    accuracies = []
    fprs = []
    f1s = []
    cms = []

    for threshold in thresholds:
        df_not_confident = time_nonan[score > threshold]

        anomaly_indices = list()
        tp = 0

        for anomaly in collisions_init:#iterate over gt

            for index, row in df_not_confident.iterrows():#iterate over predictions

                start_time = pd.Timestamp(row['start']).tz_localize(None)
                end_time = pd.Timestamp(row['end']).tz_localize(None)

                if anomaly >= start_time and anomaly <= end_time:
                    #LAB interpretation:
                    #"An anomaly is detected if I correctly find the starting of it"
                    anomaly_indices.append(index)
                    tp += 1 
                    #anomaly is considered tp if the start point of real anomaly is inbetween our detected anomalous window

        cm_anomaly = np.zeros((2, 2))
        n_samples = len(score)
        n_coll = len(collisions_init)
        n_not_collisions = n_samples - n_coll
        n_detected = len(df_not_confident)

        fp = n_detected - tp #fp if detected but not tp
        fn = n_coll - tp #fn is the number of anomaly - the nuumber of detected
        tn = n_not_collisions - fp #tn if no collision and no falsely detected
        cm_anomaly[0][0] = tn
        cm_anomaly[1][1] = tp
        cm_anomaly[0][1] = fp
        cm_anomaly[1][0] = fn

        precision = tp / (tp + fp) if tp + fp != 0 else 0
        recall = tp / (tp + fn) if tp + fn != 0 else 0

        precisions.append(precision) # (=specificity)
        recalls.append(recall) # (=TPR, sensitivity)

        fprs.append( fp / (fp + tn)) if fp+tn !=0 else fprs.append(0)
        
        accuracies.append((tp + tn) / (tp + tn + fp + fn)) if fp+tn+fn+tp !=0 else accuracies.append(0)

        f1s.append(2*(precision*recall) / (precision + recall)) if precision+recall != 0 else f1s.append(0)

        cms.append(cm_anomaly.copy())


    return  recalls, precisions, fprs, accuracies, f1s, cms, anomaly_indices

