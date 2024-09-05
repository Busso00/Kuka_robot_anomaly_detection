'''
Project: MLinApps-Proj
Author:MLinApps-Proj group 2024/08
Members:
Haochen He S307771
Federico Bussolino S317641
Marco D'almo  S301199
Youness Bouchari S323624
'''

import os
import time
import tsfel
import warnings
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import VarianceThreshold
import tensorflow as tf
import keras_tuner
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from datetime import datetime
import hashlib

ROOTDIR_DATASET_NORMAL = "normal"
ROOTDIR_DATASET_ANOMALY = "collisions"

filepath_csv = [os.path.join(ROOTDIR_DATASET_NORMAL, f"rec{r}_20220811_rbtc_0.1s.csv") for r in [0, 2, 3, 4]]
filepath_meta = [os.path.join(ROOTDIR_DATASET_NORMAL, f"rec{r}_20220811_rbtc_0.1s.metadata") for r in [0, 2, 3, 4]]


filepath_csv_anomaly = [os.path.join(ROOTDIR_DATASET_ANOMALY, f"rec{r}_collision_20220811_rbtc_0.1s.csv") for r in [1, 5]]
filepath_meta_anomaly = [os.path.join(ROOTDIR_DATASET_ANOMALY, f"rec{r}_collision_20220811_rbtc_0.1s.metadata") for r in[1, 5]]

def get_df_action(filepaths_csv, filepaths_meta, action2int=None, delimiter=";"):
    # Load dataframes
    print("Loading data.")
    # Make dataframes
    # Some classes show the output boolean parameter as True rather than true. Fix here
    dfs_meta = list()
    for filepath in filepaths_meta:
        df_m = pd.read_csv(filepath, sep=delimiter)
        df_m.str_repr = df_m.str_repr.str.replace('True', 'true')
        df_m['filepath'] = filepath
        dfs_meta.append(df_m)

    df_meta = pd.concat(dfs_meta)
    df_meta.index = pd.to_datetime(df_meta.init_timestamp.to_numpy().astype('datetime64[ms]'), format="%Y-%m-%dT%H:%M:%S.%f")
    df_meta['completed_timestamp'] = pd.to_datetime(df_meta.completed_timestamp.to_numpy().astype('datetime64[ms]'),
                                                    format="%Y-%m-%dT%H:%M:%S.%f")
    df_meta['init_timestamp'] = pd.to_datetime(df_meta.init_timestamp.to_numpy().astype('datetime64[ms]'),
                                               format="%Y-%m-%dT%H:%M:%S.%f")

    # Eventually reduce number of classes
    # df_meta['str_repr'] = df_meta.str_repr.str.split('=', expand = True,n=1)[0]
    # df_meta['str_repr'] = df_meta.str_repr.str.split('(', expand=True, n=1)[0]

    actions = df_meta.str_repr.unique()
    dfs = [pd.read_csv(filepath_csv, sep=";") for filepath_csv in filepaths_csv]
    df = pd.concat(dfs)

    # Sort columns by name !!!
    df = df.sort_index(axis=1)

    # Set timestamp as index
    df.index = pd.to_datetime(df.time.to_numpy().astype('datetime64[ms]'), format="%Y-%m-%dT%H:%M:%S.%f")
    # Drop useless columns
    columns_to_drop = [column for column in df.columns if "Abb" in column or "Temperature" in column]
    df.drop(["machine_nameKuka Robot_export_active_energy",
             "machine_nameKuka Robot_import_reactive_energy"] + columns_to_drop, axis=1, inplace=True)
    signals = df.columns

    df_action = list()
    for action in actions:
        for index, row in df_meta[df_meta.str_repr == action].iterrows():
            start = row['init_timestamp']
            end = row['completed_timestamp']
            df_tmp = df.loc[start: end].copy()
            df_tmp['action'] = action
            # Duration as string (so is not considered a feature)
            df_tmp['duration'] = str((row['completed_timestamp'] - row['init_timestamp']).total_seconds())
            df_action.append(df_tmp)
    df_action = pd.concat(df_action, ignore_index=True)
    df_action.index = pd.to_datetime(df_action.time.to_numpy().astype('datetime64[ms]'), format="%Y-%m-%dT%H:%M:%S.%f")
    df_action = df_action[~df_action.index.duplicated(keep='first')]

    # Drop NaN
    df = df.dropna(axis=0)
    df_action = df_action.dropna(axis=0)

    if action2int is None:
        action2int = dict()
        j = 1
        for label in df_action.action.unique():
            action2int[label] = j
            j += 1

    df_merged = df.merge(df_action[['action']], left_index=True, right_index=True, how="left")
    # print(f"df_merged len: {len(df_merged)}")
    # Where df_merged in NaN Kuka is in idle state
    df_idle = df_merged[df_merged['action'].isna()].copy()
    df_idle['action'] = 'idle'
    df_idle['duration'] = df_action.duration.values.astype(float).mean().astype(str)
    df_action = pd.concat([df_action, df_idle])

    # ile label must be 0 for debug mode
    action2int['idle'] = 0
    print(f"Found {len(set(df_action['action']))} different actions.")
    print("Loading data done.\n")
    df_act_toconcat = df_action['action']
    df_concat = pd.concat([df, df_act_toconcat], axis=1)

    df_action['action'] = df_action['action'].map(action2int)
    return df_action, df_concat, df_meta, action2int



def get_windowed_data(root_path="/content/drive/MyDrive/Kuka_v1/", period='0.1', window_size=40, stride=1, train=True):
    """
        -period (str in s) in ['1.0','0.1','0.01','0.05']
        -window_size = n input samples in each window (int)
        -stride = n moving position over time series datapoint (int)



    return window over all train/test timeseries,
    non contiguous timeseries (from different dataset) are handled
    avoiding overlapping (a window over 2 different timeseries data is not possible)
    """

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)


    assert window_size > 0, "at least 1 sample in each window is required"
    assert stride > 0, "moving window must move of at least 1 sample"
    assert window_size-stride >= 0, "why you are discarding precious timeseries data?"


    overlap = (window_size-stride)/window_size

    window_splits = []

    if train:

        collisions_corr = None

        filepath_csv = [os.path.join(root_path, f"normal/rec{r}_20220811_rbtc_{period}s.csv") for r in [0, 2, 3, 4]]
        filepath_meta = [os.path.join(root_path, f"normal/rec{r}_20220811_rbtc_{period}s.metadata") for r in [0, 2, 3, 4]]

        for i in range(4):
            #get one csv file at a time, discard incorrect (cross-file) time windows
            #--> start from first of at least window_size size)
            df_action, _, _, _ = get_df_action([filepath_csv[i],], [filepath_meta[i],])
            window_splits.extend(tsfel.utils.signal_processing.signal_window_splitter(df_action.copy(), window_size, overlap))

    else:

        collisions = pd.read_excel(os.path.join(root_path, "collisions/20220811_collisions_timestamp.xlsx"))
        #timestamp of collision corrected
        collisions_corr = collisions.Timestamp - pd.to_timedelta([2] * len(collisions.Timestamp), 'h')


        filepath_csv = [os.path.join(root_path, f"collisions/rec{r}_collision_20220811_rbtc_{period}s.csv") for r in [1, 5]]
        filepath_meta = [os.path.join(root_path, f"collisions/rec{r}_collision_20220811_rbtc_{period}s.metadata") for r in [1, 5]]

        for i in range(2):
            #get one csv file at a time, discard incorrect (cross-file) time windows
            #--> start from first of at least window_size size)
            df_action, _, _, _ = get_df_action([filepath_csv[i],], [filepath_meta[i],]) #get partial label
            window_splits.extend(tsfel.utils.signal_processing.signal_window_splitter(df_action.copy(), window_size, overlap))

    df_action_tot, _, _, _ = get_df_action(filepath_csv, filepath_meta)

    return window_splits, df_action_tot, collisions_corr

def get_windowed_data_for_plot(root_path="/content/drive/MyDrive/Kuka_v1/", period='0.1', window_size=100, stride=100, train=True):
    """
        -period (str in s) in ['1.0','0.1','0.01','0.05']
        -window_size = n input samples in each window (int)
        -stride = n moving position over time series datapoint (int)

    return window over all train/test timeseries,
    non contiguous timeseries (from different dataset) are handled
    avoiding overlapping (a window over 2 different timeseries data is not possible)
    """

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)


    assert window_size > 0, "at least 1 sample in each window is required"
    assert stride > 0, "moving window must move of at least 1 sample"
    assert window_size-stride >= 0, "why you are discarding precious timeseries data?"


    overlap = (window_size-stride)/window_size

    window_splits = []

    if train:

        collisions_corr = None

        filepath_csv = [os.path.join(root_path, f"normal/rec{r}_20220811_rbtc_{period}s.csv") for r in [0, 2, 3, 4]]
        filepath_meta = [os.path.join(root_path, f"normal/rec{r}_20220811_rbtc_{period}s.metadata") for r in [0, 2, 3, 4]]

        for i in range(4):
            #get one csv file at a time, discard incorrect (cross-file) time windows
            #--> start from first of at least window_size size)
            print(filepath_csv[i])
            df_action, _, _, _ = get_df_action([filepath_csv[i],], [filepath_meta[i],])
            window_splits.extend(tsfel.utils.signal_processing.signal_window_splitter(df_action.copy(), window_size, overlap))

    else:

        collisions = pd.read_excel(os.path.join(root_path, "collisions/20220811_collisions_timestamp.xlsx"))
        #timestamp of collision corrected
        collisions_corr = collisions.Timestamp - pd.to_timedelta([2] * len(collisions.Timestamp), 'h')


        filepath_csv = [os.path.join(root_path, f"collisions/rec{r}_collision_20220811_rbtc_{period}s.csv") for r in [1, 5]]
        filepath_meta = [os.path.join(root_path, f"collisions/rec{r}_collision_20220811_rbtc_{period}s.metadata") for r in [1, 5]]

        for i in range(2):
            #get one csv file at a time, discard incorrect (cross-file) time windows
            #--> start from first of at least window_size size)
            df_action, _, _, _ = get_df_action([filepath_csv[i],], [filepath_meta[i],]) #get partial label
            window_splits.extend(tsfel.utils.signal_processing.signal_window_splitter(df_action.copy(), window_size, overlap))

        print("End get windowed data for test")

    df_action_tot, _, _, _ = get_df_action(filepath_csv, filepath_meta)


    return window_splits, collisions_corr




ROOTDIR_KUKA = "/content/drive/MyDrive/Kuka_v1"

def transform_datetime_strings(datetime_str_array):
    transformed_array = []
    for dt_str in datetime_str_array.flatten():
        transformed_dt = datetime.strptime(dt_str, '%Y-%m-%dT%H:%M:%S.%f%z').strftime('%Y-%m-%d %H:%M:%S')
        transformed_array.append(transformed_dt)
    return np.array(transformed_array).reshape(datetime_str_array.shape)

def hash_string_to_float(s):
    # Hash the string using SHA-256 and get a hex digest
    hash_object = hashlib.sha256(s.encode())
    hex_dig = hash_object.hexdigest()

    # Convert the hex digest to an integer
    int_hash = int(hex_dig, 16)

    # Normalize the integer to a float between 0 and 1
    normalized_value = int_hash / 2 ** 256

    # Map the normalized value to the range -1 to 1
    float_value = 2 * normalized_value - 1

    return float_value

def sample_as_splitter(signal, window_size, overlap=0):
    """sample same size as splitter of tsfel

    Parameters
    ----------
    signal : nd-array or pandas DataFrame
        input signal
    window_size : int
        number of points of window size
    overlap : float
        percentage of overlap, value between 0 and 1 (exclusive)
        Default: 0
    Returns
    -------
    list
        list of signal samples
    """
    if not isinstance(window_size, int):
        raise SystemExit("window_size must be an integer.")
    step = int(round(window_size)) if overlap == 0 else int(round(window_size * (1 - overlap)))
    if step == 0:
        raise SystemExit(
            "Invalid overlap. " "Choose a lower overlap value.",
        )
    if len(signal) % window_size == 0 and overlap == 0:
        return [signal[i] for i in range(0, len(signal), step)]
    else:
        return [signal[i] for i in range(0, len(signal) - window_size + 1, step)]
    
def TADGANLOADER(opt):
    df_action, df, df_meta, action2int = get_df_action(filepath_csv, filepath_meta)

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    start_time = time.time()
    df_features = df

    print("--- %s seconds ---" % (time.time() - start_time))
    df_features.isnull().values.any()  # controllare se ci sono colonne con valori null -> ritorna true
    df_features_nonan = df_features.fillna(0)
    df_train = df_features_nonan

    X_train = df_train
    corr_features = tsfel.correlated_features(X_train, threshold=0.95)
    corr_features.append('time')
    X_train.drop(corr_features, inplace=True, axis=1)

    X_train_try = X_train
    X_train_try = np.asarray(X_train_try)

    X_train_features = X_train_try[:, :-1]  # All columns except the last
    if not opt['action']:
        X_train_try=X_train_features

    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_last_column = X_train_try[:, -1]

    if opt['action']:
        for i in range(len(X_train_last_column)):
            X_train_try[i][50] = hash_string_to_float(X_train_try[i][50])

    X_train_try = scaler.fit_transform(X_train_try)
    columns_to_dropi = [7, 8, 9, 10,11,12,16,17,19,21,22,27,34,35,36,42,43,44]

    if(opt['feature_analysis']):
       for col_index in sorted(columns_to_dropi, reverse=True):
         X_train_try = np.delete(X_train_try, col_index, axis=1)

    window_splits = []
    if(opt['point-based']):
        overlap = (opt['window_size'] - 1) / opt['window_size']
    else:
        overlap = (opt['window_size'] - 30) / opt['window_size']

    window_splits.extend(
        tsfel.utils.signal_processing.signal_window_splitter(X_train_try.copy(), opt['window_size'], overlap))
    window_splits = np.asarray(window_splits)
    shape = (opt['window_size'], opt['feature_num'])
    window_splits.reshape(-1, shape[0], opt['feature_num'])

    print(X_train_try.shape)
    print(window_splits.shape)

    # collision dataset caricamento

    collisions = pd.read_excel(os.path.join(ROOTDIR_DATASET_ANOMALY, "20220811_collisions_timestamp.xlsx"))
    collisions_init = collisions[collisions['Inizio/fine'] == "i"].Timestamp - pd.to_timedelta(
        [2] * len(collisions[collisions['Inizio/fine'] == "i"].Timestamp), 'h')
    collisions_end = collisions[collisions['Inizio/fine'] == "f"].Timestamp - pd.to_timedelta(
        [2] * len(collisions[collisions['Inizio/fine'] == "f"].Timestamp), 'h')

    collisions_init = collisions_init.array
    collisions_init_str = [x.strftime('%Y-%m-%d %H:%M:%S') for x in collisions_init]
    collisions_init = collisions_init_str

    collisions_end = collisions_end.array

    collisions_end_str = [x.strftime('%Y-%m-%d %H:%M:%S') for x in collisions_end]
    collisions_end = collisions_end_str

    df_action_collision, df_collision, df_meta_collision, action2int_collision = get_df_action(filepath_csv_anomaly,
                                                                                               filepath_meta_anomaly)

    start_time = time.time()

    df_features_collision = df_collision
    print("--- %s seconds ---" % (time.time() - start_time))

    df_features_collision.isnull().values.any()

    df_features_collision_nonan = df_features_collision.fillna(0)
    columns_to_keep = ["time"]
    columns_to_drop = [col for col in df_features_collision_nonan.columns if col not in columns_to_keep]

    X_collision = df_features_collision_nonan.drop(["time"], axis=1)
    df_time_only = df_features_collision_nonan.drop(columns=columns_to_drop)
    corr_features = corr_features[:-1]
    X_collision.drop(corr_features, inplace=True, axis=1)
    X_collision = np.asarray(X_collision)
    X_collision_features = X_collision[:, :-1]  # All columns except the last
    X_collision_last_column = X_collision[:, -1]
    if not opt['action']:
        X_collision=X_collision_features
    if opt['action']:
        for i in range(len(X_collision_last_column)):
            X_collision[i][50] = hash_string_to_float(X_collision[i][50])



    X_collision = scaler.transform(X_collision)
    if (opt['feature_analysis']):
      for col_index in sorted(columns_to_dropi, reverse=True):
         X_collision= np.delete(X_collision, col_index, axis=1)

    #feature_variance_analysis(X_train_try, X_collision, opt['feature_num'])

    time_splits_test = []
    window_splits_test = []
    overlap_test = (opt['window_size'] - 1) / opt['window_size']
    window_splits_test.extend(
        tsfel.utils.signal_processing.signal_window_splitter(X_collision.copy(), opt['window_size'], overlap_test))
    time_splits_test.extend(
        tsfel.utils.signal_processing.signal_window_splitter(df_time_only.copy(), opt['window_size'], overlap_test))
    window_splits_test = np.asarray(window_splits_test)
    time_splits_test = np.asarray(time_splits_test)

    window_splits_test.reshape(-1, shape[0], opt['feature_num'])
    time_splits_test_transformed = np.array([transform_datetime_strings(window) for window in time_splits_test])

    print(window_splits_test.shape)

    return window_splits, window_splits_test, collisions_init, time_splits_test_transformed, collisions_end

def reshape(da, time_step):
    return da.reshape(da.shape[0], time_step, da.shape[1]).astype("float32")

def VAELSTMLOADER(opt):
    frequency = 1/10
    duration = opt['window_size']*frequency
    filepath_csv = [os.path.join(ROOTDIR_DATASET_NORMAL, f"rec{r}_20220811_rbtc_0.1s.csv") for r in [0, 2, 3, 4]]
    filepath_meta = [os.path.join(ROOTDIR_DATASET_NORMAL, f"rec{r}_20220811_rbtc_0.1s.metadata") for r in [0, 2, 3, 4]]
    df_action, df, df_meta, action2int = get_df_action(filepath_csv, filepath_meta)

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    start_time = time.time()
    df_features = df

    print("--- %s seconds ---" % (time.time() - start_time))
    df_features.isnull().values.any()  # controllare se ci sono colonne con valori null -> ritorna true
    df_features_nonan = df_features.fillna(0)
    df_train = df_features_nonan

    X_train = df_train
    corr_features = tsfel.correlated_features(X_train, threshold=0.95)
    X_train.drop(corr_features, inplace=True, axis=1)
    X_train_try = X_train
    X_train_try = X_train_try.drop(["time"], axis=1)
    X_train_try = np.array(X_train_try)

    X_train_features = X_train_try[:, :-1]  # All columns except the last
    if not opt['action']:
        X_train_try=X_train_features

    scaler_value = opt['scaler_value']
    if scaler_value == 1:
        scaler = MinMaxScaler(feature_range=(0, 1))
    else:
        scaler = MinMaxScaler(feature_range=(scaler_value, -(scaler_value)))

    X_train_last_column = X_train_try[:, -1]

    if opt['action']:
        for i in range(len(X_train_last_column)):
            X_train_try[i][50] = hash_string_to_float(X_train_try[i][50])
    if opt['x_dim'] != 50:
        pca = PCA(n_components=opt['x_dim'])
        X_train_try = pca.fit_transform(X_train_try)
    X_train = scaler.fit_transform(X_train_try)
    # eliminate nan values
    X_train = np.nan_to_num(X_train)
    X_train = reshape(X_train, opt['time_step'])

    print(X_train_try.shape)
    #print(window_splits.shape)

    # collision dataset caricamento

    collisions = pd.read_excel("20220811_collisions_timestamp.xlsx")
    collisions_init = collisions[collisions['Inizio/fine'] == "i"].Timestamp - pd.to_timedelta(
        [2] * len(collisions[collisions['Inizio/fine'] == "i"].Timestamp), 'h')
    collisions_end = collisions[collisions['Inizio/fine'] == "f"].Timestamp - pd.to_timedelta(
        [2] * len(collisions[collisions['Inizio/fine'] == "f"].Timestamp), 'h')

    collisions_init = collisions_init.array
    collisions_init_str = [x.strftime('%Y-%m-%d %H:%M:%S') for x in collisions_init]
    collisions_init = collisions_init_str

    collisions_end = collisions_end.array

    collisions_end_str = [x.strftime('%Y-%m-%d %H:%M:%S') for x in collisions_end]
    collisions_end = collisions_end_str

    df_action, df_collision, _, _ = get_df_action(filepath_csv_anomaly, filepath_meta_anomaly)

    start_time = time.time()

    df_features_collision = df_collision
    print("--- %s seconds ---" % (time.time() - start_time))

    df_features_collision.isnull().values.any()

    df_features_collision_nonan = df_features_collision.fillna(0)
    columns_to_keep = ["time"]
    columns_to_drop = [col for col in df_features_collision_nonan.columns if col not in columns_to_keep]

    X_collision = df_features_collision_nonan.drop(["time"], axis=1)
    df_time_only = df_features_collision_nonan.drop(columns=columns_to_drop)
    X_collision.drop(corr_features, inplace=True, axis=1)
    X_collision = np.asarray(X_collision)
    X_collision_features = X_collision[:, :-1]  # All columns except the last
    X_collision_last_column = X_collision[:, -1]
    if not opt['action']:
        X_collision=X_collision_features
    if opt['action']:
        for i in range(len(X_collision_last_column)):
            X_collision[i][50] = hash_string_to_float(X_collision[i][50])
    if opt['x_dim'] != 50:
        X_collision = pca.transform(X_collision)
    X_collision = scaler.transform(X_collision)
    X_collision = reshape(X_collision, opt['time_step'])
    df_time = pd.DataFrame()
    time_conversion = pd.to_datetime(df_action.time.astype('datetime64[ms]'), format="%Y-%m-%dT%H:%M:%S.%f")
    df_time['start'] = sample_as_splitter(time_conversion, opt["window_size"], overlap=0)
    df_time['end'] = df_time['start'] + pd.to_timedelta(duration, 's')
    window_test_times = pd.DataFrame()
    window_test_times = pd.concat([window_test_times, df_time], axis=0)
    return X_train, X_collision, window_test_times


def sample_as_splitter(signal, window_size, overlap=0):
    """sample same size as splitter of tsfel

    Parameters
    ----------
    signal : nd-array or pandas DataFrame
        input signal
    window_size : int
        number of points of window size
    overlap : float
        percentage of overlap, value between 0 and 1 (exclusive)
        Default: 0
    Returns
    -------
    list
        list of signal samples
    """
    if not isinstance(window_size, int):
        raise SystemExit("window_size must be an integer.")
    step = int(round(window_size)) if overlap == 0 else int(round(window_size * (1 - overlap)))
    if step == 0:
        raise SystemExit(
            "Invalid overlap. " "Choose a lower overlap value.",
        )
    if len(signal) % window_size == 0 and overlap == 0:
        return [signal[i] for i in range(0, len(signal), step)]
    else:
        return [signal[i] for i in range(0, len(signal) - window_size + 1, step)]



def MTGFLOWLOADER_action(opt):

    #Use same overlap for test and train set
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    if opt['num_features']==50:
        mode = 0
    elif opt['num_features']==51:
        mode = 1
    else:
        assert False, "not implemented"

    frequency = 1/10
    duration = opt['window_size']*frequency
    overlap = (opt['window_size'] - opt['stride']) / opt['window_size']        
    

    print("--- LOADING TRAINING SET FOR TUNING ---")
    #NOTE: all action2int unique values must appear in training set
    #action2int and action_cols must not be changed at test time    
    X_train_action, _, _, action2int = get_df_action(filepath_csv, filepath_meta)
    
    scaler = StandardScaler()
    
    shape = (-1, opt['window_size'], opt['num_features'])

    print("--- TUNING SCALER AND CORR PARAMETERS ---")

    #int encoding
    X_train_action["action"] = X_train_action["action"].map(action2int)
   
    X_train = X_train_action.fillna(0) #1 fill null values
    
    if mode != 1:
        X_train = X_train.drop(["time", "duration", "action"], axis=1) #2 drop timestamps columns
    else:
        X_train = X_train.drop(["time", "duration"], axis=1) #2 drop timestamps columns
    
    corr_features = tsfel.correlated_features(X_train,threshold=0.95) #action should not be too much correlated
    
    X_train = X_train.drop(corr_features, axis=1) #3 drop corr features

    scaler.fit_transform(X_train)

    print("--- LOADING TRAINING SET ---")

    window_splits_train = []
    window_action_train = []

    for i in range(4):
        #get one csv file at a time, discard incorrect (cross-file) time windows
        #--> start from first of at least window_size size)
        
        X_train_action, _, _, _ = get_df_action([filepath_csv[i],], [filepath_meta[i],])
        
        print(f"--- PREPROCESSING TRAINING SPLIT {i} WITH OVERLAP {overlap} ---")
        #actual preprocessing

        #int encoding
        X_train_action["action"] = X_train_action["action"].map(action2int)
        
        window_action_train.extend(sample_as_splitter(X_train_action["action"],opt["window_size"], overlap=overlap)) #need ations mapping to rescale by action
        
        
        X_train = X_train_action.fillna(0) #1 fill null values
        
        if mode != 1:
            X_train = X_train.drop(["time","duration","action"], axis=1) #2 drop timestamps columns
        else:
            X_train = X_train.drop(["time","duration"], axis=1) #2 drop timestamps columns

        X_train = X_train.drop(corr_features, axis=1) #3 drop corr features
        X_train = scaler.transform(X_train) #4 normalize

        
        window_splits_train.extend(tsfel.utils.signal_processing.signal_window_splitter(X_train.copy(), opt['window_size'], overlap=overlap))
    
    
    #output of scaler is np array
    window_splits_train = np.concatenate(window_splits_train, axis=0)
    window_splits_train = window_splits_train.reshape(shape)

    overlap = (opt['window_size'] - 1) / opt['window_size']
    #overlap = 0

    print("window split train size:")
    print(window_splits_train.shape)
    print("window action train size:")
    print(len(window_action_train))
    assert len(window_action_train) == len(window_splits_train) , "error loading train set actions"

    print("--- LOADING TEST SET ---")
    
    window_splits_test = []
    window_action_test = []
    window_test_times = pd.DataFrame()

    for i in range(2):
        #get one csv file at a time, discard incorrect (cross-file) time windows
        #--> start from first of at least window_size size)
        X_test_action, _, _, _ = get_df_action([filepath_csv_anomaly[i],], [filepath_meta_anomaly[i],])
        
        print(f"--- PREPROCESSING TEST SPLIT {i} ---")
        
        df_time = pd.DataFrame()
        time = pd.to_datetime(X_test_action.time.astype('datetime64[ms]'), format="%Y-%m-%dT%H:%M:%S.%f")
        #NOTE: X_test_action returns always timestamp of first file
        
        df_time['start'] = sample_as_splitter(time, opt["window_size"], overlap=overlap)
        df_time['end'] = df_time['start'] + pd.to_timedelta(duration, 's')
        
        #int encoding
        X_test_action["action"] = X_test_action["action"].map(action2int)
        window_action_test.extend(sample_as_splitter(X_test_action["action"],opt["window_size"], overlap=overlap)) #need ations mapping to rescale by action
        
        X_test = X_test_action.fillna(0) #1 fill null values
        if mode != 1:
            X_test = X_test.drop(["time","duration","action"], axis=1)  #2 drop timestamps columns
        else:
            X_test = X_test.drop(["time","duration"], axis=1)

        X_test = X_test.drop(corr_features, axis=1) #3 drop corr features
        X_test = scaler.transform(X_test) #4 normalize
        
        window_splits_test.extend(tsfel.utils.signal_processing.signal_window_splitter(X_test.copy(), opt['window_size'], overlap=overlap))
        #append new timestamps
        window_test_times = pd.concat([window_test_times, df_time], axis=0)

    #output of scaler is np array
    window_splits_test = np.concatenate(window_splits_test, axis=0)
    window_splits_test = window_splits_test.reshape(shape)       
    
    print("window split test size:")
    print(window_splits_test.shape)
    print("window action test size:")
    print(len(window_action_test))
    print("timestamps")
    print(len(window_test_times))

    assert len(window_test_times) == len(window_splits_test), "time and data mismatch"
    assert len(window_action_test) == len(window_splits_test) , "error loading test set actions"

    return window_splits_train, window_action_train, window_splits_test, window_test_times, window_action_test

def MSCREDLOADER(opt):
    ROOTDIR_DATASET_NORMAL = "/content/drive/MyDrive/Kuka_v1/normal"
    ROOTDIR_DATASET_ANOMALY = "/content/drive/MyDrive/Kuka_v1/collisions"
    filepath_csv = [os.path.join(ROOTDIR_DATASET_NORMAL, f"rec{r}_20220811_rbtc_0.1s.csv") for r in [0, 2, 3, 4]]
    filepath_meta = [os.path.join(ROOTDIR_DATASET_NORMAL, f"rec{r}_20220811_rbtc_0.1s.metadata") for r in [0, 2, 3, 4]]

    filepath_csv_anomaly = [os.path.join(ROOTDIR_DATASET_ANOMALY, f"rec{r}_collision_20220811_rbtc_0.1s.csv") for r in [1, 5]]
    filepath_meta_anomaly = [os.path.join(ROOTDIR_DATASET_ANOMALY, f"rec{r}_collision_20220811_rbtc_0.1s.metadata") for r in[1, 5]]

    df_action, df, df_meta, action2int = get_df_action(filepath_csv,filepath_meta)
    df.drop(['action'], axis=1 )

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    start_time = time.time()
    df_features = df

    print("--- %s seconds ---" % (time.time() - start_time))
    df_features.isnull().values.any()  # controllare se ci sono colonne con valori null -> ritorna true
    df_features_nonan = df_features.fillna(0)
    df_train = df_features_nonan

    X_train = df_train
    corr_features = tsfel.correlated_features(X_train, threshold=0.95)
    X_train.drop(corr_features, inplace=True, axis=1)
    X_train_try = X_train
    X_train_try = X_train_try.drop(["time"], axis=1)
    X_train_try = np.array(X_train_try)

    X_train_try= X_train_try[:, :-1]  # All columns except the last

    scaler = MinMaxScaler(feature_range=(0, 1))
    pca = PCA(n_components=0.95)


    X_train_try = scaler.fit_transform(X_train_try)
    #X_train_try = pca.fit_transform(X_train_try)

    window_splits = []
    #overlap = (opt['window_size'] - 15) / opt['window_size']
    window_splits.extend(
        tsfel.utils.signal_processing.signal_window_splitter(X_train_try.copy(), opt['window_size'],opt['overlap_train']))
    window_splits = np.asarray(window_splits)
    window_splits.reshape(-1, opt['window_size'], X_train_try.shape[1])

    # collision dataset caricamento

    collisions = pd.read_excel(os.path.join(ROOTDIR_DATASET_ANOMALY, "20220811_collisions_timestamp.xlsx"))
    collisions_init = collisions[collisions['Inizio/fine'] == "i"].Timestamp - pd.to_timedelta(
        [2] * len(collisions[collisions['Inizio/fine'] == "i"].Timestamp), 'h')
    collisions_end = collisions[collisions['Inizio/fine'] == "f"].Timestamp - pd.to_timedelta(
        [2] * len(collisions[collisions['Inizio/fine'] == "f"].Timestamp), 'h')

    collisions_init = collisions_init.array
    collisions_init_str = [x.strftime('%Y-%m-%d %H:%M:%S') for x in collisions_init]
    collisions_init = collisions_init_str

    collisions_end = collisions_end.array

    collisions_end_str = [x.strftime('%Y-%m-%d %H:%M:%S') for x in collisions_end]
    collisions_end = collisions_end_str

    df_action_collision, df_collision, df_meta_collision, action2int_collision = get_df_action(filepath_csv_anomaly,filepath_meta_anomaly)
    df_collision.drop(['action'], axis=1)
    start_time = time.time()

    df_features_collision = df_collision
    print("--- %s seconds ---" % (time.time() - start_time))

    df_features_collision.isnull().values.any()

    df_features_collision_nonan = df_features_collision.fillna(0)
    columns_to_keep = ["time"]
    columns_to_drop = [col for col in df_features_collision_nonan.columns if col not in columns_to_keep]

    X_collision = df_features_collision_nonan.drop(["time"], axis=1)
    df_time_only = df_features_collision_nonan.drop(columns=columns_to_drop)
    X_collision.drop(corr_features, inplace=True, axis=1)
    X_collision = np.asarray(X_collision)
    X_collision_features = X_collision[:, :-1]  # All columns except the last
    X_collision=X_collision_features
    X_collision = scaler.transform(X_collision)
    #X_collision = pca.transform(X_collision)

    time_splits_test = []
    window_splits_test = []
    overlap_test = (opt['window_size'] - 1) / opt['window_size']
    window_splits_test.extend(
        tsfel.utils.signal_processing.signal_window_splitter(X_collision.copy(), opt['window_size'], overlap_test))
    time_splits_test.extend(
        tsfel.utils.signal_processing.signal_window_splitter(df_time_only.copy(), opt['window_size'], overlap_test))
    window_splits_test = np.asarray(window_splits_test)
    time_splits_test = np.asarray(time_splits_test)

    window_splits_test.reshape(-1,opt['window_size'], X_collision.shape[1])
    time_splits_test_transformed = np.array([transform_datetime_strings(window) for window in time_splits_test])

    return window_splits, window_splits_test, time_splits_test_transformed
