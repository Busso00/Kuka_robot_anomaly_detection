import matplotlib.pyplot as plt
import numpy as np

from matplotlib.dates import date2num, num2date
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from load_data import get_windowed_data
from load_data import get_windowed_data_for_plot
from matplotlib.colors import Normalize
import matplotlib.dates as mdates
import pandas as pd


def plot_anomaly_ts(index, data, anomaly_scores, title="Sensor_X", mapping=None, figsize=(18,5)):
    """
    Plots sensor data with anomalies highlighted in red and normal data in blue

    Args:
        index: range of values (in dataframe) to plot
        data: List or NumPy array of sensor readings.
        anomaly_scores: label or prediction of anomaly [0.0,1.0]
        title: Title for the plot (optional, defaults to "Sensor_X").
        mapping: due to incompatibilities of LineCollection ant timestamp data type we must perform a mapping later
    """

    # Separate normal and anomaly data for plotting
    
    x = np.array(index.copy()).reshape(-1,)
    y = np.array(data.copy()).reshape(-1,)
    dydx = np.array(anomaly_scores.copy()).reshape(-1,)[:-1]

    # Create a set of line segments so that we can color them individually
    # This creates the points as an N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be (numlines) x (points per line) x 2 (for x and y)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    fig, axs = plt.subplots(1, 1, sharex=True, sharey=True , figsize=figsize)

    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(dydx.min(), dydx.max())
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    # Set the values used for colormapping
    lc.set_array(dydx)
    lc.set_linewidth(2)
    line = axs.add_collection(lc)
    fig.colorbar(line, ax=axs)

    axs.set_xlim(x.min(), x.max())
    axs.set_ylim(y.min(), y.max())

    axs.set_title(title)

    if mapping is not None:

        stride = len(mapping)//10
        plt.xticks(ticks=index[::stride], 
                   labels=mapping[::stride], 
                   rotation=45)  # Adjust rotation as needed
        
        
        # Assuming your timestamps are in a list called 'mapping'
        # and your corresponding data is in a list called 'data'

        # Convert timestamps to numeric format for plotting
        timestamps_num = date2num(mapping)

        # Calculate differences between consecutive timestamps
        time_diffs = np.diff(timestamps_num)

        # Find indices where differences exceed 2 seconds
        large_diff_indices = np.where(time_diffs > 2)[0] + 1  # Add 1 for correct indexing

        # Create your plot
        plt.plot(timestamps_num, data)

        # Add vertical lines at indices with large differences
        for idx in large_diff_indices:
            
            plt.axvline(x=timestamps_num[idx] , linestyle='-', linewidth=2)

    
    
    plt.show()


def has_non_numeric_values(df, column_name):
    """
    Checks if a column in a pandas DataFrame contains non-numeric values.

    Args:
        df: The pandas DataFrame.
        column_name: The name of the column to check.

    Returns:
        True if the column contains non-numeric values, False otherwise.
    """

    return not pd.api.types.is_numeric_dtype(df[column_name])

def plot_all(df_anomaly, is_sorted=False, figsize=(18,5)):
    """
      plot all column (sensor values) in the dataframe,
      dataframe must have column anomaly
      remember to sort dataframe before passing
    """

    assert 'anomaly' in df_anomaly.keys(), "dataframe passed must have an anomaly label or anomaly score"
    assert is_sorted, "remember to sort and flag is_sorted=True"

    for key in df_anomaly.keys():
        if key == 'anomaly' or has_non_numeric_values(df_anomaly, key):
            continue
        plot_anomaly_ts(np.array([i for i in range(len(df_anomaly))]), df_anomaly[key], df_anomaly['anomaly'], title=key, mapping=df_anomaly.index)

#   NOTE: usage

#   remember to sort by index when calling head, is_sorted is just remember to set to sort the df (than you can set to true)
#   plot_all(df_anomaly.sort_index().head(1000), is_sorted=True, figsize=(18,5))

def plot_windowed_data(root_path="normal", period='0.1', window_size=100, stride=100, train=False, frame_num=0, left=True, anomaly_frame_num=0):
    window_splits, _ = get_windowed_data_for_plot(root_path, period, window_size, stride, train)
    if len(window_splits) == 0:
        print("No data available for the specified parameters.")
        return

    plt.figure(figsize=(30, 20))

    # Assuming the first column is the timestamp and the rest are sensor data
    frames = window_splits[frame_num]
    timestamps = frames['time'].array
    sensor_data = frames.iloc[:, 10].array  # Change the sensor index to select a specific sensor

    timestamps_num = mdates.date2num(timestamps)

    # Create line segments
    points = np.array([timestamps_num, sensor_data]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Normalize the sensor data for colormap
    norm = Normalize(vmin=sensor_data.min(), vmax=sensor_data.max())
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    lc.set_array(sensor_data)

    plt.gca().add_collection(lc)

    # Highlight anomaly segments
    if left:
        anomaly_indices = range(anomaly_frame_num)
    else:
        anomaly_indices = range(len(timestamps) - anomaly_frame_num, len(timestamps))

    anomaly_points = points[anomaly_indices]
    anomaly_segments = np.concatenate([anomaly_points[:-1], anomaly_points[1:]], axis=1)
    anomaly_lc = LineCollection(anomaly_segments, colors='red')
    plt.gca().add_collection(anomaly_lc)

    plt.gca().autoscale()
    plt.gca().set_xlim(timestamps_num.min(), timestamps_num.max())
    plt.colorbar(lc, label='Sensor Value')

    # Set x-axis to display dates and show every other timestamp
    plt.gca().xaxis_date()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
    plt.gca().set_xticks(timestamps_num[::2])
    plt.xticks(rotation=90)  # Rotate x-axis labels vertically

    plt.title(f"Sensor {10} Data Over {len(timestamps)} Frames starting timestamp : {timestamps[0]}"
              f" window size : {window_size}")
    plt.xlabel("Timestamp")
    plt.ylabel("Sensor Value")
    plt.tight_layout()
    plt.show()
