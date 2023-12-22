import pandas as pd
import numpy as np
from numpy.fft import fft
import os
from song.song import SONG
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import datetime

# Define the Fourier Transform function
def apply_fft(data):
    # Apply FFT
    fft_result = fft(data, axis=1)
    # Calculate Power Spectral Density
    psd = np.abs(fft_result) ** 2
    return psd

# Set the directory of data files and the range of weeks
directory = 'weekly_data'
weeks = range(23, 35)

# Get all unique respondent IDs
unique_ids = set()
for week in weeks:
    file_path = os.path.join(directory, f'weekly_data_2021_week_{week}.csv')
    week_data = pd.read_csv(file_path)
    unique_ids.update(week_data['respondent_id'].unique())

# Create a fixed timeline baseline, with intervals of two minutes
start_time = datetime.datetime(2021, 6, 7)
end_time = datetime.datetime(2021, 8, 29, 23, 59)
base_timestamps = pd.date_range(start=start_time, end=end_time, freq='2T').round('min')

# Add a new respondent ID whose voltage data is always 220 at all timestamps
new_respondent_id = "constant_voltage_220"
# unique_ids.add(new_respondent_id)

# Initialize a dictionary to store data for each respondent ID
data_by_id = {respondent_id: np.full((len(weeks), len(base_timestamps)), np.nan) for respondent_id in unique_ids}

# Read and align data for each respondent ID
for week in weeks:
    file_path = os.path.join(directory, f'weekly_data_2021_week_{week}.csv')
    week_data = pd.read_csv(file_path)
    week_data['time'] = pd.to_datetime(week_data['time']).dt.round('min')

    for respondent_id in unique_ids:  # Exclude the new respondent ID
    # for respondent_id in unique_ids:
        respondent_week_data = week_data[week_data['respondent_id'] == respondent_id]
        for i, timestamp in enumerate(base_timestamps):
            if timestamp in respondent_week_data['time'].values:
                voltage_value = respondent_week_data[respondent_week_data['time'] == timestamp]['voltage'].values[0]
                data_by_id[respondent_id][weeks.index(week), i] = voltage_value

# Define a standardization function
def standardize(data):
    mean = np.nanmean(data, axis=1, keepdims=True)
    std = np.nanstd(data, axis=1, keepdims=True)
    std[std == 0] = 1  # Prevent division by zero
    return (data - mean) / std

# Create a color map
n_respondents = len(unique_ids)
colors = cm.rainbow(np.linspace(0, 1, n_respondents))
print("data id:", data_by_id)

# Apply the SONG algorithm and plot
plt.figure(figsize=(16, 10))
for i, (respondent_id, data) in enumerate(data_by_id.items()):
    # Apply interpolation and standardization to each respondent's data
    interpolated_data = np.apply_along_axis(lambda x: np.interp(np.arange(len(x)), np.arange(len(x))[~np.isnan(x)], x[~np.isnan(x)]), axis=1, arr=data)
    print("input shape:", interpolated_data.shape)
    standardized_data = standardize(interpolated_data)
    fft_data = apply_fft(interpolated_data)
    standardized_data = fft_data

    # Apply the SONG algorithm
    song = SONG(n_components=2, n_neighbors=min(standardized_data.shape[0] - 1, 15))
    Y = song.fit_transform(standardized_data)

    # Plot the scatter plot
    plt.scatter(Y[:, 0], Y[:, 1], color=colors[i], label=f'{respondent_id}', alpha=0.7)

    # Calculate and plot the label of the center point of each cluster
    centroid = np.mean(Y, axis=0)
    plt.annotate(respondent_id, (centroid[0], centroid[1]), textcoords="offset points", xytext=(0,10), ha='center')

plt.title('SONG Visualization of All Respondents')
plt.xlabel('SONG Dimension 1')
plt.ylabel('SONG Dimension 2')
plt.legend()
plt.show()
