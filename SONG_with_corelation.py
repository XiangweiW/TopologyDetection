import pandas as pd
import numpy as np
import os
from song.song import SONG
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import datetime

# Set the directory of data files and the range of weeks
directory = 'weekly_data'
weeks = range(23, 35)

# Define a standardization function
def standardize(data):
    mean = np.nanmean(data, axis=1, keepdims=True)
    std = np.nanstd(data, axis=1, keepdims=True)
    std[std == 0] = 1  # Prevent division by zero
    return (data - mean) / std

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

# Add a new respondent ID, whose voltage data is always 220 at all timestamps
new_respondent_id = "constant_voltage_220"
unique_ids.add(new_respondent_id)

# Initialize a dictionary to store data for each respondent ID
data_by_id = {respondent_id: np.full((len(weeks), len(base_timestamps)), np.nan) for respondent_id in unique_ids}

# Assign constant voltage values for the new respondent ID
data_by_id[new_respondent_id][:] = 220

# Read and align data for each respondent ID
for week in weeks:
    file_path = os.path.join(directory, f'weekly_data_2021_week_{week}.csv')
    week_data = pd.read_csv(file_path)
    week_data['time'] = pd.to_datetime(week_data['time']).dt.round('min')

    for respondent_id in unique_ids - {new_respondent_id}:
        respondent_week_data = week_data[week_data['respondent_id'] == respondent_id]
        for i, timestamp in enumerate(base_timestamps):
            if timestamp in respondent_week_data['time'].values:
                voltage_value = respondent_week_data[respondent_week_data['time'] == timestamp]['voltage'].values[0]
                data_by_id[respondent_id][weeks.index(week), i] = voltage_value

unique_ids.remove(new_respondent_id)
unique_ids_list = list(unique_ids)  # Update unique_ids_list

# Define an interpolation function to fill NaN values in time series data
def interpolate_data(data):
    return np.apply_along_axis(
        lambda x: np.interp(
            np.arange(len(x)),
            np.arange(len(x))[~np.isnan(x)],
            x[~np.isnan(x)]
        ),
        axis=1,
        arr=data
    )

# Function to compute Pearson correlation coefficients between each sensor and all other sensors over a week
def compute_similarity(data_by_week):
    n_sensors = len(data_by_week)
    similarity_matrix = np.zeros((n_sensors, n_sensors))

    for i in range(n_sensors):
        for j in range(n_sensors):
            if i != j:
                similarity_matrix[i, j] = np.corrcoef(data_by_week[i], data_by_week[j])[0, 1]
            else:
                similarity_matrix[i, j] = 1  # Set similarity with self as NaN

    return similarity_matrix

# Transform data structure for similarity computation
transformed_data = []
for week in weeks:
    week_data = []
    for respondent_id in unique_ids_list:
        # Extract data for each sensor in a specific week
        data_by_week = [data_by_id[resp_id][weeks.index(week)] for resp_id in unique_ids_list]

        # Apply interpolation to data for each sensor
        interpolated_data_by_week = interpolate_data(np.array(data_by_week))

        # Apply standardization to data for each sensor
        standardized_data_by_week = standardize(interpolated_data_by_week)

        # Compute similarity matrix of standardized data
        week_similarity = compute_similarity(standardized_data_by_week)
        week_data.append(week_similarity[unique_ids_list.index(respondent_id)])
    transformed_data.extend(week_data)

transformed_data = np.array(transformed_data)  # Convert to NumPy array

# Ensure data shape is (12*19, 19)
print("Transformed data shape:", transformed_data.shape)

# Transform data structure for similarity computation
transformed_data = np.array(transformed_data).reshape(len(unique_ids), len(weeks), -1)

# Check for NaN or Infinite values in the data
if np.any(np.isnan(transformed_data)) or not np.all(np.isfinite(transformed_data)):
    print("Data contains NaN or Infinite values, which may cause issues with SONG.")

# # Apply the SONG algorithm and plot
# plt.figure(figsize=(16, 10))

n_respondents = len(unique_ids)
colors = cm.rainbow(np.linspace(0, 1, n_respondents))

# For each respondent_id in unique_ids_list
for i, respondent_id in enumerate(unique_ids_list):
    plt.figure(figsize=(8, 5))
    data = transformed_data[i]
    song = SONG(n_components=2, n_neighbors=min(data.shape[0] - 1, 15))
    Y = song.fit_transform(data)

    # Plot scatter plot
    plt.scatter(Y[:, 0], Y[:, 1], color=colors[i % n_respondents], alpha=0.7, label=f'{respondent_id}')

    # Add labels for each point indicating the week
    for j in range(len(Y)):
        week_number = weeks[j]
        plt.text(Y[j, 0], Y[j, 1], f'W{week_number}', fontsize=8, ha='right', va='bottom')

    plt.title(f'SONG Visualization for {respondent_id}')
    plt.xlabel('SONG Dimension 1')
    plt.ylabel('SONG Dimension 2')
    plt.legend()
    plt.show()
