import pandas as pd
import os

# Load data
file_path = 'filtered_data.csv'
data = pd.read_csv(file_path, low_memory=False)
data.drop(data.columns[[0, -2, -1]], axis=1, inplace=True)

print(data.head())

# Make sure the 'time' column exists and is converted to datetime type
if 'time' in data.columns:
    data['time'] = pd.to_datetime(data['time'])
    # Add 'year' and 'week' columns
    data['year'] = data['time'].dt.isocalendar().year
    data['week'] = data['time'].dt.isocalendar().week
else:
    raise ValueError("Column 'time' not found in the dataset")


# Get the only week in the data set
unique_weeks = data[['year', 'week']].drop_duplicates()

# Create a dictionary to store weekly data
weekly_data_dict = {}

# Go through each unique week
for _, week_info in unique_weeks.iterrows():
    year, week = week_info['year'], week_info['week']
    # Get data for a specific week
    week_data = data[(data['year'] == year) & (data['week'] == week)]

    # Store data into dictionary
    weekly_data_dict[(year, week)] = week_data

# Create a directory to save files
save_directory = 'weekly_data'
os.makedirs(save_directory, exist_ok=True)

# Save weekly data as separate CSV file
for (year, week), week_data in weekly_data_dict.items():
    filename = f'weekly_data_{year}_week_{week}.csv'
    file_path = os.path.join(save_directory, filename)
    week_data.to_csv(file_path, index=False)

# Return to the directory where the file is saved
print("Files saved in:", save_directory)
