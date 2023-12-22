import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from minisom import MiniSom
import numpy as np
from song.song import SONG


# Apply the community detection algorithm to each week
# Tally the results for each week to plot the coexistence probability matrix
weekly_communities = {
"Week 23": {0: ['04B00F5A', 10077, '897AEFCE', 'BBF6FF75', 'D42BC2B8', 'FD1C5B4B'], 1: ['8B3771A1'], 2: ['0C97921F', '3711BB6B', '9175C446'], 3: ['0FDCF4C8', 'AE1E42BD'], 4: ['0AC819FA', 10122, '769D0CBD', '8AFF3E6D', 'BEA81447', 'CFEA783C', 'DF6DC684']},
"Week 24": {0: ['04B00F5A', '0C97921F', 10077, '3711BB6B', '897AEFCE', '9175C446', 'BBF6FF75', 'D42BC2B8', 'FD1C5B4B'], 1: ['0AC819FA', 10122, '769D0CBD', '8AFF3E6D', '8B3771A1', 'BEA81447', 'CFEA783C', 'DF6DC684'], 2: ['0FDCF4C8', 'AE1E42BD']},
"Week 25": {0: ['769D0CBD', '8AFF3E6D', '8B3771A1'], 1: ['0AC819FA', 10122, 'AE1E42BD', 'BEA81447', 'CFEA783C', 'DF6DC684'], 2: ['0C97921F', '3711BB6B', '9175C446'], 3: ['0FDCF4C8'], 4: ['04B00F5A', 10077, '897AEFCE', 'BBF6FF75', 'D42BC2B8', 'FD1C5B4B']},
"Week 26": {0: ['04B00F5A', 10077, '897AEFCE', 'BBF6FF75', 'CFEA783C', 'D42BC2B8', 'FD1C5B4B'], 1: ['0AC819FA', 10122, '769D0CBD', '8AFF3E6D'], 2: ['8B3771A1'], 3: ['0FDCF4C8', 'AE1E42BD'], 4: ['0C97921F', '3711BB6B', '9175C446', 'BEA81447', 'DF6DC684']},
"Week 27": {0: ['04B00F5A', 10077, '897AEFCE', 'BBF6FF75', 'CFEA783C', 'D42BC2B8', 'FD1C5B4B'], 1: ['0AC819FA', '0FDCF4C8', 'AE1E42BD'], 2: ['0C97921F', '3711BB6B', '9175C446', 'BEA81447'], 3: [10122, '769D0CBD', '8AFF3E6D', '8B3771A1', 'DF6DC684']},
"Week 28": {0: ['04B00F5A', 10122, 'BBF6FF75', 'D42BC2B8'], 1: ['DF6DC684'], 2: ['0C97921F'], 3: ['0AC819FA', '0FDCF4C8', 'AE1E42BD'], 4: [10077], 5: ['FD1C5B4B'], 6: ['3711BB6B'], 7: ['769D0CBD'], 8: ['897AEFCE'], 9: ['BEA81447', 'CFEA783C'], 10: ['8AFF3E6D', '8B3771A1'], 11: ['9175C446']},
"Week 29": {0: ['DF6DC684'], 1: ['0FDCF4C8', 'AE1E42BD'], 2: ['04B00F5A', '0AC819FA', 10122, '769D0CBD', '8AFF3E6D', '8B3771A1', 'BBF6FF75', 'D42BC2B8'], 3: ['0C97921F', 10077, '3711BB6B', '897AEFCE', '9175C446', 'BEA81447', 'CFEA783C', 'FD1C5B4B']},
"Week 30": {0: ['DF6DC684'], 1: ['04B00F5A', '0AC819FA', 10122, '769D0CBD', '8AFF3E6D', '8B3771A1', 'BBF6FF75', 'D42BC2B8'], 2: ['0C97921F', 10077, '3711BB6B', '897AEFCE', '9175C446', 'BEA81447', 'CFEA783C', 'FD1C5B4B'], 3: ['0FDCF4C8', 'AE1E42BD']},
"Week 31": {0: ['DF6DC684'], 1: ['04B00F5A', '0AC819FA', 10122, '769D0CBD', '8AFF3E6D', '8B3771A1', 'BBF6FF75', 'D42BC2B8'], 2: ['0C97921F', 10077, '3711BB6B', '897AEFCE', '9175C446', 'BEA81447', 'CFEA783C', 'FD1C5B4B'], 3: ['0FDCF4C8', 'AE1E42BD']},
"Week 32": {0: ['DF6DC684'], 1: ['04B00F5A', '0AC819FA', 10122, '769D0CBD', '8AFF3E6D', 'AE1E42BD', 'BBF6FF75', 'D42BC2B8'], 2: ['8B3771A1'], 3: ['0FDCF4C8'], 4: ['0C97921F', 10077, '3711BB6B', '897AEFCE', '9175C446', 'BEA81447', 'CFEA783C', 'FD1C5B4B']},
"Week 33": {0: ['0C97921F', 10077, '3711BB6B', '897AEFCE', '9175C446', 'BEA81447', 'CFEA783C', 'FD1C5B4B'], 1: ['DF6DC684'], 2: ['04B00F5A', '0AC819FA', 10122, '769D0CBD', '8AFF3E6D', '8B3771A1', 'BBF6FF75', 'D42BC2B8'], 3: ['0FDCF4C8', 'AE1E42BD']},
"Week 34": {0: ['04B00F5A', 'BBF6FF75', 'D42BC2B8', 'FD1C5B4B'], 1: ['0AC819FA', 10122, '769D0CBD', '897AEFCE', '8AFF3E6D', '9175C446', 'AE1E42BD'], 2: ['8B3771A1'], 3: ['0FDCF4C8'], 4: ['0C97921F', 10077, '3711BB6B', 'BEA81447', 'CFEA783C', 'DF6DC684']}
}

# Extract all unique sensor IDs
all_ids = set()
for week_data in weekly_communities.values():
    for community_ids in week_data.values():
        all_ids.update(map(str, community_ids))  # 确保所有 ID 都转换为字符串

# Convert set to list to guarantee order
all_ids = list(all_ids)

# Initialize an empty DataFrame to store co-occurrence frequencies
co_occurrence_matrix = pd.DataFrame(0, index=all_ids, columns=all_ids)

# Calculate co-occurrence frequency
for week_data in weekly_communities.values():
    for community_ids in week_data.values():
        for id1 in community_ids:
            for id2 in community_ids:
                if id1 != id2:
                    co_occurrence_matrix.at[str(id1), str(id2)] += 1  # 使用字符串形式的 ID

# Converted to probabilities of co-occurrence (percentage)
co_occurrence_matrix = co_occurrence_matrix.div(len(weekly_communities))

# Iterate through each row of co_occurrence_matrix
for sensor_id in co_occurrence_matrix.index:
    # Get the coexistence probability of the current sensor with other sensors and sort it in descending order
    co_occurrence = co_occurrence_matrix.loc[sensor_id].sort_values(ascending=False)

    # Print each sensor and its coexistence probability with other sensors
    print(f"Sensor {sensor_id} co-occurrence probabilities:")
    print(co_occurrence)
    print("\n")


plt.figure(figsize=(10, 8))
sns.heatmap(co_occurrence_matrix, cmap='viridis', annot=True)
plt.title('Co-occurrence Probability of Sensors in the Same Community')
plt.xlabel('Sensor ID')
plt.ylabel('Sensor ID')
plt.show()


print('---------------Prediction ------------------------')
predicted_communities = {0: [10077, '0C97921F', '3711BB6B', '897AEFCE', '9175C446', 'BEA81447', 'CFEA783C', 'FD1C5B4B'], 1: ['DF6DC684'], 2: [10122, '04B00F5A', '0AC819FA', '769D0CBD', '8AFF3E6D', '8B3771A1', 'BBF6FF75', 'D42BC2B8'], 3: ['0FDCF4C8', 'AE1E42BD']}

# Analyze predicted community coexistence probabilities
for community, sensors in predicted_communities.items():
    print(f"Community {community}:")
    for i, sensor1 in enumerate(sensors):
        for sensor2 in sensors[i+1:]:
            prob = co_occurrence_matrix.at[str(sensor1), str(sensor2)]
            print(f"  {sensor1} - {sensor2}: {prob}")
    print("\n")





