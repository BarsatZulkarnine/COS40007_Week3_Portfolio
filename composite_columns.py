import pandas as pd
import numpy as np

# Define the column mapping based on your student number ending
column_mapping = {
    0: (["L5 x", "L5 y", "L5 z"], ["T12 x", "T12 y", "T12 z"]),
}

# Load the combined dataset from Step 1
combined_data = pd.read_csv('combined_dataset.csv')

def compute_composites(data):
    data = data.copy()  # Make a copy of the data to avoid the warning
    data['RMS_xy'] = np.sqrt((data.iloc[:, 1] ** 2 + data.iloc[:, 2] ** 2) / 2)
    data['RMS_yz'] = np.sqrt((data.iloc[:, 2] ** 2 + data.iloc[:, 3] ** 2) / 2)
    data['RMS_zx'] = np.sqrt((data.iloc[:, 3] ** 2 + data.iloc[:, 1] ** 2) / 2)
    data['RMS_xyz'] = np.sqrt((data.iloc[:, 1] ** 2 + data.iloc[:, 2] ** 2 + data.iloc[:, 3] ** 2) / 3)
    data['Roll'] = np.degrees(np.arctan2(data.iloc[:, 2], np.sqrt(data.iloc[:, 1] ** 2 + data.iloc[:, 3] ** 2)))
    data['Pitch'] = np.degrees(np.arctan2(data.iloc[:, 1], np.sqrt(data.iloc[:, 2] ** 2 + data.iloc[:, 3] ** 2)))
    data['RMS_xy_2'] = np.sqrt((data.iloc[:, 4] ** 2 + data.iloc[:, 5] ** 2) / 2)
    data['RMS_yz_2'] = np.sqrt((data.iloc[:, 5] ** 2 + data.iloc[:, 6] ** 2) / 2)
    data['RMS_zx_2'] = np.sqrt((data.iloc[:, 6] ** 2 + data.iloc[:, 4] ** 2) / 2)
    data['RMS_xyz_2'] = np.sqrt((data.iloc[:, 4] ** 2 + data.iloc[:, 5] ** 2 + data.iloc[:, 6] ** 2) / 3)
    data['Roll_2'] = np.degrees(np.arctan2(data.iloc[:, 5], np.sqrt(data.iloc[:, 4] ** 2 + data.iloc[:, 6] ** 2)))
    data['Pitch_2'] = np.degrees(np.arctan2(data.iloc[:, 4], np.sqrt(data.iloc[:, 5] ** 2 + data.iloc[:, 6] ** 2)))
    return data
# Select the column sets based on your student number ending
column_set_1 = compute_composites(combined_data[['Frame'] + column_mapping[0][0]])
column_set_2 = compute_composites(combined_data[['Frame'] + column_mapping[0][1]])

# Merge computed columns back into the main dataset
composite_columns = pd.concat([column_set_1, column_set_2], axis=1)
combined_data = pd.concat([combined_data, composite_columns], axis=1)

# Drop any duplicate Frame columns
combined_data = combined_data.loc[:, ~combined_data.columns.duplicated()]

# Save the new dataset with composite columns
combined_data.to_csv('combined_with_composites.csv', index=False)

print("Composite columns created and saved to 'combined_with_composites.csv'.")
