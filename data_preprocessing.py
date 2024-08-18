
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

data = pd.read_csv('combined_with_composites.csv')

def compute_features(df):
    features = {}
    for col in df.columns[1:-1]:  
        features[f'{col}_mean'] = df[col].mean()
        features[f'{col}_std'] = df[col].std()
        features[f'{col}_min'] = df[col].min()
        features[f'{col}_max'] = df[col].max()
        features[f'{col}_auc'] = np.trapz(df[col])
        features[f'{col}_peaks'] = len(find_peaks(df[col])[0])
    return pd.DataFrame([features])

num_frames = 60
features_list = []

for i in range(0, len(data), num_frames):
    frame_data = data.iloc[i:i + num_frames]
    feature_data = compute_features(frame_data)
    feature_data['class'] = frame_data['class'].iloc[0]
    features_list.append(feature_data)

features_df = pd.concat(features_list, axis=0)

features_df.to_csv('processed_features.csv', index=False)

print("Features computed and saved to 'processed_features.csv'.")
