import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from aignosi_case.config import PLOT_FIGSIZE, SEABORN_STYLE
from aignosi_case.dataset import load_hourly_data

# seaborn style settings
sns.set_theme(style=SEABORN_STYLE, rc={'figure.figsize': PLOT_FIGSIZE})

# load hourly resampled data using centralized utility
df_hourly = load_hourly_data()

# Find all rows where ALL columns are NaN
missing_mask = df_hourly.isnull().all(axis=1)

# Get the timestamps for missing hours
missing_timestamps = df_hourly.index[missing_mask]

print(f"\nFound {len(missing_timestamps)} total missing hourly data points.")

if len(missing_timestamps) > 0:
    print("Investigating contiguous blocks of missing data...")
    
    # Find where the gaps between missing timestamps are > 1 hour
    time_diffs = missing_timestamps.to_series().diff()
    
    # Find the indices where a new block starts
    start_indices = np.where(time_diffs > pd.Timedelta('1 hour'))[0]    
    all_start_indices = np.insert(start_indices, 0, 0)
    
    end_indices = np.append(start_indices - 1, len(missing_timestamps) - 1)
    
    print("\n--- Missing Data Time Frames ---")
    for start_idx, end_idx in zip(all_start_indices, end_indices):
        start_time = pd.to_datetime(missing_timestamps[start_idx])
        end_time = pd.to_datetime(missing_timestamps[end_idx])
        
        # Calculate duration and count
        count = end_idx - start_idx + 1
        duration = pd.Timedelta(hours=count)
        
        print(f"Block found:")
        print(f"  Start:    {start_time}")
        print(f"  End:      {end_time}")
        print(f"  Count:    {count} missing hours")
        print(f"  Duration: {duration}\n")
        
else:
    print("No missing data blocks found.")