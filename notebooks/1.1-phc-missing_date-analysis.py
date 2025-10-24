from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# seaborn style settings
sns.set_theme(style="whitegrid", rc={'figure.figsize':(12,6)})

# load dataset and setting first column as datetime and index
csv_filename = 'MiningProcess_Flotation_Plant_Database.csv'
workspace_root = Path('/home/pedrocosme/aignosi/aignosi-case')  
data_path = workspace_root / 'data' / 'raw' / csv_filename

df = pd.read_csv(data_path, parse_dates=[0], index_col=0, decimal=',')

# downsampling to hourly frequency by aggregating 20-second data using mean over each hour
df_hourly = df.resample('h').mean()

# --- New Code to Find Missing Data Blocks ---

# 1. Find all rows where ALL columns are NaN
#    (This indicates a full shutdown, as suggested by your output)
missing_mask = df_hourly.isnull().all(axis=1)

# 2. Get the timestamps for these missing hours
missing_timestamps = df_hourly.index[missing_mask]

print(f"\nFound {len(missing_timestamps)} total missing hourly data points.")

if len(missing_timestamps) > 0:
    print("Investigating contiguous blocks of missing data...")
    
    # 3. Find where the gaps between missing timestamps are > 1 hour
    #    This indicates the start of a new *block* of missing data
    time_diffs = missing_timestamps.to_series().diff()
    
    # Find the indices where a new block starts
    # A new block starts where the diff is not 1 hour, or at the very beginning (index 0)
    start_indices = np.where(time_diffs > pd.Timedelta('1 hour'))[0]    
    all_start_indices = np.insert(start_indices, 0, 0)
    
    # The end indices are the ones *before* the new starts, plus the very last item
    end_indices = np.append(start_indices - 1, len(missing_timestamps) - 1)
    
    # 4. Now, loop through these start/end indices to print the blocks
    print("\n--- Missing Data Time Frames ---")
    for start_idx, end_idx in zip(all_start_indices, end_indices):
        start_time = pd.to_datetime(missing_timestamps[start_idx])
        end_time = pd.to_datetime(missing_timestamps[end_idx])
        
        # Calculate duration and count
        # Use the count of hourly entries to avoid index/timedelta type issues
        count = end_idx - start_idx + 1
        duration = pd.Timedelta(hours=count)
        
        print(f"Block found:")
        print(f"  Start:    {start_time}")
        print(f"  End:      {end_time}")
        print(f"  Count:    {count} missing hours")
        print(f"  Duration: {duration}\n")
        
else:
    print("No missing data blocks found.")