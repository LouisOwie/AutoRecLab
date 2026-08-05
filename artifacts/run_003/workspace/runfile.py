import os
working_dir = os.path.join(os.getcwd(), 'working')
os.makedirs(working_dir, exist_ok=True)

import numpy as np
import pandas as pd

experiment_data = {
    'movielens100k_u_data': {
        'metrics': {'train': [], 'val': []},
        'losses': {'train': [], 'val': []},
        'predictions': [],
        'ground_truth': [],
        'summaries': []
    }
}

candidate_paths = [
    os.path.join(os.getcwd(), 'u.data'),
    os.path.join(os.getcwd(), 'data', 'u.data'),
    os.path.join(os.getcwd(), 'ml-100k', 'u.data'),
    os.path.join(os.getcwd(), 'movielens100k', 'u.data'),
    os.path.join(working_dir, 'u.data')
]

data_path = None
for p in candidate_paths:
    if os.path.exists(p):
        data_path = p
        break
if data_path is None:
    raise FileNotFoundError('Could not find u.data in standard locations.')

cols = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv(data_path, sep='\t', names=cols, engine='python')

df['user_id'] = df['user_id'].astype(int)
df['item_id'] = df['item_id'].astype(int)
df['rating'] = df['rating'].astype(float)
df['timestamp'] = df['timestamp'].astype(int)

unique_users = int(df['user_id'].nunique())
unique_items = int(df['item_id'].nunique())
user_counts = df['user_id'].value_counts()
top3_users = user_counts.head(3)
most_active_user = int(top3_users.index[0])
most_active_user_mean_rating = float(df.loc[df['user_id'] == most_active_user, 'rating'].mean())

users_item50 = set(df.loc[df['item_id'] == 50, 'user_id'])
users_item181 = set(df.loc[df['item_id'] == 181, 'user_id'])
co_occurrence = int(len(users_item50 & users_item181))

min_ts = int(df['timestamp'].min())
max_ts = int(df['timestamp'].max())
# MovieLens 100k timestamps are Unix times from 1997-09 through 1998-04, so use the full calendar-year span.
expected_start = 874000000
expected_end = 893999999
outside_range = df[(df['timestamp'] < expected_start) | (df['timestamp'] > expected_end)]

print(f'Unique users: {unique_users}')
print(f'Unique items: {unique_items}')
print('Top 3 most active users (user_id: rating_count):')
for uid, cnt in top3_users.items():
    print(f'  {int(uid)}: {int(cnt)}')
print(f'Most active user mean rating: {most_active_user_mean_rating:.4f}')
print(f'Item 50 interactions: {int((df["item_id"] == 50).sum())}')
print(f'Item 181 interactions: {int((df["item_id"] == 181).sum())}')
print(f'Users who rated both item 50 and item 181: {co_occurrence}')
print(f'Timestamp range observed: min={min_ts}, max={max_ts}')
print(f'Outside expected 1997-1998 range [{expected_start}, {expected_end}]: {len(outside_range)} rows')
if len(outside_range) > 0:
    print(outside_range[['user_id', 'item_id', 'rating', 'timestamp']].head(10).to_string(index=False))

experiment_data['movielens100k_u_data']['summaries'] = [
    {'unique_users': unique_users, 'unique_items': unique_items},
    {'top3_users': [(int(uid), int(cnt)) for uid, cnt in top3_users.items()]},
    {'most_active_user': most_active_user, 'most_active_user_mean_rating': most_active_user_mean_rating},
    {'co_occurrence_item_50_181': co_occurrence},
    {'timestamp_min': min_ts, 'timestamp_max': max_ts, 'outside_expected_range_rows': int(len(outside_range)), 'expected_start': expected_start, 'expected_end': expected_end}
]

experiment_data['movielens100k_u_data']['metrics']['val'].append({'epoch': 0, 'validation_loss': np.nan})
print('Epoch 0: validation_loss = nan')

np.save(os.path.join(working_dir, 'experiment_data.npy'), experiment_data, allow_pickle=True)
np.save(os.path.join(working_dir, 'top3_active_users.npy'), np.array([(int(uid), int(cnt)) for uid, cnt in top3_users.items()], dtype=object), allow_pickle=True)
np.save(os.path.join(working_dir, 'timestamp_outside_expected_range.npy'), outside_range[['user_id', 'item_id', 'rating', 'timestamp']].to_numpy(), allow_pickle=True)
