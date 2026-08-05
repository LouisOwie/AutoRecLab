import os
working_dir = os.path.join(os.getcwd(), 'working')
os.makedirs(working_dir, exist_ok=True)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

try:
    import lenskit  # optional, kept only for environment reporting / compatibility
    HAS_LENSKIT = True
except Exception:
    lenskit = None
    HAS_LENSKIT = False

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    plt = None
    HAS_MPL = False

experiment_data = {
    'movielens100k': {
        'metrics': {'train': [], 'val': []},
        'losses': {'train': [], 'val': []},
        'predictions': [],
        'ground_truth': [],
        'timestamps': [],
        'top_users': [],
        'summary': {}
    }
}

possible_paths = [
    'u.data',
    './u.data',
    './ml-100k/u.data',
    './data/u.data',
    './data/ml-100k/u.data'
]

data_path = next((p for p in possible_paths if os.path.exists(p)), None)
if data_path is None:
    raise FileNotFoundError('Could not find u.data in expected locations.')

cols = ['user', 'item', 'rating', 'timestamp']
ratings = pd.read_csv(data_path, sep='\t', names=cols, header=None)
ratings = ratings.astype({'user': np.int64, 'item': np.int64, 'rating': np.float64, 'timestamp': np.int64})
ratings = ratings.sort_values(['user', 'item', 'timestamp']).reset_index(drop=True)

n_users = int(ratings['user'].nunique())
n_items = int(ratings['item'].nunique())

user_counts = ratings.groupby('user').size().sort_values(ascending=False)
top3 = user_counts.head(3)
most_active_user = int(top3.index[0])
most_active_mean = float(ratings.loc[ratings['user'] == most_active_user, 'rating'].mean())

users_item_50 = set(ratings.loc[ratings['item'] == 50, 'user'].unique().tolist())
users_item_181 = set(ratings.loc[ratings['item'] == 181, 'user'].unique().tolist())
cooccurrence = int(len(users_item_50 & users_item_181))

expected_start = int(pd.Timestamp('1997-09-20 00:00:00', tz='UTC').timestamp())
expected_end = int(pd.Timestamp('1998-04-22 23:59:59', tz='UTC').timestamp())
min_ts = int(ratings['timestamp'].min())
max_ts = int(ratings['timestamp'].max())
outside = ratings[(ratings['timestamp'] < expected_start) | (ratings['timestamp'] > expected_end)].copy()

train_df, val_df = train_test_split(ratings[['user', 'item', 'rating']], test_size=0.2, random_state=42)
item_means = train_df.groupby('item')['rating'].mean()
global_mean = float(train_df['rating'].mean())
val_pred = val_df['item'].map(item_means).fillna(global_mean).to_numpy(dtype=np.float64)
val_true = val_df['rating'].to_numpy(dtype=np.float64)
train_pred = train_df['item'].map(item_means).fillna(global_mean).to_numpy(dtype=np.float64)
train_true = train_df['rating'].to_numpy(dtype=np.float64)
val_rmse = float(np.sqrt(mean_squared_error(val_true, val_pred)))
train_rmse = float(np.sqrt(mean_squared_error(train_true, train_pred)))

epoch = 1
experiment_data['movielens100k']['metrics']['train'].append({'epoch': epoch, 'rmse': train_rmse})
experiment_data['movielens100k']['metrics']['val'].append({'epoch': epoch, 'rmse': val_rmse})
experiment_data['movielens100k']['losses']['train'].append({'epoch': epoch, 'loss': train_rmse})
experiment_data['movielens100k']['losses']['val'].append({'epoch': epoch, 'loss': val_rmse})
experiment_data['movielens100k']['predictions'] = val_pred.tolist()
experiment_data['movielens100k']['ground_truth'] = val_true.tolist()
experiment_data['movielens100k']['timestamps'] = ratings['timestamp'].to_numpy(dtype=np.int64)
experiment_data['movielens100k']['top_users'] = [(int(uid), int(cnt)) for uid, cnt in top3.items()]
experiment_data['movielens100k']['summary'] = {
    'unique_users': n_users,
    'unique_items': n_items,
    'most_active_user': most_active_user,
    'most_active_user_mean_rating': most_active_mean,
    'item50_item181_cooccurrence': cooccurrence,
    'min_timestamp': min_ts,
    'max_timestamp': max_ts,
    'outside_expected_count': int(len(outside)),
    'lenskit_available': HAS_LENSKIT,
    'data_path': data_path
}

print(f'Epoch {epoch}: validation_loss = {val_rmse:.4f}')
print(f'Unique user_ids: {n_users}')
print(f'Unique item_ids: {n_items}')
print('Top 3 most active users by rating count:')
for uid, cnt in top3.items():
    print(f'  user {int(uid)}: {int(cnt)} ratings')
print(f'Mean rating for most active user #{most_active_user}: {most_active_mean:.4f}')
print(f'Users who rated both item 50 and item 181: {cooccurrence}')
print(f'Timestamp min: {min_ts} ({pd.to_datetime(min_ts, unit="s", utc=True)})')
print(f'Timestamp max: {max_ts} ({pd.to_datetime(max_ts, unit="s", utc=True)})')
if len(outside) == 0:
    print('No timestamps fall outside the expected 1997-09-20 to 1998-04-22 range.')
else:
    print(f'Timestamps outside expected range: {len(outside)}')
    print(outside[['user', 'item', 'rating', 'timestamp']].head(10).to_string(index=False))

np.save(os.path.join(working_dir, 'movielens100k_val_predictions.npy'), val_pred)
np.save(os.path.join(working_dir, 'movielens100k_val_ground_truth.npy'), val_true)
np.save(os.path.join(working_dir, 'movielens100k_timestamps.npy'), ratings['timestamp'].to_numpy(dtype=np.int64))
np.save(os.path.join(working_dir, 'movielens100k_top3_users.npy'), np.array([(int(uid), int(cnt)) for uid, cnt in top3.items()], dtype=np.int64))
np.save(os.path.join(working_dir, 'movielens100k_summary.npy'), np.array([
    n_users, n_items, most_active_user, cooccurrence, min_ts, max_ts, int(len(outside))
], dtype=np.int64))
np.save(os.path.join(working_dir, 'movielens100k_most_active_user_mean_rating.npy'), np.array([most_active_mean], dtype=np.float64))
if len(outside) > 0:
    np.save(os.path.join(working_dir, 'movielens100k_outside_timestamps.npy'), outside[['user', 'item', 'rating', 'timestamp']].to_numpy())
else:
    np.save(os.path.join(working_dir, 'movielens100k_outside_timestamps.npy'), np.empty((0, 4), dtype=np.float64))

if HAS_MPL:
    plt.figure(figsize=(8, 4))
    plt.hist(ratings['timestamp'].to_numpy(dtype=np.int64), bins=50, color='steelblue', edgecolor='black')
    plt.title('MovieLens100k Timestamp Distribution')
    plt.xlabel('Unix Timestamp')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, 'movielens100k_timestamp_distribution.png'), dpi=150)
    plt.close()
else:
    print('matplotlib not available; skipping timestamp plot.')

np.save(os.path.join(working_dir, 'experiment_data.npy'), experiment_data, allow_pickle=True)
print(f'Validation RMSE: {val_rmse:.4f}')
print(f'LensKit available: {HAS_LENSKIT}')