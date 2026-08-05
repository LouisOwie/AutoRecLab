import os
working_dir = os.path.join(os.getcwd(), 'working')
os.makedirs(working_dir, exist_ok=True)

import numpy as np
import pandas as pd

# Optional lenskit import to satisfy research environment preference, but the analysis does not depend on it.
try:
    import lenskit  # noqa: F401
except Exception:
    lenskit = None

experiment_data = {
    'movielens100k': {
        'metrics': {'train': [], 'val': []},
        'losses': {'train': [], 'val': []},
        'predictions': [],
        'ground_truth': [],
        'extra': {}
    }
}

candidate_paths = [
    os.path.join(os.getcwd(), 'u.data'),
    os.path.join(os.getcwd(), 'ml-100k', 'u.data'),
    os.path.join(working_dir, 'u.data'),
]
data_path = None
for p in candidate_paths:
    if os.path.exists(p):
        data_path = p
        break
if data_path is None:
    raise FileNotFoundError('Could not find u.data in expected locations.')

ratings = pd.read_csv(data_path, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'], engine='python')
ratings['user_id'] = ratings['user_id'].astype(np.int64)
ratings['item_id'] = ratings['item_id'].astype(np.int64)
ratings['rating'] = ratings['rating'].astype(np.float32)
ratings['timestamp'] = ratings['timestamp'].astype(np.int64)

unique_users = int(ratings['user_id'].nunique())
unique_items = int(ratings['item_id'].nunique())
user_counts = ratings.groupby('user_id').size().sort_values(ascending=False)
top3_users = user_counts.head(3)
most_active_user = int(top3_users.index[0])
most_active_user_mean_rating = float(ratings.loc[ratings['user_id'] == most_active_user, 'rating'].mean())

users_item50 = set(ratings.loc[ratings['item_id'] == 50, 'user_id'].tolist())
users_item181 = set(ratings.loc[ratings['item_id'] == 181, 'user_id'].tolist())
cooccurrence_50_181 = int(len(users_item50 & users_item181))

min_ts = int(ratings['timestamp'].min())
max_ts = int(ratings['timestamp'].max())
expected_min = 874724710
expected_max = 893286638
out_of_range = ratings[(ratings['timestamp'] < expected_min) | (ratings['timestamp'] > expected_max)]
out_of_range_count = int(len(out_of_range))

experiment_data['movielens100k']['extra'] = {
    'unique_users': unique_users,
    'unique_items': unique_items,
    'top3_users': [(int(uid), int(cnt)) for uid, cnt in top3_users.items()],
    'most_active_user': most_active_user,
    'most_active_user_mean_rating': most_active_user_mean_rating,
    'item50_users': int(len(users_item50)),
    'item181_users': int(len(users_item181)),
    'cooccurrence_50_181': cooccurrence_50_181,
    'timestamp_min': min_ts,
    'timestamp_max': max_ts,
    'expected_timestamp_min': expected_min,
    'expected_timestamp_max': expected_max,
    'out_of_range_count': out_of_range_count,
}
experiment_data['movielens100k']['metrics']['val'].append({
    'timestamp_min': min_ts,
    'timestamp_max': max_ts,
    'out_of_range_count': out_of_range_count,
})

np.save(os.path.join(working_dir, 'experiment_data.npy'), experiment_data)
np.save(os.path.join(working_dir, 'movielens100k_top3_user_counts.npy'), top3_users.values.astype(np.int64))
np.save(os.path.join(working_dir, 'movielens100k_top3_user_ids.npy'), top3_users.index.values.astype(np.int64))
np.save(os.path.join(working_dir, 'movielens100k_timestamp_flags.npy'), np.array([min_ts, max_ts, expected_min, expected_max, out_of_range_count], dtype=np.int64))

# Optional visualization: only if matplotlib is available.
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    plt.bar([str(u) for u in top3_users.index], top3_users.values)
    plt.xlabel('User ID')
    plt.ylabel('Rating Count')
    plt.title('MovieLens100k: Top 3 Most Active Users')
    plt.tight_layout()
    plot_path = os.path.join(working_dir, 'movielens100k_top3_active_users.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
except Exception:
    pass

print(f'Unique users: {unique_users}')
print(f'Unique items: {unique_items}')
print('Top 3 most active users (user_id, rating_count):')
for uid, cnt in top3_users.items():
    print(f'  {int(uid)}: {int(cnt)}')
print(f'Most active user {most_active_user} mean rating: {most_active_user_mean_rating:.4f}')
print(f'Users who rated both item 50 and item 181: {cooccurrence_50_181}')
print(f'Timestamp range observed: [{min_ts}, {max_ts}]')
print(f'Expected timestamp range: [{expected_min}, {expected_max}]')
print(f'Out-of-range timestamps count: {out_of_range_count}')
if out_of_range_count > 0:
    print('Example out-of-range rows:')
    print(out_of_range.head().to_string(index=False))
