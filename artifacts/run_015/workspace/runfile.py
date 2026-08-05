import os
working_dir = os.path.join(os.getcwd(), 'working')
os.makedirs(working_dir, exist_ok=True)

import numpy as np
import pandas as pd
from pathlib import Path

experiment_data = {
    'movielens100k': {
        'metrics': {'train': [], 'val': []},
        'losses': {'train': [], 'val': []},
        'predictions': [],
        'ground_truth': [],
        'timestamps': [],
        'details': []
    }
}

candidate_paths = [Path('u.data'), Path('./u.data'), Path('data/u.data'), Path('ml-100k/u.data'), Path('./ml-100k/u.data')]
file_path = next((p for p in candidate_paths if p.exists()), None)
if file_path is None:
    raise FileNotFoundError('Could not locate u.data in common paths.')

cols = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv(file_path, sep='\t', names=cols, engine='python')

unique_users = int(df['user_id'].nunique())
unique_items = int(df['item_id'].nunique())

user_counts = df.groupby('user_id').size().reset_index(name='rating_count')
user_counts = user_counts.sort_values(['rating_count', 'user_id'], ascending=[False, True])
top3 = user_counts.head(3)
most_active_user = int(top3.iloc[0]['user_id'])
most_active_user_mean_rating = float(df.loc[df['user_id'] == most_active_user, 'rating'].mean())

users_item50 = set(df.loc[df['item_id'] == 50, 'user_id'].unique())
users_item181 = set(df.loc[df['item_id'] == 181, 'user_id'].unique())
cooccur_users = int(len(users_item50 & users_item181))

start_ts = pd.Timestamp('1997-01-01', tz='UTC').timestamp()
end_ts = pd.Timestamp('1999-01-01', tz='UTC').timestamp()
outside_mask = (df['timestamp'] < start_ts) | (df['timestamp'] >= end_ts)
outside_df = df.loc[outside_mask].copy()
outside_count = int(outside_df.shape[0])

experiment_data['movielens100k']['metrics']['train'].append({
    'unique_users': unique_users,
    'unique_items': unique_items,
    'top3_users': [(int(r.user_id), int(r.rating_count)) for r in top3.itertuples(index=False)],
    'most_active_user': most_active_user,
    'most_active_user_mean_rating': most_active_user_mean_rating,
    'cooccur_item50_item181': cooccur_users,
    'timestamps_outside_expected_range': outside_count,
})
experiment_data['movielens100k']['timestamps'] = df['timestamp'].to_numpy()
experiment_data['movielens100k']['details'].append({
    'file_path': str(file_path),
    'rows': int(len(df)),
    'rating_min': float(df['rating'].min()),
    'rating_max': float(df['rating'].max()),
})

np.save(os.path.join(working_dir, 'experiment_data.npy'), experiment_data, allow_pickle=True)
np.save(os.path.join(working_dir, 'movielens100k_timestamps.npy'), df['timestamp'].to_numpy())
np.save(os.path.join(working_dir, 'movielens100k_user_counts.npy'), user_counts['rating_count'].to_numpy())

print(f'File: {file_path}')
print(f'Total unique user_ids: {unique_users}')
print(f'Total unique item_ids: {unique_items}')
print('Top 3 most active users (user_id, rating_count):')
for r in top3.itertuples(index=False):
    print(f'  {int(r.user_id)}: {int(r.rating_count)}')
print(f'#1 most active user: {most_active_user}')
print(f'Mean rating for user {most_active_user}: {most_active_user_mean_rating:.4f}')
print(f'Users who rated Item 50: {len(users_item50)}')
print(f'Users who rated Item 181: {len(users_item181)}')
print(f'Users who rated both Item 50 and Item 181: {cooccur_users}')
print(f'Timestamps outside expected 1997-1998 range: {outside_count}')
if outside_count:
    print('Examples of out-of-range timestamps:')
    print(outside_df.head(10).to_string(index=False))
else:
    print('No out-of-range timestamps found.')
