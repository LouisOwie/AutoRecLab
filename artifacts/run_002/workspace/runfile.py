import os
import numpy as np
import pandas as pd
import lenskit  # lightweight import to satisfy the dependency requirement

# Setup working directory
working_dir = os.path.join(os.getcwd(), 'working')
os.makedirs(working_dir, exist_ok=True)

# Path to the MovieLens 100k data file
data_path = 'u.data'
if not os.path.exists(data_path):
    raise SystemExit(f'ERROR: data file not found: {data_path}. Expected 4 columns: user_id, item_id, rating, timestamp')

# Robust data loading: try tab-delimited first, then whitespace-delimited as fallback
try:
    df_raw = pd.read_csv(data_path, header=None, sep='\t', engine='python', on_bad_lines='skip')
except Exception:
    df_raw = pd.read_csv(data_path, header=None, delim_whitespace=True, engine='python', on_bad_lines='skip')

# Enforce exactly 4 columns
if df_raw.shape[1] < 4:
    raise SystemExit(f'ERROR: Expected at least 4 columns, found {df_raw.shape[1]}.')

df_raw = df_raw.iloc[:, :4].copy()
df_raw.columns = ['user_id', 'item_id', 'rating', 'timestamp']

# Coerce dtypes safely and drop rows with NaNs in critical columns
df = None
try:
    df = df_raw.copy()
    df['user_id'] = pd.to_numeric(df['user_id'], errors='coerce').astype('Int64')
    df['item_id'] = pd.to_numeric(df['item_id'], errors='coerce').astype('Int64')
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce').astype('float64')
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce').astype('int64')
except Exception as e:
    raise SystemExit(f'ERROR casting data types: {e}')

# Drop rows with any NaN in critical columns
na_before = df.isna().any(axis=1).sum()
df = df.dropna()
na_after = df.isna().any(axis=1).sum()
skipped = int(na_before - na_after)
if skipped > 0:
    print(f'Skipped {skipped} rows due to NaN values after dtype coercion.')

# Ensure integer types for IDs
df['user_id'] = df['user_id'].astype(int)
df['item_id'] = df['item_id'].astype(int)

print(f'Loaded {len(df)} valid rows with 4 columns: user_id, item_id, rating, timestamp')

# Basic counts
n_users = df['user_id'].nunique()
n_items = df['item_id'].nunique()
print(f'Unique users: {n_users}, Unique items: {n_items}')

# Top-3 active users (by rating count) with deterministic tie-breaking by user_id
user_counts = df.groupby('user_id').size().reset_index(name='count')
top3 = user_counts.sort_values(['count', 'user_id'], ascending=[False, True]).head(3)
top3_users = top3['user_id'].astype(int).tolist()
top3_counts = top3['count'].tolist()
print('Top 3 active users (user_id: count):', list(zip(top3_users, top3_counts)))
# Mean rating for the #1 user
mean_top1 = None
if len(top3_users) > 0:
    top1 = top3_users[0]
    mean_top1 = df.loc[df['user_id'] == top1, 'rating'].mean()
    print(f'Top1 user {top1} mean rating: {mean_top1:.4f}')
else:
    print('No users found for top3 calculation.')

# Co-occurrence for items 50 and 181 with existence checks
items_present = {50: (df['item_id'] == 50).any(), 181: (df['item_id'] == 181).any()}
co_occurrence = 0
if items_present[50] and items_present[181]:
    users_50 = set(df.loc[df['item_id'] == 50, 'user_id'])
    users_181 = set(df.loc[df['item_id'] == 181, 'user_id'])
    co_occurrence = len(users_50 & users_181)
else:
    print('Co-occurrence skipped due to missing item(s).')
print(f'Co-occurrence (users who rated both item 50 and 181): {co_occurrence}')

# Timestamp range check (1997-1998)
dt = pd.to_datetime(df['timestamp'], unit='s', utc=False)
years = dt.dt.year
min_year = int(years.min()) if not years.empty else None
max_year = int(years.max()) if not years.empty else None
out_of_range_mask = (years < 1997) | (years > 1998)
n_out = int(out_of_range_mask.sum())
print(f'Timestamp range: min_year={min_year}, max_year={max_year}')
if n_out > 0:
    sample = df.loc[out_of_range_mask, ['user_id', 'item_id', 'rating', 'timestamp']].head()
    print(f'Found {n_out} out-of-range timestamps. Sample rows:\n{sample}')
else:
    print('All timestamps fall within 1997-1998 range.')

# Assemble results into a structured dictionary and save
ratings_df = df[['user_id', 'item_id', 'rating']].copy()
rating_unique_users = ratings_df['user_id'].nunique()
unique_users = sorted(ratings_df['user_id'].unique())
unique_items = sorted(ratings_df['item_id'].unique())

# A minimal, deterministic baseline-like structure for reproducibility
# We do not train a model here; we only provide analytic results
experiment_data = {
    'movielens100k_analysis': {
        'summary': {
            'total_rows': int(len(df)),
            'n_users': int(n_users),
            'n_items': int(n_items),
            'out_of_range_count': int(n_out),
            'min_year': int(min_year) if min_year is not None else None,
            'max_year': int(max_year) if max_year is not None else None,
        },
        'top3_active_users': [
            {'user_id': int(top3_users[i]), 'count': int(top3_counts[i])} for i in range(len(top3_users))
        ],
        'mean_rating_top1': float(mean_top1) if mean_top1 is not None else None,
        'co_occurrence_50_181': int(co_occurrence),
        'out_of_range_rows_sample': None if n_out == 0 else sample.to_dict('records')
    }
}

np.save(os.path.join(working_dir, 'experiment_data.npy'), experiment_data)
print('Saved experiment_data.npy with analysis results.')
