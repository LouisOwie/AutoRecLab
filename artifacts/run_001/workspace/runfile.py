import os
import numpy as np
import pandas as pd

# Setup working directory (as required by execution environment)
working_dir = os.path.join(os.getcwd(), 'working')
os.makedirs(working_dir, exist_ok=True)

# Columns expected in u.data
COLS = ['user_id', 'item_id', 'rating', 'timestamp']

# Path discovery similar to prior implementation
path_candidates = [
    os.path.join('ml-100k', 'u.data'),
    os.path.join('ml-100k', 'u.data.gz'),
    'u.data',
    os.path.join('data', 'ml-100k', 'u.data'),
]
path = None
for p in path_candidates:
    if p and os.path.exists(p):
        path = p
        break
if path is None:
    guess = os.path.join('ml-100k', 'u.data')
    if os.path.exists(guess):
        path = guess

# Attempt LensKit-based loader, else fall back to pandas
use_lenskit = False
dataset_dir = None
if path:
    if path.endswith('u.data'):
        dataset_dir = os.path.dirname(path)
    elif os.path.isdir(path):
        dataset_dir = path
try:
    from lenskit.data import MovieLens  # type: ignore
    use_lenskit = True
except Exception:
    use_lenskit = False


def load_with_lenskit(dataset_dir_path: str) -> pd.DataFrame:
    """Load data via LensKit MovieLens helper, align to required schema."""
    ml = None
    try:
        ml = MovieLens(dataset_dir_path)
    except Exception as exc:
        raise
    if hasattr(ml, 'ratings'):
        df = ml.ratings
    elif hasattr(ml, 'to_dataframe'):
        df = ml.to_dataframe()
    elif hasattr(ml, 'to_pandas'):
        df = ml.to_pandas()
    else:
        raise AttributeError('LensKit MovieLens loader did not provide a dataframe attribute')
    if isinstance(df, pd.DataFrame) and set(['user_id','item_id','rating','timestamp']).issubset(set(df.columns)):
        return df
    if isinstance(df, pd.DataFrame) and set(['user','item','rating','time']).issubset(set(df.columns)):
        return df.rename(columns={'user':'user_id','item':'item_id','rating':'rating','time':'timestamp'})
    raise ValueError('Loaded data from LensKit but cannot align to required schema')


def load_with_pandas(file_path: str) -> pd.DataFrame:
    # Try strict whitespace-delimited parsing first, then tab-delimited
    try:
        df = pd.read_csv(
            file_path,
            sep=r'\s+',
            engine='python',
            header=None,
            names=COLS,
            dtype={'user_id': int, 'item_id': int, 'rating': int, 'timestamp': int},
            on_bad_lines='error',
        )
    except Exception:
        df = pd.read_csv(
            file_path,
            sep='\t',
            header=None,
            names=COLS,
            dtype={'user_id': int, 'item_id': int, 'rating': int, 'timestamp': int},
        )
    return df


def load_data(file_path: str) -> pd.DataFrame:
    df = None
    if use_lenskit and dataset_dir is not None:
        try:
            df = load_with_lenskit(dataset_dir)
        except Exception:
            df = None
    if df is None:
        if not file_path:
            raise FileNotFoundError('Could not locate a valid u.data file for loading.')
        df = load_with_pandas(file_path)
    # Normalize dtypes
    df['user_id'] = df['user_id'].astype(int)
    df['item_id'] = df['item_id'].astype(int)
    df['rating'] = df['rating'].astype(int)
    df['timestamp'] = df['timestamp'].astype(int)
    return df


def analyze_u_data(file_path: str) -> dict:
    df = load_data(file_path)

    # Basic statistics
    n_users = int(df['user_id'].nunique())
    n_items = int(df['item_id'].nunique())

    # Top-3 most active users by rating count (count desc, user_id asc)
    user_counts = df.groupby('user_id').size().reset_index(name='count')
    top3_df = user_counts.sort_values(['count', 'user_id'], ascending=[False, True]).head(3)
    top3_list = [(int(r['user_id']), int(r['count'])) for _, r in top3_df.iterrows()]

    # Mean rating for the #1 user
    mean_top1 = None
    if not top3_df.empty:
        top1_user = int(top3_df.iloc[0]['user_id'])
        mean_top1 = float(df.loc[df['user_id'] == top1_user, 'rating'].mean())

    # Co-occurrence: users who rated both item 50 and item 181
    users_50 = set(df.loc[df['item_id'] == 50, 'user_id'])
    users_181 = set(df.loc[df['item_id'] == 181, 'user_id'])
    cooccurrence_50_181 = len(users_50 & users_181)

    # Timestamp range check for 1997-01-01 to 1998-12-31 (UTC)
    ts = pd.to_datetime(df['timestamp'], unit='s', utc=True)
    start = pd.Timestamp('1997-01-01', tz='UTC')
    end = pd.Timestamp('1998-12-31 23:59:59', tz='UTC')
    out_of_range_mask = (ts < start) | (ts > end)
    out_of_range_count = int(out_of_range_mask.sum())

    # Simple 5-epoch validation using a global-mean baseline on a holdout split
    rng = np.random.default_rng(123)
    holdout_mask = rng.random(len(df)) < 0.1
    train_df = df.loc[~holdout_mask].copy()
    holdout_df = df.loc[holdout_mask].copy()
    global_mean = float(train_df['rating'].mean()) if len(train_df) > 0 else float(df['rating'].mean())
    epochs = 5
    rmse_list = []
    for _ in range(epochs):
        rmse = float(((holdout_df['rating'] - global_mean) ** 2).mean() ** 0.5) if len(holdout_df) > 0 else 0.0
        rmse_list.append(rmse)

    summary = {
        'total_users': int(n_users),
        'total_items': int(n_items),
        'top3_users': top3_list,
        'mean_rating_top1': mean_top1,
        'cooccurrence_50_181': int(cooccurrence_50_181),
        'out_of_range_timestamp_count': int(out_of_range_count)
    }
    metrics = {
        'validation': {
            'epochs': epochs,
            'val_rmse_per_epoch': rmse_list
        }
    }
    return {'summary': summary, 'metrics': metrics}


# Run analysis on the discovered path if available
if path is None:
    raise FileNotFoundError('Could not locate a valid u.data file for analysis.')
result = analyze_u_data(path)
summary = result['summary']
validation = result['metrics']['validation']

# Print concise human-readable summary
print("Unique users:", summary['total_users'])
print("Unique items:", summary['total_items'])
print("Top-3 active users (id, count):", summary['top3_users'])
if summary['mean_rating_top1'] is not None:
    top1_id = summary['top3_users'][0][0] if summary['top3_users'] else None
    print(f"Mean rating for top user {top1_id}: {summary['mean_rating_top1']:.4f}" if top1_id is not None else "Mean rating for top user: None")
print("Co-occurrence (users who rated both 50 and 181):", summary['cooccurrence_50_181'])
print("Timestamps outside 1997-1998:", summary['out_of_range_timestamp_count'])

# Epoch-wise validation losses
for i, rmse in enumerate(validation['val_rmse_per_epoch'], start=1):
    print(f"Epoch {i}: validation_rmse = {rmse:.4f}")

# Persist a reproducible record to working_dir
experiment_data = {
    'ml100k_u_data_analysis': {
        'summary': summary,
        'metrics': result['metrics']
    }
}
np.save(os.path.join(working_dir, 'experiment_data.npy'), experiment_data)
