import os
working_dir = os.path.join(os.getcwd(), 'working')
os.makedirs(working_dir, exist_ok=True)

import time
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

try:
    import matplotlib.pyplot as plt  # optional
except Exception:
    plt = None

from lenskit import batch
from lenskit.algorithms.als import ImplicitMF
from lenskit.algorithms.item_knn import ItemItem
from lenskit.algorithms.basic import Popular

experiment_data = {
    'ml100k': {'metrics': {'train': [], 'val': []}, 'losses': {'train': [], 'val': []}, 'predictions': [], 'ground_truth': []},
    'amazon_vg': {'metrics': {'train': [], 'val': []}, 'losses': {'train': [], 'val': []}, 'predictions': [], 'ground_truth': []},
    'lastfm': {'metrics': {'train': [], 'val': []}, 'losses': {'train': [], 'val': []}, 'predictions': [], 'ground_truth': []},
}


def _standardize_cols(df):
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def load_movielens(path='u.data'):
    df = pd.read_csv(path, sep='\t', names=['user', 'item', 'rating', 'timestamp'])
    df = df[df['rating'] > 3].copy()
    df['user'] = df['user'].astype(str)
    df['item'] = df['item'].astype(str)
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce').fillna(0).astype(np.int64)
    df['rating'] = 1.0
    return df[['user', 'item', 'rating', 'timestamp']]


def load_amazon(path='VideoGames.csv'):
    df = pd.read_csv(path)
    df = _standardize_cols(df)
    user_col = next((c for c in ['reviewerid', 'user_id', 'user', 'reviewer_id'] if c in df.columns), None)
    item_col = next((c for c in ['asin', 'item_id', 'item', 'product_id'] if c in df.columns), None)
    rating_col = next((c for c in ['overall', 'rating', 'score'] if c in df.columns), None)
    time_col = next((c for c in ['unixreviewtime', 'timestamp', 'time'] if c in df.columns), None)
    if user_col is None or item_col is None or rating_col is None:
        raise ValueError('Could not identify Amazon columns')
    df = df[[user_col, item_col, rating_col] + ([time_col] if time_col else [])].copy()
    df = df[pd.to_numeric(df[rating_col], errors='coerce') > 3].copy()
    out = pd.DataFrame({'user': df[user_col].astype(str), 'item': df[item_col].astype(str)})
    out['timestamp'] = pd.to_numeric(df[time_col], errors='coerce').fillna(0).astype(np.int64) if time_col else np.arange(len(out), dtype=np.int64)
    out['rating'] = 1.0
    return out[['user', 'item', 'rating', 'timestamp']]


def load_lastfm(path='user_taggedartists-timestamps.dat'):
    df = pd.read_csv(path, sep='\t')
    df = _standardize_cols(df)
    user_col = next((c for c in ['userid', 'user_id', 'user'] if c in df.columns), None)
    item_col = next((c for c in ['artistid', 'artist_id', 'item', 'artist'] if c in df.columns), None)
    time_col = next((c for c in ['timestamp', 'time'] if c in df.columns), None)
    if user_col is None or item_col is None:
        raise ValueError('Could not identify LastFM columns')
    out = pd.DataFrame({'user': df[user_col].astype(str), 'item': df[item_col].astype(str)})
    out['timestamp'] = pd.to_numeric(df[time_col], errors='coerce').fillna(0).astype(np.int64) if time_col else np.arange(len(out), dtype=np.int64)
    out['rating'] = 1.0
    return out[['user', 'item', 'rating', 'timestamp']]


def core_filter(df, min_uc=5, min_ic=5):
    df = df.copy()
    changed = True
    while changed:
        changed = False
        keep_u = df.groupby('user').size()
        keep_u = keep_u[keep_u >= min_uc].index
        new_df = df[df['user'].isin(keep_u)]
        changed |= (len(new_df) != len(df))
        df = new_df
        keep_i = df.groupby('item').size()
        keep_i = keep_i[keep_i >= min_ic].index
        new_df = df[df['item'].isin(keep_i)]
        changed |= (len(new_df) != len(df))
        df = new_df
    return df.reset_index(drop=True)


def split_user_holdout(df, seed, test_frac=0.2):
    rng = np.random.default_rng(seed)
    train_parts = []
    test_parts = []
    for _, grp in df.groupby('user', sort=False):
        idx = grp.index.to_numpy()
        perm = rng.permutation(idx)
        n_test = max(1, int(np.floor(len(idx) * test_frac)))
        test_idx = perm[:n_test]
        train_idx = perm[n_test:]
        if len(train_idx) == 0:
            train_idx = test_idx[:1]
            test_idx = test_idx[1:]
        train_parts.append(df.loc[train_idx])
        if len(test_idx) > 0:
            test_parts.append(df.loc[test_idx])
    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True) if test_parts else df.iloc[:0].copy()
    return train, test


def build_model(name):
    if name == 'ALS':
        return ImplicitMF(features=50, iterations=10, reg=0.1, random_state=42)
    if name == 'ItemKNN':
        return ItemItem(nnbrs=100, min_nbrs=1)
    if name == 'Pop':
        return Popular()
    raise ValueError(name)


def recommend_df(model, users, n):
    recs = batch.recommend(model, users, n=n)
    if recs is None:
        return pd.DataFrame(columns=['user', 'item', 'rank', 'score'])
    cols = {c.lower(): c for c in recs.columns}
    rename = {}
    for target in ['user', 'item', 'rank', 'score']:
        if target in cols:
            rename[cols[target]] = target
    recs = recs.rename(columns=rename)
    if 'rank' not in recs.columns:
        recs['rank'] = recs.groupby('user').cumcount() + 1
    if 'score' not in recs.columns:
        recs['score'] = np.nan
    return recs[['user', 'item', 'rank', 'score']]


def _rank_metrics(recs, truth, k):
    if recs.empty or not truth:
        return np.nan, np.nan
    users = sorted(set(recs['user']).intersection(truth.keys()))
    if not users:
        return np.nan, np.nan
    p_list = []
    n_list = []
    for u in users:
        topk = recs[recs['user'] == u].sort_values('rank').head(k)['item'].tolist()
        rel = set(truth[u])
        hits = [1 if i in rel else 0 for i in topk]
        p_list.append(sum(hits) / float(k))
        dcg = sum(hit / np.log2(idx + 2) for idx, hit in enumerate(hits))
        ideal = min(len(rel), k)
        idcg = sum(1.0 / np.log2(idx + 2) for idx in range(ideal))
        n_list.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(p_list)), float(np.mean(n_list))


def evaluate_recommender(model, test, k_values=(1, 5, 10)):
    truth = test.groupby('user')['item'].apply(list).to_dict()
    recs = recommend_df(model, list(truth.keys()), n=max(k_values))
    metrics = {}
    for k in k_values:
        p, n = _rank_metrics(recs, truth, k)
        metrics[f'Precision@{k}'] = p
        metrics[f'nDCG@{k}'] = n
    return metrics, recs, truth


def load_dataset(name):
    if name == 'ml100k':
        return load_movielens()
    if name == 'amazon_vg':
        return load_amazon()
    if name == 'lastfm':
        return load_lastfm()
    raise ValueError(name)


datasets = ['ml100k', 'amazon_vg', 'lastfm']
algorithms = ['ALS', 'ItemKNN', 'Pop']
seeds = [11, 22, 33, 44, 55]
all_results = []
start_time = time.time()

for dname in datasets:
    df = core_filter(load_dataset(dname))
    df = df[['user', 'item', 'rating', 'timestamp']].copy()
    np.save(os.path.join(working_dir, f'{dname}_preprocessed.npy'), df.to_records(index=False))
    for seed in seeds:
        train, test = split_user_holdout(df, seed)
        np.save(os.path.join(working_dir, f'{dname}_seed{seed}_train.npy'), train.to_records(index=False))
        np.save(os.path.join(working_dir, f'{dname}_seed{seed}_test.npy'), test.to_records(index=False))
        for alg in algorithms:
            model = build_model(alg)
            model.fit(train[['user', 'item', 'rating']])
            metrics, recs, truth = evaluate_recommender(model, test)
            row = {'dataset': dname, 'seed': seed, 'algorithm': alg, **metrics}
            all_results.append(row)
            experiment_data[dname]['metrics']['val'].append([seed, alg] + [metrics[f'Precision@{k}'] for k in [1, 5, 10]] + [metrics[f'nDCG@{k}'] for k in [1, 5, 10]])
            experiment_data[dname]['predictions'].append(recs.head(1000).to_records(index=False))
            experiment_data[dname]['ground_truth'].append(pd.DataFrame({'user': list(truth.keys()), 'items': list(truth.values())}).to_records(index=False))
            print(f'{dname} seed={seed} alg={alg} ' + ' '.join([f'{k}={v:.4f}' for k, v in metrics.items()]))

results_df = pd.DataFrame(all_results)
results_df.to_csv(os.path.join(working_dir, 'all_results.csv'), index=False)
summary = results_df.groupby(['dataset', 'algorithm']).agg(['mean', 'std'])
print('\nSummary by dataset/algorithm:')
print(summary)

analysis_rows = []
for dname in datasets:
    sub = results_df[results_df['dataset'] == dname]
    for metric in [f'Precision@{k}' for k in [1, 5, 10]] + [f'nDCG@{k}' for k in [1, 5, 10]]:
        piv = sub.pivot(index='seed', columns='algorithm', values=metric)
        for a1, a2 in [('ALS', 'ItemKNN'), ('ALS', 'Pop'), ('ItemKNN', 'Pop')]:
            x = piv[a1].to_numpy()
            y = piv[a2].to_numpy()
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() > 1:
                t, p = stats.ttest_rel(x[mask], y[mask])
                mean_diff = float(np.mean(x[mask] - y[mask]))
            else:
                t, p, mean_diff = np.nan, np.nan, np.nan
            analysis_rows.append([dname, metric, a1, a2, t, p, mean_diff])
analysis_df = pd.DataFrame(analysis_rows, columns=['dataset', 'metric', 'alg1', 'alg2', 't_stat', 'p_value', 'mean_diff'])
analysis_df.to_csv(os.path.join(working_dir, 'statistical_analysis.csv'), index=False)
print('\nStatistical analysis (paired t-tests across seeds):')
print(analysis_df)

np.save(os.path.join(working_dir, 'experiment_data.npy'), experiment_data)
np.save(os.path.join(working_dir, 'results_array.npy'), results_df.to_records(index=False))
np.save(os.path.join(working_dir, 'analysis_array.npy'), analysis_df.to_records(index=False))

if plt is not None:
    for dname in datasets:
        sub = results_df[results_df['dataset'] == dname]
        fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
        metrics = [f'Precision@{k}' for k in [1, 5, 10]] + [f'nDCG@{k}' for k in [1, 5, 10]]
        for ax, metric in zip(axes.flat, metrics):
            data = [sub[sub['algorithm'] == alg][metric].values for alg in algorithms]
            ax.boxplot(data, labels=algorithms)
            ax.set_title(metric)
            ax.grid(True, alpha=0.3)
        fig.suptitle(f'{dname} split-seed sensitivity')
        fig.savefig(os.path.join(working_dir, f'{dname}_seed_sensitivity.png'), dpi=150)
        plt.close(fig)
else:
    print('matplotlib is unavailable; skipping plots.')

print(f'Finished in {time.time() - start_time:.1f}s')