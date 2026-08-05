import os
working_dir = os.path.join(os.getcwd(), 'working')
os.makedirs(working_dir, exist_ok=True)

import math
import numpy as np
import pandas as pd
from scipy import stats

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from lenskit import topn
from lenskit.algorithms import als, basic, item_knn

np.random.seed(42)

DATASETS = {
    'MovieLens100K': {
        'path': os.path.join(os.getcwd(), 'u.data'),
        'kind': 'movielens'
    },
    'AmazonVideoGames': {
        'path': os.path.join(os.getcwd(), 'VideoGames.csv'),
        'kind': 'amazon'
    },
    'LastFM': {
        'path': os.path.join(os.getcwd(), 'user_taggedartists-timestamps.dat'),
        'kind': 'lastfm'
    },
}
SEEDS = [11, 22, 33, 44, 55]
KS = [1, 5, 10]
ALGO_NAMES = ['ALS', 'ItemKNN', 'Pop']

experiment_data = {
    ds: {
        'metrics': {'train': [], 'val': []},
        'losses': {'train': [], 'val': []},
        'predictions': [],
        'ground_truth': [],
        'meta': []
    }
    for ds in DATASETS
}


def standardize_columns(df):
    cmap = {str(c).strip().lower(): c for c in df.columns}
    return cmap


def load_dataset(name, meta):
    p = meta['path']
    kind = meta['kind']
    if not os.path.exists(p):
        raise FileNotFoundError(f'Missing file for {name}: {p}')

    if kind == 'movielens':
        df = pd.read_csv(p, sep='\t', header=None, names=['user', 'item', 'rating', 'timestamp'])
        df = df[df['rating'] > 3].copy()
    elif kind == 'amazon':
        df = pd.read_csv(p)
        cols = standardize_columns(df)
        user_col = cols.get('reviewerid', list(df.columns)[0])
        item_col = cols.get('asin', list(df.columns)[1])
        rating_col = cols.get('overall', list(df.columns)[2])
        df = df[[user_col, item_col, rating_col]].copy()
        df.columns = ['user', 'item', 'rating']
        df = df[df['rating'] > 3].copy()
    elif kind == 'lastfm':
        df = pd.read_csv(p, sep='\t')
        cols = standardize_columns(df)
        user_col = cols.get('userid', cols.get('user_id', list(df.columns)[0]))
        item_col = cols.get('artistid', cols.get('artist_id', cols.get('artist', list(df.columns)[1])))
        df = df[[user_col, item_col]].copy()
        df.columns = ['user', 'item']
        df['rating'] = 1.0
    else:
        raise ValueError(name)

    df['user'] = df['user'].astype(str)
    df['item'] = df['item'].astype(str)
    if 'rating' not in df.columns:
        df['rating'] = 1.0
    df = df.groupby(['user', 'item'], as_index=False)['rating'].max()
    return df[['user', 'item', 'rating']].copy()


def core_filter(df, min_uc=5, min_ic=5):
    changed = True
    while changed:
        changed = False
        ucounts = df.groupby('user').size()
        keep_u = ucounts[ucounts >= min_uc].index
        df2 = df[df['user'].isin(keep_u)]
        icounts = df2.groupby('item').size()
        keep_i = icounts[icounts >= min_ic].index
        df3 = df2[df2['item'].isin(keep_i)]
        if len(df3) != len(df):
            changed = True
        df = df3
    return df.reset_index(drop=True)


def user_holdout_split(df, seed, frac=0.2):
    rng = np.random.default_rng(seed)
    train_parts, test_parts = [], []
    for _, g in df.groupby('user', sort=False):
        idx = g.index.to_numpy()
        n_test = max(1, int(round(len(idx) * frac)))
        if len(idx) <= 1:
            continue
        n_test = min(n_test, len(idx) - 1)
        test_idx = rng.choice(idx, size=n_test, replace=False)
        train_idx = np.setdiff1d(idx, test_idx)
        train_parts.append(df.loc[train_idx])
        test_parts.append(df.loc[test_idx])
    return pd.concat(train_parts, ignore_index=True), pd.concat(test_parts, ignore_index=True)


def build_algorithms():
    return {
        'ALS': als.BiasedMF(features=20, iterations=15, reg=0.1),
        'ItemKNN': item_knn.ItemItem(20),
        'Pop': basic.PopularityRecommender(),
    }


def fit_recommend(algo, train, users, k=10):
    train_ui = train[['user', 'item']].copy()
    algo.fit(train_ui)
    recs = topn.recommend(algo, train_ui, users=users, k=k, exclude_known=True)
    return recs


def eval_metrics(recs, test, ks=(1, 5, 10)):
    gt = test.groupby('user')['item'].apply(set).to_dict()
    recs = recs.copy()
    if len(recs) == 0:
        return {f'precision@{k}': 0.0 for k in ks} | {f'ndcg@{k}': 0.0 for k in ks}
    recs['rank'] = recs.groupby('user').cumcount() + 1
    out = {}
    for k in ks:
        precs, ndcgs = [], []
        for u, true_items in gt.items():
            user_recs = recs[(recs['user'] == u) & (recs['rank'] <= k)]['item'].tolist()
            hits = [1 if i in true_items else 0 for i in user_recs]
            precs.append(sum(hits) / k)
            dcg = sum(h / math.log2(r + 2) for r, h in enumerate(hits))
            ideal = min(len(true_items), k)
            idcg = sum(1.0 / math.log2(r + 2) for r in range(ideal)) if ideal > 0 else 1.0
            ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
        out[f'precision@{k}'] = float(np.mean(precs))
        out[f'ndcg@{k}'] = float(np.mean(ndcgs))
    return out

all_results = []
for ds_name, meta in DATASETS.items():
    print(f'Loading {ds_name}...')
    df = load_dataset(ds_name, meta)
    df = core_filter(df, 5, 5)
    print(f'{ds_name}: {len(df)} interactions, {df.user.nunique()} users, {df.item.nunique()} items')

    for seed in SEEDS:
        train, test = user_holdout_split(df, seed=seed, frac=0.2)
        users = test['user'].unique()
        algos = build_algorithms()
        for algo_name, algo in algos.items():
            recs = fit_recommend(algo, train, users, k=10)
            metrics = eval_metrics(recs, test, ks=KS)
            row = {'dataset': ds_name, 'seed': seed, 'algorithm': algo_name}
            row.update(metrics)
            all_results.append(row)
            experiment_data[ds_name]['metrics']['val'].append(row)
            experiment_data[ds_name]['predictions'].append(recs[['user', 'item']].to_numpy())
            experiment_data[ds_name]['ground_truth'].append(test[['user', 'item']].to_numpy())
            experiment_data[ds_name]['meta'].append({'seed': seed, 'algorithm': algo_name})
            print(ds_name, seed, algo_name, metrics)

results = pd.DataFrame(all_results)
results.to_csv(os.path.join(working_dir, 'seed_level_results.csv'), index=False)

summary = results.groupby(['dataset', 'algorithm']).agg(['mean', 'std'])
summary.to_csv(os.path.join(working_dir, 'summary_results.csv'))
print('\nAggregate results (mean ± std over seeds):')
print(summary)

stat_rows = []
for ds in results['dataset'].unique():
    print(f'\nStatistical comparison on {ds}:')
    sub = results[results['dataset'] == ds]
    for metric in [f'precision@{k}' for k in KS] + [f'ndcg@{k}' for k in KS]:
        pivot = sub.pivot(index='seed', columns='algorithm', values=metric)
        for i, a1 in enumerate(ALGO_NAMES):
            for a2 in ALGO_NAMES[i + 1:]:
                x = pivot[a1].dropna().values
                y = pivot[a2].dropna().values
                if len(x) > 1 and len(x) == len(y):
                    t, p = stats.ttest_rel(x, y)
                    stat_rows.append([ds, metric, a1, a2, float(t), float(p)])
                    print(f'{metric}: {a1} vs {a2} paired t-test p={p:.4g}')

stat_df = pd.DataFrame(stat_rows, columns=['dataset', 'metric', 'algo1', 'algo2', 't', 'p'])
stat_df.to_csv(os.path.join(working_dir, 'stat_tests.csv'), index=False)

np.save(os.path.join(working_dir, 'experiment_data.npy'), experiment_data)
np.save(os.path.join(working_dir, 'results_array.npy'), results.to_numpy())

if plt is not None:
    for ds in results['dataset'].unique():
        sub = results[results['dataset'] == ds]
        fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
        metrics = [f'precision@{k}' for k in KS] + [f'ndcg@{k}' for k in KS]
        for j, metric in enumerate(metrics):
            ax = axes.flat[j]
            for algo in ALGO_NAMES:
                vals = sub[sub['algorithm'] == algo].sort_values('seed')[metric].values
                ax.plot(SEEDS, vals, marker='o', label=algo)
            ax.set_title(f'{ds} - {metric}')
            ax.set_xlabel('seed')
            ax.set_ylabel(metric)
            ax.grid(True, alpha=0.3)
        axes.flat[0].legend()
        fig.savefig(os.path.join(working_dir, f'{ds}_seed_sensitivity.png'), dpi=150)
        plt.close(fig)

print('\nSaved outputs to:', working_dir)
print(results.head())
