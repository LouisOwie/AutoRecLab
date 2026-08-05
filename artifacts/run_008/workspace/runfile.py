import os
working_dir = os.path.join(os.getcwd(), 'working')
os.makedirs(working_dir, exist_ok=True)

import numpy as np
import pandas as pd
from scipy import stats

from lenskit.algorithms.basic import Popular
from lenskit.algorithms.als import ImplicitMF
from lenskit.algorithms.item_knn import ItemItem
from lenskit import batch

experiment_data = {
    'ml100k': {'metrics': {'train': [], 'val': []}, 'losses': {'train': [], 'val': []}, 'predictions': [], 'ground_truth': []},
    'amazon_vg': {'metrics': {'train': [], 'val': []}, 'losses': {'train': [], 'val': []}, 'predictions': [], 'ground_truth': []},
    'lastfm': {'metrics': {'train': [], 'val': []}, 'losses': {'train': [], 'val': []}, 'predictions': [], 'ground_truth': []},
}

SEEDS = [11, 22, 33, 44, 55]
KS = [1, 5, 10]


def k_core_filter(df, user_col='user', item_col='item', min_uc=5, min_ic=5):
    df = df[[user_col, item_col]].dropna().drop_duplicates().copy()
    while True:
        ucnt = df[user_col].value_counts()
        icnt = df[item_col].value_counts()
        keep_u = ucnt[ucnt >= min_uc].index
        keep_i = icnt[icnt >= min_ic].index
        new_df = df[df[user_col].isin(keep_u) & df[item_col].isin(keep_i)]
        if len(new_df) == len(df):
            break
        df = new_df
    return df.reset_index(drop=True)


def load_ml100k(path):
    df = pd.read_csv(path, sep='\t', header=None, names=['user', 'item', 'rating', 'timestamp'])
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df = df[df['rating'] > 3][['user', 'item']]
    return k_core_filter(df)


def load_amazon(path):
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, header=None)
    cols = {str(c).lower(): c for c in df.columns}
    ucol = cols.get('user_id') or cols.get('userid') or cols.get('user') or df.columns[0]
    icol = cols.get('item_id') or cols.get('asin') or cols.get('item') or df.columns[1]
    rcol = cols.get('rating') or cols.get('overall') or (df.columns[2] if len(df.columns) > 2 else None)
    rename_map = {ucol: 'user', icol: 'item'}
    if rcol is not None:
        rename_map[rcol] = 'rating'
    df = df.rename(columns=rename_map)
    if 'rating' not in df.columns:
        raise ValueError('Amazon file must contain a rating/overall column')
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df = df[df['rating'] > 3][['user', 'item']]
    return k_core_filter(df)


def load_lastfm(path):
    df = pd.read_csv(path, sep='\t')
    cols = {str(c).lower(): c for c in df.columns}
    ucol = cols.get('userid') or cols.get('user') or df.columns[0]
    icol = cols.get('artistid') or cols.get('artist') or cols.get('itemid') or df.columns[1]
    df = df.rename(columns={ucol: 'user', icol: 'item'})
    df = df[['user', 'item']]
    return k_core_filter(df)


def user_holdout(df, seed=42, test_ratio=0.2):
    rng = np.random.default_rng(seed)
    train_parts, test_parts = [], []
    for _, udf in df.groupby('user', sort=False):
        idx = np.arange(len(udf))
        rng.shuffle(idx)
        n = len(udf)
        n_test = max(1, int(round(n * test_ratio)))
        if n - n_test < 1:
            n_test = n - 1
        if n_test <= 0:
            train_parts.append(udf)
            continue
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]
        train_parts.append(udf.iloc[train_idx])
        test_parts.append(udf.iloc[test_idx])
    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame(columns=df.columns)
    return train, test


def precision_at_k(recs, truth, k):
    topk = recs[:k]
    if k == 0:
        return 0.0
    return sum(1 for x in topk if x in truth) / k


def ndcg_at_k(recs, truth, k):
    topk = recs[:k]
    dcg = 0.0
    for i, item in enumerate(topk):
        if item in truth:
            dcg += 1.0 / np.log2(i + 2)
    ideal_hits = min(len(truth), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg


def evaluate_algo(algo, train, test, dataset_name, algo_name, seed):
    algo.fit(train)
    users = test['user'].drop_duplicates().tolist()
    recs = batch.recommend(algo, users, 10)
    truth_map = test.groupby('user')['item'].apply(set).to_dict()
    pred_map = recs.groupby('user')['item'].apply(list).to_dict() if len(recs) else {}

    user_metrics = []
    for u in users:
        truth = truth_map.get(u, set())
        pred = pred_map.get(u, [])
        row = {'user': u}
        for k in KS:
            row[f'P@{k}'] = precision_at_k(pred, truth, k)
            row[f'nDCG@{k}'] = ndcg_at_k(pred, truth, k)
        user_metrics.append(row)
    um = pd.DataFrame(user_metrics)
    mean_metrics = um.drop(columns=['user']).mean().to_dict() if len(um) else {}
    for k in KS:
        mean_metrics.setdefault(f'P@{k}', 0.0)
        mean_metrics.setdefault(f'nDCG@{k}', 0.0)

    val_loss = float('nan')
    print(f'{dataset_name} | {algo_name} | seed={seed}: validation_loss = {val_loss}')

    result_row = {'dataset': dataset_name, 'algorithm': algo_name, 'seed': seed, **mean_metrics}
    experiment_data[dataset_name]['metrics']['val'].append({'seed': seed, 'algorithm': algo_name, **mean_metrics})
    experiment_data[dataset_name]['losses']['val'].append({'seed': seed, 'algorithm': algo_name, 'validation_loss': val_loss})
    experiment_data[dataset_name]['predictions'].append({'seed': seed, 'algorithm': algo_name, 'rows': recs[['user', 'item']].to_dict('records') if len(recs) else []})
    experiment_data[dataset_name]['ground_truth'].append({'seed': seed, 'algorithm': algo_name, 'rows': test[['user', 'item']].to_dict('records')})
    return pd.DataFrame([result_row]), um


def summarize_stats(results):
    metric_cols = [c for c in results.columns if '@' in c]
    summary = results.groupby(['dataset', 'algorithm'])[metric_cols].agg(['mean', 'std'])
    print('\nMean ± std across seeds:')
    print(summary)

    stats_rows = []
    for (ds, alg), g in results.groupby(['dataset', 'algorithm']):
        g = g.sort_values('seed')
        for metric in metric_cols:
            vals = g[metric].astype(float).values
            mean_v = float(np.mean(vals)) if len(vals) else np.nan
            std_v = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
            cv = float(std_v / mean_v) if len(vals) > 1 and mean_v != 0 else np.nan
            base = vals[0] if len(vals) else np.nan
            diffs = vals[1:] - base if len(vals) > 1 else np.array([])
            t_stat, p_val = (np.nan, np.nan)
            if len(diffs) > 1 and np.std(diffs, ddof=1) > 0:
                t_stat, p_val = stats.ttest_1samp(diffs, 0.0)
            stats_rows.append({
                'dataset': ds,
                'algorithm': alg,
                'metric': metric,
                'mean': mean_v,
                'std': std_v,
                'cv_across_seeds': cv,
                't_stat_vs_seed11': t_stat,
                'p_value': p_val
            })
    stat_df = pd.DataFrame(stats_rows)
    print('\nShort statistical analysis (seed sensitivity):')
    print(stat_df.to_string(index=False))
    return summary, stat_df


def plot_results(results, dataset_name):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f'Skipping plots for {dataset_name}: {e}')
        return
    metric_cols = [c for c in results.columns if '@' in c]
    sub = results[results['dataset'] == dataset_name]
    algos = list(sub['algorithm'].unique())
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    axes = axes.ravel()
    for ax, metric in zip(axes, metric_cols):
        means = sub.groupby('algorithm')[metric].mean().reindex(algos)
        stds = sub.groupby('algorithm')[metric].std().reindex(algos).fillna(0)
        ax.bar(range(len(algos)), means.values, yerr=stds.values, capsize=4)
        ax.set_xticks(range(len(algos)))
        ax.set_xticklabels(algos, rotation=20)
        ax.set_title(metric)
        ymax = float((means + stds).max()) if len(means) else 1.0
        ax.set_ylim(0, max(1e-6, ymax) * 1.2)
    fig.tight_layout()
    fig.savefig(os.path.join(working_dir, f'{dataset_name}_seed_sensitivity_metrics.png'), dpi=150)
    plt.close(fig)


datasets = {
    'ml100k': load_ml100k('u.data'),
    'amazon_vg': load_amazon('VideoGames.csv'),
    'lastfm': load_lastfm('user_taggedartists-timestamps.dat')
}

all_results = []
all_user_metrics = []

for ds_name, df in datasets.items():
    print(f'\nLoaded {ds_name}: {len(df)} interactions, {df.user.nunique()} users, {df.item.nunique()} items')
    for seed in SEEDS:
        train, test = user_holdout(df, seed=seed, test_ratio=0.2)
        train = train.copy()
        test = test.copy()
        train['rating'] = 1.0
        test['rating'] = 1.0
        algos = {
            'ALS': ImplicitMF(features=50, iterations=15, reg=0.1, weight=40),
            'ItemKNN': ItemItem(nnbrs=20, min_nbrs=1, min_sim=1.0e-6, center=False),
            'Pop': Popular()
        }
        for algo_name, algo in algos.items():
            res, um = evaluate_algo(algo, train[['user', 'item', 'rating']], test[['user', 'item', 'rating']], ds_name, algo_name, seed)
            all_results.append(res)
            um['dataset'] = ds_name
            um['algorithm'] = algo_name
            um['seed'] = seed
            all_user_metrics.append(um)

results_df = pd.concat(all_results, ignore_index=True)
user_metrics_df = pd.concat(all_user_metrics, ignore_index=True)
summary_df, stat_df = summarize_stats(results_df)

print('\nFinal aggregated results:')
print(results_df.to_string(index=False))

for ds_name in datasets:
    plot_results(results_df, ds_name)
    ds_res = results_df[results_df['dataset'] == ds_name]
    np.savez_compressed(
        os.path.join(working_dir, f'{ds_name}_metrics.npz'),
        results=ds_res.to_records(index=False),
        user_metrics=user_metrics_df[user_metrics_df['dataset'] == ds_name].to_records(index=False)
    )

np.save(os.path.join(working_dir, 'experiment_data.npy'), experiment_data, allow_pickle=True)
results_df.to_csv(os.path.join(working_dir, 'all_results.csv'), index=False)
user_metrics_df.to_csv(os.path.join(working_dir, 'all_user_metrics.csv'), index=False)
summary_df.to_csv(os.path.join(working_dir, 'summary_mean_std.csv'))
stat_df.to_csv(os.path.join(working_dir, 'seed_variation_stats.csv'), index=False)

print(f'\nSaved outputs to: {working_dir}')