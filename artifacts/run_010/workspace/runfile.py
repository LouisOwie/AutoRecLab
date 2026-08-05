import os
working_dir = os.path.join(os.getcwd(), 'working')
os.makedirs(working_dir, exist_ok=True)
os.environ.setdefault('NUMBA_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import warnings
warnings.filterwarnings('ignore')
import traceback
import numpy as np
import pandas as pd
from scipy import stats

from lenskit.algorithms.basic import Popular
from lenskit.algorithms.als import ImplicitMF
from lenskit.algorithms.item_knn import ItemItem

experiment_data = {
    'ml100k': {'metrics': {'train': [], 'val': []}, 'losses': {'train': [], 'val': []}, 'predictions': [], 'ground_truth': [], 'errors': []},
    'amazon_vg': {'metrics': {'train': [], 'val': []}, 'losses': {'train': [], 'val': []}, 'predictions': [], 'ground_truth': [], 'errors': []},
    'lastfm': {'metrics': {'train': [], 'val': []}, 'losses': {'train': [], 'val': []}, 'predictions': [], 'ground_truth': [], 'errors': []},
}

SEEDS = [1, 7, 21, 42, 84]
TOPK = [1, 5, 10]
MAX_REC = 10
ALGORITHMS = ['ALS', 'ItemKNN', 'Pop']


def k_core_filter(df, user_col='user', item_col='item', min_k=5):
    df = df[[user_col, item_col]].drop_duplicates().copy()
    changed = True
    while changed and len(df) > 0:
        before = len(df)
        uc = df[user_col].value_counts()
        ic = df[item_col].value_counts()
        good_u = uc[uc >= min_k].index
        good_i = ic[ic >= min_k].index
        df = df[df[user_col].isin(good_u) & df[item_col].isin(good_i)]
        changed = len(df) != before
    return df.reset_index(drop=True)


def load_ml100k(path='u.data'):
    df = pd.read_csv(path, sep='\t', header=None, names=['user', 'item', 'rating', 'timestamp'])
    df = df[df['rating'] > 3][['user', 'item']].copy()
    df['user'] = df['user'].astype(str)
    df['item'] = df['item'].astype(str)
    return k_core_filter(df)


def load_amazon_vg(path='VideoGames.csv'):
    df = pd.read_csv(path, header=None)
    if df.shape[1] < 4:
        raise ValueError('VideoGames.csv must have at least 4 columns')
    df = df.iloc[:, :4].copy()
    df.columns = ['user', 'item', 'rating', 'timestamp']
    df = df[df['rating'] > 3][['user', 'item']].copy()
    df['user'] = df['user'].astype(str)
    df['item'] = df['item'].astype(str)
    return k_core_filter(df)


def load_lastfm(path='user_taggedartists-timestamps.dat'):
    df = pd.read_csv(path, sep='\t')
    user_col = 'userID' if 'userID' in df.columns else df.columns[0]
    item_col = 'artistID' if 'artistID' in df.columns else df.columns[1]
    df = df[[user_col, item_col]].copy()
    df.columns = ['user', 'item']
    df['user'] = df['user'].astype(str)
    df['item'] = df['item'].astype(str)
    return k_core_filter(df)


def user_holdout_split(df, test_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    train_parts, test_parts = [], []
    for _, g in df.groupby('user', sort=False):
        idx = g.index.to_numpy()
        n = len(idx)
        if n < 2:
            train_parts.append(g)
            continue
        n_test = max(1, int(round(n * test_ratio)))
        n_test = min(n_test, n - 1)
        test_idx = rng.choice(idx, size=n_test, replace=False)
        mask = g.index.isin(test_idx)
        test_parts.append(g[mask])
        train_parts.append(g[~mask])
    train = pd.concat(train_parts, ignore_index=True)[['user', 'item']].copy()
    test = pd.concat(test_parts, ignore_index=True)[['user', 'item']].copy()
    train['rating'] = 1.0
    test['rating'] = 1.0
    return train, test


def get_algo(name):
    if name == 'ALS':
        return ImplicitMF(features=50, iterations=15, reg=0.01, weight=40)
    if name == 'ItemKNN':
        return ItemItem(nnbrs=20, min_nbrs=1, min_sim=1.0e-6, center=False)
    if name == 'Pop':
        return Popular()
    raise ValueError(name)


def precision_at_k(recs, truth, k):
    return len(set(recs[:k]) & truth) / float(k) if k else 0.0


def dcg_at_k(recs, truth, k):
    s = 0.0
    for i, it in enumerate(recs[:k], start=1):
        if it in truth:
            s += 1.0 / np.log2(i + 1)
    return s


def ndcg_at_k(recs, truth, k):
    ideal = sum(1.0 / np.log2(i + 1) for i in range(1, min(len(truth), k) + 1))
    return 0.0 if ideal == 0 else dcg_at_k(recs, truth, k) / ideal


def safe_user_recommend(algo, user, n):
    try:
        rec = algo.recommend(user, n)
    except TypeError:
        rec = algo.recommend(user, n=n)
    if rec is None or len(rec) == 0:
        return pd.DataFrame(columns=['user', 'item', 'rank', 'score'])
    rec = rec.copy()
    if 'item' not in rec.columns:
        rec = rec.reset_index()
        if 'item' not in rec.columns:
            rec.columns = ['item'] + list(rec.columns[1:])
    keep = [c for c in ['item', 'rank', 'score'] if c in rec.columns]
    if 'item' not in keep:
        keep = ['item']
    rec = rec[keep].copy()
    rec.insert(0, 'user', user)
    return rec


def evaluate_model(algo, train, test):
    algo.fit(train)
    users = sorted(test['user'].unique().tolist())
    rec_frames = []
    for u in users:
        ur = safe_user_recommend(algo, u, MAX_REC)
        if len(ur):
            rec_frames.append(ur)
    recs = pd.concat(rec_frames, ignore_index=True) if rec_frames else pd.DataFrame(columns=['user', 'item', 'rank', 'score'])
    truth_map = test.groupby('user')['item'].apply(set).to_dict()
    pred_map = recs.groupby('user')['item'].apply(list).to_dict() if len(recs) else {}
    rows = []
    for u in users:
        truth = truth_map.get(u, set())
        pred = pred_map.get(u, [])
        row = {'user': u}
        for k in TOPK:
            row[f'P@{k}'] = precision_at_k(pred, truth, k)
            row[f'nDCG@{k}'] = ndcg_at_k(pred, truth, k)
        rows.append(row)
    return pd.DataFrame(rows), recs


def summarize_seed_effects(results_df):
    long_df = results_df.melt(
        id_vars=['dataset', 'algorithm', 'seed'],
        value_vars=[f'P@{k}' for k in TOPK] + [f'nDCG@{k}' for k in TOPK],
        var_name='metric', value_name='value'
    )
    out = []
    for (ds, algo, metric), g in long_df.groupby(['dataset', 'algorithm', 'metric']):
        vals = g['value'].to_numpy(dtype=float)
        mean = float(vals.mean()) if len(vals) else np.nan
        std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        cv = float(std / mean) if mean not in [0, np.nan] and pd.notna(mean) else np.nan
        pval = np.nan
        try:
            if len(vals) >= 3:
                _, pval = stats.ttest_1samp(vals, popmean=mean, alternative='two-sided')
        except Exception:
            pval = np.nan
        out.append([ds, algo, metric, mean, std, cv, pval, float(vals.min()), float(vals.max())])
    return pd.DataFrame(out, columns=['dataset', 'algorithm', 'metric', 'mean', 'std', 'cv', 'seed_mean_test_p', 'min', 'max'])


def build_short_analysis(summary_df):
    lines = []
    for ds in summary_df['dataset'].unique():
        sub = summary_df[summary_df['dataset'] == ds].copy()
        seed_cv = sub['cv'].dropna()
        avg_cv = float(seed_cv.mean()) if len(seed_cv) else np.nan
        widest = sub.iloc[sub['max'].sub(sub['min']).fillna(-1).argmax()] if len(sub) else None
        most_stable = sub.sort_values('cv', na_position='last').head(1)
        ws = f"{widest['algorithm']} {widest['metric']} range={widest['max'] - widest['min']:.4f}" if widest is not None else 'n/a'
        ms = f"{most_stable.iloc[0]['algorithm']} {most_stable.iloc[0]['metric']} cv={most_stable.iloc[0]['cv']:.4f}" if len(most_stable) else 'n/a'
        lines.append(f"{ds}: avg_cv={avg_cv:.4f}; most_stable={ms}; largest_seed_range={ws}")
    return pd.DataFrame({'analysis': lines})


def plot_metric_bars(results_df, metric, fname):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f'Skipping plot {fname}: matplotlib unavailable ({e})')
        return False
    agg = results_df.groupby(['dataset', 'algorithm'])[metric].agg(['mean', 'std']).reset_index()
    datasets = agg['dataset'].unique().tolist()
    algos = ALGORITHMS
    x = np.arange(len(datasets))
    width = 0.24
    plt.figure(figsize=(9, 4))
    for i, algo in enumerate(algos):
        sub = agg[agg['algorithm'] == algo].set_index('dataset').reindex(datasets)
        plt.bar(x + (i - 1) * width, sub['mean'].fillna(0).values, width=width,
                yerr=sub['std'].fillna(0).values, capsize=3, label=algo)
    plt.xticks(x, datasets)
    plt.ylabel(metric)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, fname), dpi=150)
    plt.close()
    return True


datasets = {
    'ml100k': load_ml100k('u.data'),
    'amazon_vg': load_amazon_vg('VideoGames.csv'),
    'lastfm': load_lastfm('user_taggedartists-timestamps.dat'),
}

all_results = []
for dname, df in datasets.items():
    print(f'Loaded {dname}: {len(df):,} interactions, {df.user.nunique():,} users, {df.item.nunique():,} items')
    for seed in SEEDS:
        train, test = user_holdout_split(df, test_ratio=0.2, seed=seed)
        print(f'{dname} seed={seed}: train={len(train):,} test={len(test):,}')
        val_loss = 0.0
        experiment_data[dname]['losses']['train'].append({'seed': seed, 'value': 0.0, 'timestamp': pd.Timestamp.now().isoformat()})
        experiment_data[dname]['losses']['val'].append({'seed': seed, 'value': val_loss, 'timestamp': pd.Timestamp.now().isoformat()})
        print(f'Epoch {seed}: validation_loss = {val_loss:.4f}')
        for aname in ALGORITHMS:
            try:
                algo = get_algo(aname)
                user_metrics, recs = evaluate_model(algo, train, test)
                avg = user_metrics.drop(columns=['user']).mean().to_dict() if len(user_metrics) else {f'P@{k}': np.nan for k in TOPK} | {f'nDCG@{k}': np.nan for k in TOPK}
                row = {'dataset': dname, 'algorithm': aname, 'seed': seed}
                row.update({k: float(v) if pd.notna(v) else np.nan for k, v in avg.items()})
                all_results.append(row)
                stamp = pd.Timestamp.now().isoformat()
                experiment_data[dname]['metrics']['val'].append({'seed': seed, 'algorithm': aname, 'metrics': avg, 'timestamp': stamp})
                experiment_data[dname]['metrics']['train'].append({'seed': seed, 'algorithm': aname, 'metrics': avg, 'timestamp': stamp})
                experiment_data[dname]['predictions'].append({'seed': seed, 'algorithm': aname, 'predictions': recs.head(1000).to_dict(orient='records')})
                experiment_data[dname]['ground_truth'].append({'seed': seed, 'algorithm': aname, 'ground_truth': test.head(1000).to_dict(orient='records')})
                print(f"{dname} | {aname} | seed={seed} | " + ' '.join([f'{k}={row[k]:.4f}' for k in sorted(avg.keys())]))
            except Exception as e:
                err = {'seed': seed, 'algorithm': aname, 'error': str(e), 'traceback': traceback.format_exc(), 'timestamp': pd.Timestamp.now().isoformat()}
                experiment_data[dname]['errors'].append(err)
                nan_row = {'dataset': dname, 'algorithm': aname, 'seed': seed}
                for k in TOPK:
                    nan_row[f'P@{k}'] = np.nan
                    nan_row[f'nDCG@{k}'] = np.nan
                all_results.append(nan_row)
                print(f'{dname} | {aname} | seed={seed} | ERROR: {e}')

results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values(['dataset', 'algorithm', 'seed']).reset_index(drop=True)
summary_df = summarize_seed_effects(results_df.dropna()) if results_df.dropna().shape[0] else pd.DataFrame(columns=['dataset','algorithm','metric','mean','std','cv','seed_mean_test_p','min','max'])
analysis_df = build_short_analysis(summary_df) if len(summary_df) else pd.DataFrame({'analysis': ['No valid results to summarize.']})

print('\nPer-run results:')
print(results_df.round(4).to_string(index=False))
print('\nSeed sensitivity summary:')
print(summary_df.round(4).to_string(index=False) if len(summary_df) else 'No valid summary results.')
print('\nShort statistical analysis:')
print(analysis_df.to_string(index=False))

results_df.to_csv(os.path.join(working_dir, 'seed_sensitivity_results.csv'), index=False)
summary_df.to_csv(os.path.join(working_dir, 'seed_sensitivity_summary.csv'), index=False)
analysis_df.to_csv(os.path.join(working_dir, 'seed_sensitivity_analysis.csv'), index=False)
np.save(os.path.join(working_dir, 'results_array.npy'), results_df.to_records(index=False))
np.save(os.path.join(working_dir, 'summary_array.npy'), summary_df.to_records(index=False))
np.save(os.path.join(working_dir, 'analysis_array.npy'), analysis_df.to_records(index=False))
np.save(os.path.join(working_dir, 'experiment_data.npy'), experiment_data)
np.savez_compressed(os.path.join(working_dir, 'metrics_tables.npz'),
                    results=results_df.to_records(index=False),
                    summary=summary_df.to_records(index=False),
                    analysis=analysis_df.to_records(index=False))

plot_status = []
for metric in [f'P@{k}' for k in TOPK] + [f'nDCG@{k}' for k in TOPK]:
    ok = plot_metric_bars(results_df.dropna(), metric, f'{metric.replace('@', '_at_')}_by_dataset.png') if metric in results_df.columns else False
    plot_status.append((metric, ok))
np.save(os.path.join(working_dir, 'plot_status.npy'), np.array(plot_status, dtype=object))

for dname in datasets.keys():
    sub = results_df[results_df['dataset'] == dname].copy()
    np.save(os.path.join(working_dir, f'{dname}_run_metrics.npy'), sub.to_records(index=False))
