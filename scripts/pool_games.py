"""
Who earns credit in a forecasting pool?
Computes, for every coalition of Walmart stores (M5), the out-of-sample error of a pooled
global forecaster on each member's own series, under two validation protocols:
  R = random 5-fold over rows (common practice in data valuation)
  W = walk-forward, quarterly expanding-window origins (deployment-faithful)
Saves per-coalition, per-store, per-week errors; all game theory is done afterwards.
"""
import os, sys, json, time, itertools, argparse
import numpy as np, pandas as pd

p = argparse.ArgumentParser()
p.add_argument('--data_dir', required=True)          # folder with calendar.csv, sales_train_evaluation.csv
p.add_argument('--out', default='pool_results')
p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
p.add_argument('--lgbm', type=int, default=1)
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
T0 = time.time()
def log(*a): print(f'[{time.time()-T0:7.1f}s]', *a, flush=True)

STORES = ['CA_1', 'CA_2', 'CA_3', 'CA_4', 'TX_1', 'TX_2', 'TX_3', 'WI_1', 'WI_2', 'WI_3']
N = len(STORES)
NW = 277                      # full weeks in d_1..d_1939
FIRST_T, LAST_T = 52, 272     # usable target weeks (0-index): need lag 52; 273-276 dropped
EVAL0 = 156                   # evaluation weeks 156..272 (117 weeks), same for both protocols
ORIGINS = list(range(156, 273, 13))   # 9 quarterly origins
NEV = LAST_T - EVAL0 + 1
LAGS = [1, 2, 3, 4, 5, 6, 8, 13, 26, 52]
SIZES = [12, 12, 20, 20, 30, 30, 45, 45, 60, 60]   # items per store (firm size), shuffled per seed
LAMBDA = 1.0

# ---------------- data ----------------
cal = pd.read_csv(f'{args.data_dir}/calendar.csv')
sales = pd.read_csv(f'{args.data_dir}/sales_train_evaluation.csv')
dcols = [f'd_{i}' for i in range(1, NW * 7 + 1)]
Yd = sales[dcols].to_numpy(np.float32)
Yw = Yd.reshape(len(sales), NW, 7).sum(2)            # weekly sales
del Yd
cal = cal.iloc[:NW * 7]
snap = {st: cal[f'snap_{st}'].to_numpy().reshape(NW, 7).sum(1) for st in ['CA', 'TX', 'WI']}
event = cal['event_name_1'].notna().to_numpy().reshape(NW, 7).sum(1)
DEPTS = sorted(sales['dept_id'].unique())
log('loaded', Yw.shape, 'depts', DEPTS)

def build_rows(Y, store_of, dept_of):
    """Y: [n_series, NW] weekly sales. Returns feature matrix etc. for target weeks FIRST_T..LAST_T."""
    s = Y[:, :104].mean(1)                              # level scale from burn-in (weeks 0-103)
    dnaive = (np.diff(Y[:, :104], axis=1) ** 2).mean(1)  # naive one-step MSE in burn-in (MSSE denominator)
    Z = Y / s[:, None]
    ts = np.arange(FIRST_T, LAST_T + 1)
    feats = []
    for L in LAGS: feats.append(Z[:, ts - L])
    feats.append(np.stack([Z[:, t - 4:t].mean(1) for t in ts], 1))
    feats.append(np.stack([Z[:, t - 13:t].mean(1) for t in ts], 1))
    n = len(Y); T = len(ts)
    feats.append(np.tile(np.sin(2 * np.pi * ts / 52.18), (n, 1)))
    feats.append(np.tile(np.cos(2 * np.pi * ts / 52.18), (n, 1)))
    st_state = [STORES[k][:2] for k in store_of]
    feats.append(np.stack([snap[st][ts] for st in st_state]) / 7.0)
    feats.append(np.tile(event[ts] / 7.0, (n, 1)))
    X = np.stack(feats, -1)                              # [n, T, f]
    D = np.zeros((n, T, len(DEPTS)), np.float32)
    for j, d in enumerate(dept_of): D[j, :, DEPTS.index(d)] = 1.0
    X = np.concatenate([X, D, np.ones((n, T, 1), np.float32)], -1)   # dept one-hot + intercept
    y = Z[:, ts]
    w = (s ** 2 / dnaive)                                # (z - zhat)^2 * w = scaled squared error
    return dict(X=X.reshape(n * T, -1).astype(np.float64), y=y.reshape(-1).astype(np.float64),
                firm=np.repeat(store_of, T), week=np.tile(ts, n), w=np.repeat(w, T),
                series=np.repeat(np.arange(n), T), nser=np.bincount(store_of, minlength=N))

def design_items(seed):
    rng = np.random.default_rng(seed)
    sizes = rng.permutation(SIZES)
    rows, st_of, dp_of, chosen = [], [], [], {}
    for k, st in enumerate(STORES):
        idx = np.where(sales['store_id'].to_numpy() == st)[0]
        Y = Yw[idx]
        ok = (Y[:, :104].mean(1) >= 1.0) & ((np.diff(Y[:, :104], axis=1) ** 2).mean(1) > 0)
        pick = rng.choice(idx[ok], sizes[k], replace=False)
        chosen[st] = sales['item_id'].to_numpy()[pick].tolist()
        rows.append(Yw[pick]); st_of += [k] * len(pick); dp_of += sales['dept_id'].to_numpy()[pick].tolist()
    return build_rows(np.concatenate(rows), np.array(st_of), dp_of), dict(sizes=sizes.tolist(), items=chosen)

def design_depts():
    rows, st_of, dp_of = [], [], []
    for k, st in enumerate(STORES):
        for d in DEPTS:
            m = (sales['store_id'].to_numpy() == st) & (sales['dept_id'].to_numpy() == d)
            rows.append(Yw[m].sum(0)); st_of.append(k); dp_of.append(d)
    return build_rows(np.stack(rows), np.array(st_of), dp_of), dict(sizes=[len(DEPTS)] * N)

COAL = list(range(1, 2 ** N))
def members(c, n=N): return [i for i in range(n) if c >> i & 1]

# ---------------- ridge games via sufficient statistics ----------------
def ridge_game(R, seed):
    X, y, firm, week, w = R['X'], R['y'], R['firm'], R['week'], R['w']
    nser = R['nser'].astype(np.float64); f = X.shape[1]
    E = {'R': np.zeros((2 ** N, N, NEV), np.float32), 'W': np.zeros((2 ** N, N, NEV), np.float32)}
    rng = np.random.default_rng(1000 + seed)
    fold = rng.integers(0, 5, len(y))
    splits = [('R', k, fold != k, (fold == k) & (week >= EVAL0)) for k in range(5)]
    splits += [('W', o, week < o, (week >= o) & (week < o + 13)) for o in ORIGINS]
    reg = LAMBDA * np.eye(f); reg[-1, -1] = 0.0     # do not penalise the intercept
    for proto, key, trm, tem in splits:
        G = np.zeros((N, f, f)); b = np.zeros((N, f))
        for i in range(N):
            mi = trm & (firm == i)
            G[i] = X[mi].T @ X[mi]; b[i] = X[mi].T @ y[mi]
        tests = []
        for i in range(N):
            mi = tem & (firm == i)
            tests.append((X[mi], y[mi], w[mi], week[mi] - EVAL0))
        Gc = np.zeros((2 ** N, f, f)); bc = np.zeros((2 ** N, f))
        for c in COAL:                               # build coalition sums incrementally
            low = c & -c; i = low.bit_length() - 1; rest = c ^ low
            Gc[c] = Gc[rest] + G[i]; bc[c] = bc[rest] + b[i]
        betas = np.linalg.solve(Gc[1:] + reg, bc[1:][..., None])[..., 0]
        for c in COAL:
            beta = betas[c - 1]
            for i in members(c):
                Xt, yt, wt, wk = tests[i]
                se = (yt - Xt @ beta) ** 2 * wt
                E[proto][c, i] += np.bincount(wk, weights=se, minlength=NEV) / nser[i]
    return E

# ---------------- LightGBM sub-game (6 stores, 63 coalitions) ----------------
SUB = [0, 1, 4, 5, 7, 8]   # CA_1, CA_2, TX_1, TX_2, WI_1, WI_2
def lgbm_game(R, seed):
    import lightgbm as lgb
    X, y, firm, week, w = R['X'], R['y'], R['firm'], R['week'], R['w']
    nser = R['nser'].astype(np.float64); n = len(SUB)
    E = {'R': np.zeros((2 ** n, n, NEV), np.float32), 'W': np.zeros((2 ** n, n, NEV), np.float32)}
    rng = np.random.default_rng(1000 + seed); fold = rng.integers(0, 5, len(y))
    splits = [('R', k, fold != k, (fold == k) & (week >= EVAL0)) for k in range(5)]
    splits += [('W', o, week < o, (week >= o) & (week < o + 13)) for o in ORIGINS]
    params = dict(n_estimators=150, learning_rate=0.08, num_leaves=31, min_child_samples=50, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=-1, random_state=seed)
    for si, (proto, key, trm, tem) in enumerate(splits):
        for c in range(1, 2 ** n):
            mem = [SUB[j] for j in members(c, n)]
            mtr = trm & np.isin(firm, mem)
            m = lgb.LGBMRegressor(**params).fit(X[mtr], y[mtr])
            for j in members(c, n):
                mi = tem & (firm == SUB[j])
                se = (y[mi] - m.predict(X[mi])) ** 2 * w[mi]
                E[proto][c, j] += np.bincount(week[mi] - EVAL0, weights=se, minlength=NEV) / nser[SUB[j]]
        log(f'  lgbm split {si + 1}/{len(splits)} done')
    return E

meta = dict(stores=STORES, origins=ORIGINS, eval0=EVAL0, n_eval_weeks=NEV, lags=LAGS, lam=LAMBDA,
            sub=[STORES[i] for i in SUB], depts=DEPTS, designs={})
out = {}
R, info = design_depts(); log('dept design rows', len(R['y']))
E = ridge_game(R, 0); out['dept_R'], out['dept_W'] = E['R'], E['W']; meta['designs']['dept'] = info
log('dept game done')
for sd in args.seeds:
    R, info = design_items(sd); log(f'item design seed {sd}: rows', len(R['y']), 'sizes', info['sizes'])
    E = ridge_game(R, sd); out[f'item{sd}_R'], out[f'item{sd}_W'] = E['R'], E['W']
    meta['designs'][f'item{sd}'] = info
    log(f'item seed {sd} ridge game done')
    if args.lgbm and sd == args.seeds[0]:
        E = lgbm_game(R, sd); out[f'lgbm{sd}_R'], out[f'lgbm{sd}_W'] = E['R'], E['W']
        log('lgbm sub-game done')
np.savez_compressed(f'{args.out}/games.npz', **out)
json.dump(meta, open(f'{args.out}/meta.json', 'w'), indent=1)
log('saved', list(out.keys()))
