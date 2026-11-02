import json, sys, itertools, math, numpy as np
from scipy.optimize import linprog
from scipy.stats import kendalltau
D = sys.argv[1]
Z = np.load(f'{D}/games.npz'); meta = json.load(open(f'{D}/meta.json'))
rng = np.random.default_rng(7)

def members(c, n): return [i for i in range(n) if c >> i & 1]
def shapley_matrix(n):
    A = np.zeros((n, 2 ** n))
    for c in range(1, 2 ** n):
        m = members(c, n); s = len(m)
        for i in range(n):
            if c >> i & 1:
                A[i, c] += math.factorial(s - 1) * math.factorial(n - s) / math.factorial(n)
                A[i, c ^ (1 << i)] -= math.factorial(s - 1) * math.factorial(n - s) / math.factorial(n)
    return A
def game_from_E(E, weeks=None):
    """E[c, i, w]: per-week mean scaled sq error of firm i under coalition c's model. v(S) = sum_i err_i(alone) - err_i(S)."""
    n = E.shape[1]
    Ew = E if weeks is None else E[:, :, weeks]
    err = Ew.mean(2)                                  # [2^n, n]
    alone = np.array([err[1 << i, i] for i in range(n)])
    v = np.zeros(2 ** n)
    for c in range(1, 2 ** n):
        m = members(c, n); v[c] = sum(alone[i] - err[c, i] for i in m)
    return v, err, alone
def core_eps(v, n):
    # least core: min eps s.t. x(S) >= v(S) - eps for proper S, x(N) = v(N)
    rows, rhs = [], []
    for c in range(1, 2 ** n - 1):
        a = np.zeros(n + 1); a[members(c, n)] = -1; a[-1] = -1; rows.append(a); rhs.append(-v[c])
    res = linprog(np.r_[np.zeros(n), 1], A_ub=np.array(rows), b_ub=np.array(rhs),
                  A_eq=np.r_[np.ones(n), 0][None], b_eq=[v[-1]], bounds=[(None, None)] * (n + 1), method='highs')
    return res.x[-1], res.x[:n]
def excess(x, v, n):
    ex = [(v[c] - x[members(c, n)].sum(), c) for c in range(1, 2 ** n - 1)]
    return max(ex)
def block_boot_weeks(nw, B=1000, block=4):
    nb = int(np.ceil(nw / block)); idx = []
    for _ in range(B):
        st = rng.integers(0, nw - block + 1, nb); idx.append(np.concatenate([np.arange(s, s + block) for s in st])[:nw])
    return idx
def superadd_viol(v, n):
    tot = bad = 0
    for a in range(1, 2 ** n):
        rest = (2 ** n - 1) ^ a
        b = rest
        while b:
            if a < b:
                tot += 1; bad += v[a | b] < v[a] + v[b] - 1e-12
            b = (b - 1) & rest
    return bad / tot

res = {}
NEV = meta['n_eval_weeks']; boots = block_boot_weeks(NEV)
quarters = [np.arange(o - meta['eval0'], min(o - meta['eval0'] + 13, NEV)) for o in meta['origins']]
games = sorted({k.rsplit('_', 1)[0] for k in Z.files})
for g in games:
    n = Z[f'{g}_R'].shape[1]; A = shapley_matrix(n); stores = meta['sub'] if g.startswith('lgbm') else meta['stores']
    r = {'stores': stores}
    for P in ['R', 'W']:
        E = Z[f'{g}_{P}']
        v, err, alone = game_from_E(E); phi = A @ v
        grand = err[-1]
        eps, xlc = core_eps(v, n); mx, cmax = excess(phi, v, n)
        bphi = np.array([A @ game_from_E(E, b)[0] for b in boots])
        bgain = np.array([(lambda e: (np.diag(e[[1 << i for i in range(n)]]) - e[-1]) / np.diag(e[[1 << i for i in range(n)]]))(E[:, :, b].mean(2)) for b in boots])
        loo = np.array([v[-1] - v[(2 ** n - 1) ^ (1 << i)] for i in range(n)])
        r[P] = dict(v_grand=float(v[-1]), phi=phi.tolist(), phi_lo=np.percentile(bphi, 2.5, 0).tolist(),
                    phi_hi=np.percentile(bphi, 97.5, 0).tolist(), phi_rank=(-phi).argsort().argsort().tolist(),
                    alone_err=alone.tolist(), grand_err=grand.tolist(),
                    gain_pct=((alone - grand) / alone * 100).tolist(),
                    gain_lo=(np.percentile(bgain, 2.5, 0) * 100).tolist(), gain_hi=(np.percentile(bgain, 97.5, 0) * 100).tolist(),
                    least_core_eps=float(eps), core_nonempty=bool(eps <= 1e-12), shapley_max_excess=float(mx),
                    shapley_in_core=bool(mx <= 1e-12), blocking_coalition=[stores[i] for i in members(cmax, n)],
                    superadd_violation_frac=float(superadd_viol(v, n)), loo=loo.tolist(),
                    n_negative_v=int((v[1:] < 0).sum()), boot_phi=bphi.tolist())
    # R vs W comparison with bootstrap
    pR, pW = np.array(r['R']['phi']), np.array(r['W']['phi'])
    bR, bW = np.array(r['R'].pop('boot_phi')), np.array(r['W'].pop('boot_phi'))
    taus = [kendalltau(bR[b], bW[b])[0] for b in range(len(bR))]
    r['kendall_tau'] = float(kendalltau(pR, pW)[0]); r['tau_lo'], r['tau_hi'] = map(float, np.percentile(taus, [2.5, 97.5]))
    # pairwise rank flips: P(order of i,j differs between R and W) under bootstrap
    flips = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j: flips[i, j] = np.mean(np.sign(bR[:, i] - bR[:, j]) != np.sign(bW[:, i] - bW[:, j]))
    r['flip_prob'] = flips.tolist()
    r['sign_change'] = [stores[i] for i in range(n) if np.sign(pR[i]) != np.sign(pW[i])]
    # quarterly drift (walk-forward)
    EW = Z[f'{g}_W']
    qphi = [A @ game_from_E(EW, q)[0] for q in quarters]
    r['W_quarterly_phi'] = np.array(qphi).tolist()
    r['W_quarterly_rank_tau_vs_first'] = [float(kendalltau(qphi[0], q)[0]) for q in qphi]
    # cheap estimators vs exact (walk-forward game)
    vW, _, _ = game_from_E(EW)
    exact = A @ vW; loo = np.array(r['W']['loo'])
    def mc(m):
        est = np.zeros(n)
        for _ in range(m):
            perm = rng.permutation(n); c = 0
            for i in perm: est[i] += vW[c | (1 << i)] - vW[c]; c |= 1 << i
        return est / m
    r['estimators'] = {'loo_tau': float(kendalltau(exact, loo)[0]),
                       'loo_rel_err': float(np.abs(loo - exact).sum() / np.abs(exact).sum())}
    for m in [10, 50, 200]:
        errs, tt = [], []
        for _ in range(50):
            e = mc(m); errs.append(np.abs(e - exact).sum() / np.abs(exact).sum()); tt.append(kendalltau(e, exact)[0])
        r['estimators'][f'mc{m}_rel_err'] = float(np.mean(errs)); r['estimators'][f'mc{m}_tau'] = float(np.mean(tt))
    res[g] = r
    print(g, 'tau R-W', round(r['kendall_tau'], 2), 'core W', r['W']['core_nonempty'], 'eps', round(r['W']['least_core_eps'], 5),
          'gainW', np.round(r['W']['gain_pct'], 1))
json.dump(res, open(f'{D}/analysis.json', 'w'), indent=1)
