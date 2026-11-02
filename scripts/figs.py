import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import sys
R = json.load(open(sys.argv[1] if len(sys.argv) > 1 else 'results/analysis.json'))
OUT = sys.argv[2] if len(sys.argv) > 2 else 'figures'
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']
TXT, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.edgecolor': MUTED, 'axes.labelcolor': TXT,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True})
G = R['lgbm0']; st = G['stores']; n = len(st); x = np.arange(n)

# Fig 1: (a) average pooling gain by store, R vs W (LightGBM); (b) mechanism: how much random folds lower errors
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.55), gridspec_kw=dict(width_ratios=[1.15, 1]))
w = 0.38
for k, (P, col, lab) in enumerate([('R', C[1], 'Random 5-fold'), ('W', C[0], 'Walk-forward')]):
    g = np.array(G[P]['gain_pct']); lo = np.array(G[P]['gain_lo']); hi = np.array(G[P]['gain_hi'])
    ax[0].bar(x + (k - 0.5) * w, g, w * 0.92, color=col, label=lab, zorder=2)
    ax[0].errorbar(x + (k - 0.5) * w, g, yerr=[g - lo, hi - g], fmt='none', ecolor=TXT, elinewidth=0.8, capsize=1.8, zorder=3)
ax[0].axhline(0, color=MUTED, lw=0.8); ax[0].set_xticks(x); ax[0].set_xticklabels(st)
ax[0].set_ylabel('Error reduction from\njoining the full pool (%)'); ax[0].set_title('(a) Who gains from pooling', loc='left', fontsize=9, color=TXT)
ax[0].set_ylim(-8, 20); ax[0].legend(frameon=False, loc='upper center', ncol=2, fontsize=7.5)
dA = np.array(G['W']['alone_err']) - np.array(G['R']['alone_err']); dG = np.array(G['W']['grand_err']) - np.array(G['R']['grand_err'])
ax[1].bar(x - w / 2, dA, w * 0.92, color=C[3], label='Store alone', zorder=2)
ax[1].bar(x + w / 2, dG, w * 0.92, color=C[2], label='Full pool', zorder=2)
ax[1].axhline(0, color=MUTED, lw=0.8); ax[1].set_xticks(x); ax[1].set_xticklabels(st)
ax[1].set_ylabel('Error flattered by random\nfolds (W minus R, MSSE)'); ax[1].set_title('(b) Leakage helps the store alone more', loc='left', fontsize=9, color=TXT)
ax[1].legend(frameon=False, loc='upper left', fontsize=7.5)
fig.tight_layout(); fig.savefig(f'{OUT}/fig1.png', dpi=250, bbox_inches='tight'); plt.close(fig)

# Fig 2: Shapley values with 95% CI, R vs W, LightGBM 6 stores and ridge item0 10 stores
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw=dict(width_ratios=[1, 1.5]))
for ax, gname, title in [(axes[0], 'lgbm0', '(a) LightGBM pool, 6 stores'), (axes[1], 'item0', '(b) Linear pool, 10 stores')]:
    g = R[gname]; s = g['stores']; xx = np.arange(len(s))
    for k, (P, col, mk, lab) in enumerate([('R', C[1], 's', 'Random 5-fold'), ('W', C[0], 'o', 'Walk-forward')]):
        ph = np.array(g[P]['phi']); lo = np.array(g[P]['phi_lo']); hi = np.array(g[P]['phi_hi'])
        ax.errorbar(xx + (k - 0.5) * 0.28, ph, yerr=[ph - lo, hi - ph], fmt=mk, color=col, ms=4.8, elinewidth=1.1, capsize=1.8, label=lab, zorder=3)
    ax.axhline(0, color=MUTED, lw=0.8); ax.set_xticks(xx); ax.set_xticklabels(s, rotation=0 if len(s) <= 6 else 45, fontsize=7.6)
    ax.set_title(title, loc='left', fontsize=9, color=TXT)
axes[0].set_ylabel('Shapley value (MSSE\nreduction credited)'); axes[0].legend(frameon=False, loc='upper left', fontsize=7.5)
fig.tight_layout(); fig.savefig(f'{OUT}/fig2.png', dpi=250, bbox_inches='tight'); plt.close(fig)

# Fig 3: quarterly rank heatmap (walk-forward, LightGBM)
q = np.array(G['W_quarterly_phi']); ranks = (-q).argsort(1).argsort(1) + 1   # 1 = largest credit that quarter
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list('b', ['#0d3b78', '#2a78d6', '#a9c8ef', '#eef4fc'])
fig, ax = plt.subplots(figsize=(7.0, 2.2))
ax.imshow(ranks.T, cmap=cmap, aspect='auto', vmin=1, vmax=n)
for i in range(q.shape[0]):
    for j in range(n):
        ax.text(i, j, ranks[i, j], ha='center', va='center', fontsize=7.5, color='white' if ranks[i, j] <= 2 else TXT)
ax.set_yticks(range(n)); ax.set_yticklabels(st); ax.set_xticks(range(q.shape[0])); ax.set_xticklabels([f'Q{i+1}' for i in range(q.shape[0])])
ax.grid(False); [sp.set_visible(False) for sp in ax.spines.values()]
ax.set_xlabel('Walk-forward quarter (rank 1 = largest Shapley credit that quarter)')
fig.tight_layout(); fig.savefig(f'{OUT}/fig3.png', dpi=250, bbox_inches='tight'); plt.close(fig)
print('ok')
