# -*- coding: utf-8 -*-
"""
图 4-1  训练样本量对预测表现的影响
=====================================================================
输出：Fig4-1_data_efficiency.png (600 dpi) 与 .pdf (矢量，投稿用)
画布：Elsevier 双栏宽 190 mm

数据源：ablation_decompose_results.csv
  配置名与论文记号的对应：
    MLP_0   → M0   无 DSM 信息，直接预测
    MLP_F   → M1   DSM 作特征，直接预测（直接预测网络）
    Full_RF → M2   DSM 作特征 + 残差修正（本文模型，无软约束）
    XGB_0 / XGB_F  → 不含 / 含 DSM 特征的 XGBoost
    RF_0  / RF_F   → 不含 / 含 DSM 特征的随机森林

调参入口集中在「参数区」，改完直接重跑。
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════
# 参数区
# ═══════════════════════════════════════════════════════════════════════
HERE     = Path(__file__).resolve().parent
DATA     = HERE / 'ablation_decompose_results.csv'
OUT_DIR  = HERE
OUT_NAME = 'Fig4-1_data_efficiency'

MM    = 1 / 25.4
FIG_W = 190 * MM          # 双栏 190；单栏改 90
FIG_H = 66 * MM

FONT_STACK = ['Times New Roman', 'Liberation Serif', 'Nimbus Roman', 'DejaVu Serif']
FS_BASE  = 8              # 轴标题
FS_TICK  = 7.5
FS_LEG   = 7
FS_ANNO  = 7.5

# 要画哪些曲线：(csv 中的配置名, 图例文字, 颜色键)
SERIES = [
    ('MLP_0',   'M0','m0'),
    ('MLP_F',   'M1','m1'),
    ('Full_RF', 'M2', 'ours'),
    ('XGB_0',   'XGBoost no DSM(X1)','dsm'),
    ('XGB_F',   'XGBoost with DSM features(X2)', 'xgb'),
]
HIGHLIGHT = 'Full_RF'     # 加粗显示的配置
DASHED    = {'MLP_0'}     # 用虚线的配置，便于与 M1 区分

# Okabe–Ito 色盲友好配色
C = {'dsm': '#7F7F7F', 'rf': '#009E73', 'xgb': '#0072B2',
     'm1': '#D55E00', 'ours': '#000000', 'm0': '#CC79A7'}
MK = {'dsm': 's', 'rf': '^', 'xgb': 'D', 'm1': 'v', 'ours': 'o', 'm0': 'x'}

SHOW_BAND   = False        # (a) 是否画 ±1 标准差阴影
BAND_ALPHA  = 0.10
YLIM_R2     = (-0.9, 1.05)
YTICKS_RMSE = [20, 30, 50, 80, 130]

ANNOTATE_GAP = False       # (b) 是否标注 N=50 处的 RMSE 倍数
GAP_N        = 50

# 图例放置方式：
#   'shared'  两面板共用一份图例，置于图底（默认）
#   'panel_a' 仅 (a) 内部放图例
#   'both'    两个面板各放一份
LEGEND_MODE = 'shared'
LEGEND_NCOL = 5           # shared 模式下的列数
LEGEND_Y    = 0      # shared 模式下图例的垂直位置，负值表示位于图下方

mpl.rcParams.update({
    'font.family': 'serif', 'font.serif': FONT_STACK,
    'mathtext.fontset': 'stix',
    'font.size': FS_BASE, 'axes.labelsize': FS_BASE, 'axes.titlesize': FS_BASE,
    'xtick.labelsize': FS_TICK, 'ytick.labelsize': FS_TICK, 'legend.fontsize': FS_LEG,
    'axes.linewidth': 0.6, 'lines.linewidth': 1.0,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.minor.width': 0.5, 'ytick.minor.width': 0.5,
    'xtick.major.size': 2.6, 'ytick.major.size': 2.6,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
    'legend.frameon': False, 'axes.grid': False,
    'savefig.dpi': 600, 'figure.dpi': 140,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})


def panel_label(ax, s, dx=-0.16, dy=1.02):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=FS_BASE + 0.5,
            fontweight='bold', va='bottom', ha='left')


def style_of(key):
    """返回该配置的线宽、标记大小、线型、标记填充色。"""
    hi = (key == HIGHLIGHT)
    return (1.6 if hi else 1.0,
            4.2 if hi else 3.4,
            '--' if key in DASHED else '-',
            None if hi else 'white')


def draw():
    d = pd.read_csv(DATA)
    r2 = d.groupby(['config', 'N']).R2.agg(['mean', 'std'])
    rm = d.groupby(['config', 'N']).RMSE.agg(['mean', 'std'])
    N = sorted(d.N.unique())

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H))

    # ── (a) 决定系数 ───────────────────────────────────────────────────
    ax = axes[0]
    for key, lab, c in SERIES:
        m = np.array([r2.loc[(key, n), 'mean'] for n in N])
        s = np.array([r2.loc[(key, n), 'std'] for n in N])
        lw, ms, ls, mfc = style_of(key)
        ax.plot(N, m, marker=MK[c], color=C[c], label=lab, lw=lw, ms=ms, ls=ls,
                mfc=mfc if mfc else C[c], mew=1.0, zorder=3)
        if SHOW_BAND:
            ax.fill_between(N, m - s, m + s, color=C[c], alpha=BAND_ALPHA, lw=0)
    ax.axhline(0, color='k', lw=.5, ls=':')
    ax.set_xscale('log'); ax.set_xticks(N); ax.set_xticklabels(N)
    ax.set_xlabel('Training set size $N$')
    ax.set_ylabel(r'Test $R^2$')
    ax.set_ylim(*YLIM_R2)
    if LEGEND_MODE in ('panel_a', 'both'):
        ax.legend(loc='lower right')
    panel_label(ax, '(a)')

    # ── (b) 均方根误差 ─────────────────────────────────────────────────
    ax = axes[1]
    for key, lab, c in SERIES:
        m = np.array([rm.loc[(key, n), 'mean'] for n in N])
        lw, ms, ls, mfc = style_of(key)
        ax.plot(N, m, marker=MK[c], color=C[c], label=lab, lw=lw, ms=ms, ls=ls,
                mfc=mfc if mfc else C[c], mew=1.0, zorder=3)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xticks(N); ax.set_xticklabels(N)
    ax.set_yticks(YTICKS_RMSE); ax.set_yticklabels(YTICKS_RMSE)
    ax.set_xlabel('Training set size $N$')
    ax.set_ylabel('Test RMSE (kN)')
    if LEGEND_MODE == 'both':
        ax.legend(loc='lower left')
    panel_label(ax, '(b)')

    if ANNOTATE_GAP:
        hi = rm.loc[('MLP_F', GAP_N), 'mean']      # M1
        lo = rm.loc[(HIGHLIGHT, GAP_N), 'mean']    # M2
        ax.annotate('', xy=(GAP_N, lo * 1.025), xytext=(GAP_N, hi * 0.975),
                    arrowprops=dict(arrowstyle='<->', lw=.7, color='k',
                                    shrinkA=1.5, shrinkB=1.5))
        ax.text(GAP_N * 1.12, np.sqrt(hi * lo), f'{hi / lo:.1f}$\\times$',
                fontsize=FS_ANNO, ha='left', va='center')

    fig.tight_layout(w_pad=2.0)

    if LEGEND_MODE == 'shared':
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center',
                   bbox_to_anchor=(0.5, LEGEND_Y), ncol=LEGEND_NCOL,
                   frameon=False, fontsize=FS_LEG,
                   columnspacing=1.4, handlelength=1.9, handletextpad=0.45)

    for ext in ('png', 'pdf'):
        fig.savefig(OUT_DIR / f'{OUT_NAME}.{ext}')
    plt.close(fig)
    print(f'saved: {OUT_NAME}.png / .pdf')


if __name__ == '__main__':
    draw()