# -*- coding: utf-8 -*-
"""
图 4-2 / 图 4-3  SHAP 归因图
=====================================================================
输出：
  Fig4-2_shap_attribution.png / .pdf   三面板：总量归因、残差归因、跨方法对照
  Fig4-3_shap_dependence.png  / .pdf   双面板：板厚与紧固件间距的依赖关系

数据源：shap_values.npz（由打过补丁的 shap_analysis.py 生成）
  npz 中的键：
    names            26 个特征名
    X_test           测试集标准化特征矩阵 (156, 26)
    sv_pinn_delta    PINN 残差 ΔP 的 GradientSHAP (156, 26)
    sv_xgb_residual  XGB 残差 r 的 TreeSHAP (156, 26)
    sv_xgb_total     XGB 总承载力 P 的 TreeSHAP (156, 26)

依赖图的横轴会从标准化值还原为原始单位，需要读取原始数据集以取得
均值与标准差；若找不到数据集文件，则退回标准化坐标并在轴标签注明。

画布：Elsevier 双栏宽 190 mm。字体 Times New Roman，公式用 STIX。
运行：python fig4_shap.py
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════
# 参数区
# ═══════════════════════════════════════════════════════════════════════
HERE    = Path(__file__).resolve().parent
NPZ     = HERE / 'shap_values.npz'
DATASET = HERE / 'All_Dataset_cleaned1.csv'     # 用于还原原始单位，可缺省
OUT_DIR = HERE

MM    = 1 / 25.4
FIG_W = 190 * MM
H_ATTR = 76 * MM          # 图 4-2 高度
H_DEP  = 66 * MM          # 图 4-3 高度

FONT_STACK = ['Times New Roman', 'Liberation Serif', 'Nimbus Roman', 'DejaVu Serif']
FS_BASE, FS_TICK, FS_LEG, FS_ANNO = 8, 7.0, 7, 6.8

TOP_K = 12                # 条形图展示的特征数
DEP_FEATURES = ['t(mm)', 'FS (mm)']     # 依赖图要画的两个特征

# 特征名 → 图中显示名（数学符号）
LABEL = {
    'L(mm)': '$L$', 't(mm)': '$t$', 'h(mm)': '$h$', 'b(mm)': '$b$',
    'Area(mm)2': '$A$', 'FS (mm)': '$FS$', 'Fy(Mpa)': '$F_y$', 'E(Mpa)': '$E$',
    'λc_b': r'$\lambda_c$', 'λ(crl,crd)_s': r'$\lambda_{(crl,crd),s}$',
    'λ(crl,crd)_ms': r'$\lambda_{(crl,crd),ms}$', 'KL/rb': '$KL/r_b$',
    'Pne_b': '$P_{ne}$', 'PDSM_s(kN)': r'$P_{\mathrm{DSM},s}$',
    'PDSM_ms(kN)': r'$P_{\mathrm{DSM},ms}$', 'Pcre (kN)': '$P_{cre}$',
    'h_over_t': '$h/t$', 'b_over_t': '$b/t$', 'L_over_h': '$L/h$',
    'norm_PDSM': r'$P_{\mathrm{DSM},s}/P_y$',
    'BC_Fixed': 'BC: fixed', 'BC_Pinned': 'BC: pinned', 'BC_Semi-Fixed': 'BC: semi-fixed',
    'SEC_Closed': 'Sec: closed', 'SEC_Half-Closed': 'Sec: half-closed', 'SEC_Open': 'Sec: open',
}
# 依赖图的轴标题与单位
DEP_LABEL = {'t(mm)': 'Thickness $t$ (mm)', 'FS (mm)': 'Fastener spacing $FS$ (mm)'}

# Okabe–Ito 配色
C_TOTAL = '#009E73'       # 总量归因
C_DELTA = '#0072B2'       # PINN 残差
C_XGBR  = '#D55E00'       # XGB 残差
C_PT    = '#4C72B0'       # 依赖图散点
INK     = '#1A1A1A'
GREY    = '#7F7F7F'

mpl.rcParams.update({
    'font.family': 'serif', 'font.serif': FONT_STACK, 'mathtext.fontset': 'stix',
    'font.size': FS_BASE, 'axes.labelsize': FS_BASE, 'axes.titlesize': FS_BASE,
    'xtick.labelsize': FS_TICK, 'ytick.labelsize': FS_TICK, 'legend.fontsize': FS_LEG,
    'axes.linewidth': 0.6, 'lines.linewidth': 1.0,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.6, 'ytick.major.size': 2.6,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
    'legend.frameon': False, 'axes.grid': False,
    'savefig.dpi': 600, 'figure.dpi': 140,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})


def plab(ax, s, dx=-0.20, dy=1.03):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=FS_BASE + 0.5,
            fontweight='bold', va='bottom', ha='left')


def disp(n):
    return LABEL.get(n, n)


def load():
    z = np.load(NPZ, allow_pickle=True)
    return {k: z[k] for k in z.files}


def barh(ax, imp, names, color, xlabel, k=TOP_K, annotate_top=False):
    """横向条形图，按重要性降序，最重要的排在最上方。"""
    order = np.argsort(imp)[::-1][:k][::-1]
    yy = np.arange(len(order))
    ax.barh(yy, imp[order], height=0.68, color=color,
            edgecolor='black', linewidth=0.35, zorder=3)
    ax.set_yticks(yy)
    ax.set_yticklabels([disp(names[i]) for i in order])
    ax.set_xlabel(xlabel)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.tick_params(axis='y', length=0)
    if annotate_top:
        top = order[-1]
        ax.text(imp[top] * 0.97, yy[-1], f'{imp[top]:.1f}', ha='right',
                va='center', fontsize=FS_ANNO, color='white', zorder=4)
    return order


# ═══════════════════════════════════════════════════════════════════════
# 图 4-2  归因对比
# ═══════════════════════════════════════════════════════════════════════
def fig_attribution(D):
    names = [str(x) for x in D['names']]
    tot   = np.abs(D['sv_xgb_total']).mean(axis=0)
    dlt   = np.abs(D['sv_pinn_delta']).mean(axis=0)
    xgbr  = np.abs(D['sv_xgb_residual']).mean(axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, H_ATTR))

    # (a) XGB 解释总承载力
    ax = axes[0]
    barh(ax, tot, names, C_TOTAL, r'mean $|$SHAP$|$ (kN)', annotate_top=True)
    ax.set_title('XGBoost on total capacity $P$', fontsize=FS_BASE)
    plab(ax, '(a)')

    # (b) PINN 解释残差
    ax = axes[1]
    barh(ax, dlt, names, C_DELTA, r'mean $|$SHAP$|$ (kN)')
    ax.set_title(r'Residual correction network on $\Delta P$', fontsize=FS_BASE)
    plab(ax, '(b)')

    # (c) 跨方法对照，归一化后并列
    ax = axes[2]
    a = dlt / (dlt.max() + 1e-12)
    b = xgbr / (xgbr.max() + 1e-12)
    order = np.argsort(a + b)[::-1][:TOP_K][::-1]
    yy = np.arange(len(order))
    ax.barh(yy + 0.19, a[order], height=0.36, color=C_DELTA,
            edgecolor='black', linewidth=0.35, label=r'PINN $\Delta P$', zorder=3)
    ax.barh(yy - 0.19, b[order], height=0.36, color=C_XGBR,
            edgecolor='black', linewidth=0.35, label='XGBoost residual', zorder=3)
    ax.set_yticks(yy); ax.set_yticklabels([disp(names[i]) for i in order])
    ax.set_xlabel(r'normalised mean $|$SHAP$|$')
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.tick_params(axis='y', length=0)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2,
              columnspacing=1.4, handlelength=1.4)
    ax.set_title('Cross-method comparison', fontsize=FS_BASE)
    plab(ax, '(c)')

    fig.tight_layout(w_pad=2.4)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT_DIR / f'Fig4-2_shap_attribution.{ext}')
    plt.close(fig)
    print('saved: Fig4-2_shap_attribution')


# ═══════════════════════════════════════════════════════════════════════
# 图 4-3  依赖关系
# ═══════════════════════════════════════════════════════════════════════
def fig_dependence(D):
    names = [str(x) for x in D['names']]
    X, sv = D['X_test'], D['sv_pinn_delta']

    # 还原原始单位：标准化参数在训练子集上拟合，此处用全库近似，仅作坐标显示
    raw_ok, mu, sg = False, {}, {}
    if DATASET.exists():
        try:
            df = pd.read_csv(DATASET, encoding='utf-8-sig')
            for f in DEP_FEATURES:
                if f in df.columns:
                    mu[f], sg[f] = df[f].mean(), df[f].std(ddof=0)
            raw_ok = len(mu) == len(DEP_FEATURES)
        except Exception:
            raw_ok = False

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W * 0.66, H_DEP))
    for ax, f, tag in zip(axes, DEP_FEATURES, 'ab'):
        j = names.index(f)
        x = X[:, j] * sg[f] + mu[f] if raw_ok else X[:, j]
        ax.scatter(x, sv[:, j], s=9, alpha=.55, color=C_PT, edgecolors='none', zorder=3)
        ax.axhline(0, color='k', ls='--', lw=.7)
        ax.set_xlabel(DEP_LABEL.get(f, disp(f)) if raw_ok
                      else f'{disp(f)} (standardised)')
        ax.set_ylabel(rf'SHAP value for {disp(f)} (kN)')
        plab(ax, f'({tag})')
    fig.tight_layout(w_pad=2.4)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT_DIR / f'Fig4-3_shap_dependence.{ext}')
    plt.close(fig)
    print('saved: Fig4-3_shap_dependence' + ('' if raw_ok else '  [横轴为标准化值]'))


# ═══════════════════════════════════════════════════════════════════════
def main():
    if not NPZ.exists():
        raise SystemExit(f'未找到 {NPZ.name}，请先运行打过补丁的 shap_analysis.py')
    D = load()
    fig_attribution(D)
    fig_dependence(D)

    # 顺带打印正文所需的关键数值，便于核对
    names = [str(x) for x in D['names']]
    for key, tag in [('sv_xgb_total', '总承载力'), ('sv_pinn_delta', 'PINN 残差'),
                     ('sv_xgb_residual', 'XGB 残差')]:
        imp = np.abs(D[key]).mean(axis=0)
        od = np.argsort(imp)[::-1][:8]
        print(f'\n{tag} 前 8 位：')
        for r, i in enumerate(od, 1):
            print(f'  {r:>2}. {names[i]:<18} {imp[i]:.2f}')
        print(f'  末位（共 {len(imp)} 项）: {imp.min():.2f}')


if __name__ == '__main__':
    main()