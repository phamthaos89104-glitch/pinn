# -*- coding: utf-8 -*-
"""
图 3-1 基线分解架构示意图
=====================================================================
输出：Fig3-1_architecture.png (600 dpi) 与 .pdf (矢量，投稿用)
画布：Elsevier 双栏宽 190 mm

调参入口集中在下面的「参数区」，改完直接重跑即可。
坐标系为 0–100 (x) × 0–50 (y)，与画布尺寸无关，挪动方块只改这些数。
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════
# 参数区
# ═══════════════════════════════════════════════════════════════════════
OUT_DIR   = Path('.')            # 输出目录
OUT_NAME  = 'Fig3-1_architecture'

MM        = 1 / 25.4
FIG_W     = 190 * MM             # 双栏 190 mm；单栏改 90
FIG_H     = 92 * MM

# 字体：Times New Roman。若系统无此字体，回退到 Liberation Serif（度量兼容）
FONT_STACK = ['Times New Roman', 'Liberation Serif', 'Nimbus Roman', 'DejaVu Serif']

FS_BOX    = 7.5      # 方块正文
FS_SMALL  = 7.0      # 输入框明细
FS_LABEL  = 7.5      # 支路标签
FS_NOTE   = 6.8      # 斜体注释
FS_LOSS   = 6.6      # 损失框
FS_PLUS   = 13       # 加号

# 配色（Okabe–Ito 色盲友好体系）
INK     = '#1A1A1A'                        # 主线条与文字
GREY,  GREY_E  = '#F2F2F2', '#9A9A9A'      # 输入
WARM,  WARM_E  = '#FBE6D4', '#D55E00'      # 力学支路（橙）
COOL,  COOL_E  = '#DCE9F5', '#0072B2'      # 数据支路（蓝）
MUTE,  MUTE_E  = '#EFEFEF', '#B0B0B0'      # 软约束（灰，次要）
NOTE_C  = '#444444'
DASH_C  = '#999999'

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': FONT_STACK,
    'mathtext.fontset': 'stix',            # 公式配 Times 风格
    'font.size': FS_BOX,
    'savefig.dpi': 600, 'figure.dpi': 150,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03,
    'pdf.fonttype': 42, 'ps.fonttype': 42,  # 字体嵌入为 TrueType，投稿要求
})


# ═══════════════════════════════════════════════════════════════════════
# 绘图辅助
# ═══════════════════════════════════════════════════════════════════════
def box(ax, x, y, w, h, text, fc, ec, fs=FS_BOX, lw=0.9,
        weight='normal', ls='-', tc=INK, pad=0.02, r=1.2):
    """圆角方块。(x, y) 为左下角，w/h 为宽高。"""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f'round,pad={pad},rounding_size={r}',
        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fs, color=tc, fontweight=weight, zorder=3, linespacing=1.45)


def arrow(ax, x1, y1, x2, y2, ec=INK, lw=0.9, ls='-',
          style='-|>', rad=0.0, ms=5):
    """箭头。rad 控制弯曲程度：正为逆时针弧，负为顺时针弧，0 为直线。"""
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=ms,
        color=ec, linewidth=lw, linestyle=ls,
        connectionstyle=f'arc3,rad={rad}',
        shrinkA=1.5, shrinkB=1.5, zorder=4))


# ═══════════════════════════════════════════════════════════════════════
# 主图
# ═══════════════════════════════════════════════════════════════════════
def draw():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, 100)
    ax.set_ylim(16, 72)
    ax.axis('off')

    # ── 输入特征 ───────────────────────────────────────────────────────
    box(ax, 1.5, 39.5, 15, 19,
        'Input features (26)\n\n'
        'Geometry & material (8)\n'
        'DSM-derived (8)\n'
        'Derived ratios (4)\n'
        'Boundary & Categorical(6)',
        GREY, GREY_E, fs=FS_SMALL)

    # ── 力学支路（上） ─────────────────────────────────────────────────
    box(ax, 24, 56, 22, 11.5,
        'AISI S100 Direct Strength Method\n'
        r'$P_n=\min(P_{ne},P_{nl},P_{nd})$',
        WARM, WARM_E, fs=FS_BOX)

    box(ax, 52, 57.5, 15, 8.5,
        'Analytical baseline\n' r'$P_{\mathrm{DSM},s}$',
        WARM, WARM_E, fs=8, weight='bold')

    ax.text(35, 69.2, 'Mechanics-based branch  (fixed, not trained)',
            ha='center', fontsize=FS_LABEL, color=WARM_E, fontweight='bold')

    # ── 数据支路（下） ─────────────────────────────────────────────────
    box(ax, 24,   31.5, 12.5, 12, 'Input\nprojection\n\n' r'$26\rightarrow128$',
        COOL, COOL_E, fs=FS_SMALL)
    box(ax, 38.5, 31.5, 15,   12, 'Pre-activation\nresidual blocks\n\n128–256–256–128',
        COOL, COOL_E, fs=FS_SMALL)
    box(ax, 55.5, 31.5, 11.5, 12, 'Output\nhead\n\n' r'$\Delta P$',
        COOL, COOL_E, fs=FS_SMALL)

    ax.text(38.5, 44.8, 'Data-driven branch  (learns the deviation only)',
            ha='center', fontsize=FS_LABEL, color=COOL_E, fontweight='bold')

    # ── 求和节点与输出 ─────────────────────────────────────────────────
    cx, cy = 74.5, 49.5
    ax.add_patch(Circle((cx, cy), 2.6, facecolor='white',
                        edgecolor=INK, linewidth=1.1, zorder=3))
    ax.text(cx, cy, '+', ha='center', va='center', fontsize=FS_PLUS, zorder=4)

    box(ax, 82.5, 43.5, 16, 12,
        'Prediction\n\n' r'$P_{pred}=P_{\mathrm{DSM},s}+\Delta P$',
        '#FFFFFF', INK, fs=8, lw=1.2, weight='bold')

    # ── 主干连线 ───────────────────────────────────────────────────────
    arrow(ax, 16.8, 52,   24, 61.5, rad= 0.18)        # 输入 → DSM
    arrow(ax, 16.8, 45,   24, 37.5, rad=-0.18)        # 输入 → 网络
    arrow(ax, 46.2, 61.5, 52, 61.5)                   # DSM 计算 → 基线
    arrow(ax, 36.7, 37.5, 38.5, 37.5)                 # 投影 → 残差塔
    arrow(ax, 53.7, 37.5, 55.5, 37.5)                 # 残差塔 → 输出头
    arrow(ax, 67.2, 61.5, cx, cy + 2.9, rad=-0.22)    # 基线 → 求和
    arrow(ax, 67.2, 37.5, cx, cy - 2.9, rad= 0.25)    # ΔP  → 求和
    arrow(ax, cx + 2.9, cy, 82.5, cy, lw=1.2)         # 求和 → 预测

    # ── 损失与软约束（虚线，次要） ─────────────────────────────────────
    box(ax, 62.0, 19.0, 36, 8.5,
        'Loss\n'
        r'$\mathcal{L}=\mathcal{L}_{data}'
        r'+w_{phy}(\mathcal{L}_{upper}+\mathcal{L}_{dsm})+w_{mono}\mathcal{L}_{mono}$'
        '\nsoft constraints: tested and found inert',
        MUTE, MUTE_E, fs=FS_LOSS, ls='--', lw=0.8, tc='#555555')

    arrow(ax, 88.0, 43.3, 88.0, 27.7, ls='--', ec=DASH_C, lw=0.8)   # 预测 → 损失
    arrow(ax, 62.0, 23.3, 61.4, 31.5, ls='--', ec=DASH_C, lw=0.8, rad=0.25)  # 梯度回传

    # ── 注释 ───────────────────────────────────────────────────────────
    # ax.text(90.5, 61.5,
    #         'The baseline enters with a fixed\n'
    #         'unit coefficient, so the network\n'
    #         'never has to learn it from data',
    #         ha='center', va='center', fontsize=FS_NOTE,
    #         color=NOTE_C, style='italic', linespacing=1.55)


    # ── 残差块细节（缩小的放大视图，置于数据支路下方）─────────────────
    bx, by, bw, bh = 23.0, 19.0, 38.0, 8.5          # 细节框位置与尺寸
    ax.add_patch(FancyBboxPatch(
        (bx, by), bw, bh,
        boxstyle='round,pad=0.02,rounding_size=0.9',
        facecolor='#FAFCFE', edgecolor=COOL_E, linewidth=0.55,
        linestyle=(0, (3, 2)), zorder=1))
    ax.text(bx + 0.9, by + bh - 1.5, 'Residual block  (256, 256, 128)',
            fontsize=6.4, color=COOL_E, fontweight='bold', va='center')
    ax.text(bx + bw - 0.9, by + bh - 1.5, r'$\times\,3$',
            fontsize=6.0, color=NOTE_C, ha='right', va='center')

    # 引出线：从主图残差塔指向细节框
    arrow(ax, 46.0, 31.4, 46.0, by + bh + 0.05, ls=(0, (2.5, 2)),
          ec=COOL_E, lw=0.6, style='-|>', ms=4)

    yb, hb, wb, gap = by + 1.4, 3.2, 4.7, 0.7        # 小方块
    xs = [bx + 3.2 + i * (wb + gap) for i in range(5)]
    labs = ['LN', r'$W_1$', 'GELU', 'Drop', r'$W_2$']
    for x, lab in zip(xs, labs):
        box(ax, x, yb, wb, hb, lab, COOL, COOL_E, fs=5.6, lw=0.5, r=0.5)
    for i in range(len(xs) - 1):
        arrow(ax, xs[i] + wb, yb + hb / 2, xs[i + 1], yb + hb / 2, ms=3.2, lw=0.6)

    sx, sy = xs[-1] + wb + 2.4, yb + hb / 2
    ax.add_patch(Circle((sx, sy), 1.15, facecolor='white', edgecolor=INK,
                        linewidth=0.8, zorder=3))
    ax.text(sx, sy, '+', ha='center', va='center', fontsize=7, zorder=4)
    arrow(ax, xs[-1] + wb, sy, sx - 1.15, sy, ms=3.2, lw=0.6)

    ax.text(bx + 1.5, sy, r'$\boldsymbol{z}$', ha='right', va='center', fontsize=6.4)
    arrow(ax, bx + 1.8, sy, xs[0], sy, ms=3.2, lw=0.6)
    arrow(ax, sx + 1.15, sy, bx + bw - 4.2, sy, ms=3.2, lw=0.6)
    ax.text(bx + bw - 3.9, sy, r'$\boldsymbol{z}_{out}$', ha='left',
            va='center', fontsize=6.4)

    ax.add_patch(FancyArrowPatch(
        (bx + 2.1, sy + 0.3), (sx, sy + 1.15), arrowstyle='-|>',
        mutation_scale=3.2, color=INK, linewidth=0.6,
        connectionstyle='arc3,rad=-0.22', shrinkA=0.6, shrinkB=0.6, zorder=4))

    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(OUT_DIR / f'{OUT_NAME}.{ext}')
    plt.close(fig)
    print(f'saved: {OUT_NAME}.png / .pdf')


if __name__ == '__main__':
    draw()