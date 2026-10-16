"""
CFS 拼合柱轴压承载力预测 —— 步骤 4：SHAP 可解释性分析（补充）
=====================================================================
本脚本复用步骤 2（2_main_model.py）的特征工程、网络与训练组件。

定位：支撑"工程可信度/可解释性"，揭示残差 ΔP 在 DSM 基线之上修正了什么。
      不替代数据效率主线，作为独立的可解释性小节。

关键设计（针对残差架构 P = PDSM_s + ΔP）：
  · 对 PINN，SHAP 解释的是网络输出 **残差 ΔP**，而非总预测 P。
    （若解释 P，PDSM_s 会平凡地压倒一切、近乎循环论证。）
  · 用 XGB 在 **残差 r = Ptest − PDSM_s** 上做精确 TreeSHAP 作交叉验证：
    两法若指向同一批特征，则"残差捕捉 DSM 未建模效应"的结论更稳健。
  · 另对 XGB 在总量 P 上做 TreeSHAP 作 sanity（应显示 PDSM_s 主导），印证残差架构合理。

解释注意：PDSM_s、Pne_b、Pcre、λc、几何量彼此高度相关，SHAP 在共线特征间的
          归因会被任意拆分——讲"方向"和"特征组"，不要把精确排名当真。

本版另将 SHAP 数值导出为 shap_values.npz 与 shap_importance.csv，
供论文插图脚本按期刊版式重绘，计算逻辑与原版完全一致。

运行：python shap_analysis.py
依赖：pip install shap （其余同步骤 2）
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import torch
import shap
from xgboost import XGBRegressor

# ── 复用步骤 2 组件 ───────────────────────────────────────────────────────────
import importlib.util, sys
_MM_PATH = Path(__file__).resolve().parent / "main_model.py"
_spec = importlib.util.spec_from_file_location("main_model", _MM_PATH)
mm = importlib.util.module_from_spec(_spec); sys.modules["main_model"] = mm
_spec.loader.exec_module(mm)

BASE_DIR = Path(__file__).resolve().parent
DEVICE   = mm.DEVICE

# 论文配色
C_BLUE, C_ORANGE, C_GREEN = "#378ADD", "#D85A30", "#639922"
TOP_K = 15           # 重要性图展示前 K 个特征
N_BG  = 100          # GradientSHAP 背景样本数


# ─────────────────────────────────────────────────────────────────────────────
# 1. 准备：训练 Full PINN + XGB，组织 SHAP 所需数据
# ─────────────────────────────────────────────────────────────────────────────
def prepare(cfg=None):
    cfg = cfg or mm.CFG
    X, y, phy_raw, df = mm.load_and_engineer(cfg["data_path"])
    splits, phy, scaler, i_tr, i_vl, i_te = mm.split_dataset(X, y, phy_raw)
    names = list(X.columns)
    in_dim = X.shape[1]

    # 训练 Full PINN（残差修正）
    print("\n训练 Full PINN（用于 ΔP 的 GradientSHAP）...")
    pinn = mm.ResidualCorrectionNet(in_dim, cfg["hidden_dims"], cfg["dropout"])
    pinn, _ = mm.train_model(pinn, splits, phy, mode="pinn", cfg=cfg, verbose=False)
    pinn.eval()

    X_tr_np = splits["train"][0].numpy()
    X_te_np = splits["test"][0].numpy()
    y_tr    = splits["train"][1].numpy()
    y_te    = splits["test"][1].numpy()

    # 残差目标 r = Ptest − PDSM_s（与 PINN 的 ΔP 同义，供 XGB 交叉验证）
    r_tr = y_tr - phy["train"]["PDSM_s"]

    return dict(pinn=pinn, names=names,
                X_tr_np=X_tr_np, X_te_np=X_te_np,
                y_tr=y_tr, y_te=y_te, r_tr=r_tr,
                phy_test=phy["test"])


# ─────────────────────────────────────────────────────────────────────────────
# 2. SHAP 计算
# ─────────────────────────────────────────────────────────────────────────────
class _Delta2D(torch.nn.Module):
    """把残差网的一维输出 ΔP 包成 (n,1)，以适配 shap.GradientExplainer。"""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x).unsqueeze(-1)


def shap_pinn_delta(D):
    """对 PINN 输出残差 ΔP 做 GradientSHAP。"""
    model = _Delta2D(D["pinn"]).to(DEVICE).eval()
    rng = np.random.RandomState(mm.SEED)
    bg_idx = rng.choice(len(D["X_tr_np"]), size=min(N_BG, len(D["X_tr_np"])), replace=False)
    bg = torch.tensor(D["X_tr_np"][bg_idx], dtype=torch.float32).to(DEVICE)
    te = torch.tensor(D["X_te_np"], dtype=torch.float32).to(DEVICE)

    explainer = shap.GradientExplainer(model, bg)
    sv = explainer.shap_values(te)
    if isinstance(sv, list):
        sv = sv[0]
    sv = np.asarray(sv)
    if sv.ndim == 3:          # (n, feat, outputs) → 取单输出
        sv = sv[:, :, 0]
    return sv


def shap_xgb(target, X_tr, X_te):
    """对 XGB 做精确 TreeSHAP。target 可为总量 P 或残差 r。"""
    xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5,
                       subsample=0.8, colsample_bytree=0.8,
                       random_state=mm.SEED, n_jobs=-1, verbosity=0)
    xgb.fit(X_tr, target)
    explainer = shap.TreeExplainer(xgb)
    sv = explainer.shap_values(X_te)
    return np.asarray(sv), xgb


# ─────────────────────────────────────────────────────────────────────────────
# 3. 绘图
# ─────────────────────────────────────────────────────────────────────────────
def plot_beeswarm(sv, X, names, title, fname):
    plt.figure()
    shap.summary_plot(sv, X, feature_names=names, max_display=TOP_K, show=False)
    plt.title(title, fontsize=11)
    out = BASE_DIR / fname
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {fname}")


def plot_bar(sv, names, title, fname, color=C_BLUE):
    imp = np.abs(sv).mean(axis=0)
    order = np.argsort(imp)[::-1][:TOP_K][::-1]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh(range(len(order)), imp[order], color=color, alpha=0.85)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([names[i] for i in order], fontsize=9)
    ax.set_xlabel("mean(|SHAP value|)")
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    out = BASE_DIR / fname
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {fname}")


def plot_dependence(idx, sv, X, names, fname):
    fig, ax = plt.subplots(figsize=(6, 4.6))
    sc = ax.scatter(X[:, idx], sv[:, idx], c=X[:, idx], cmap="viridis",
                    s=18, alpha=0.7, edgecolors="none")
    ax.axhline(0, color="gray", lw=1, ls="--", alpha=0.6)
    ax.set_xlabel(f"{names[idx]} (standardized)")
    ax.set_ylabel(f"SHAP value for {names[idx]}")
    ax.set_title(f"Dependence — {names[idx]}", fontsize=11)
    plt.colorbar(sc, ax=ax, label=names[idx])
    plt.tight_layout()
    out = BASE_DIR / fname
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {fname}")


def plot_compare_importance(sv_a, sv_b, names, label_a, label_b, fname):
    imp_a = np.abs(sv_a).mean(axis=0)
    imp_b = np.abs(sv_b).mean(axis=0)
    imp_a = imp_a / (imp_a.max() + 1e-12)
    imp_b = imp_b / (imp_b.max() + 1e-12)
    order = np.argsort(imp_a + imp_b)[::-1][:TOP_K][::-1]
    yy = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8, 5.8))
    ax.barh(yy + 0.2, imp_a[order], height=0.4, color=C_BLUE, alpha=0.85, label=label_a)
    ax.barh(yy - 0.2, imp_b[order], height=0.4, color=C_ORANGE, alpha=0.85, label=label_b)
    ax.set_yticks(yy)
    ax.set_yticklabels([names[i] for i in order], fontsize=9)
    ax.set_xlabel("normalized mean(|SHAP|)")
    ax.set_title("Residual driver comparison: PINN-dP vs XGB-residual", fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = BASE_DIR / fname
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. 主流程
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  SHAP 可解释性分析  (paper-styled)")
    print("=" * 60)
    D = prepare()
    names = D["names"]

    # (A) PINN 残差 ΔP — GradientSHAP
    print("\n[A] PINN 残差 ΔP — GradientSHAP")
    sv_pinn = shap_pinn_delta(D)
    plot_beeswarm(sv_pinn, D["X_te_np"], names,
                  "SHAP - PINN residual dP (what the network corrects beyond DSM)",
                  "shap_A1_pinn_delta_beeswarm.png")
    plot_bar(sv_pinn, names, "Feature importance - PINN residual dP",
             "shap_A2_pinn_delta_bar.png", color=C_BLUE)
    imp = np.abs(sv_pinn).mean(axis=0)
    numeric_mask = [not (n.startswith("BC_") or n.startswith("SEC_") or n.startswith("FM_"))
                    for n in names]
    top_num = [i for i in np.argsort(imp)[::-1] if numeric_mask[i]][:4]
    for k, idx in enumerate(top_num, 1):
        safe = names[idx].replace("/", "_").replace("(", "").replace(")", "").replace(" ", "")
        plot_dependence(idx, sv_pinn, D["X_te_np"], names,
                        f"shap_A3_pinn_delta_dependence_{k}_{safe}.png")

    # (B) XGB 残差 r = Ptest − PDSM — TreeSHAP（交叉验证）
    print("\n[B] XGB on residual r = Ptest - PDSM - TreeSHAP")
    sv_xgb_r, _ = shap_xgb(D["r_tr"], D["X_tr_np"], D["X_te_np"])
    plot_beeswarm(sv_xgb_r, D["X_te_np"], names,
                  "SHAP - XGB on residual (Ptest - PDSM) [cross-check of dP drivers]",
                  "shap_B1_xgb_residual_beeswarm.png")
    plot_bar(sv_xgb_r, names, "Feature importance - XGB residual (Ptest - PDSM)",
             "shap_B2_xgb_residual_bar.png", color=C_ORANGE)

    # (C) 对比：PINN-ΔP vs XGB-残差
    print("\n[C] 对比 PINN-dP 与 XGB-残差 的归因")
    plot_compare_importance(sv_pinn, sv_xgb_r, names,
                            "PINN dP (GradientSHAP)", "XGB residual (TreeSHAP)",
                            "shap_C_compare_residual_importance.png")

    # (D) sanity：XGB 总量 P（应显示 PDSM_s 主导）
    print("\n[D] XGB on total P - TreeSHAP (sanity: PDSM_s dominates)")
    sv_xgb_p, _ = shap_xgb(D["y_tr"], D["X_tr_np"], D["X_te_np"])
    plot_beeswarm(sv_xgb_p, D["X_te_np"], names,
                  "SHAP - XGB on total capacity P (PDSM_s dominates -> residual arch justified)",
                  "shap_D1_xgb_total_beeswarm.png")
    plot_bar(sv_xgb_p, names, "Feature importance - XGB on total capacity P",
             "shap_D2_xgb_total_bar.png", color=C_GREEN)

    # ── 导出 SHAP 数值，供论文绘图脚本重绘（不影响上述计算）──────────────
    out_npz = BASE_DIR / "shap_values.npz"
    np.savez_compressed(
        out_npz,
        names=np.array(names, dtype=object),   # 特征名，26 维
        X_test=D["X_te_np"],                   # 测试集标准化特征矩阵 (156, 26)
        sv_pinn_delta=sv_pinn,                 # PINN 残差 ΔP 的 GradientSHAP  (156, 26)
        sv_xgb_residual=sv_xgb_r,              # XGB 残差 r 的 TreeSHAP        (156, 26)
        sv_xgb_total=sv_xgb_p,                 # XGB 总承载力 P 的 TreeSHAP    (156, 26)
        y_test=D["y_te"],                      # 实测承载力
        pdsm_test=np.asarray(D["phy_test"]["PDSM_s"]),
    )
    print(f"\n  → SHAP 数值已导出: {out_npz.name}")
    print("     键: names, X_test, sv_pinn_delta, sv_xgb_residual, sv_xgb_total, y_test, pdsm_test")

    # 同时导出各特征的平均绝对 SHAP 值，便于直接查表
    import csv
    with open(BASE_DIR / "shap_importance.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["feature", "pinn_delta", "xgb_residual", "xgb_total"])
        ia = np.abs(sv_pinn).mean(axis=0)
        ib = np.abs(sv_xgb_r).mean(axis=0)
        ic = np.abs(sv_xgb_p).mean(axis=0)
        for k, nm in enumerate(names):
            w.writerow([nm, round(float(ia[k]), 4), round(float(ib[k]), 4), round(float(ic[k]), 4)])
    print("  → 平均绝对 SHAP 值已导出: shap_importance.csv")

    print("\n完成。所有 SHAP 图与数值已保存到脚本目录。")


if __name__ == "__main__":
    main()