# -*- coding: utf-8 -*-
"""
FS = 0 试件对 SHAP 归因的敏感性检验
=====================================================================
背景：数据库中有 34 个试件采用连续连接、无间断紧固件，其 FS 记为 0
      （a/ri 亦为 0）。这些试件的 FS 语义与其余试件不同，需检验第 4.4.2 节
      中紧固件间距 FS 进入残差驱动量前列这一结论是否依赖于它们。

两级检验：
  L1（不重训，约 2 min）
      沿用同一模型，仅把 SHAP 的解释样本换成"排除 FS=0 后的测试集"，
      比较重要性排名。回答：FS 的排位是否由这些点撑起。

  L2（需重训，约 20 min）
      保持 SEED=42 的原划分不变，仅从训练索引中剔除 FS=0 的样本后重训。
      验证集与测试集原样不动，故测试指标可直接对比。
      同时跑一个"随机剔除同样数量"的对照，以区分样本量效应与特殊样本效应。
      回答：模型学到的 FS 关系是否由这些点塑造。

运行：
    python fs_sensitivity.py              # 仅 L1
    python fs_sensitivity.py --l2         # L1 + L2
    python fs_sensitivity.py --l2 --seeds 3   # L2 重复 3 个种子

依赖：与 shap_analysis.py 相同（torch, shap, xgboost）
放置：与 main_model.py 同目录
"""
import argparse, importlib.util, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import shap
from xgboost import XGBRegressor

BASE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("main_model", BASE / "main_model.py")
mm = importlib.util.module_from_spec(_spec); sys.modules["main_model"] = mm
_spec.loader.exec_module(mm)

DEVICE = mm.DEVICE
N_BG   = 100          # GradientSHAP 背景样本数
TOP_K  = 15
FEAT   = "FS (mm)"    # 关注的特征


# ─────────────────────────────────────────────────────────────────────
# 工具
# ─────────────────────────────────────────────────────────────────────
class _Delta2D(torch.nn.Module):
    """把残差网的一维输出 ΔP 包成 (n,1) 以适配 GradientExplainer。"""
    def __init__(self, net):
        super().__init__(); self.net = net

    def forward(self, x):
        return self.net(x).unsqueeze(-1)


def gradient_shap(net, X_bg, X_eval):
    model = _Delta2D(net).to(DEVICE).eval()
    rng = np.random.RandomState(mm.SEED)
    bg_idx = rng.choice(len(X_bg), size=min(N_BG, len(X_bg)), replace=False)
    bg = torch.tensor(X_bg[bg_idx], dtype=torch.float32).to(DEVICE)
    te = torch.tensor(X_eval, dtype=torch.float32).to(DEVICE)
    sv = shap.GradientExplainer(model, bg).shap_values(te)
    if isinstance(sv, list):
        sv = sv[0]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 0]
    return sv


def tree_shap_residual(X_tr, r_tr, X_eval):
    xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5,
                       subsample=0.8, colsample_bytree=0.8,
                       random_state=mm.SEED, n_jobs=-1, verbosity=0)
    xgb.fit(X_tr, r_tr)
    return np.asarray(shap.TreeExplainer(xgb).shap_values(X_eval))


def ranking(sv, names):
    """返回 DataFrame：特征、平均绝对 SHAP、排名。"""
    imp = np.abs(sv).mean(axis=0)
    order = np.argsort(imp)[::-1]
    return pd.DataFrame({
        "feature": [names[i] for i in order],
        "mean_abs_shap": imp[order],
        "rank": np.arange(1, len(order) + 1),
    })


def compare(rk_a, rk_b, label_a, label_b, k=TOP_K):
    """并排打印两组排名。"""
    a = rk_a.set_index("feature"); b = rk_b.set_index("feature")
    feats = list(rk_a.feature[:k])
    print(f"\n  {'feature':<22}{label_a:>22}{label_b:>22}{'Δrank':>8}")
    print("  " + "-" * 74)
    for f in feats:
        ra, va = int(a.loc[f, "rank"]), a.loc[f, "mean_abs_shap"]
        if f in b.index:
            rb, vb = int(b.loc[f, "rank"]), b.loc[f, "mean_abs_shap"]
            d = rb - ra
            print(f"  {f:<22}{f'#{ra}  {va:.3f}':>22}{f'#{rb}  {vb:.3f}':>22}{d:>+8d}")
        else:
            print(f"  {f:<22}{f'#{ra}  {va:.3f}':>22}{'—':>22}{'':>8}")


def fs_summary(rk, tag):
    row = rk[rk.feature == FEAT]
    if len(row):
        print(f"    {tag:<34} {FEAT} → 第 {int(row['rank'].iloc[0])} 位 "
              f"(mean|SHAP| = {row['mean_abs_shap'].iloc[0]:.3f})")


# ─────────────────────────────────────────────────────────────────────
# 数据准备
# ─────────────────────────────────────────────────────────────────────
def load_all():
    X, y, phy_raw, df = mm.load_and_engineer(mm.CFG["data_path"])
    splits, phy, scaler, i_tr, i_vl, i_te = mm.split_dataset(X, y, phy_raw)
    fs0 = (df[FEAT].values == 0)
    names = list(X.columns)
    print(f"\n  FS = 0 试件共 {fs0.sum()} 个 —— "
          f"train {fs0[i_tr].sum()} / val {fs0[i_vl].sum()} / test {fs0[i_te].sum()}")
    return dict(X=X, y=y, phy_raw=phy_raw, df=df, splits=splits, phy=phy,
                i_tr=i_tr, i_vl=i_vl, i_te=i_te, fs0=fs0, names=names)


def train_pinn(splits, phy, cfg=None):
    cfg = cfg or mm.CFG
    net = mm.ResidualCorrectionNet(splits["train"][0].shape[1],
                                   cfg["hidden_dims"], cfg["dropout"])
    net, _ = mm.train_model(net, splits, phy, mode="pinn", cfg=cfg, verbose=False)
    net.eval()
    return net


# ─────────────────────────────────────────────────────────────────────
# L1：不重训，只换解释样本
# ─────────────────────────────────────────────────────────────────────
def level1(D):
    print("\n" + "=" * 78)
    print("  L1  同一模型，比较『全部测试集』与『排除 FS=0 后的测试集』的归因")
    print("=" * 78)

    net = train_pinn(D["splits"], D["phy"])
    X_tr = D["splits"]["train"][0].numpy()
    X_te = D["splits"]["test"][0].numpy()
    y_tr = D["splits"]["train"][1].numpy()
    r_tr = y_tr - D["phy"]["train"]["PDSM_s"]
    keep = ~D["fs0"][D["i_te"]]        # 测试集中非 FS=0 的样本

    sv_all  = gradient_shap(net, X_tr, X_te)
    sv_keep = gradient_shap(net, X_tr, X_te[keep])
    rk_all, rk_keep = ranking(sv_all, D["names"]), ranking(sv_keep, D["names"])
    print(f"\n[PINN 残差 ΔP — GradientSHAP]  n = {len(X_te)} → {keep.sum()}")
    compare(rk_all, rk_keep, "all test", "excl. FS=0")

    xs_all  = tree_shap_residual(X_tr, r_tr, X_te)
    xs_keep = tree_shap_residual(X_tr, r_tr, X_te[keep])
    xk_all, xk_keep = ranking(xs_all, D["names"]), ranking(xs_keep, D["names"])
    print(f"\n[XGB 残差 r = Pt − PDSM_s — TreeSHAP]  n = {len(X_te)} → {keep.sum()}")
    compare(xk_all, xk_keep, "all test", "excl. FS=0")

    print("\n  小结：")
    fs_summary(rk_all,  "PINN, 全部测试集");  fs_summary(rk_keep, "PINN, 排除 FS=0")
    fs_summary(xk_all,  "XGB,  全部测试集");  fs_summary(xk_keep, "XGB,  排除 FS=0")
    return dict(rk_all=rk_all, rk_keep=rk_keep, xk_all=xk_all, xk_keep=xk_keep)


# ─────────────────────────────────────────────────────────────────────
# L2：剔除训练样本后重训（划分不变）
# ─────────────────────────────────────────────────────────────────────
def _subset_splits(D, drop_mask_in_train):
    """保持 val / test 不变，仅从 train 中剔除指定样本。"""
    i_tr = D["i_tr"][~drop_mask_in_train]
    splits = {k: (v[0].clone(), v[1].clone()) for k, v in D["splits"].items()}
    Xtr_full, ytr_full = D["splits"]["train"]
    keep_idx = np.where(~drop_mask_in_train)[0]
    splits["train"] = (Xtr_full[keep_idx].clone(), ytr_full[keep_idx].clone())
    phy = {k: dict(v) for k, v in D["phy"].items()}
    phy["train"] = {k: (v[keep_idx] if isinstance(v, np.ndarray) else v)
                    for k, v in D["phy"]["train"].items()}
    return splits, phy, i_tr


def level2(D, n_seeds=1):
    print("\n" + "=" * 78)
    print("  L2  剔除训练集中的 FS=0 样本后重训（val / test 保持不变）")
    print("=" * 78)

    in_train_fs0 = D["fs0"][D["i_tr"]]
    n_drop = int(in_train_fs0.sum())
    X_te = D["splits"]["test"][0].numpy()
    y_te = D["splits"]["test"][1].numpy()

    results = {}

    # 基准：完整训练集
    net = train_pinn(D["splits"], D["phy"])
    sv = gradient_shap(net, D["splits"]["train"][0].numpy(), X_te)
    results["full"] = ranking(sv, D["names"])
    P = mm.predict(net, D["splits"], D["phy"], mode="pinn")
    m = mm.compute_metrics(y_te, P)
    print(f"\n  [基准] n_train = {len(D['i_tr'])}  R2 = {m['R2']:.4f}  RMSE = {m['RMSE']:.2f}")

    # 处理组：剔除 FS=0
    sp, ph, _ = _subset_splits(D, in_train_fs0)
    net2 = train_pinn(sp, ph)
    sv2 = gradient_shap(net2, sp["train"][0].numpy(), X_te)
    results["drop_fs0"] = ranking(sv2, D["names"])
    P2 = mm.predict(net2, sp, ph, mode="pinn")
    m2 = mm.compute_metrics(y_te, P2)
    print(f"  [剔除 FS=0] n_train = {len(sp['train'][0])}  "
          f"R2 = {m2['R2']:.4f}  RMSE = {m2['RMSE']:.2f}")

    # 对照组：随机剔除同样数量（区分样本量效应）
    rand_ranks = []
    for s in range(n_seeds):
        rng = np.random.RandomState(2000 + s)
        mask = np.zeros(len(D["i_tr"]), dtype=bool)
        mask[rng.choice(len(D["i_tr"]), n_drop, replace=False)] = True
        sp3, ph3, _ = _subset_splits(D, mask)
        net3 = train_pinn(sp3, ph3)
        sv3 = gradient_shap(net3, sp3["train"][0].numpy(), X_te)
        rk3 = ranking(sv3, D["names"])
        rand_ranks.append(int(rk3[rk3.feature == FEAT]["rank"].iloc[0]))
        P3 = mm.predict(net3, sp3, ph3, mode="pinn")
        m3 = mm.compute_metrics(y_te, P3)
        print(f"  [随机剔除 {n_drop} 条, seed {s}] R2 = {m3['R2']:.4f}  "
              f"RMSE = {m3['RMSE']:.2f}  {FEAT} 排名 #{rand_ranks[-1]}")
        if s == 0:
            results["drop_random"] = rk3

    print(f"\n[PINN 残差 ΔP — 训练集变化前后]")
    compare(results["full"], results["drop_fs0"], "full train", "drop FS=0")

    print("\n  小结：")
    fs_summary(results["full"],     "完整训练集")
    fs_summary(results["drop_fs0"], "剔除 FS=0 后")
    if rand_ranks:
        print(f"    {'随机剔除同数量（对照）':<34} {FEAT} → 第 "
              f"{'/'.join('#'+str(r) for r in rand_ranks)} 位")
    return results


# ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l2", action="store_true", help="额外运行 L2（需重训）")
    ap.add_argument("--seeds", type=int, default=1, help="L2 随机对照的种子数")
    ap.add_argument("--csv", action="store_true", help="导出排名 CSV")
    args = ap.parse_args()

    D = load_all()
    r1 = level1(D)
    out = {f"L1_{k}": v for k, v in r1.items()}

    if args.l2:
        r2 = level2(D, n_seeds=args.seeds)
        out.update({f"L2_{k}": v for k, v in r2.items()})

    if args.csv:
        for k, v in out.items():
            v.to_csv(BASE / f"fs_sensitivity_{k}.csv", index=False, encoding="utf-8-sig")
        print(f"\n  已导出 {len(out)} 个排名 CSV")

    print("\n" + "=" * 78)
    print("  判读：若 FS 的排位变化在 ±2 名以内，且与随机剔除对照相当，")
    print("        则第 4.4.2 节的归因结论不依赖于 FS = 0 试件。")
    print("=" * 78)


if __name__ == "__main__":
    main()