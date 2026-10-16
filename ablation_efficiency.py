"""
CFS 拼合柱轴压承载力预测 —— 步骤 3：消融实验 + 数据效率分析
=====================================================================
本脚本复用步骤 2（2_main_model.py）的特征工程、网络、损失与训练组件。

核心叙事：
  · 全量数据上的消融（附录）显示三类物理损失项 ΔR² 落在噪声内、违反率为 0，
    看似"物理无用"——这是一个由数据分布决定的「零结果」，不是失败。
  · 数据效率扫描把这一现象解释清楚：物理先验的收益是训练量 N 的递减函数，
    在小样本区显著、在全量区饱和；且收益主要来自 DSM 架构端（特征 + 残差），
    而非损失端软约束。

实验设计：
  (1) 数据效率主扫描：2×2 析因 × 训练量 × 种子
        Factor F（DSM 知识）：有(Full/A1) vs 无(D1/D2)
        Factor L（物理损失）：有(Full/D1) vs 无(A1/D2)
        训练量 N ∈ {50,100,200,400,725}，每格 5 个种子 → 100 次训练
        —— 验证/测试集全程固定；标准化器只在子样本上拟合（防泄漏）；
           按破坏模式分层下采样；结果即时写盘，支持断点续跑。
  (2) 全量数据消融（附录）：在全训练集上逐项关闭物理损失（upper/dsm/mono），
        报告 ΔR² 与物理一致性指标（违反率 ViolRate）。

运行：
  python 3_ablation_efficiency.py            # 跑数据效率主扫描 + 出图
  python 3_ablation_efficiency.py --ablation # 额外跑全量消融（附录表）
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# ── 复用步骤 2 的全部组件 ─────────────────────────────────────────────────────
import importlib.util, sys
_MM_PATH = Path(__file__).resolve().parent / "main_model.py"
_spec = importlib.util.spec_from_file_location("main_model", _MM_PATH)
mm = importlib.util.module_from_spec(_spec); sys.modules["main_model"] = mm
_spec.loader.exec_module(mm)

BASE_DIR = Path(__file__).resolve().parent
DEVICE   = mm.DEVICE
FM_ORDER = mm.FM_ORDER
SEED     = mm.SEED

# 效率扫描的轻量训练配置（独立于主模型的 1000-epoch 设置）
EFF_CFG = dict(mm.CFG)
EFF_CFG.update({"epochs": 400, "patience": 40})

TRAIN_SIZES = [50, 100, 200, 400, 725]
RUN_SEEDS   = [0, 1, 2, 3, 4]

# 2×2 析因：use_phy = DSM 架构端物理；use_loss = 损失端物理
FACTORIAL = {
    "Full": dict(use_phy=True,  use_loss=True),    # DSM 架构 + 物理损失
    "A1"  : dict(use_phy=True,  use_loss=False),   # DSM 架构，无物理损失
    "D1"  : dict(use_phy=False, use_loss=True),    # 纯数据 + 物理损失（损失端惰性对照）
    "D2"  : dict(use_phy=False, use_loss=False),   # 纯数据基线
}

EFF_CSV = BASE_DIR / "data_efficiency_results.csv"
ABL_CSV = BASE_DIR / "full_ablation_results.csv"


# ─────────────────────────────────────────────────────────────────────────────
# 1. 特征构建（可选移除 DSM 知识列）
# ─────────────────────────────────────────────────────────────────────────────
def build_feature_matrices(df):
    """返回 (X_phy, X_nophy)：前者含 DSM 特征，后者剔除所有 DSM 相关列。"""
    geo_mat = ["L(mm)", "t(mm)", "h(mm)", "b(mm)", "Area(mm)2", "FS (mm)",
               "Fy(Mpa)", "E(Mpa)"]
    dsm_feats = ["λc_b", "λ(crl,crd)_s", "λ(crl,crd)_ms",
                 "KL/rb", "Pne_b", "PDSM_s(kN)", "PDSM_ms(kN)", "Pcre (kN)"]
    df = df.copy()
    df["h_over_t"]  = df["h(mm)"]      / df["t(mm)"].clip(lower=1e-6)
    df["b_over_t"]  = df["b(mm)"]      / df["t(mm)"].clip(lower=1e-6)
    df["L_over_h"]  = df["L(mm)"]      / df["h(mm)"].clip(lower=1e-6)
    df["norm_PDSM"] = df["PDSM_s(kN)"] / df["Py(kN)"].clip(lower=1e-6)
    derived_geo = ["h_over_t", "b_over_t", "L_over_h"]   # 纯几何衍生
    derived_dsm = ["norm_PDSM"]                          # 含 DSM 信息的衍生

    bc  = pd.get_dummies(df["BC"],                prefix="BC").astype(float)
    sec = pd.get_dummies(df["Sections_Category"], prefix="SEC").astype(float)
    # fmd = pd.get_dummies(df["FM_merged"],         prefix="FM").astype(float)

    X_phy = pd.concat([df[geo_mat + dsm_feats + derived_geo + derived_dsm],
                       bc, sec], axis=1).fillna(0).astype(np.float32)
    X_nophy = pd.concat([df[geo_mat + derived_geo],
                         bc, sec], axis=1).fillna(0).astype(np.float32)
    return X_phy, X_nophy


# ─────────────────────────────────────────────────────────────────────────────
# 2. 固定的 val/test 划分（全程不变）+ 训练池
# ─────────────────────────────────────────────────────────────────────────────
def fixed_split(y, fm_labels):
    idx = np.arange(len(y))
    i_pool, i_test = train_test_split(idx, test_size=mm.CFG["test_size"],
                                      stratify=fm_labels, random_state=SEED)
    val_rel = mm.CFG["val_size"] / (1 - mm.CFG["test_size"])
    i_train_pool, i_val = train_test_split(i_pool, test_size=val_rel,
                                           stratify=fm_labels[i_pool], random_state=SEED)
    return i_train_pool, i_val, i_test


def subsample_train(i_train_pool, fm_labels, N, seed):
    """在训练池内按破坏模式分层下采样到 N 条。"""
    if N >= len(i_train_pool):
        return i_train_pool
    i_sub, _ = train_test_split(i_train_pool, train_size=N,
                                stratify=fm_labels[i_train_pool], random_state=seed)
    return i_sub


# ─────────────────────────────────────────────────────────────────────────────
# 3. 单次训练（支持 residual / direct + 可选物理损失）
# ─────────────────────────────────────────────────────────────────────────────
def train_variant(X_tr, y_tr, X_vl, y_vl, phy_tr, phy_vl,
                  use_phy, use_loss, cfg, seed):
    torch.manual_seed(seed)
    in_dim = X_tr.shape[1]
    if use_phy:
        model = mm.ResidualCorrectionNet(in_dim, cfg["hidden_dims"], cfg["dropout"])
        mode = "pinn"
    else:
        model = mm.PlainMLP(in_dim, cfg["hidden_dims"], cfg["dropout"])
        mode = "mlp"
    model = model.to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"], eta_min=1e-6)
    phy_sched = mm.PhysicsWeightScheduler(cfg["w_phy_start"], cfg["w_phy_end"], cfg["epochs"])

    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    pdsm = torch.tensor(phy_tr["PDSM_s"], dtype=torch.float32)
    py   = torch.tensor(phy_tr["Py"], dtype=torch.float32)
    lc   = torch.tensor(phy_tr["lambda_c"], dtype=torch.float32)
    ll   = torch.tensor(phy_tr["lambda_crl"], dtype=torch.float32)
    bs = min(cfg["batch_size"], len(yt))
    loader = DataLoader(TensorDataset(Xt, yt, pdsm, py, lc, ll),
                        batch_size=bs, shuffle=True, drop_last=False)

    Xv = torch.tensor(X_vl, dtype=torch.float32).to(DEVICE)
    yv = torch.tensor(y_vl, dtype=torch.float32).to(DEVICE)
    pdsm_v = torch.tensor(phy_vl["PDSM_s"], dtype=torch.float32).to(DEVICE)

    best_val, best_state, wait = np.inf, None, 0
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        w_phy = phy_sched.get(epoch)
        for Xb, yb, pd_b, py_b, lc_b, ll_b in loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            pd_b, py_b = pd_b.to(DEVICE), py_b.to(DEVICE)
            lc_b, ll_b = lc_b.to(DEVICE), ll_b.to(DEVICE)
            out = model(Xb)
            P_pred = (pd_b + out) if use_phy else out
            loss = nn.functional.huber_loss(P_pred, yb, delta=30.0)
            if use_loss:
                L_up  = mm.physics_loss_upper_bound(P_pred, py_b)
                L_dsm = mm.physics_loss_dsm_proximity(P_pred, pd_b, ll_b)
                L_mono = mm.monotonicity_loss(P_pred, lc_b)
                loss = loss + w_phy * (L_up + L_dsm) + cfg["w_mono"] * L_mono
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sch.step()

        model.eval()
        with torch.no_grad():
            out_v = model(Xv)
            P_v = (pdsm_v + out_v) if use_phy else out_v
            val = nn.functional.huber_loss(P_v, yv).item()
        if val < best_val - 1e-4:
            best_val, wait = val, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


def eval_variant(model, X_te, y_te, phy_te, use_phy):
    Xt = torch.tensor(X_te, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        out = model(Xt)
        if use_phy:
            pdsm = torch.tensor(phy_te["PDSM_s"], dtype=torch.float32).to(DEVICE)
            P = (pdsm + out).cpu().numpy()
        else:
            P = out.cpu().numpy()
    P = np.clip(P, 1e-3, None)
    r2 = r2_score(y_te, P)
    rmse = np.sqrt(mean_squared_error(y_te, P))
    ratio = y_te / P
    within10 = float(np.mean(np.abs(ratio - 1.0) <= 0.10))
    viol = float(np.mean(P > phy_te["Py"]))         # 违反率：超过屈服承载力
    return dict(R2=r2, RMSE=rmse, Within10=within10, ViolRate=viol)


# ─────────────────────────────────────────────────────────────────────────────
# 4. 数据效率主扫描
# ─────────────────────────────────────────────────────────────────────────────
def run_efficiency_sweep(sizes=TRAIN_SIZES, seeds=RUN_SEEDS, cfg=EFF_CFG):
    X, y, phy_raw, df = mm.load_and_engineer(mm.CFG["data_path"])
    X_phy, X_nophy = build_feature_matrices(df)
    fm_labels = np.array([FM_ORDER.index(f) for f in phy_raw["FM_merged"]])
    i_pool, i_val, i_test = fixed_split(y, fm_labels)

    # 断点续跑：读取已完成的 (config,N,seed)
    done = set()
    if EFF_CSV.exists():
        prev = pd.read_csv(EFF_CSV)
        done = set(zip(prev["config"], prev["N"], prev["seed"]))
        print(f"  发现已完成记录 {len(done)} 条，将跳过。")

    rows = []
    total = len(FACTORIAL) * len(sizes) * len(seeds)
    cnt = 0
    for cname, fac in FACTORIAL.items():
        Xmat = X_phy if fac["use_phy"] else X_nophy
        Xmat_np = Xmat.values
        for N in sizes:
            for s in seeds:
                cnt += 1
                if (cname, N, s) in done:
                    continue
                i_sub = subsample_train(i_pool, fm_labels, N, s)

                scaler = StandardScaler()
                X_tr = scaler.fit_transform(Xmat_np[i_sub]).astype(np.float32)
                X_vl = scaler.transform(Xmat_np[i_val]).astype(np.float32)
                X_te = scaler.transform(Xmat_np[i_test]).astype(np.float32)

                def slice_phy(ix):
                    return {k: (v[ix] if isinstance(v, np.ndarray) else v)
                            for k, v in phy_raw.items()}
                phy_tr, phy_vl, phy_te = slice_phy(i_sub), slice_phy(i_val), slice_phy(i_test)

                model = train_variant(X_tr, y[i_sub], X_vl, y[i_val],
                                      phy_tr, phy_vl, fac["use_phy"], fac["use_loss"],
                                      cfg, seed=1000 + s)
                m = eval_variant(model, X_te, y[i_test], phy_te, fac["use_phy"])
                row = dict(config=cname, N=N, seed=s, **m)
                rows.append(row)
                print(f"  [{cnt:>3}/{total}] {cname:<5} N={N:<4} seed={s}  "
                      f"R2={m['R2']:.4f} RMSE={m['RMSE']:.2f} Viol={m['ViolRate']:.4f}")
                # 即时写盘
                pd.DataFrame([row]).to_csv(
                    EFF_CSV, mode="a", header=not EFF_CSV.exists(), index=False)

    print(f"\n效率扫描完成 → {EFF_CSV.name}")
    return pd.read_csv(EFF_CSV)


# ─────────────────────────────────────────────────────────────────────────────
# 5. 聚合 + 三联图（R² / RMSE / 跨种子 std(R²) vs N）
# ─────────────────────────────────────────────────────────────────────────────
def summarize_and_plot(res: pd.DataFrame):
    agg = res.groupby(["config", "N"]).agg(
        R2_mean=("R2", "mean"), R2_std=("R2", "std"),
        RMSE_mean=("RMSE", "mean"), Viol_mean=("ViolRate", "mean")).reset_index()

    # gap(N) = R2_Full - R2_D2
    piv = agg.pivot(index="N", columns="config", values="R2_mean")
    if {"Full", "D2"}.issubset(piv.columns):
        gap = (piv["Full"] - piv["D2"]).reset_index().rename(columns={0: "gap"})
        gap.columns = ["N", "gap"]
        print("\ngap(N) = R2_Full - R2_D2:")
        for _, r in gap.iterrows():
            print(f"  N={int(r['N']):>4}: gap = {r['gap']:+.4f}")

    colors = {"Full": "#378ADD", "A1": "#7F77DD", "D1": "#BA7517", "D2": "#D85A30"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    titles = ["(a) Test R^2 vs Training Size",
              "(b) Test RMSE vs Training Size",
              "(c) Stability: std(R^2) across seeds"]
    ycols  = ["R2_mean", "RMSE_mean", "R2_std"]
    ylabs  = ["Test R^2", "Test RMSE (kN)", "std(R^2) across seeds"]
    for ax, title, yc, yl in zip(axes, titles, ycols, ylabs):
        for cname in ["Full", "A1", "D1", "D2"]:
            sub = agg[agg["config"] == cname].sort_values("N")
            if sub.empty:
                continue
            ax.plot(sub["N"], sub[yc], "-o", color=colors[cname], lw=1.8,
                    ms=5, label=cname)
            if yc == "R2_mean":
                lo = sub["R2_mean"] - sub["R2_std"]
                hi = sub["R2_mean"] + sub["R2_std"]
                ax.fill_between(sub["N"], lo, hi, color=colors[cname], alpha=0.12)
        ax.set_xlabel("Training samples N")
        ax.set_ylabel(yl); ax.set_title(title)
        ax.set_xscale("log"); ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    plt.suptitle("Data Efficiency — Physics Prior Benefit vs Training Size",
                 fontsize=13, fontweight="bold", y=1.03)
    out = BASE_DIR / "data_efficiency_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → 数据效率三联图已保存: {out.name}")
    agg.to_csv(BASE_DIR / "data_efficiency_summary.csv", index=False)
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# 6. 全量数据消融（附录）：逐项关闭物理损失
# ─────────────────────────────────────────────────────────────────────────────
def run_full_ablation(cfg=None):
    cfg = cfg or EFF_CFG
    X, y, phy_raw, df = mm.load_and_engineer(mm.CFG["data_path"])
    splits, phy, scaler, i_tr, i_vl, i_te = mm.split_dataset(X, y, phy_raw)
    in_dim = X.shape[1]
    P_true = splits["test"][1].numpy()
    Py_te = phy["test"]["Py"]

    configs = {
        "Full (all on)"        : dict(use_upper=True,  use_dsm=True,  use_mono=True),
        "no upper-bound"       : dict(use_upper=False, use_dsm=True,  use_mono=True),
        "no DSM-proximity"     : dict(use_upper=True,  use_dsm=False, use_mono=True),
        "no monotonicity"      : dict(use_upper=True,  use_dsm=True,  use_mono=False),
        "no physics (data only)": dict(use_upper=False, use_dsm=False, use_mono=False),
    }
    rows = []
    for name, flags in configs.items():
        net = mm.ResidualCorrectionNet(in_dim, cfg["hidden_dims"], cfg["dropout"])
        net, _ = mm.train_model(net, splits, phy, mode="pinn", cfg=cfg,
                                loss_flags=flags, verbose=False)
        P = mm.predict(net, splits, phy, mode="pinn")
        met = mm.compute_metrics(P_true, P)
        viol = float(np.mean(P > Py_te))
        rows.append(dict(config=name, R2=met["R2"], RMSE=met["RMSE"],
                         CoV=met["CoV"], ViolRate=viol))
        print(f"  {name:<24} R2={met['R2']:.4f}  RMSE={met['RMSE']:.2f}  "
              f"CoV={met['CoV']:.4f}  Viol={viol:.4f}")
    out = pd.DataFrame(rows)
    out.to_csv(ABL_CSV, index=False)
    print(f"\n全量消融完成 → {ABL_CSV.name}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation", action="store_true", help="额外运行全量数据消融（附录）")
    args = ap.parse_args()

    print("=" * 60)
    print("  数据效率扫描  (2x2 factorial x sizes x seeds)")
    print("=" * 60)
    res = run_efficiency_sweep()
    summarize_and_plot(res)

    if args.ablation:
        print("\n" + "=" * 60)
        print("  全量数据消融（附录）")
        print("=" * 60)
        run_full_ablation()


if __name__ == "__main__":
    main()