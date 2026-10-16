# -*- coding: utf-8 -*-
"""问题1（重复性）：重复分层 K 折交叉验证 + 配对 Wilcoxon 检验，
给出表5-1各模型的均值±标准差，并统计验证"PINN≈MLP≈XGB（全量精度无优势）"。

【v2 修订】树模型此前用 i_tr + i_vl 拟合、神经网络只用 i_tr 训练，样本量不一致；
  现统一为只用 i_tr，使表 5-1 各模型的训练数据完全相同。

★ 本脚本【完全自包含】：只需与 main_model.py 放在同一文件夹即可，不依赖 _cv_utils.py。
运行：python cv_significance.py
输出：cv_table51_results.csv（逐折逐模型指标）+ 屏幕汇总
"""
import os, importlib.util
import numpy as np, pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from scipy.stats import wilcoxon

# ── 按绝对路径加载同目录的 main_model.py（此机制已在你环境验证可用）──
_MM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main_model.py")
assert os.path.exists(_MM), f"未找到 {_MM}；请把本脚本放到 main_model.py 所在文件夹"
_spec = importlib.util.spec_from_file_location("main_model", _MM)
mm = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(mm)


# ── 内联工具：给定折索引，训练 DSM/RF/XGB/MLP/PINN 并返回指标 ──
def _rf():
    return RandomForestRegressor(n_estimators=300, max_features=0.6,
                                 min_samples_leaf=2, random_state=mm.SEED, n_jobs=-1)

def _xgb():
    return XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5,
                        subsample=0.8, colsample_bytree=0.8,
                        random_state=mm.SEED, n_jobs=-1, verbosity=0)

def _splits(Xdf, y, phy_raw, i_tr, i_vl, i_te):
    sc = StandardScaler()
    Xtr = sc.fit_transform(Xdf.iloc[i_tr]).astype(np.float32)
    Xvl = sc.transform(Xdf.iloc[i_vl]).astype(np.float32)
    Xte = sc.transform(Xdf.iloc[i_te]).astype(np.float32)
    t = lambda a: torch.tensor(a, dtype=torch.float32)
    splits = {"train": (t(Xtr), t(y[i_tr])), "val": (t(Xvl), t(y[i_vl])), "test": (t(Xte), t(y[i_te]))}
    phy = {n: {k: (v[ix] if isinstance(v, np.ndarray) else v) for k, v in phy_raw.items()}
           for n, ix in [("train", i_tr), ("val", i_vl), ("test", i_te)]}
    return splits, phy

def fit_eval_models(Xdf, y, phy_raw, i_tr, i_vl, i_te, cfg, seed):
    out, yte = {}, y[i_te]
    out["dsm"] = mm.compute_metrics(yte, phy_raw["PDSM_s"][i_te])
    # ★ v2：树模型只用 i_tr 拟合（原为 i_tr + i_vl），与神经网络训练样本量一致
    i_fit = i_tr
    Xfit, Xte_raw = Xdf.iloc[i_fit].values, Xdf.iloc[i_te].values
    out["rf"]  = mm.compute_metrics(yte, _rf().fit(Xfit, y[i_fit]).predict(Xte_raw))
    out["xgb"] = mm.compute_metrics(yte, _xgb().fit(Xfit, y[i_fit]).predict(Xte_raw))
    splits, phy = _splits(Xdf, y, phy_raw, i_tr, i_vl, i_te)
    in_dim = splits["train"][0].shape[1]
    torch.manual_seed(seed)
    net = mm.PlainMLP(in_dim, cfg["hidden_dims"], cfg["dropout"])
    net, _ = mm.train_model(net, splits, phy, mode="mlp", cfg=cfg, verbose=False)
    out["mlp"] = mm.compute_metrics(yte, mm.predict(net, splits, phy, mode="mlp"))
    torch.manual_seed(seed)
    net = mm.ResidualCorrectionNet(in_dim, cfg["hidden_dims"], cfg["dropout"])
    net, _ = mm.train_model(net, splits, phy, mode="pinn", cfg=cfg,
                            loss_flags=dict(use_upper=True, use_dsm=True, use_mono=True), verbose=False)
    out["pinn"] = mm.compute_metrics(yte, mm.predict(net, splits, phy, mode="pinn"))
    return out


# ── 实验：重复分层 K 折 ──
K, REPEATS = 5, 2          # 5×2=10 折评估；要更稳可设 (10, 3)
OUT = "cv_table51_results.csv"

X, y, phy_raw, df = mm.load_and_engineer(mm.CFG["data_path"])
fm = np.array([mm.FM_ORDER.index(f) for f in phy_raw["FM_merged"]])

rows, fid = [], 0
for rep in range(REPEATS):
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=100 + rep)
    for tr, te in skf.split(np.arange(len(y)), fm):
        tr2, vl = train_test_split(tr, test_size=0.15, stratify=fm[tr], random_state=100 + rep)
        res = fit_eval_models(X, y, phy_raw, tr2, vl, te, cfg=mm.CFG, seed=1000 + fid)
        for mdl, met in res.items():
            rows.append(dict(fold=fid, model=mdl, **{k: met[k] for k in ("R2", "RMSE", "MAE", "CoV")}))
        fid += 1
        print(f"fold {fid:>2}:  " + "  ".join(f"{m}={res[m]['R2']:.3f}"
                                              for m in ["dsm", "rf", "xgb", "mlp", "pinn"]))

res = pd.DataFrame(rows); res.to_csv(OUT, index=False)

print(f"\n=== 表5-1（{fid} 折 CV，均值±标准差）===")
for mdl in ["dsm", "rf", "xgb", "mlp", "pinn"]:
    s = res[res.model == mdl]
    print(f"  {mdl:<6} R²={s.R2.mean():.3f}±{s.R2.std():.3f}  "
          f"RMSE={s.RMSE.mean():.2f}±{s.RMSE.std():.2f}  CoV={s.CoV.mean():.3f}±{s.CoV.std():.3f}")

print("\n=== 配对 Wilcoxon（PINN vs 基线，按折配对 R²）===")
pinn = res[res.model == "pinn"].sort_values("fold").R2.values
for mdl in ["dsm", "rf", "xgb", "mlp"]:
    other = res[res.model == mdl].sort_values("fold").R2.values
    stat, p = wilcoxon(pinn, other)
    verdict = "★ 显著差异" if p < 0.05 else "无显著差异 (p>0.05)"
    print(f"  PINN vs {mdl:<4}:  ΔR²(均值)={pinn.mean()-other.mean():+.4f}   p={p:.3f}   → {verdict}")