# -*- coding: utf-8 -*-
"""问题2（外部验证）：(a) 按试件系列分组 K 折（LOSO 代理）；(b) 物理外推划分。
检验泛化：预期 DSM 锚定的 PINN 在未见系列/外推区的退化显著小于纯黑箱。

【v2 修订】
  1. 物理外推划分改为 5 个种子重复（原为单次），报均值±标准差；外推域本身由物理
     判据确定、固定不变，仅训练/验证划分与网络初始化随种子变化。
  2. 树模型此前用 i_tr + i_vl 拟合、神经网络只用 i_tr 训练，样本量不一致；现统一为
     只用 i_tr，保证各模型训练数据完全相同。
  3. 新增外推域上的物理约束违反率 ViolRate = mean(P_pred > Py) 与实测预测比统计
     (Pm, CoV)，用于检验"物理约束在外推场景下是否由非激活转为激活"。

★ 本脚本【完全自包含】：只需与 main_model.py 放在同一文件夹，不依赖 _cv_utils.py。
运行：python generalization_tests.py
输出：generalization_results.csv + 屏幕汇总
"""
import os, importlib.util
import numpy as np, pandas as pd
import torch
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# ── 按绝对路径加载同目录的 main_model.py ──
_MM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main_model.py")
assert os.path.exists(_MM), f"未找到 {_MM}；请把本脚本放到 main_model.py 所在文件夹"
_spec = importlib.util.spec_from_file_location("main_model", _MM)
mm = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(mm)


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

def fit_eval_models(Xdf, y, phy_raw, i_tr, i_vl, i_te, seed=42):
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
    mlp_net = mm.PlainMLP(in_dim, mm.CFG["hidden_dims"], mm.CFG["dropout"])
    mlp_net, _ = mm.train_model(mlp_net, splits, phy, mode="mlp", cfg=mm.CFG, verbose=False)
    out["mlp"] = mm.compute_metrics(yte, mm.predict(mlp_net, splits, phy, mode="mlp"))
    torch.manual_seed(seed)
    net = mm.ResidualCorrectionNet(in_dim, mm.CFG["hidden_dims"], mm.CFG["dropout"])
    net, _ = mm.train_model(net, splits, phy, mode="pinn", cfg=mm.CFG,
                            loss_flags=dict(use_upper=True, use_dsm=True, use_mono=True), verbose=False)
    out["pinn"] = mm.compute_metrics(yte, mm.predict(net, splits, phy, mode="pinn"))

    # ★ v2：记录各模型在测试域上的上界违反率 mean(P_pred > Py)
    Py_te = phy_raw["Py"][i_te]
    preds = {"dsm": phy_raw["PDSM_s"][i_te],
             "rf":  _rf().fit(Xfit, y[i_fit]).predict(Xte_raw),
             "xgb": _xgb().fit(Xfit, y[i_fit]).predict(Xte_raw),
             "mlp": mm.predict(mlp_net, splits, phy, mode="mlp"),
             "pinn": mm.predict(net, splits, phy, mode="pinn")}
    for k, P in preds.items():
        out[k]["ViolRate"] = float(np.mean(np.asarray(P) > Py_te))
    return out


X, y, phy_raw, df = mm.load_and_engineer(mm.CFG["data_path"])
fm = np.array([mm.FM_ORDER.index(f) for f in phy_raw["FM_merged"]])
rows = []

# ── (a) 按试件系列分组 5 折 ──
prefix = df["Test_Label"].astype(str).str.extract(r'^([A-Za-z\-_&]+)')[0].fillna("NA")
print(f"试件系列组数: {prefix.nunique()}")
print("\n===== (a) 按系列分组 5 折（泛化到未见系列）=====")
for k, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, y, groups=prefix)):
    tr2, vl = train_test_split(tr, test_size=0.15, random_state=42)
    res = fit_eval_models(X, y, phy_raw, tr2, vl, te, seed=1000 + k)
    for m, met in res.items():
        rows.append(dict(block="series_cv", fold=k, seed=k, model=m, R2=met["R2"],
                         RMSE=met["RMSE"], Pm=met["MeanRatio"], CoV=met["CoV"],
                         ViolRate=met["ViolRate"]))
    print(f"  fold{k}:  " + "  ".join(f"{m}={res[m]['R2']:.3f}" for m in ["dsm", "xgb", "mlp", "pinn"]))
g = pd.DataFrame([r for r in rows if r["block"] == "series_cv"])
print("\n  分组CV均值±标准差:")
for m in ["dsm", "rf", "xgb", "mlp", "pinn"]:
    s = g[g.model == m]
    print(f"    {m:<5} R²={s.R2.mean():.3f}±{s.R2.std():.3f}")

# ── (b) 物理外推划分（v2：5 种子重复）──
EXTRAP_SEEDS = [0, 1, 2, 3, 4]

def extrapolate(name, train_mask):
    """外推域由物理判据固定；训练/验证划分与网络初始化随种子变化。"""
    i_in, i_out = np.where(train_mask)[0], np.where(~train_mask)[0]
    print(f"\n[{name}]  训练域 n={len(i_in)} → 外推测试 n={len(i_out)}")
    per_model = {m: [] for m in ["dsm", "rf", "xgb", "mlp", "pinn"]}
    for sd in EXTRAP_SEEDS:
        tr, vl = train_test_split(i_in, test_size=0.15, random_state=200 + sd)
        res = fit_eval_models(X, y, phy_raw, tr, vl, i_out, seed=2000 + sd)
        for m, met in res.items():
            per_model[m].append(met)
            rows.append(dict(block=name, fold=sd, seed=sd, model=m, R2=met["R2"],
                             RMSE=met["RMSE"], Pm=met["MeanRatio"], CoV=met["CoV"],
                             ViolRate=met["ViolRate"]))
        print(f"    seed={sd}:  " + "  ".join(f"{m}={res[m]['R2']:+.3f}"
                                              for m in ["dsm", "xgb", "mlp", "pinn"]))
    print(f"    {'模型':<6}{'R² 均值±标准差':>20}{'RMSE':>10}{'Pm':>8}{'违反率':>9}")
    for m in ["dsm", "rf", "xgb", "mlp", "pinn"]:
        r2 = np.array([x["R2"] for x in per_model[m]])
        rm = np.mean([x["RMSE"] for x in per_model[m]])
        pm = np.mean([x["MeanRatio"] for x in per_model[m]])
        vr = np.mean([x["ViolRate"] for x in per_model[m]])
        print(f"    {m:<6}{r2.mean():>12.3f}±{r2.std():<7.3f}{rm:>10.1f}{pm:>8.3f}{vr:>9.4f}")

KLrb, t, cat = df["KL/rb"].values, df["t(mm)"].values, df["Sections_Category"].values
q_kl, q_t = np.quantile(KLrb, 0.75), np.quantile(t, 0.75)
print("\n===== (b) 物理外推（训练域内 → 测试域外，5 种子重复）=====")
extrapolate(f"细长比外推(训练KL/rb<={q_kl:.0f}, 测试更长柱)", KLrb <= q_kl)
extrapolate(f"厚度外推(训练t<={q_t:.2f}, 测试更厚)", t <= q_t)
extrapolate("截面外推(训练Open+Half-Closed, 测试Closed)", cat != "Closed")

pd.DataFrame(rows).to_csv("generalization_results.csv", index=False)
print("\n→ 已写出 generalization_results.csv")