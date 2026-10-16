# -*- coding: utf-8 -*-
"""问题3（贡献分解的彻底性）：把"DSM 知识"拆成两条独立通道（特征 vs 残差架构），
并加入树模型（± DSM 特征）的数据效率曲线。

  特征通道 = MLP_F − MLP_0 ；  架构通道 = Full_RF − MLP_F ；  树是否受益于DSM特征 = XGB_F − XGB_0

【v2 修订】树模型此前用 i_sub + i_val 拟合，而神经网络只用 i_sub 训练（验证集仅作早停），
  在 N=50 时树实际见到 206 个样本、是神经网络的 4 倍，对照不公平且高估树基线。
  现改为树模型同样只用 i_sub 拟合，与神经网络的训练样本量严格一致。

★ 本脚本【完全自包含】：只需与 main_model.py、ablation_efficiency.py 放在同一文件夹，
  不依赖 _cv_utils.py。
运行：python ablation_decompose.py
输出：ablation_decompose_results.csv + 屏幕分解表
"""
import os, importlib.util
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# ── 按绝对路径加载同目录模块 ──
_here = os.path.dirname(os.path.abspath(__file__))
def _load(fname):
    p = os.path.join(_here, fname)
    assert os.path.exists(p), f"未找到 {p}；请把本脚本放到 {fname} 所在文件夹"
    s = importlib.util.spec_from_file_location(fname[:-3], p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
mm = _load("main_model.py")
ab = _load("ablation_efficiency.py")

def _rf():
    return RandomForestRegressor(n_estimators=300, max_features=0.6,
                                 min_samples_leaf=2, random_state=mm.SEED, n_jobs=-1)
def _xgb():
    return XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5,
                        subsample=0.8, colsample_bytree=0.8,
                        random_state=mm.SEED, n_jobs=-1, verbosity=0)

SIZES, SEEDS, CFG = [50, 100, 200, 400, 725], [0, 1, 2, 3, 4], ab.EFF_CFG
OUT = "ablation_decompose_results.csv"

X, y, phy_raw, df = mm.load_and_engineer(mm.CFG["data_path"])
Xphy, Xnophy = ab.build_feature_matrices(df)               # 含 / 不含 DSM 特征
fm = np.array([mm.FM_ORDER.index(f) for f in phy_raw["FM_merged"]])
i_pool, i_val, i_test = ab.fixed_split(y, fm)

def slice_phy(ix):
    return {k: (v[ix] if isinstance(v, np.ndarray) else v) for k, v in phy_raw.items()}

# NN：(用哪个特征矩阵, 是否残差架构)。残差架构内部以 PDSM 为基准。
NN = {"Full_RF": ("phy", True), "MLP_F": ("phy", False), "MLP_0": ("nophy", False)}
TREE = {"XGB_F": ("phy", "xgb"), "XGB_0": ("nophy", "xgb"),
        "RF_F": ("phy", "rf"),   "RF_0": ("nophy", "rf")}

rows = []
for N in SIZES:
    for s in SEEDS:
        i_sub = ab.subsample_train(i_pool, fm, N, s)
        phy_tr, phy_vl, phy_te = slice_phy(i_sub), slice_phy(i_val), slice_phy(i_test)
        for name, (feat, resid) in NN.items():
            Xm = Xphy if feat == "phy" else Xnophy
            sc = StandardScaler()
            Xtr = sc.fit_transform(Xm.values[i_sub]).astype(np.float32)
            Xvl = sc.transform(Xm.values[i_val]).astype(np.float32)
            Xte = sc.transform(Xm.values[i_test]).astype(np.float32)
            net = ab.train_variant(Xtr, y[i_sub], Xvl, y[i_val], phy_tr, phy_vl,
                                   use_phy=resid, use_loss=False, cfg=CFG, seed=1000 + s)
            met = ab.eval_variant(net, Xte, y[i_test], phy_te, use_phy=resid)
            rows.append(dict(config=name, N=N, seed=s, R2=met["R2"], RMSE=met["RMSE"]))
        for name, (feat, kind) in TREE.items():
            Xm = Xphy if feat == "phy" else Xnophy
            # ★ v2：只用 i_sub 拟合（原为 i_sub + i_val），与神经网络训练样本量一致
            i_fit = i_sub
            mdl = (_xgb() if kind == "xgb" else _rf()).fit(Xm.values[i_fit], y[i_fit])
            m = mm.compute_metrics(y[i_test], mdl.predict(Xm.values[i_test]))
            rows.append(dict(config=name, N=N, seed=s, R2=m["R2"], RMSE=m["RMSE"]))
        print(f"  N={N:<4} seed={s} done")

res = pd.DataFrame(rows); res.to_csv(OUT, index=False)
agg = res.groupby(["config", "N"]).R2.mean().unstack(0)
print("\n=== 各配置 R²（5种子均值）===")
print(agg.round(4))

print("\n=== 贡献分解（R² 均值）===")
print(f"  {'N':>5}{'特征通道(MLP_F-MLP_0)':>24}{'架构通道(Full-MLP_F)':>24}{'树受益DSM(XGB_F-XGB_0)':>26}")
for N in SIZES:
    a = agg.loc[N]
    print(f"  {N:>5}{a['MLP_F']-a['MLP_0']:>24.4f}{a['Full_RF']-a['MLP_F']:>24.4f}{a['XGB_F']-a['XGB_0']:>26.4f}")

print("\n=== 与树模型基线的公平对照（同训练样本量）===")
print(f"  {'N':>5}{'Full_RF':>10}{'XGB_F':>10}{'XGB_0':>10}{'Full-XGB_F':>13}{'Full-XGB_0':>13}")
for N in SIZES:
    a = agg.loc[N]
    print(f"  {N:>5}{a['Full_RF']:>10.4f}{a['XGB_F']:>10.4f}{a['XGB_0']:>10.4f}"
          f"{a['Full_RF']-a['XGB_F']:>13.4f}{a['Full_RF']-a['XGB_0']:>13.4f}")

# 跨种子标准差（稳定性）
sd = res.groupby(["config", "N"]).R2.std().unstack(0)
print("\n=== 跨种子 R² 标准差（稳定性，越小越好）===")
print(sd.round(4).to_string())
sd.to_csv("ablation_decompose_std.csv")