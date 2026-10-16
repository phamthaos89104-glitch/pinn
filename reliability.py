# -*- coding: utf-8 -*-
"""
reliability_phi.py
按 AISI S100 第 K2.1.1 节计算各模型的抗力系数 phi_c（分破坏模式 + 总体）。

复用 main_model 的特征工程、数据划分与模型，保证与主结果（表5-1/5-2）同源：
  phi_c = Cphi * Mm * Fm * Pm * exp(-beta0 * sqrt(VM^2 + VF^2 + C_P*VP^2 + VQ^2))
其中 Pm = mean(Pt/Ppred)，VP = CoV(Pt/Ppred) = std/mean，
C_P = (1 + 1/n) * m / (m - 2)，m = n - 1，为 AISI S100 规定的样本量修正系数。
总体按 n = 156 计算，分破坏模式时按各模式自身的试件数计算。

用法（在项目目录下，与 main_model.py 同级）：
    python reliability_phi.py
输出：控制台分模式 phi_c 表 + reliability_results.csv
"""
import sys, csv
from pathlib import Path
import numpy as np

# 复用主模型组件（与 ablation_efficiency.py / shap_analysis.py 同样的导入方式）
import main_model as mm
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

BASE_DIR = Path(__file__).resolve().parent

# ── AISI S100 K2.1.1 可靠度参数（与论文 4.4.4 一致）──────────────────────
CPHI, MM, FM = 1.52, 1.10, 1.00
BETA0        = 2.5
VM, VF, VQ   = 0.10, 0.05, 0.21
PHI_TARGET   = 0.85   # 规范建议下限


def c_p(n: int) -> float:
    """AISI S100 K2.1.1 样本量修正系数：C_P = (1 + 1/n) * m / (m - 2)，m = n - 1。"""
    m = n - 1
    return (1.0 + 1.0 / n) * m / (m - 2)


def phi_c(Pm: float, VP: float, n: int) -> float:
    """由实测-预测比的均值 Pm、变异系数 VP 与试件数 n 计算抗力系数。"""
    return CPHI * MM * FM * Pm * np.exp(
        -BETA0 * np.sqrt(VM**2 + VF**2 + c_p(n) * VP**2 + VQ**2))


def ratio_stats(P_true: np.ndarray, P_pred: np.ndarray):
    """返回 (n, Pm, VP)；VP 为总体 CoV（ddof=0，与 compute_metrics 一致）。"""
    r = P_true / np.clip(P_pred, 1e-6, None)
    Pm = float(r.mean())
    VP = float(r.std() / r.mean()) if r.mean() != 0 else float("nan")
    return len(r), Pm, VP


def main():
    print("=" * 64)
    print("  可靠度分析：分破坏模式抗力系数 phi_c (AISI S100 K2.1.1)")
    print("=" * 64)

    # 1) 复用主模型的加载、划分（同 SEED，保证与表5-1同一测试集）
    X, y, phy_raw, df = mm.load_and_engineer(mm.CFG["data_path"])
    splits, phy, scaler, i_tr, i_vl, i_te = mm.split_dataset(X, y, phy_raw)
    in_dim  = X.shape[1]
    P_true  = splits["test"][1].numpy()
    fm_test = np.asarray(phy["test"]["FM_merged"])
    print(f"  测试集样本数: {len(P_true)} | 特征维度: {in_dim}")

    # 2) 训练四个模型（与 main() 完全一致的顺序与超参）
    print("  训练 PINN ...")
    pinn = mm.ResidualCorrectionNet(in_dim, mm.CFG["hidden_dims"], mm.CFG["dropout"])
    pinn, _ = mm.train_model(pinn, splits, phy, mode="pinn")

    print("  训练 纯MLP ...")
    mlp = mm.PlainMLP(in_dim, mm.CFG["hidden_dims"], mm.CFG["dropout"])
    mlp, _ = mm.train_model(mlp, splits, phy, mode="mlp")

    print("  训练 RF / XGBoost ...")
    Xtr, ytr = splits["train"][0].numpy(), splits["train"][1].numpy()
    Xte      = splits["test"][0].numpy()
    rf = RandomForestRegressor(n_estimators=300, max_features=0.6,
                               min_samples_leaf=2, random_state=mm.SEED, n_jobs=-1).fit(Xtr, ytr)
    xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5,
                       subsample=0.8, colsample_bytree=0.8,
                       random_state=mm.SEED, n_jobs=-1, verbosity=0).fit(Xtr, ytr)

    # 3) 各模型测试集预测
    preds = {
        "DSM" : np.asarray(phy["test"]["PDSM_s"]),
        "RF"  : np.clip(rf.predict(Xte), 1e-3, None),
        "XGB" : np.clip(xgb.predict(Xte), 1e-3, None),
        "MLP" : mm.predict(mlp,  splits, phy, mode="mlp"),
        "PINN": mm.predict(pinn, splits, phy, mode="pinn"),
    }

    # 4) 分模式 + 总体计算 phi_c
    rows = []
    modes = list(mm.FM_ORDER)          # ["L","L+F","F","D","L+D"]
    for model, P_pred in preds.items():
        # 总体
        n, Pm, VP = ratio_stats(P_true, P_pred)
        rows.append([model, "ALL", n, round(Pm, 4), round(VP, 4),
                     round(phi_c(Pm, VP, n), 4)])
        # 分模式
        for fm in modes:
            mask = (fm_test == fm)
            if mask.sum() < 3:         # 样本过少不单独报 phi_c
                rows.append([model, fm, int(mask.sum()), "-", "-", "-"])
                continue
            n, Pm, VP = ratio_stats(P_true[mask], P_pred[mask])
            rows.append([model, fm, n, round(Pm, 4), round(VP, 4),
                         round(phi_c(Pm, VP, n), 4)])

    # 5) 打印
    hdr = f"{'模型':<6}{'模式':<6}{'N':>4}{'Pm':>9}{'VP(CoV)':>10}{'phi_c':>9}  安全?"
    print("\n" + hdr)
    print("-" * len(hdr))
    for model, fm, n, Pm, VP, pc in rows:
        flag = ""
        if pc != "-":
            flag = "OK" if pc >= PHI_TARGET else "< 0.85"
        sep = "\n" if fm == "ALL" else ""   # 每个模型前空行
        print(f"{sep}{model:<6}{fm:<6}{n:>4}{str(Pm):>9}{str(VP):>10}{str(pc):>9}  {flag}")

    # 6) 写 CSV
    out = BASE_DIR / "reliability_results.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["model", "failure_mode", "N", "Pm", "VP", "phi_c", "safe(>=0.85)"])
        for model, fm, n, Pm, VP, pc in rows:
            safe = "" if pc == "-" else ("yes" if pc >= PHI_TARGET else "no")
            w.writerow([model, fm, n, Pm, VP, pc, safe])
    print(f"\n  已写出: {out}")
    print(f"  参数: Cphi={CPHI}, Mm={MM}, Fm={FM}, beta0={BETA0}, "
          f"VM={VM}, VF={VF}, VQ={VQ}")
    print(f"  样本量修正: C_P = (1+1/n)*m/(m-2), m = n-1;  n=156 时 C_P = {c_p(156):.4f}")


if __name__ == "__main__":
    main()