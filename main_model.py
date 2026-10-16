"""
CFS 拼合柱轴压承载力预测 —— 步骤 2：主模型
=====================================================================
数据集  : All_Dataset_cleaned1.csv
目标变量 : Ptest(kN) — 极限轴压承载力
物理先验 : DSM 直接强度法（AISI S100）柱强度方程
架构     : 残差修正网络  P_pred = PDSM_s + ΔP

模型与基准：DSM(PDSM_s) / RandomForest / XGBoost / 纯MLP / PINN(本文)

运行：python main_model.py
依赖：pip install torch scikit-learn pandas numpy matplotlib xgboost

说明：本文件中的特征工程、数据划分、网络、损失、训练、评估函数均可被
      步骤 3（消融/效率）与步骤 4（SHAP）直接 import 复用。

【v2 新增】物理约束激活率记录（见 3.5 节）：三个物理损失函数在计算时顺带统计
      约束的实际触发情况（越界样本比例、进入约束的样本比例、单调性过阈值对数等），
      不参与梯度、不改变任何损失数值与训练结果。训练结束打印摘要，main() 另导出
      activation_log.csv（逐 epoch）。因记录器位于损失函数内部，所有 import 本模块的
      脚本均自动获得该记录，无需额外运行。

【v2 新增】main() 另导出 main_table1.csv / main_table2.csv，正文数字直接取自 CSV，
      避免从控制台手抄造成的版本错位。
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# ─────────────────────────────────────────────────────────────────────────────
# 0. 全局设置
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = Path(__file__).resolve().parent           # 以脚本目录为基准

CFG = {
    "data_path"   : str(BASE_DIR / "All_Dataset_cleaned1.csv"),
    "target"      : "Ptest(kN)",
    "test_size"   : 0.15,
    "val_size"    : 0.15,
    # 网络结构
    "hidden_dims" : [128, 256, 256, 128],
    "dropout"     : 0.10,
    # 训练参数
    "lr"          : 3e-4,
    "weight_decay": 1e-5,
    "epochs"      : 1000,
    "batch_size"  : 128,
    "patience"    : 60,
    # 物理损失权重（自适应调度，见 PhysicsWeightScheduler）
    "w_phy_start" : 0.05,    # 前期小，让网络先拟合数据
    "w_phy_end"   : 0.30,    # 后期增大，强化物理约束
    "w_mono"      : 0.10,    # 单调性约束权重
}

FM_ORDER  = ["L", "L+F", "F", "D", "L+D"]
FM_COLORS = {
    "L"  : "#378ADD", "L+F": "#7F77DD", "F"  : "#639922",
    "D"  : "#BA7517", "L+D": "#D85A30",
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. 数据加载与特征工程
# ─────────────────────────────────────────────────────────────────────────────
def load_and_engineer(path: str):
    df = pd.read_csv(path, encoding="utf-8-sig")

    # 1.1 数值特征
    geo_mat = ["L(mm)", "t(mm)", "h(mm)", "b(mm)", "Area(mm)2", "FS (mm)",
               "Fy(Mpa)", "E(Mpa)"]
    dsm_feats = ["λc_b", "λ(crl,crd)_s", "λ(crl,crd)_ms",
                 "KL/rb", "Pne_b", "PDSM_s(kN)", "PDSM_ms(kN)", "Pcre (kN)"]

    # 1.2 衍生特征
    df["h_over_t"]  = df["h(mm)"]      / df["t(mm)"].clip(lower=1e-6)
    df["b_over_t"]  = df["b(mm)"]      / df["t(mm)"].clip(lower=1e-6)
    df["L_over_h"]  = df["L(mm)"]      / df["h(mm)"].clip(lower=1e-6)
    df["norm_PDSM"] = df["PDSM_s(kN)"] / df["Py(kN)"].clip(lower=1e-6)
    derived = ["h_over_t", "b_over_t", "L_over_h", "norm_PDSM"]

    # 1.3 类别特征独热编码
    bc_dummies  = pd.get_dummies(df["BC"],                prefix="BC").astype(float)
    sec_dummies = pd.get_dummies(df["Sections_Category"], prefix="SEC").astype(float)
    # fm_dummies  = pd.get_dummies(df["FM_merged"],         prefix="FM").astype(float)

    # 1.4 拼接特征矩阵
    feat_df = pd.concat(
        [df[geo_mat + dsm_feats + derived], bc_dummies, sec_dummies],#去除破坏模式FM
        axis=1
    ).fillna(0).astype(np.float32)

    y = df[CFG["target"]].values.astype(np.float32)

    # 1.5 物理约束辅助列（保留原始物理量，不归一化）
    phy_raw = {
        "PDSM_s"    : df["PDSM_s(kN)"].values.astype(np.float32),
        "Py"        : df["Py(kN)"].values.astype(np.float32),
        "lambda_c"  : df["λc_b"].values.astype(np.float32),
        "lambda_crl": df["λ(crl,crd)_s"].values.astype(np.float32),
        "FM_merged" : df["FM_merged"].to_numpy(dtype=str),
    }

    print(f"特征维度: {feat_df.shape[1]}")
    return feat_df, y, phy_raw, df


# ─────────────────────────────────────────────────────────────────────────────
# 2. 数据集划分（按破坏模式分层，70/15/15）
# ─────────────────────────────────────────────────────────────────────────────
def split_dataset(X, y, phy_raw, seed=SEED):
    idx = np.arange(len(y))
    fm_labels = np.array([FM_ORDER.index(f) for f in phy_raw["FM_merged"]])

    i_tv, i_test = train_test_split(
        idx, test_size=CFG["test_size"], stratify=fm_labels, random_state=seed)
    val_rel = CFG["val_size"] / (1 - CFG["test_size"])
    i_train, i_val = train_test_split(
        i_tv, test_size=val_rel, stratify=fm_labels[i_tv], random_state=seed)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X.iloc[i_train]).astype(np.float32)
    X_vl = scaler.transform(X.iloc[i_val]).astype(np.float32)
    X_te = scaler.transform(X.iloc[i_test]).astype(np.float32)

    def t(arr):
        return torch.tensor(arr, dtype=torch.float32)

    splits = {
        "train": (t(X_tr), t(y[i_train])),
        "val"  : (t(X_vl), t(y[i_val])),
        "test" : (t(X_te), t(y[i_test])),
    }
    phy = {}
    for name, ix in [("train", i_train), ("val", i_val), ("test", i_test)]:
        phy[name] = {k: (v[ix] if isinstance(v, np.ndarray) else v)
                     for k, v in phy_raw.items()}

    print(f"\n划分结果 — Train:{len(i_train)}  Val:{len(i_val)}  Test:{len(i_test)}")
    for name, ix in [("train", i_train), ("val", i_val), ("test", i_test)]:
        counts = {fm: int((fm_labels[ix] == FM_ORDER.index(fm)).sum()) for fm in FM_ORDER}
        print(f"  {name:5s}: {counts}")
    return splits, phy, scaler, i_train, i_val, i_test


# ─────────────────────────────────────────────────────────────────────────────
# 3. 网络架构
# ─────────────────────────────────────────────────────────────────────────────
class ResBlock(nn.Module):
    """Pre-activation 残差块"""
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.block(x)


class ResidualCorrectionNet(nn.Module):
    """残差修正网络：P_pred = PDSM_s + ΔP（网络只学残差 ΔP）"""
    def __init__(self, in_dim: int, hidden_dims: list, dropout: float):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden_dims[0]), nn.GELU())
        layers = []
        for i in range(len(hidden_dims) - 1):
            if hidden_dims[i] != hidden_dims[i + 1]:
                layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
                layers.append(nn.GELU())
            layers.append(ResBlock(hidden_dims[i + 1], dropout))
        self.tower = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dims[-1]), nn.Linear(hidden_dims[-1], 64),
            nn.GELU(), nn.Linear(64, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.head(self.tower(self.proj(x))).squeeze(-1)


class PlainMLP(nn.Module):
    """基准纯 MLP（无物理约束，直接预测 Ptest）；架构与残差网相同便于公平对比"""
    def __init__(self, in_dim: int, hidden_dims: list, dropout: float):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden_dims[0]), nn.GELU())
        layers = []
        for i in range(len(hidden_dims) - 1):
            if hidden_dims[i] != hidden_dims[i + 1]:
                layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
                layers.append(nn.GELU())
            layers.append(ResBlock(hidden_dims[i + 1], dropout))
        self.tower = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dims[-1]), nn.Linear(hidden_dims[-1], 64),
            nn.GELU(), nn.Linear(64, 1), nn.Softplus(),   # 输出 > 0
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.head(self.tower(self.proj(x))).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# 3.5 物理约束激活率记录器
#   目的：把“物理约束是否被触发”从间接推断变成直接测量。
#   三个损失函数在计算时顺带记录激活统计，不参与梯度、不改变任何损失数值。
#   任何 import 本模块的脚本（消融/效率/泛化）都会自动获得记录。
#   用 set_activation_recording(False) 可关闭。
# ─────────────────────────────────────────────────────────────────────────────
RECORD_ACTIVATION = True
_ACT_BATCH = {}      # 当前 batch 的统计
_ACT_EPOCH = []      # 当前 epoch 累积的各 batch 统计


def set_activation_recording(flag: bool):
    """全局开关；关闭后所有记录逻辑短路，开销为零。"""
    global RECORD_ACTIVATION
    RECORD_ACTIVATION = flag


def _act(key, value):
    if RECORD_ACTIVATION:
        _ACT_BATCH[key] = float(value)


def _flush_batch():
    """把当前 batch 的统计推入 epoch 缓冲。由 total_loss_pinn 在末尾调用。"""
    if RECORD_ACTIVATION and _ACT_BATCH:
        _ACT_EPOCH.append(dict(_ACT_BATCH))
        _ACT_BATCH.clear()


def pop_epoch_activation():
    """取出并清空本 epoch 的激活统计，返回各项均值。"""
    global _ACT_EPOCH
    if not _ACT_EPOCH:
        return {}
    keys = _ACT_EPOCH[0].keys()
    out = {k: float(np.mean([r.get(k, np.nan) for r in _ACT_EPOCH])) for k in keys}
    _ACT_EPOCH = []
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 4. 物理损失函数
# ─────────────────────────────────────────────────────────────────────────────
def physics_loss_upper_bound(P_pred, Py):
    """上界约束：P_pred 不应超过屈服承载力 Py。软约束 L = mean(relu(P_pred - Py))"""
    if RECORD_ACTIVATION:
        with torch.no_grad():
            r = (P_pred / Py.clamp(min=1e-6))
            _act("upper_frac_viol", (P_pred > Py).float().mean())   # 越界样本比例
            _act("upper_max_ratio", r.max())                        # 最大 P/Py
            _act("upper_mean_ratio", r.mean())
    return torch.relu(P_pred - Py).mean()


def physics_loss_dsm_proximity(P_pred, PDSM_s, lambda_crl):
    """DSM 接近性：λ_crl ≤ 0.776（局部屈曲不控制）时，预测应在 PDSM_s 的 ±30% 内"""
    mask = lambda_crl <= 0.776
    pred_m = P_pred[mask]
    if RECORD_ACTIVATION:
        with torch.no_grad():
            _act("dsm_frac_in_scope", mask.float().mean())          # 进入约束的样本比例
    if pred_m.numel() == 0:
        _act("dsm_frac_outside_band", 0.0)
        return torch.tensor(0.0, device=P_pred.device)
    upper = 1.30 * PDSM_s[mask]
    lower = 0.70 * PDSM_s[mask]
    if RECORD_ACTIVATION:
        with torch.no_grad():
            out_band = ((pred_m > upper) | (pred_m < lower)).float().mean()
            _act("dsm_frac_outside_band", out_band)                 # 受约样本中越带比例
    loss = torch.relu(pred_m - upper) + torch.relu(lower - pred_m)
    return loss.mean()


def monotonicity_loss(P_pred, lambda_c):
    """单调性：整体细长比 λc 增大时承载力应不增。仅对 Δλc > 0.10 的相邻对施加"""
    sort_idx = torch.argsort(lambda_c)
    P_sorted = P_pred[sort_idx]
    lam_sorted = lambda_c[sort_idx]
    delta = lam_sorted[1:] - lam_sorted[:-1]
    mask = delta > 0.10
    if RECORD_ACTIVATION:
        with torch.no_grad():
            n_pair = max(int(delta.numel()), 1)
            _act("mono_n_pairs_total", n_pair)                      # 相邻对总数
            _act("mono_n_pairs_gated", int(mask.sum()))             # 过阈值的对数
            _act("mono_frac_pairs_gated", float(mask.sum()) / n_pair)
    if mask.sum() == 0:
        _act("mono_frac_viol", 0.0)
        return torch.tensor(0.0, device=P_pred.device)
    violation = torch.relu(P_sorted[1:][mask] - P_sorted[:-1][mask])
    if RECORD_ACTIVATION:
        with torch.no_grad():
            _act("mono_frac_viol", (violation > 0).float().mean())  # 受约对中违反比例
    return violation.mean()


def total_loss_pinn(delta_P, P_true, PDSM_s, Py, lambda_c, lambda_crl,
                    w_phy=0.20, w_mono=0.10,
                    use_upper=True, use_dsm=True, use_mono=True):
    """
    PINN 总损失（P_pred = PDSM_s + ΔP）
    L_total = Huber(P_pred, P_true) + w_phy*(L_upper + L_dsm) + w_mono*L_mono
    布尔开关供步骤 3 的消融逐项关闭损失项。
    """
    P_pred = PDSM_s + delta_P
    L_data  = nn.functional.huber_loss(P_pred, P_true, delta=30.0)
    L_upper = physics_loss_upper_bound(P_pred, Py) if use_upper else torch.tensor(0.0, device=P_pred.device)
    L_dsm   = physics_loss_dsm_proximity(P_pred, PDSM_s, lambda_crl) if use_dsm else torch.tensor(0.0, device=P_pred.device)
    L_mono  = monotonicity_loss(P_pred, lambda_c) if use_mono else torch.tensor(0.0, device=P_pred.device)
    L_phy   = L_upper + L_dsm
    L_total = L_data + w_phy * L_phy + (w_mono * L_mono if use_mono else 0.0)
    _flush_batch()
    return L_total, L_data.item(), float(L_phy), float(L_mono)


# ─────────────────────────────────────────────────────────────────────────────
# 5. 物理权重调度器
# ─────────────────────────────────────────────────────────────────────────────
class PhysicsWeightScheduler:
    """前期权重小（先拟合数据），预热后线性增大到 w_end"""
    def __init__(self, w_start, w_end, total_epochs, warmup_ratio=0.15):
        self.w_start = w_start
        self.w_end   = w_end
        self.total_epochs = total_epochs
        self.warmup_end = int(total_epochs * warmup_ratio)

    def get(self, epoch: int) -> float:
        if epoch <= self.warmup_end:
            return self.w_start
        progress = (epoch - self.warmup_end) / (self.total_epochs - self.warmup_end)
        return self.w_start + (self.w_end - self.w_start) * min(progress, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 6. 训练函数
# ─────────────────────────────────────────────────────────────────────────────
def train_model(model, splits, phy, mode="pinn", cfg=None,
                loss_flags=None, verbose=True):
    """
    mode = "pinn" → 残差修正 + 物理损失；mode = "mlp" → 纯数据 Huber
    cfg        : 覆盖默认 CFG（消融/效率扫描用），None 则用全局 CFG
    loss_flags : dict(use_upper/use_dsm/use_mono)，供消融逐项关闭
    """
    assert mode in ("pinn", "mlp")
    cfg = cfg or CFG
    flags = {"use_upper": True, "use_dsm": True, "use_mono": True}
    if loss_flags:
        flags.update(loss_flags)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                  weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"], eta_min=1e-6)
    phy_sched = PhysicsWeightScheduler(cfg["w_phy_start"], cfg["w_phy_end"], cfg["epochs"])

    X_tr, y_tr = splits["train"]
    X_vl, y_vl = splits["val"]
    PDSM_tr  = torch.tensor(phy["train"]["PDSM_s"],     dtype=torch.float32)
    Py_tr    = torch.tensor(phy["train"]["Py"],         dtype=torch.float32)
    lam_c_tr = torch.tensor(phy["train"]["lambda_c"],   dtype=torch.float32)
    lam_l_tr = torch.tensor(phy["train"]["lambda_crl"], dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(X_tr, y_tr, PDSM_tr, Py_tr, lam_c_tr, lam_l_tr),
        batch_size=cfg["batch_size"], shuffle=True, drop_last=False)

    history = {"train_loss": [], "val_loss": [], "L_data": [], "L_phy": [], "L_mono": []}
    history["activation"] = []          # 每 epoch 的物理约束激活统计
    best_val, best_state, wait = np.inf, None, 0
    model = model.to(DEVICE)

    if verbose:
        print(f"\n{'─'*55}\n  Training [{mode.upper()}]  device={DEVICE}  "
              f"params={sum(p.numel() for p in model.parameters()):,}\n{'─'*55}")
        print(f"{'Ep':>5} {'Train':>9} {'Val':>9} {'L_data':>8} {'L_phy':>7} {'w_phy':>7}")

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        ep_total = ep_ld = ep_lp = ep_lm = 0.0
        w_phy = phy_sched.get(epoch)
        for Xb, yb, pdsm_b, py_b, lc_b, ll_b in loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            pdsm_b, py_b = pdsm_b.to(DEVICE), py_b.to(DEVICE)
            lc_b, ll_b = lc_b.to(DEVICE), ll_b.to(DEVICE)

            if mode == "pinn":
                delta_P = model(Xb)
                loss, ld, lp, lm = total_loss_pinn(
                    delta_P, yb, pdsm_b, py_b, lc_b, ll_b,
                    w_phy=w_phy, w_mono=cfg["w_mono"], **flags)
            else:
                P_pred = model(Xb)
                loss = nn.functional.huber_loss(P_pred, yb, delta=30.0)
                ld, lp, lm = loss.item(), 0.0, 0.0

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ep_total += loss.item(); ep_ld += ld; ep_lp += lp; ep_lm += lm

        scheduler.step()
        n = len(loader)
        ep_total /= n; ep_ld /= n; ep_lp /= n; ep_lm /= n

        model.eval()
        with torch.no_grad():
            X_vl_d, y_vl_d = X_vl.to(DEVICE), y_vl.to(DEVICE)
            if mode == "pinn":
                PDSM_vl = torch.tensor(phy["val"]["PDSM_s"], dtype=torch.float32).to(DEVICE)
                P_vl = PDSM_vl + model(X_vl_d)
            else:
                P_vl = model(X_vl_d)
            val_loss = nn.functional.huber_loss(P_vl, y_vl_d).item()

        for k, v in zip(["train_loss", "val_loss", "L_data", "L_phy", "L_mono"],
                        [ep_total, val_loss, ep_ld, ep_lp, ep_lm]):
            history[k].append(v)

        act = pop_epoch_activation()
        if act:
            act["epoch"] = epoch
            act["w_phy"] = w_phy
            history["activation"].append(act)

        if verbose and (epoch % 50 == 0 or epoch == 1):
            print(f"{epoch:>5} {ep_total:>9.3f} {val_loss:>9.3f} "
                  f"{ep_ld:>8.3f} {ep_lp:>7.3f} {w_phy:>7.3f}")

        if val_loss < best_val - 1e-4:
            best_val, wait = val_loss, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg["patience"]:
                if verbose:
                    print(f"\n  Early stop @ epoch {epoch}  best_val={best_val:.4f}")
                break

    model.load_state_dict(best_state)
    model.eval()
    if verbose:
        print(f"  训练完成  best_val_loss={best_val:.4f}")
        if history["activation"]:
            print_activation_summary(history["activation"])
        print()
    return model, history


def print_activation_summary(act_list):
    """打印物理约束激活率摘要（全程平均 / 末 20% epoch 平均）。"""
    a = pd.DataFrame(act_list)
    tail = a.iloc[int(len(a) * 0.8):] if len(a) >= 5 else a
    rows = [
        ("上界  P>Py 的样本比例",          "upper_frac_viol",       "{:.4%}"),
        ("上界  max(P/Py)",                "upper_max_ratio",       "{:.3f}"),
        ("DSM   进入约束的样本比例",        "dsm_frac_in_scope",     "{:.2%}"),
        ("DSM   受约样本中越出±30%带比例",  "dsm_frac_outside_band", "{:.4%}"),
        ("单调  过阈值(Δλc>0.10)的对数",    "mono_n_pairs_gated",    "{:.1f}"),
        ("单调  过阈值对占相邻对比例",       "mono_frac_pairs_gated", "{:.2%}"),
        ("单调  受约对中违反比例",           "mono_frac_viol",        "{:.4%}"),
    ]
    print(f"\n  {'─'*58}")
    print(f"  物理约束激活率  ({len(a)} epochs)")
    print(f"  {'指标':<34}{'全程均值':>12}{'末20%epoch':>13}")
    print(f"  {'─'*58}")
    for label, key, fmt in rows:
        if key not in a.columns:
            continue
        print(f"  {label:<34}{fmt.format(a[key].mean()):>12}{fmt.format(tail[key].mean()):>13}")
    print(f"  {'─'*58}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. 预测 & 评估
# ─────────────────────────────────────────────────────────────────────────────
def predict(model, splits, phy, mode="pinn", split="test"):
    model.eval()
    X = splits[split][0].to(DEVICE)
    with torch.no_grad():
        if mode == "pinn":
            PDSM = torch.tensor(phy[split]["PDSM_s"], dtype=torch.float32).to(DEVICE)
            P_pred = (PDSM + model(X)).cpu().numpy()
        else:
            P_pred = model(X).cpu().numpy()
    return np.clip(P_pred, 1e-3, None)


def compute_metrics(P_true, P_pred, label=""):
    ratio = P_true / np.clip(P_pred, 1e-6, None)
    mean_r, std_r = ratio.mean(), ratio.std()
    return {"label": label,
            "R2": r2_score(P_true, P_pred),
            "RMSE": np.sqrt(mean_squared_error(P_true, P_pred)),
            "MAE": mean_absolute_error(P_true, P_pred),
            "MeanRatio": mean_r, "StdRatio": std_r, "CoV": std_r / mean_r}


def evaluate_all(results_dict, P_true, fm_test):
    model_keys = ["dsm", "rf", "xgb", "mlp", "pinn"]
    print(f"\n{'═'*75}\n  Table 1 — 模型整体指标（测试集）\n{'═'*75}")
    print(f"  {'模型':<22} {'R²':>7} {'RMSE':>8} {'MAE':>8} {'均值比':>9} {'标准差':>8} {'CoV':>7}")
    print("─" * 75)
    for k in model_keys:
        m = results_dict[k]
        print(f"  {m['label']:<22} {m['R2']:>7.4f} {m['RMSE']:>8.2f} {m['MAE']:>8.2f} "
              f"{m['MeanRatio']:>9.4f} {m['StdRatio']:>8.4f} {m['CoV']:>7.4f}")

    fm_metrics = {k: {} for k in model_keys}
    fm_counts = {}
    for fm in FM_ORDER:
        mask = fm_test == fm
        if mask.sum() < 3:
            continue
        fm_counts[fm] = int(mask.sum())
        for k in model_keys:
            fm_metrics[k][fm] = compute_metrics(P_true[mask], results_dict[k]["P_pred"][mask])

    header = f"  {'FM':<6} {'N':>4}  " + "".join(f"{l:>9}" for l in ["DSM", "RF", "XGB", "MLP", "PINN"])
    for metric_key, metric_name in [("R2", "R²"), ("RMSE", "RMSE (kN)"), ("MeanRatio", "均值比 Pt/Pp")]:
        print(f"\n{'═'*75}\n  Table 2 — 分破坏模式 {metric_name}\n{'═'*75}\n{header}\n{'─'*75}")
        for fm in FM_ORDER:
            if fm not in fm_counts:
                continue
            row = f"  {fm:<6} {fm_counts[fm]:>4}  "
            for k in model_keys:
                row += f"{fm_metrics[k][fm][metric_key]:>9.3f}"
            print(row)
    return fm_metrics, fm_counts


# ─────────────────────────────────────────────────────────────────────────────
# 8. 可视化（六面板主结果图）
# ─────────────────────────────────────────────────────────────────────────────
def plot_results(history_pinn, history_mlp, results_dict, P_true, fm_test):
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(history_pinn["train_loss"], lw=1.5, color="#378ADD", label="PINN Train")
    ax.plot(history_pinn["val_loss"],   lw=1.5, color="#378ADD", ls="--", label="PINN Val")
    ax.plot(history_mlp["train_loss"],  lw=1.5, color="#D85A30", label="MLP Train")
    ax.plot(history_mlp["val_loss"],    lw=1.5, color="#D85A30", ls="--", label="MLP Val")
    ax.set_yscale("log"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (log)")
    ax.set_title("(a) Training Curves"); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(history_pinn["L_data"], lw=1.5, color="#378ADD", label="L_data")
    ax.plot(history_pinn["L_phy"],  lw=1.5, color="#639922", label="L_physics")
    ax.plot(history_pinn["L_mono"], lw=1.5, color="#BA7517", ls=":", label="L_mono")
    ax.set_yscale("log"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (log)")
    ax.set_title("(b) PINN Physics Loss Terms"); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[0, 2])
    P_pinn = results_dict["pinn"]["P_pred"]
    for fm in FM_ORDER:
        mask = fm_test == fm
        if mask.sum() == 0:
            continue
        ax.scatter(P_pinn[mask], P_true[mask], c=FM_COLORS[fm], label=fm,
                   alpha=0.55, s=18, edgecolors="none")
    lim = max(P_true.max(), P_pinn.max()) * 1.08
    ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.4, label="1:1 line")
    ax.set_xlabel("PINN Ppred (kN)"); ax.set_ylabel("Ptest (kN)")
    ax.set_title(f"(c) PINN  R²={results_dict['pinn']['R2']:.4f}")
    ax.legend(fontsize=7, markerscale=1.5); ax.set_xlim(0, lim); ax.set_ylim(0, lim)

    ax = fig.add_subplot(gs[1, 0])
    model_order = ["DSM", "RF", "XGB", "MLP", "PINN"]
    model_colors = ["#BA7517", "#639922", "#7F77DD", "#D85A30", "#378ADD"]
    ratio_data = [P_true / np.clip(results_dict[k]["P_pred"], 1e-6, None)
                  for k in ["dsm", "rf", "xgb", "mlp", "pinn"]]
    bp = ax.boxplot(ratio_data, labels=model_order, patch_artist=True,
                    medianprops=dict(color="white", lw=2),
                    flierprops=dict(marker="o", markersize=3, alpha=0.4))
    for patch, c in zip(bp["boxes"], model_colors):
        patch.set_facecolor(c); patch.set_alpha(0.75)
    ax.axhline(1.0, color="gray", ls="--", lw=1.2, alpha=0.7)
    ax.set_ylabel("Pt / Ppred"); ax.set_title("(d) Prediction Ratio — All Models")

    ax = fig.add_subplot(gs[1, 1])
    fm_ratio_data, fm_labs = [], []
    for fm in FM_ORDER:
        mask = fm_test == fm
        if mask.sum() < 3:
            continue
        fm_ratio_data.append(P_true[mask] / np.clip(P_pinn[mask], 1e-6, None))
        fm_labs.append(fm)
    bp2 = ax.boxplot(fm_ratio_data, labels=fm_labs, patch_artist=True,
                     medianprops=dict(color="white", lw=2),
                     flierprops=dict(marker="o", markersize=3, alpha=0.4))
    for patch, fm in zip(bp2["boxes"], fm_labs):
        patch.set_facecolor(FM_COLORS[fm]); patch.set_alpha(0.75)
    ax.axhline(1.0, color="gray", ls="--", lw=1.2, alpha=0.7)
    ax.set_ylabel("Pt / Ppred"); ax.set_title("(e) PINN Ratio by Failure Mode")

    ax = fig.add_subplot(gs[1, 2])
    P_dsm = results_dict["dsm"]["P_pred"]
    ax.scatter(P_dsm,  P_true, c="#BA7517", alpha=0.35, s=12, edgecolors="none",
               label=f"DSM  R²={results_dict['dsm']['R2']:.3f}")
    ax.scatter(P_pinn, P_true, c="#378ADD", alpha=0.45, s=12, edgecolors="none",
               label=f"PINN R²={results_dict['pinn']['R2']:.3f}")
    lim = max(P_true.max(), P_pinn.max(), P_dsm.max()) * 1.08
    ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.4)
    ax.set_xlabel("Ppred (kN)"); ax.set_ylabel("Ptest (kN)")
    ax.set_title("(f) DSM vs PINN Scatter")
    ax.legend(fontsize=8); ax.set_xlim(0, lim); ax.set_ylim(0, lim)

    plt.suptitle("PINN vs Baselines — CFS Built-up Column Axial Capacity",
                 fontsize=13, fontweight="bold", y=1.01)
    out = BASE_DIR / "pinn_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → 主结果图已保存: {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. 主函数
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  CFS Built-up Column — PINN Capacity Prediction")
    print("=" * 55)
    print(f"  Device: {DEVICE}")

    print("\n[1/5] 加载数据与特征工程 ...")
    X, y, phy_raw, df = load_and_engineer(CFG["data_path"])
    splits, phy, scaler, i_tr, i_vl, i_te = split_dataset(X, y, phy_raw)
    in_dim = X.shape[1]
    P_true  = splits["test"][1].numpy()
    fm_test = phy["test"]["FM_merged"]

    print("\n[2/5] 训练 PINN ...")
    pinn_net = ResidualCorrectionNet(in_dim, CFG["hidden_dims"], CFG["dropout"])
    pinn_net, history_pinn = train_model(pinn_net, splits, phy, mode="pinn")

    print("\n[3/5] 训练纯 MLP（基准）...")
    mlp_net = PlainMLP(in_dim, CFG["hidden_dims"], CFG["dropout"])
    mlp_net, history_mlp = train_model(mlp_net, splits, phy, mode="mlp")

    print("\n[4/5] 训练传统 ML 基准 ...")
    X_tr_np, y_tr_np = splits["train"][0].numpy(), splits["train"][1].numpy()
    X_te_np = splits["test"][0].numpy()
    rf = RandomForestRegressor(n_estimators=300, max_features=0.6,
                               min_samples_leaf=2, random_state=SEED, n_jobs=-1)
    rf.fit(X_tr_np, y_tr_np)
    P_rf = np.clip(rf.predict(X_te_np), 1e-3, None)
    xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5,
                       subsample=0.8, colsample_bytree=0.8,
                       random_state=SEED, n_jobs=-1, verbosity=0)
    xgb.fit(X_tr_np, y_tr_np)
    P_xgb = np.clip(xgb.predict(X_te_np), 1e-3, None)

    print("\n[5/5] 评估与可视化 ...")
    P_pinn = predict(pinn_net, splits, phy, mode="pinn")
    P_mlp  = predict(mlp_net,  splits, phy, mode="mlp")
    P_dsm  = phy["test"]["PDSM_s"]

    results = {
        "dsm" : {**compute_metrics(P_true, P_dsm,  "DSM (PDSM_s)"), "P_pred": P_dsm},
        "rf"  : {**compute_metrics(P_true, P_rf,   "Random Forest"), "P_pred": P_rf},
        "xgb" : {**compute_metrics(P_true, P_xgb,  "XGBoost"),       "P_pred": P_xgb},
        "mlp" : {**compute_metrics(P_true, P_mlp,  "Plain MLP"),     "P_pred": P_mlp},
        "pinn": {**compute_metrics(P_true, P_pinn, "PINN (ours)"),   "P_pred": P_pinn},
    }
    fm_metrics, fm_counts = evaluate_all(results, P_true, fm_test)

    # 导出表 1 / 表 2 数字，避免手抄控制台
    pd.DataFrame([{k: results[m][k] for k in
                   ("label", "R2", "RMSE", "MAE", "MeanRatio", "StdRatio", "CoV")}
                  for m in ["dsm", "rf", "xgb", "mlp", "pinn"]]
                 ).to_csv(BASE_DIR / "main_table1.csv", index=False, encoding="utf-8-sig")
    rows_fm = []
    for m in ["dsm", "rf", "xgb", "mlp", "pinn"]:
        for fm, met in fm_metrics[m].items():
            rows_fm.append(dict(model=m, FM=fm, N=fm_counts[fm],
                                **{k: met[k] for k in ("R2", "RMSE", "MAE", "MeanRatio", "CoV")}))
    pd.DataFrame(rows_fm).to_csv(BASE_DIR / "main_table2.csv", index=False, encoding="utf-8-sig")
    print("  → 已写出: main_table1.csv, main_table2.csv")

    # 导出物理约束激活率逐 epoch 日志
    if history_pinn.get("activation"):
        pd.DataFrame(history_pinn["activation"]).to_csv(
            BASE_DIR / "activation_log.csv", index=False, encoding="utf-8-sig")
        print("  → 已写出: activation_log.csv")

    torch.save({
        "model_state": pinn_net.state_dict(), "config": CFG,
        "scaler_mean": scaler.mean_, "scaler_std": scaler.scale_,
        "feat_names": list(X.columns), "in_dim": in_dim,
        "history": history_pinn, "seed": SEED,
    }, BASE_DIR / "pinn_cfs_best.pt")
    print(f"  模型已保存: pinn_cfs_best.pt")

    plot_results(history_pinn, history_mlp, results, P_true, fm_test)


if __name__ == "__main__":
    main()