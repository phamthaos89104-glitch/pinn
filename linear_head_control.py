# -*- coding: utf-8 -*-
"""M1-linear 输出头对照：N=50/725，各 5 个种子，共 10 次新训练。

放在 cc_0902 目录后运行：python linear_head_control.py
只检查配置、结构及两轮训练：python linear_head_control.py --check-only
在其他目录运行：python linear_head_control.py --project-dir "D:/.../cc_0902"

复用 ablation_efficiency.py 的训练函数、数据划分、特征和 EFF_CFG。
仅在本进程中临时将直接预测网络构造器设为 ResidualCorrectionNet，
仍传入 use_phy=False、use_loss=False，即线性输出直接预测总承载力。
原文件不被修改。M1-softplus 与 M2 的参照值读取历史逐种子结果。
"""
import argparse
from contextlib import contextmanager
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
import torch

SIZES = [50, 725]
SEEDS = [0, 1, 2, 3, 4]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_experiment(project):
    path = project / "ablation_efficiency.py"
    spec = importlib.util.spec_from_file_location("linear_control_ablation", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def linear_direct_constructor(mm):
    original = mm.PlainMLP
    mm.PlainMLP = mm.ResidualCorrectionNet
    try:
        yield
    finally:
        mm.PlainMLP = original


def read_reference(project):
    original = pd.read_csv(project / "ablation_decompose_results.csv")
    mask = original.N.isin(SIZES) & original.seed.isin(SEEDS)
    reference = original.loc[
        mask & original.config.isin(["MLP_F", "Full_RF"]),
        ["config", "N", "seed", "R2", "RMSE"],
    ].copy()
    if reference.duplicated(["config", "N", "seed"]).any():
        raise ValueError("历史参照存在重复的 config/N/seed。")
    expected = {(c, n, s) for c in ["MLP_F", "Full_RF"] for n in SIZES for s in SEEDS}
    if set(zip(reference.config, reference.N, reference.seed)) != expected:
        raise ValueError("历史结果缺少指定规模或种子的 M1/M2 记录。")
    if not np.isfinite(reference[["R2", "RMSE"]].to_numpy()).all():
        raise ValueError("历史参照包含非有限指标。")

    # 两份独立保存的 M2 结果应相符；避免误用不同实验配置的参照值。
    efficiency = pd.read_csv(project / "data_efficiency_results.csv")
    efficiency = efficiency.loc[
        (efficiency.config == "A1") & efficiency.N.isin(SIZES)
        & efficiency.seed.isin(SEEDS), ["N", "seed", "R2", "RMSE"]
    ]
    matched = reference.query("config == 'Full_RF'").merge(
        efficiency, on=["N", "seed"], validate="one_to_one", suffixes=("_a", "_b")
    )
    if len(matched) != 10:
        raise ValueError("无法在效率扫描结果中匹配全部 10 个 M2 参照。")
    for metric in ["R2", "RMSE"]:
        if not np.allclose(matched[metric + "_a"], matched[metric + "_b"], rtol=0, atol=1e-7):
            raise ValueError(f"两份 M2 历史结果的 {metric} 不一致。")
    reference["model"] = reference.config.map({"MLP_F": "M1-softplus", "Full_RF": "M2"})
    reference["origin"] = "historical"
    return reference.drop(columns="config")


def prepare(ab):
    mm = ab.mm
    _, y, raw, df = mm.load_and_engineer(mm.CFG["data_path"])
    X, _ = ab.build_feature_matrices(df)
    fm = np.array([mm.FM_ORDER.index(f) for f in raw["FM_merged"]])
    pool, val, test = ab.fixed_split(y, fm)
    if (len(pool), len(val), len(test)) != (725, 156, 156):
        raise ValueError("划分与既有实验 725/156/156 不一致。")
    if X.shape[1] != 26:
        raise ValueError(f"期望 26 维特征，实际为 {X.shape[1]}。")
    jobs = {}
    split_records = []
    for n in SIZES:
        for seed in SEEDS:
            train = ab.subsample_train(pool, fm, n, seed)
            scaler = StandardScaler()
            arrays = (
                scaler.fit_transform(X.values[train]).astype(np.float32),
                scaler.transform(X.values[val]).astype(np.float32),
                scaler.transform(X.values[test]).astype(np.float32),
            )
            phy = [{k: v[ix] for k, v in raw.items()} for ix in (train, val, test)]
            jobs[n, seed] = (arrays, phy, train, val, test)
            for split, indices in [("train", train), ("val", val), ("test", test)]:
                split_records.extend(
                    {"N": n, "seed": seed, "split": split,
                     "row_index": int(i), "Test_ID": str(df.iloc[i]["Test_ID"])}
                    for i in indices
                )
    return X, y, df, jobs, pd.DataFrame(split_records)


def verify_contract(ab, X, y, job):
    mm = ab.mm
    original_cls = mm.PlainMLP
    torch.manual_seed(1000)
    direct = original_cls(X.shape[1], mm.CFG["hidden_dims"], mm.CFG["dropout"])
    torch.manual_seed(1000)
    linear = mm.ResidualCorrectionNet(X.shape[1], mm.CFG["hidden_dims"], mm.CFG["dropout"])
    if not isinstance(direct.head[-1], torch.nn.Softplus):
        raise ValueError("当前 PlainMLP 的输出头已变化，请重新核查对照定义。")
    a, b = direct.state_dict(), linear.state_dict()
    if a.keys() != b.keys() or any(not torch.equal(a[k], b[k]) for k in a):
        raise ValueError("相同种子下两个网络的参数初始化不一致。")
    direct.eval()
    linear.eval()
    sample = torch.tensor(job[0][0][:8])
    with torch.no_grad():
        torch.testing.assert_close(direct(sample), torch.nn.functional.softplus(linear(sample)))

    # 两轮集成检查：直接线性输出应与“零 DSM 基线”的残差训练等价。
    arrays, phy, tr, vl, _ = job
    cfg = dict(ab.EFF_CFG, epochs=2, patience=2)
    with linear_direct_constructor(mm):
        direct_model = ab.train_variant(
            arrays[0], y[tr], arrays[1], y[vl], phy[0], phy[1],
            use_phy=False, use_loss=False, cfg=cfg, seed=1000,
        )
    zero_phy = [{**p, "PDSM_s": np.zeros_like(p["PDSM_s"])} for p in phy[:2]]
    residual_model = ab.train_variant(
        arrays[0], y[tr], arrays[1], y[vl], zero_phy[0], zero_phy[1],
        use_phy=True, use_loss=False, cfg=cfg, seed=1000,
    )
    test_tensor = torch.tensor(arrays[2][:16], device=ab.DEVICE)
    with torch.no_grad():
        torch.testing.assert_close(
            direct_model(test_tensor), residual_model(test_tensor), rtol=1e-5, atol=1e-5
        )
    if mm.PlainMLP is not original_cls:
        raise RuntimeError("临时网络构造器没有恢复。")
    print("检查通过：参数初始化一致、输出差异仅为 Softplus、直接训练不叠加 DSM。", flush=True)


def write_tables(reference, rows, output):
    new = pd.DataFrame(rows)
    new.to_csv(output / "linear_results.csv", index=False, encoding="utf-8-sig")
    combined = pd.concat([reference, new], ignore_index=True)
    combined.to_csv(output / "all_results.csv", index=False, encoding="utf-8-sig")
    summary = combined.groupby(["N", "model", "origin"])[["R2", "RMSE"]].agg(["mean", "std", "count"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.to_csv(output / "summary.csv", encoding="utf-8-sig")
    paired = combined.pivot(index=["N", "seed"], columns="model", values=["R2", "RMSE"])
    paired.columns = [f"{metric}_{model}" for metric, model in paired.columns]
    paired["R2_gain_M2_over_linear"] = paired["R2_M2"] - paired["R2_M1-linear"]
    paired["R2_gain_linear_over_softplus"] = paired["R2_M1-linear"] - paired["R2_M1-softplus"]
    paired["RMSE_reduction_M2_over_linear"] = paired["RMSE_M1-linear"] - paired["RMSE_M2"]
    paired.to_csv(output / "paired_comparison.csv", encoding="utf-8-sig")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    project = args.project_dir.resolve()
    required = ["main_model.py", "ablation_efficiency.py", "Ablation decompose.py",
                "All_Dataset_cleaned1.csv", "ablation_decompose_results.csv", "data_efficiency_results.csv"]
    for name in required:
        if not (project / name).is_file():
            raise FileNotFoundError(project / name)
    ab = load_experiment(project)
    reference = read_reference(project)
    X, y, df, jobs, split_table = prepare(ab)
    print(f"设备：{ab.DEVICE}；配置：{ab.EFF_CFG}", flush=True)
    print("10 个 M2 历史参照在两份结果文件中完全一致。", flush=True)
    verify_contract(ab, X, y, jobs[50, 0])
    if args.check_only:
        print("检查完成；正式运行时去掉 --check-only。")
        return

    output = args.output_dir or project / "linear_head_control_results" / datetime.now().strftime("%Y%m%d_%H%M%S")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "predictions").mkdir()
    (output / "checkpoints").mkdir()
    split_table.to_csv(output / "splits.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "experiment": "M1-linear matched-head control", "sizes": SIZES, "subsample_seeds": SEEDS,
        "network_seeds": [1000 + s for s in SEEDS], "cfg": ab.EFF_CFG,
        "features": list(X.columns), "project_dir": str(project),
        "script_sha256": sha256(__file__), "source_sha256": {f: sha256(project / f) for f in required},
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__,
        "sklearn": sklearn.__version__, "device": ab.DEVICE,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "training_huber_delta": 30.0, "validation_huber_delta": 1.0,
        "prediction_lower_clip": 0.001,
        "reference_provenance": "Historical CSVs; original runs contain no environment or split hashes.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果目录：{output}", flush=True)
    rows = []
    for (n, seed), (arrays, phy, tr, vl, te) in jobs.items():
        print(f"\n[{len(rows) + 1}/10] M1-linear N={n}, seed={seed} 开始", flush=True)
        started = time.perf_counter()
        with linear_direct_constructor(ab.mm):
            model = ab.train_variant(
                arrays[0], y[tr], arrays[1], y[vl], phy[0], phy[1],
                use_phy=False, use_loss=False, cfg=ab.EFF_CFG, seed=1000 + seed,
            )
        met = ab.eval_variant(model, arrays[2], y[te], phy[2], use_phy=False)
        with torch.no_grad():
            raw = model(torch.tensor(arrays[2], device=ab.DEVICE)).cpu().numpy()
        clipped = np.clip(raw, 1e-3, None)
        pd.DataFrame({
            "Test_ID": df.iloc[te]["Test_ID"].to_numpy(), "P_true": y[te],
            "P_pred_raw": raw, "P_pred": clipped, "PDSM": phy[2]["PDSM_s"],
        }).to_csv(output / "predictions" / f"N{n}_seed{seed}.csv", index=False, encoding="utf-8-sig")
        torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                   output / "checkpoints" / f"N{n}_seed{seed}.pt")
        rows.append({"model": "M1-linear", "origin": "new", "N": n, "seed": seed,
                     **met, "NegativeRate_raw": float(np.mean(raw < 0)),
                     "elapsed_seconds": time.perf_counter() - started})
        summary = write_tables(reference, rows, output)
        print(f"完成：R2={met['R2']:.6f}, RMSE={met['RMSE']:.4f}, 耗时={rows[-1]['elapsed_seconds']:.1f}s", flush=True)
        del model
    print("\n全部完成：\n" + summary.round(6).to_string(), flush=True)
    print(f"\n输出：{output}", flush=True)


if __name__ == "__main__":
    main()
