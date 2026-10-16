"""
冷弯薄壁钢（CFS）拼合柱轴压承载力预测 —— 步骤 1：数据清洗
=====================================================================
输入：All Dataset.csv      （原始库，注意文件名带空格）
输出：All_Dataset_cleaned1.csv

清洗流程：
  1. 删除全空列 / 无 Ptest 的空行
  2. 去除所有字符串列首尾空格（兼容新版 pandas 的 string/str dtype）
  3. fne_b 列：字符串 → 数值
  4. 破坏模式归并：16 类 → 5 类（L / F / D / L+F / L+D）
  5. 数字编码 FM_class（0~4）

运行：python 1_data_cleaning.py
"""

from pathlib import Path
import numpy as np
import pandas as pd

# ── 路径：以脚本所在目录为基准，避免 PyCharm 工作目录不一致 ──────────────────
BASE_DIR = Path(__file__).resolve().parent
IN_PATH  = BASE_DIR / "All Dataset.csv"            # 原始文件名带空格
OUT_PATH = BASE_DIR / "All_Dataset_cleaned1.csv"


def main():
    # ── 1. 读取原始数据 ──────────────────────────────────────────────────────
    df = pd.read_csv(IN_PATH, encoding="utf-8-sig")
    df = df.dropna(axis=1, how="all")              # 删除全空列（CSV 末尾多余空列）
    print(f"原始行数: {len(df)}，列数: {df.shape[1]}")

    # ── 2. 删除无承载力标签的空行 ────────────────────────────────────────────
    df = df.dropna(subset=["Ptest(kN)"])
    print(f"删除空行后: {len(df)} 行")

    # ── 3. 去除所有字符串列首尾空格 ──────────────────────────────────────────
    #   关键修复：新版 pandas(>=2/3) 字符串列 dtype 可能是 'string'/'str' 而非
    #   'object'，旧写法 `df[col].dtype == object` 会漏掉这些列，导致带尾空格的
    #   破坏模式标签（如 'L '、'F '）未被清洗、在后续 map 中被静默丢弃。
    #   改用「非数值列」判定，覆盖所有文本列。
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype("string").str.strip()

    print(f"\n去除空格后 Failure Mode 类别数: {df['Failure Mode'].nunique()}")
    print(df["Failure Mode"].value_counts())

    # ── 4. fne_b 列：字符串 → 数值 ───────────────────────────────────────────
    df["fne_b"] = pd.to_numeric(df["fne_b"], errors="coerce")
    print(f"\nfne_b 转换后 NaN 数: {df['fne_b'].isna().sum()}")

    # ── 5. 破坏模式归并：16 类 → 5 类 ────────────────────────────────────────
    #   L    局部屈曲          ← L
    #   F    整体/弯扭屈曲     ← F
    #   D    畸变屈曲          ← D
    #   L+F  局部+整体屈曲     ← L+F, L+FT, L+IFT, L+FT+F, L+IF, IF
    #   L+D  局部+畸变及混合   ← L+D, L+D+F, D+F, L+D+FT, (L)+(D)+F, FT+D, Hinge
    mapping = {
        "L": "L", "F": "F", "D": "D",
        "L+F": "L+F", "L+FT": "L+F", "L+IFT": "L+F",
        "L+FT+F": "L+F", "L+IF": "L+F", "IF": "L+F",
        "L+D": "L+D", "L+D+F": "L+D", "D+F": "L+D",
        "L+D+FT": "L+D", "(L)+(D)+F": "L+D", "FT+D": "L+D", "Hinge": "L+D",
    }
    df["FM_merged"] = df["Failure Mode"].map(mapping)

    unmapped = df[df["FM_merged"].isna()]["Failure Mode"].unique()
    if len(unmapped) > 0:
        print(f"\n警告：未映射类别 → {unmapped}")
    else:
        print("\n所有类别映射成功，无遗漏")
    print(f"\n归并后分布:\n{df['FM_merged'].value_counts()}")

    # ── 6. 数字编码 FM_class ─────────────────────────────────────────────────
    label_map = {"L": 0, "F": 1, "D": 2, "L+F": 3, "L+D": 4}
    df["FM_class"] = df["FM_merged"].map(label_map)

    # ── 7. 最终检查 ──────────────────────────────────────────────────────────
    print(f"\n最终行数: {len(df)}，列数: {df.shape[1]}")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    print(f"\n剩余缺失值:\n{missing}" if len(missing) else "\n无缺失值")

    # ── 8. 输出 ──────────────────────────────────────────────────────────────
    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n清洗完成 → {OUT_PATH.name}")


if __name__ == "__main__":
    main()