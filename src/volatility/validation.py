import pandas as pd

def chronological_split(
    data: pd.DataFrame,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
):
    df = data.sort_values("date").copy()
    dates = pd.Index(df["date"].drop_duplicates())
    n_dates = len(dates)

    train_end = int(n_dates * train_ratio)
    val_end = int(n_dates * (train_ratio + val_ratio))

    if not (0 < train_end < val_end < n_dates):
        raise ValueError("日期数量不足，无法划分三个非空区间。")

    val_start = dates[train_end]
    test_start = dates[val_end]

    # 3. 训练集：特征日期、目标日期都早于验证期
    train_df = df.loc[
        (df["date"] < val_start)
        & (df["target_date"] < val_start)
    ].copy()

    # 4. 验证集：特征日期位于验证期，目标不能进入测试期
    val_df = df.loc[
        (df["date"] >= val_start)
        & (df["date"] < test_start)
        & (df["target_date"] < test_start)
    ].copy()

    # 5. 测试集：最后一段日期
    test_df = df.loc[
        df["date"] >= test_start
    ].copy()

    if any(part.empty for part in (train_df, val_df, test_df)):
        raise ValueError("边界处理后存在空数据集，请调整划分比例。")

    return train_df, val_df, test_df
