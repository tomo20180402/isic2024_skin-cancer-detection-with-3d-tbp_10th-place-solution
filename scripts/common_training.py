import numpy as np
import polars as pl
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score


def calc_metric(y_pred: np.array, y_true: np.array) -> float:
    """partial area under the ROC curve (pAUC) 
    """
    min_tpr = 0.80
    max_fpr = abs(1 - min_tpr)
    v_gt = abs(y_true - 1)
    v_pred = np.array([1.0 - x for x in y_pred])
    partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    partial_auc = 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)
    return partial_auc


def add_fold(train_metadata_df: pl.DataFrame, seed: int, n_fold: int) -> pl.DataFrame:
    """StratifiedGroupKFold, group='patient_id'
    """
    sgkf = StratifiedGroupKFold(n_splits=n_fold, shuffle=True, random_state=seed)

    X = train_metadata_df.clone()
    y = train_metadata_df['target'].clone()
    groups = train_metadata_df['patient_id'].clone()

    train_metadata_df = train_metadata_df.with_columns(pl.lit(None).alias('fold'))

    train_metadata_df = train_metadata_df.to_pandas()
    for ifold, (_train_index, test_index) in enumerate(sgkf.split(X, y, groups)):
        train_metadata_df.loc[test_index, 'fold'] = ifold
    train_metadata_df = pl.from_pandas(train_metadata_df)

    return train_metadata_df