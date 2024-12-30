import os
from typing import List, Dict, Tuple
import copy
from tqdm import tqdm
import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
import xgboost as xgb
import catboost as catb
from scripts.config import Config as cfg
from scripts import image_training
from scripts.logger import create_logger
logger = create_logger()


def make_gbdt_dirs(ifold: int, model_seed: int) -> Tuple[str, str, str]:
    lgb_dir = cfg.OUTPUT_DIR / 'lgb' / f'fold_{ifold}' / f'model_seed_{model_seed}'
    os.makedirs(lgb_dir, exist_ok=True)
    xgb_dir = cfg.OUTPUT_DIR / 'xgb' / f'fold_{ifold}' / f'model_seed_{model_seed}'
    os.makedirs(xgb_dir, exist_ok=True)
    catb_dir = cfg.OUTPUT_DIR / 'catb' / f'fold_{ifold}' / f'model_seed_{model_seed}'
    os.makedirs(catb_dir, exist_ok=True)
    return lgb_dir, xgb_dir, catb_dir


def get_gbdt_params(model_seed: int) -> Tuple[Dict, Dict, Dict]:
    lgb_params = copy.deepcopy(cfg.LGB_PARAMS)
    lgb_params['seed'] = model_seed
    xgb_params = copy.deepcopy(cfg.XGB_PARAMS)
    xgb_params['seed'] = model_seed
    catb_params = copy.deepcopy(cfg.CATB_PARAMS)
    catb_params['random_state'] = model_seed
    return lgb_params, xgb_params, catb_params


def get_train_by_set(train: pd.DataFrame, start_idx: int, end_idx: int) -> pd.DataFrame:
    _train = pd.concat([
        train.loc[start_idx:end_idx].loc[train['target']==0].sample(frac=1.0, random_state=420).reset_index(),
        train.loc[train['target']==1],
    ], axis=0)
    _train_aug = pd.concat([
        train.loc[start_idx:end_idx].loc[train['target']==0].sample(frac=1.0, random_state=520).reset_index(),
        train.loc[train['target']==1],
    ], axis=0)
    attribution_cols = [col for col in _train_aug.columns if col.startswith('attribution_')]
    _train_aug.loc[:, attribution_cols] = 0  # attribution = 0 @ data aug
    train_set = pd.concat([_train, _train_aug], axis=0).reset_index(drop=True)
    return train_set


def train_lgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    lgb_params: Dict,
    lgb_dir: str,
    iset: int) -> np.ndarray:

    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_valid = lgb.Dataset(X_valid, label=y_valid)
    lgb_model = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=350,
        valid_sets=[lgb_valid],
        callbacks=[
            lgb.log_evaluation(10000),
        ],
    )
    lgb_path = lgb_dir / f'lgb_{iset}.txt'
    lgb_model.save_model(lgb_path)
    lgb_model = lgb.Booster(model_file=lgb_path)
    lgb_y_pred = lgb_model.predict(X_valid)
    return lgb_y_pred


def train_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    xgb_params: Dict,
    xgb_dir: str,
    iset: int) -> np.ndarray:

    xgb_train = xgb.DMatrix(X_train, label=y_train)
    xgb_valid = xgb.DMatrix(X_valid, label=y_valid)
    xgb_model = xgb.train(
        xgb_params,
        xgb_train,
        num_boost_round=400,
        evals=[(xgb_valid, 'validation')],
        verbose_eval=False,
    )
    xgb_path = xgb_dir / f'xgb_{iset}.json'
    xgb_model.save_model(xgb_path)
    xgb_model = xgb.Booster()
    xgb_model.load_model(xgb_path)
    xgb_y_pred = xgb_model.predict(xgb_valid)
    return xgb_y_pred


def train_catb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    catb_params: Dict,
    catb_dir: str,
    iset: int) -> np.ndarray:

    catb_train = catb.Pool(X_train, y_train)
    catb_valid = catb.Pool(X_valid, y_valid)
    catb_model = catb.train(
        params=catb_params,
        dtrain=catb_train,
        eval_set=catb_valid,
    )
    catb_path = catb_dir / f'catb_{iset}.cbm'
    catb_model.save_model(catb_path)
    catb_model = catb.CatBoostClassifier()
    catb_model.load_model(catb_path)
    catb_y_pred = catb_model.predict(catb_valid, prediction_type='Probability')[:, 1]
    return catb_y_pred


def run(train_metadata_df: pl.DataFrame) -> None:
    logger.info(f'start: 2nd stage training')

    if cfg.PATTERN == 1:
        feature_cols = cfg.FEATURE_COLS_P1
    elif cfg.PATTERN == 2:
        feature_cols = cfg.FEATURE_COLS_P2
    else:
        raise ValueError(f'cfg.PATTERN = {cfg.PATTERN}')

    y_preds = []
    y_trues = []
    isics = []

    for ifold in range(cfg.N_FOLD):
        logger.info(f'# fold {ifold}')

        train = train_metadata_df.filter(pl.col('fold')!=ifold).to_pandas().sample(frac=1.0, random_state=42).reset_index(drop=True)
        width = train.shape[0] / cfg.N_TRAIN_SET

        valid = train_metadata_df.filter(pl.col('fold')==ifold)
        X_valid = valid.select(feature_cols).to_pandas()
        y_valid = valid.select('target').to_numpy().flatten()

        y_preds_set = []

        for model_seed in cfg.MODEL_SEEDS:
            logger.info(f'## model_seed {model_seed}')

            # mkdirs
            lgb_dir, xgb_dir, catb_dir = make_gbdt_dirs(ifold, model_seed)

            # params
            lgb_params, xgb_params, catb_params = get_gbdt_params(model_seed)

            # train by set
            for iset in tqdm(range(cfg.N_TRAIN_SET)):
                start_idx = int(width) * iset
                end_idx = int(width) * (iset+1)

                train_set = get_train_by_set(train, start_idx, end_idx)
                X_train = train_set[feature_cols]
                y_train = train_set['target']

                # lgb
                lgb_y_pred = train_lgb(X_train, y_train, X_valid, y_valid, lgb_params, lgb_dir, iset)

                # xgb
                xgb_y_pred = train_xgb(X_train, y_train, X_valid, y_valid, xgb_params, xgb_dir, iset)

                # catb
                catb_y_pred = train_catb(X_train, y_train, X_valid, y_valid, catb_params, catb_dir, iset)

                # ensemble
                y_pred = (lgb_y_pred + xgb_y_pred + catb_y_pred) / 3
                y_preds_set.append(y_pred)

        y_preds_set_mean = np.mean(y_preds_set, axis=0)
        y_true = y_valid.copy()
        isic = valid.select('isic_id').to_numpy().flatten()

        score = image_training.calc_metric(y_preds_set_mean, y_true)
        logger.info(f'score = {score}')

        y_preds.append(y_preds_set_mean)
        y_trues.append(y_true)
        isics.append(isic)

    y_preds = np.hstack(y_preds)
    y_trues = np.hstack(y_trues)
    isics = np.hstack(isics)

    whole_score = image_training.calc_metric(y_preds, y_trues)
    logger.info(f'whole_score = {whole_score}')