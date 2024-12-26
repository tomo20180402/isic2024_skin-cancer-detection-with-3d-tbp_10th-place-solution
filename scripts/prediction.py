import os
from pathlib import Path
from typing import List, Dict, Tuple, Union
from PIL import Image
from tqdm import tqdm
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import albumentations as A
import lightgbm as lgb
import xgboost as xgb
import catboost as catb
from scripts.config import Config as cfg
from scripts import image_dataset, image_model
from scripts.logger import create_logger
logger = create_logger()


def get_full_paths(directory: Path) -> List[str]:
    full_paths = []
    for root, _dirs, files in os.walk(directory):
        for file in files:
            full_path = os.path.join(root, file)
            full_paths.append(full_path)
    return full_paths


def get_img_model_paths(exp: str) -> List[str]:
    model_dirs = [
        Path(f'/kaggle/input/isic-tomo20180402-models/model_{exp}/model_{exp}/'),
        Path(f'/kaggle/input/isic-tomo20180402-models/model_{exp}_2/model_{exp}_2'),
        Path(f'/kaggle/input/isic-tomo20180402-models/model_{exp}_3/model_{exp}_3'),
        Path(f'/kaggle/input/isic-tomo20180402-models/model_{exp}_4/model_{exp}_4'),
        Path(f'/kaggle/input/isic-tomo20180402-models/model_{exp}_5/model_{exp}_5'),
    ]
    model_paths = []
    for model_dir in model_dirs:
        model_paths += sorted([path for path in get_full_paths(model_dir / 'pt') if 'best_' not in path])  # not best model
    logger.info(f'get_img_model_paths: exp = {exp}, n_model_paths = {len(model_paths)}')
    return model_paths


def get_gbdt_model_paths(exp: str, gbdt: str) -> List[str]:
    model_dirs = [
        Path(f'/kaggle/input/isic-tomo20180402-models/model_{exp}/model_{exp}/{gbdt}'),
        Path(f'/kaggle/input/isic-tomo20180402-models/model_{exp}_2/model_{exp}_2/{gbdt}'),
        Path(f'/kaggle/input/isic-tomo20180402-models/model_{exp}_3/model_{exp}_3/{gbdt}'),
        Path(f'/kaggle/input/isic-tomo20180402-models/model_{exp}_4/model_{exp}_4/{gbdt}'),
        Path(f'/kaggle/input/isic-tomo20180402-models/model_{exp}_5/model_{exp}_5/{gbdt}'),
    ]
    model_paths = []
    for model_dir in model_dirs:
        model_paths += sorted(get_full_paths(model_dir))
    logger.info(f'get_gbdt_model_paths: exp = {exp}, gbdt = {gbdt}, n_model_paths = {len(model_paths)}')
    return model_paths


def load_image_models(img_model_paths: List[str]) -> List[torch.nn.Module]:
    img_models = []
    for model_path in img_model_paths:
        img_model = image_model.EfficientNetBinaryClassifier()
        img_model.load_state_dict(torch.load(model_path))
        img_model = img_model.to(cfg.DEVICE)
        img_models.append(img_model)
    return img_models


def load_lgb(model_path: str) -> lgb.Booster:
    lgb_model = lgb.Booster(model_file=model_path)
    return lgb_model


def load_xgb(model_path: str) -> xgb.Booster:
    xgb_model = xgb.Booster()
    xgb_model.load_model(model_path)
    return xgb_model


def load_catb(model_path: str) -> catb.CatBoostClassifier:
    catb_model = catb.CatBoostClassifier()
    catb_model.load_model(model_path)
    return catb_model


def predict(
        test_metadata_df: pl.DataFrame,
        test_dataloader: DataLoader,
        img_models: List[torch.nn.Module],
        lgb_models: List[lgb.Booster],
        xgb_models: List[xgb.Booster],
        catb_models: List[catb.CatBoostClassifier],
        feature_cols: List[str],
        y_pred_img_name: str) -> np.ndarray:
    ### 1st stage
    preds_list = []
    for imgs, _ in tqdm(test_dataloader, total=len(test_dataloader)):
        imgs = imgs.to(cfg.DEVICE)
        with torch.no_grad():
            _preds_list = []
            for img_model in img_models:
                outputs = img_model(imgs)
                preds = nn.Softmax()(outputs)[:, 1].cpu()
                _preds_list.append(preds)
            preds_list.append(torch.vstack(_preds_list))
        torch.cuda.empty_cache()

    img_preds = torch.hstack(preds_list).numpy().mean(axis=0)
    img_pred_df = pl.DataFrame(img_preds, schema=[y_pred_img_name])

    ### 2nd stage
    test_metadata_with_pred_df = pl.concat([test_metadata_df, img_pred_df], how='horizontal')
    # lgb
    lgb_pred_list = []
    for lgb_model in lgb_models:
        lgb_pred = lgb_model.predict(
            test_metadata_with_pred_df.select(feature_cols + [y_pred_img_name]).to_pandas()
        )
        lgb_pred_list.append(lgb_pred)
    lgb_pred = np.mean(lgb_pred_list, axis=0)
    # xgb
    xgb_test = xgb.DMatrix(test_metadata_with_pred_df.select(feature_cols + [y_pred_img_name]).to_pandas())
    xgb_pred_list = []
    for xgb_model in xgb_models:
        xgb_pred = xgb_model.predict(xgb_test)
        xgb_pred_list.append(xgb_pred)
    xgb_pred = np.mean(xgb_pred_list, axis=0)
    # catb
    catb_test = catb.Pool(test_metadata_with_pred_df.select(feature_cols + [y_pred_img_name]).to_pandas())
    catb_pred_list = []
    for catb_model in catb_models:
        catb_pred = catb_model.predict(catb_test, prediction_type='Probability')[:, 1]
        catb_pred_list.append(catb_pred)
    catb_pred = np.mean(catb_pred_list, axis=0)        
    # ens
    y_pred = (lgb_pred + xgb_pred + catb_pred) / 3

    return y_pred


def predict_batch(
        test_metadata_df: pl.DataFrame,
        test_imgs: List[Image.Image],
        img_models: List[torch.nn.Module],
        lgb_models: List[lgb.Booster],
        xgb_models: List[xgb.Booster],
        catb_models: List[catb.CatBoostClassifier],
        val_transform: Union[A.Compose, transforms.Compose],
        dataset_func: Union[image_dataset.CustomDatasetP1, image_dataset.CustomDatasetP2],
        feature_cols: List[str],
        y_pred_img_name: str) -> np.ndarray:

    n_batch = len(test_imgs) // cfg.BS_PRED + 1

    y_preds = []

    for ibatch in range(n_batch):
        print(f'batch: {ibatch+1} / {n_batch}')
        start_idx, end_idx = cfg.BS_PRED*ibatch, cfg.BS_PRED*(ibatch+1)

        test_dataloader = DataLoader(
            dataset_func(
                test_imgs[start_idx:end_idx],
                [-1] * len(test_imgs[start_idx:end_idx]),  # dummy target
                val_transform,
            ),
            batch_size=cfg.BS,
            shuffle=False,
            num_workers=cfg.N_WORKER,
            prefetch_factor=2,
            pin_memory=True,
        )

        y_pred = predict(
            test_metadata_df=test_metadata_df[start_idx:end_idx],
            test_dataloader=test_dataloader,
            img_models=img_models,
            lgb_models=lgb_models,
            xgb_models=xgb_models,
            catb_models=catb_models,
            feature_cols=feature_cols,
            y_pred_img_name=y_pred_img_name)

        y_preds.append(y_pred)

    return np.hstack(y_preds)