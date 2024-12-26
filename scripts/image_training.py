import os
from typing import List, Dict, Tuple
import random
import copy
from PIL import Image
from tqdm import tqdm
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from scripts.config import Config as cfg
from scripts import image_dataset, image_model, common_training
from scripts.logger import create_logger
logger = create_logger()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def generate_train_and_valid_idx(train_metadata_df: pl.DataFrame, ifold: int) -> Tuple[Dict[int, List[int]], List[int], List[int]]:
    _train_metadata_df = train_metadata_df.select(['fold', 'target', 'patient_id']).to_pandas()

    # train index
    train_pos_idxs = _train_metadata_df.loc[(_train_metadata_df['fold']!=ifold) & (_train_metadata_df['target']==1), :].index.tolist()
    train_idx_dict = {}
    for iepoch in range(cfg.N_EPOCH):
        train_neg_idxs = _train_metadata_df.loc[
            (_train_metadata_df['fold']!=ifold) & (_train_metadata_df['target']==0), :].sample(
            frac=1.0, random_state=iepoch*10+ifold).groupby('patient_id').head(50).index.tolist()  # up to 50 per patien_id
        train_idx = train_neg_idxs + train_pos_idxs  # train all pos
        random.seed(iepoch)
        random.shuffle(train_idx)
        train_idx_dict[iepoch] = train_idx

    # valid index
    valid_neg_idxs = _train_metadata_df.loc[
        (_train_metadata_df['fold']==ifold) & (_train_metadata_df['target']==0), :].sample(
        frac=1.0, random_state=cfg.SEED).groupby('patient_id').head(50).index.tolist()  # up to 50 per patien_id
    valid_pos_idxs = _train_metadata_df.loc[(_train_metadata_df['fold']==ifold) & (_train_metadata_df['target']==1), :].index.tolist()
    valid_idx = valid_neg_idxs + valid_pos_idxs
    valid_idx_all = np.where(train_metadata_df['fold']==ifold)[0].tolist()

    return train_idx_dict, valid_idx, valid_idx_all


def calc_steps_per_epoch(train_metadata_df: pl.DataFrame, train_imgs: List[Image.Image], train_idx_dict: Dict[int, List[int]]) -> int:
    _train_imgs = [train_imgs[i] for i in train_idx_dict[cfg.N_EPOCH-1]]
    train_labels = train_metadata_df['target'].to_numpy()[train_idx_dict[cfg.N_EPOCH-1]]
    train_transform_p1, train_transform_p2 = image_dataset.generate_train_transforms(image_size=cfg.IMAGE_SIZE)
    if cfg.PATTERN == 1:
        train_dataset = image_dataset.CustomDatasetP1(imgs=_train_imgs, labels=train_labels, transform=train_transform_p1)
    elif cfg.PATTERN == 2:
        train_dataset = image_dataset.CustomDatasetP2(imgs=_train_imgs, labels=train_labels, transform=train_transform_p2)
    else:
        raise ValueError(f'cfg.PATTERN = {cfg.PATTERN}')
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.BS,
        shuffle=True,
        num_workers=cfg.N_WORKER,
        prefetch_factor=2,
        pin_memory=True,
    )
    steps_per_epoch = len(train_dataloader)
    return steps_per_epoch


def generate_valid_dataloader(train_metadata_df: pl.DataFrame, train_imgs: List[Image.Image], valid_idx: List[int], valid_idx_all: List[int]) -> Tuple[DataLoader, DataLoader]:
    valid_imgs = [train_imgs[i] for i in valid_idx]
    valid_imgs_all = [train_imgs[i] for i in valid_idx_all]
    valid_labels = train_metadata_df['target'].to_numpy()[valid_idx]
    valid_labels_all = train_metadata_df['target'].to_numpy()[valid_idx_all]
    valid_transform_p1, valid_transform_p2 = image_dataset.generate_valid_transforms(image_size=cfg.IMAGE_SIZE)
    if cfg.PATTERN == 1:
        valid_dataset = image_dataset.CustomDatasetP1(imgs=valid_imgs, labels=valid_labels, transform=valid_transform_p1)
        valid_all_dataset = image_dataset.CustomDatasetP1(imgs=valid_imgs_all, labels=valid_labels_all, transform=valid_transform_p1)
    elif cfg.PATTERN == 2:
        valid_dataset = image_dataset.CustomDatasetP2(imgs=valid_imgs, labels=valid_labels, transform=valid_transform_p2)
        valid_all_dataset = image_dataset.CustomDatasetP2(imgs=valid_imgs_all, labels=valid_labels_all, transform=valid_transform_p2)
    else:
        raise ValueError(f'cfg.PATTERN = {cfg.PATTERN}')
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=cfg.BS,
        shuffle=False,
        num_workers=cfg.N_WORKER,
        prefetch_factor=2,
        pin_memory=True,
    )
    valid_dataloader_all = DataLoader(
        valid_all_dataset,
        batch_size=cfg.BS,
        shuffle=False,
        num_workers=cfg.N_WORKER,
        prefetch_factor=2,
        pin_memory=True,
    )
    return valid_dataloader, valid_dataloader_all


def train_model(
    train_metadata_df: pl.DataFrame, train_imgs: List[Image.Image], train_idx_dict: Dict[int, List[int]], steps_per_epoch: int, valid_dataloader: DataLoader
) -> Tuple[nn.Module, np.ndarray, nn.Module, np.ndarray]:
    seed_everything(cfg.SEED)

    train_transform_p1, train_transform_p2 = image_dataset.generate_train_transforms(image_size=cfg.IMAGE_SIZE)
    model = image_model.EfficientNetBinaryClassifier(pretrained=True)
    model = model.to(cfg.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.0005, epochs=cfg.N_EPOCH, steps_per_epoch=steps_per_epoch, pct_start=0.1, anneal_strategy='cos')

    lr_history = []
    best_pauc = -1
    best_model = None
    best_preds = None

    for iepoch in range(cfg.N_EPOCH):
        logger.info(f'Epoch {iepoch+1}/{cfg.N_EPOCH}')
        logger.info('-' * 10)

        _train_imgs = [train_imgs[i] for i in train_idx_dict[iepoch]]
        train_labels = train_metadata_df['target'].to_numpy()[train_idx_dict[iepoch]]
        if cfg.PATTERN == 1:
            train_dataset = image_dataset.CustomDatasetP1(imgs=_train_imgs, labels=train_labels, transform=train_transform_p1)
        elif cfg.PATTERN == 2:
            train_dataset = image_dataset.CustomDatasetP2(imgs=_train_imgs, labels=train_labels, transform=train_transform_p2)
        else:
            raise ValueError(f'cfg.PATTERN = {cfg.PATTERN}')
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=cfg.BS,
            shuffle=True,
            num_workers=cfg.N_WORKER,
            prefetch_factor=2,
            pin_memory=True,
        )

        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)

        for phase in ['train', 'valid']:
            logger.info(f'Phase {phase}')
            if phase == 'train':
                model.train()
                dataloader = train_dataloader
            else:
                model.eval()
                dataloader = valid_dataloader

            running_loss = 0.0
            total = 0
            preds_list = []
            labels_list = []

            for imgs, labels in tqdm(dataloader, total=len(dataloader)):
                imgs = imgs.to(cfg.DEVICE)
                labels = labels.to(cfg.DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(imgs)
                    preds = nn.Softmax()(outputs)[:, 1]
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                    preds_list.append(preds.cpu())
                    labels_list.append(labels.cpu())

                running_loss += loss.item() * labels.size(0)
                total += labels.size(0)

            preds = torch.hstack(preds_list).detach().numpy()

            epoch_loss = running_loss / total
            pauc = common_training.calc_metric(y_pred=preds, y_true=torch.hstack(labels_list).detach().numpy())

            if (phase == 'valid') & (pauc > best_pauc):
                logger.info(f'{phase} epoch: {iepoch} loss: {epoch_loss:.4f} lr: {current_lr:.4f} pauc: {pauc:.4f}, highest score.')
                best_pauc = pauc
                best_model = copy.deepcopy(model)
                best_preds = copy.deepcopy(preds)
            else:
                logger.info(f'{phase} epoch: {iepoch} loss: {epoch_loss:.4f} lr: {current_lr:.4f} pauc: {pauc:.4f}')

    return model, preds, best_model, best_preds


def get_oof(valid_dataloader_all: DataLoader, model: nn.Module) -> np.ndarray:
    """Get OOF of all data
    """
    preds_list = []
    for imgs, _ in tqdm(valid_dataloader_all, total=len(valid_dataloader_all)):
        imgs = imgs.to(cfg.DEVICE)
        with torch.no_grad():
            outputs = model(imgs)
            preds = nn.Softmax()(outputs)[:, 1].cpu().numpy()
            preds_list.append(preds)
        torch.cuda.empty_cache()
    return np.hstack(preds_list)


def run(train_metadata_df: pl.DataFrame, train_imgs: List[Image.Image]) -> None:
    """Train & evaluate models
    """
    logger.info(f'start: 1st stage training')

    img_model_list, preds_img_list, best_img_model_list, best_preds_img_list, isic_ids_img_list = [], [], [], [], []
    preds_all_img_list, best_preds_all_img_list, isic_ids_all_img_list = [], [], []

    for ifold in range(cfg.N_FOLD):
        logger.info(f'fold {ifold}')
        # generate indices
        train_idx_dict, valid_idx, valid_idx_all = generate_train_and_valid_idx(train_metadata_df=train_metadata_df, ifold=ifold)

        # generate valid dataloader
        valid_dataloader, valid_dataloader_all = generate_valid_dataloader(train_metadata_df, train_imgs, valid_idx, valid_idx_all)

        # train & evaluate
        steps_per_epoch = calc_steps_per_epoch(train_metadata_df=train_metadata_df, train_imgs=train_imgs, train_idx_dict=train_idx_dict)
        img_model, preds_img, best_img_model, best_preds_img = train_model(train_metadata_df, train_imgs, train_idx_dict, steps_per_epoch, valid_dataloader)
        img_model_list.append(img_model)
        preds_img_list.append(preds_img)
        best_img_model_list.append(best_img_model)
        best_preds_img_list.append(best_preds_img)
        isic_ids_img_list.append(train_metadata_df['isic_id'].to_numpy()[valid_idx])

        # evaluate with all data
        preds_all_img = get_oof(valid_dataloader_all, img_model)
        best_preds_all_img = get_oof(valid_dataloader_all, best_img_model)
        all_pauc = common_training.calc_metric(y_pred=preds_all_img, y_true=train_metadata_df['target'].to_numpy()[valid_idx_all])
        best_all_pauc = common_training.calc_metric(y_pred=best_preds_all_img, y_true=train_metadata_df['target'].to_numpy()[valid_idx_all])
        logger.info(f'all_pauc: {all_pauc:.4f} best_all_pauc: {best_all_pauc:.4f}')
        preds_all_img_list.append(preds_all_img)
        best_preds_all_img_list.append(best_preds_all_img)
        isic_ids_all_img_list.append(train_metadata_df['isic_id'].to_numpy()[valid_idx_all])

    # save model
    img_model_dir = cfg.OUTPUT_DIR / 'pt'
    os.makedirs(img_model_dir, exist_ok=True)
    for i, model in enumerate(img_model_list):
        torch.save(model.state_dict(), img_model_dir / f'img_model_{i}.pt')
    for i, model in enumerate(best_img_model_list):
        torch.save(model.state_dict(), img_model_dir / f'best_img_model_{i}.pt')
    # save pred result
    result_img_df = pd.DataFrame({
        'isic_id': np.hstack(isic_ids_img_list),
        'y_pred_img': np.hstack(preds_img_list),
        'best_y_pred_img': np.hstack(best_preds_img_list),
    })
    result_img_df.to_csv(img_model_dir / 'result_img.csv', index=False)
    # save pred result, all
    result_all_img_df = pd.DataFrame({
        'isic_id': np.hstack(isic_ids_all_img_list),
        'y_pred_img': np.hstack(preds_all_img_list),
        'best_y_pred_img': np.hstack(best_preds_all_img_list),
    })
    result_all_img_df.to_csv(img_model_dir / 'result_all_img.csv', index=False)