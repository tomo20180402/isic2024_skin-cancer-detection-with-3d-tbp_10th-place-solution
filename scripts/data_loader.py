from io import BytesIO
from typing import List, Dict, Tuple
from pathlib import Path
import h5py
from PIL import Image
from tqdm import tqdm
import polars as pl
from scripts.config import Config as cfg
from scripts.logger import create_logger
logger = create_logger()


def load_metadata(data_dir: Path) -> Tuple[pl.DataFrame, pl.DataFrame]:
    if cfg.IS_DEBUG:
        logger.info('debug mode')
        n_rows = 10_000
    else:
        n_rows = None
    train_metadata_df = pl.read_csv(data_dir / 'train-metadata.csv', n_rows=n_rows)
    test_metadata_df = pl.read_csv(data_dir / 'test-metadata.csv')
    return train_metadata_df, test_metadata_df


def load_images(image_hdf5_path: Path, metadata_df: pl.DataFrame) -> Tuple[List[Image.Image], List[str]]:
    image_hdf5 = h5py.File(image_hdf5_path, 'r')
    imgs = []
    isic_ids = []
    for isic_id in tqdm(metadata_df['isic_id'].to_list()):
        img = Image.open(BytesIO(image_hdf5[isic_id][()]))
        imgs.append(img)
        isic_ids.append(isic_id)
    return imgs, isic_ids