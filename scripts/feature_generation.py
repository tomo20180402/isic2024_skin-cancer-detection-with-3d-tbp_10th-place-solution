from typing import List, Dict, Tuple
import re
import numpy as np
import polars as pl
from scripts.config import Config as cfg
from scripts.logger import create_logger
logger = create_logger()


def metadata_preprocess(metadata_df: pl.DataFrame) -> pl.DataFrame:
    # convert age_approx from str to int
    metadata_df = metadata_df.with_columns(
        pl.when(pl.col('age_approx').cast(str)=='NA').then(None)
        .otherwise(pl.col('age_approx'))
        .cast(pl.Float64)
        .alias('age_approx')
    )
    # fill nan by median
    metadata_df = metadata_df.with_columns(
        pl.col(pl.Float64).fill_nan(pl.col(pl.Float64).median()),
    )
    return metadata_df


def add_metadata_feature(metadata_df: pl.DataFrame) -> pl.DataFrame:
    metadata_df = metadata_df.with_columns(
        lesion_size_ratio = pl.col('tbp_lv_minorAxisMM') / pl.col('clin_size_long_diam_mm'),
        lesion_shape_index = pl.col('tbp_lv_areaMM2') / (pl.col('tbp_lv_perimeterMM') ** 2),
        hue_contrast = (pl.col('tbp_lv_H') - pl.col('tbp_lv_Hext')).abs(),
        luminance_contrast = (pl.col('tbp_lv_L') - pl.col('tbp_lv_Lext')).abs(),
        lesion_color_difference = (pl.col('tbp_lv_deltaA') ** 2 + pl.col('tbp_lv_deltaB') ** 2 + pl.col('tbp_lv_deltaL') ** 2).sqrt(),
        border_complexity = pl.col('tbp_lv_norm_border') + pl.col('tbp_lv_symm_2axis'),
        color_uniformity = pl.col('tbp_lv_color_std_mean') / (pl.col('tbp_lv_radial_color_std_max') + cfg.ERR),
        position_distance_3d = (pl.col('tbp_lv_x') ** 2 + pl.col('tbp_lv_y') ** 2 + pl.col('tbp_lv_z') ** 2).sqrt(),
        perimeter_to_area_ratio = pl.col('tbp_lv_perimeterMM') / pl.col('tbp_lv_areaMM2'),
        area_to_perimeter_ratio = pl.col('tbp_lv_areaMM2') / pl.col('tbp_lv_perimeterMM'),
        lesion_visibility_score = pl.col('tbp_lv_deltaLBnorm') + pl.col('tbp_lv_norm_color'),
        combined_anatomical_site = pl.col('anatom_site_general') + '_' + pl.col('tbp_lv_location'),
        symmetry_border_consistency = pl.col('tbp_lv_symm_2axis') * pl.col('tbp_lv_norm_border'),
        consistency_symmetry_border = pl.col('tbp_lv_symm_2axis') * pl.col('tbp_lv_norm_border') / (pl.col('tbp_lv_symm_2axis') + pl.col('tbp_lv_norm_border')),
        color_consistency = pl.col('tbp_lv_stdL') / pl.col('tbp_lv_Lext'),
        consistency_color = pl.col('tbp_lv_stdL') * pl.col('tbp_lv_Lext') / (pl.col('tbp_lv_stdL') + pl.col('tbp_lv_Lext')),
        size_age_interaction = pl.col('clin_size_long_diam_mm') * pl.col('age_approx'),
        hue_color_std_interaction = pl.col('tbp_lv_H') * pl.col('tbp_lv_color_std_mean'),
        lesion_severity_index = (pl.col('tbp_lv_norm_border') + pl.col('tbp_lv_norm_color') + pl.col('tbp_lv_eccentricity')) / 3,
        color_contrast_index = pl.col('tbp_lv_deltaA') + pl.col('tbp_lv_deltaB') + pl.col('tbp_lv_deltaL') + pl.col('tbp_lv_deltaLBnorm'),
        log_lesion_area = (pl.col('tbp_lv_areaMM2') + 1).log(),
        normalized_lesion_size = pl.col('clin_size_long_diam_mm') / pl.col('age_approx'),
        mean_hue_difference = (pl.col('tbp_lv_H') + pl.col('tbp_lv_Hext')) / 2,
        std_dev_contrast = ((pl.col('tbp_lv_deltaA') ** 2 + pl.col('tbp_lv_deltaB') ** 2 + pl.col('tbp_lv_deltaL') ** 2) / 3).sqrt(),
        color_shape_composite_index = (pl.col('tbp_lv_color_std_mean') + pl.col('tbp_lv_area_perim_ratio') + pl.col('tbp_lv_symm_2axis')) / 3,
        lesion_orientation_3d = pl.arctan2(pl.col('tbp_lv_y'), pl.col('tbp_lv_x')),
        overall_color_difference = (pl.col('tbp_lv_deltaA') + pl.col('tbp_lv_deltaB') + pl.col('tbp_lv_deltaL')) / 3,
        symmetry_perimeter_interaction = pl.col('tbp_lv_symm_2axis') * pl.col('tbp_lv_perimeterMM'),
        comprehensive_lesion_index = (pl.col('tbp_lv_area_perim_ratio') + pl.col('tbp_lv_eccentricity') + pl.col('tbp_lv_norm_color') + pl.col('tbp_lv_symm_2axis')) / 4,
        color_variance_ratio = pl.col('tbp_lv_color_std_mean') / pl.col('tbp_lv_stdLExt'),
        border_color_interaction = pl.col('tbp_lv_norm_border') * pl.col('tbp_lv_norm_color'),
        border_color_interaction_2 = pl.col('tbp_lv_norm_border') * pl.col('tbp_lv_norm_color') / (pl.col('tbp_lv_norm_border') + pl.col('tbp_lv_norm_color')),
        size_color_contrast_ratio = pl.col('clin_size_long_diam_mm') / pl.col('tbp_lv_deltaLBnorm'),
        age_normalized_nevi_confidence_2 = (pl.col('clin_size_long_diam_mm')**2 + pl.col('age_approx')**2).sqrt(),
        color_asymmetry_index = pl.col('tbp_lv_radial_color_std_max') * pl.col('tbp_lv_symm_2axis'),
        volume_approximation_3d = pl.col('tbp_lv_areaMM2') * (pl.col('tbp_lv_x')**2 + pl.col('tbp_lv_y')**2 + pl.col('tbp_lv_z')**2).sqrt(),
        color_range = (pl.col('tbp_lv_L') - pl.col('tbp_lv_Lext')).abs() + (pl.col('tbp_lv_A') - pl.col('tbp_lv_Aext')).abs() + (pl.col('tbp_lv_B') - pl.col('tbp_lv_Bext')).abs(),
        shape_color_consistency = pl.col('tbp_lv_eccentricity') * pl.col('tbp_lv_color_std_mean'),
        border_length_ratio = pl.col('tbp_lv_perimeterMM') / (2 * np.pi * (pl.col('tbp_lv_areaMM2') / np.pi).sqrt()),
        age_size_symmetry_index = pl.col('age_approx') * pl.col('clin_size_long_diam_mm') * pl.col('tbp_lv_symm_2axis'),
        index_age_size_symmetry = pl.col('age_approx') * pl.col('tbp_lv_areaMM2') * pl.col('tbp_lv_symm_2axis'),
        area_age_ratio = pl.col('tbp_lv_areaMM2') / pl.col('age_approx'),
        symmetry_eccentricity_interaction = pl.col('tbp_lv_symm_2axis') * pl.col('tbp_lv_eccentricity'),
        border_complexity_normalized = pl.col('tbp_lv_norm_border') / (pl.col('tbp_lv_areaMM2').sqrt() + cfg.ERR),
        color_variation_index = pl.col('tbp_lv_color_std_mean') * pl.col('tbp_lv_radial_color_std_max'),
        lesion_flatness = (pl.col('tbp_lv_x')**2 + pl.col('tbp_lv_y')**2).sqrt() / (pl.col('tbp_lv_z') + cfg.ERR),
        relative_hue_difference = (pl.col('tbp_lv_H') - pl.col('tbp_lv_Hext')).abs() / (pl.col('tbp_lv_H') + pl.col('tbp_lv_Hext') + cfg.ERR),
        area_diameter_ratio = pl.col('tbp_lv_areaMM2') / (pl.col('clin_size_long_diam_mm')**2 + cfg.ERR),
        color_geometric_mean = (pl.col('tbp_lv_L') * pl.col('tbp_lv_A') * pl.col('tbp_lv_B')) ** (1/3),
        tbp_lv_norm_color_per_long_diam = pl.col('tbp_lv_norm_color') / pl.col('clin_size_long_diam_mm').log1p(),
        norm_L = (pl.col('tbp_lv_stdL') + pl.col('tbp_lv_color_std_mean')) / pl.col('tbp_lv_L'),
        norm_A = (pl.col('tbp_lv_stdL') + pl.col('tbp_lv_color_std_mean')) / abs((pl.col('tbp_lv_A')) + cfg.ERR),
        norm_B = (pl.col('tbp_lv_stdL') + pl.col('tbp_lv_color_std_mean')) / abs((pl.col('tbp_lv_B')) + cfg.ERR),
        area_ratio = (np.pi * (pl.col('clin_size_long_diam_mm')/2)**2) / pl.col('tbp_lv_areaMM2'),
        clin_size_long_diam_mm_per_age = pl.col('clin_size_long_diam_mm') / pl.col('age_approx'),
    ).with_columns(
        shape_complexity_index = pl.col('border_complexity') + pl.col('lesion_shape_index'),
        weighted_area_ratio = (pl.col('area_ratio') * (pl.col('tbp_lv_areaMM2'))).log1p(),
    ).with_columns(
        [((pl.col(col) - pl.col(col).mean().over('patient_id')) / (pl.col(col).std().over('patient_id') + cfg.ERR)).alias(f'{col}_patient_norm') for col in (cfg.NUM_FEATURE_COLS)] + \
        [((pl.col(col) - pl.col(col).mean().over(['patient_id', 'tbp_tile_type'])) / \
          (pl.col(col).std().over(['patient_id', 'tbp_tile_type']) + cfg.ERR)).alias(f'{col}_patient_ttype_norm') for col in (cfg.NUM_FEATURE_COLS)] + \
        [((pl.col(col) - pl.col(col).mean().over(['patient_id', 'tbp_lv_location'])) / \
          (pl.col(col).std().over(['patient_id', 'tbp_lv_location']) + cfg.ERR)).alias(f'{col}_patient_loc_norm') for col in (cfg.NUM_FEATURE_COLS)] + \
        [((pl.col(col) - pl.col(col).mean().over(['patient_id', 'tbp_lv_location_simple'])) / \
          (pl.col(col).std().over(['patient_id', 'tbp_lv_location_simple']) + cfg.ERR)).alias(f'{col}_patient_locsim_norm') for col in (cfg.NUM_FEATURE_COLS)] + \
        [((pl.col(col) - pl.col(col).mean().over(['patient_id', 'attribution'])) / \
          (pl.col(col).std().over(['patient_id', 'attribution']) + cfg.ERR)).alias(f'{col}_patient_attr_norm') for col in (cfg.NUM_FEATURE_COLS)]
    ).with_columns(
        count_per_patient = pl.col('isic_id').count().over('patient_id'),
        count_per_patient_ttype = pl.col('isic_id').count().over(['patient_id', 'tbp_tile_type']),
        count_per_patient_loc = pl.col('isic_id').count().over(['patient_id', 'tbp_lv_location']),
        count_per_patient_locsim = pl.col('isic_id').count().over(['patient_id', 'tbp_lv_location_simple']),
        count_per_patient_attr = pl.col('isic_id').count().over(['patient_id', 'attribution']),
    )
    return metadata_df


def convert_categorical_dtype(metadata_df: pl.DataFrame) -> pl.DataFrame:
    metadata_df = metadata_df.with_columns(pl.col(cfg.CAT_FEATURE_COLS).cast(pl.Categorical))
    return metadata_df


def generate_categories_dict(metadata_df: pl.DataFrame) -> Dict[str, List[str]]:
    categories_dict = {}
    for col in cfg.CAT_FEATURE_COLS:
        categories_dict[col] = metadata_df[col].unique().sort().to_list()
    return categories_dict


def get_dummies(metadata_df: pl.DataFrame, cat_feature_cols: List[str], categories_dict: Dict[str, List[str]]) -> pl.DataFrame:
    for col in cat_feature_cols:
        for cat in categories_dict[col]:
            dummy_col_name = f"{col}_{cat}"
            metadata_df = metadata_df.with_columns(pl.when(pl.col(col) == cat).then(1).otherwise(0).alias(dummy_col_name))
    metadata_df = metadata_df.drop(cat_feature_cols)
    return metadata_df


def align_test_columns_to_train(train_metadata_df: pl.DataFrame, test_metadata_df: pl.DataFrame) -> pl.DataFrame:
    missing_cols = set(train_metadata_df.columns) - set(test_metadata_df.columns)
    for col in missing_cols:
        test_metadata_df = test_metadata_df.with_columns(pl.lit(0).alias(col))
    test_metadata_df = test_metadata_df.select(train_metadata_df.columns)
    return test_metadata_df


def clean_column_names(metadata_df: pl.DataFrame) -> pl.DataFrame:
    metadata_df = metadata_df.to_pandas()
    clean_names = [re.sub(r'[^\w\s]', '_', col) for col in metadata_df.columns]
    clean_names = [name.replace(' ', '_') for name in clean_names]
    clean_names = [f'col_{name}' if name[0].isdigit() else name for name in clean_names]
    metadata_df = pl.from_pandas(metadata_df.rename(columns=dict(zip(metadata_df.columns, clean_names))))
    return metadata_df


def align_test_dtypes_to_train(train_metadata_df: pl.DataFrame, test_metadata_df: pl.DataFrame) -> pl.DataFrame:
    train_dtype_dict = {c:d for c,d in zip(train_metadata_df.columns, train_metadata_df.dtypes)}
    test_metadata_df = test_metadata_df.cast(train_dtype_dict)
    return test_metadata_df


def fill_inf_with_median(test_metadata_df: pl.DataFrame, feature_cols: List[str]) -> pl.DataFrame:
    medians = test_metadata_df.select([pl.col(f).median().alias(f"{f}_median") for f in feature_cols])
    expressions = [pl.when(pl.col(f).is_infinite()).then(medians[f"{f}_median"]).otherwise(pl.col(f)).alias(f) for f in feature_cols]
    return test_metadata_df.with_columns(expressions)


def run(train_metadata_df: pl.DataFrame, test_metadata_df: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame]:
    logger.info(f'start: metadata_preprocess')
    train_metadata_df = metadata_preprocess(metadata_df=train_metadata_df)
    test_metadata_df = metadata_preprocess(metadata_df=test_metadata_df)

    logger.info(f'start: add_metadata_feature')
    train_metadata_df = add_metadata_feature(metadata_df=train_metadata_df)
    test_metadata_df = add_metadata_feature(metadata_df=test_metadata_df)

    logger.info(f'start: convert_categorical_dtype')
    train_metadata_df = convert_categorical_dtype(metadata_df=train_metadata_df)
    test_metadata_df = convert_categorical_dtype(metadata_df=test_metadata_df)

    logger.info(f'start: generate_categories_dict')
    categories_dict = generate_categories_dict(metadata_df=train_metadata_df)

    logger.info(f'start: get_dummies')
    train_metadata_df = get_dummies(metadata_df=train_metadata_df, cat_feature_cols=cfg.CAT_FEATURE_COLS, categories_dict=categories_dict)
    test_metadata_df = get_dummies(metadata_df=test_metadata_df, cat_feature_cols=cfg.CAT_FEATURE_COLS, categories_dict=categories_dict)

    logger.info(f'start: align_test_columns_to_train')
    test_metadata_df = align_test_columns_to_train(train_metadata_df=train_metadata_df, test_metadata_df=test_metadata_df)

    logger.info(f'start: clean_column_names')
    train_metadata_df = clean_column_names(metadata_df=train_metadata_df)
    test_metadata_df = clean_column_names(metadata_df=test_metadata_df)

    logger.info(f'start: align_test_dtypes_to_train')
    test_metadata_df = align_test_dtypes_to_train(train_metadata_df=train_metadata_df, test_metadata_df=test_metadata_df)

    logger.info(f'start: fill_inf_with_median')
    test_metadata_df = fill_inf_with_median(test_metadata_df=test_metadata_df, feature_cols=cfg.FEATURE_COLS_P1)

    return train_metadata_df, test_metadata_df