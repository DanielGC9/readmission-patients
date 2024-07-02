""" 
This module is used for data quality checks
"""

import sys
import pandas as pd
import numpy as np
import warnings
sys.path.append('.')
from src.main.utils.parameter_log import loggin_custom
from src.main.utils.utils_functions import (check_column_type,
                                            equal_columns,
                                            check_unique_nulls,
                                            )

warnings.filterwarnings('ignore')
logger = loggin_custom()


def data_qa(data_df):
    """
        The `data_qa` function performs various data quality checks and transformations on a DataFrame
        related to diabetes patient data.

        Args:
                data_df: The `data_qa` function you provided seems to be performing various data quality checks
        and transformations on a DataFrame named `data_df`. Here is a breakdown of the steps it is taking:

        Returns:
                The function `data_qa` is returning a cleaned and processed DataFrame `df_diabetes` after
        performing various data quality checks and transformations on the input DataFrame `data_df`.

    """

    logger.info("Ordering Columns")

    col_ord = ['encounter_id', 'patient_nbr', 'name', 'race', 'gender', 'age', 'weight', 'admission_type_id',
            'discharge_disposition_id', 'admission_source_id', 'time_in_hospital', 'payer_code',
            'payer_code_2', 'medical_specialty', 'num_lab_procedures', 'num_procedures', 'num_medications',
            'number_outpatient', 'number_emergency', 'number_inpatient', 'diag_1', 'diag_2', 'diag_3',
            'number_diagnoses', 'max_glu_serum', 'A1Cresult', 'change', 'diabetesMed',

            'glyburide', 'glyburide-metformin', 'glyburide-metformin_2', 'troglitazone', 'troglitazone_2',
            'metformin-pioglitazone', 'metformin-pioglitazone_2', 'metformin-rosiglitazone', 'rosiglitazone',
            'chlorpropamide', 'examide', 'tolazamide', 'citoglipton', 'miglitol', 'glipizide', 'insulin', 
            'glipizide-metformin', 'glimepiride', 'acarbose', 'acetohexamide', 'repaglinide', 'nateglinide',
            'glimepiride-pioglitazone', 'pioglitazone', 'tolbutamide', 'metformin',

            'out', 'US', 'readmitted']


    df_diabetes = data_df[col_ord]
    
    logger.debug(f'df_diabetes:\n {df_diabetes.head()}')
    
    logger.info("Checking atypical rows")
 
    atipical_rows = df_diabetes[df_diabetes['patient_nbr'] == 'abcde']

    logger.info(f'atipical_rows:\n {atipical_rows}')

    df_diabetes.drop(df_diabetes[(df_diabetes['out'] == 'abcde')].index, inplace=True)
    df_diabetes.reset_index(drop=True)

    logger.info(f"Shape of the dataset: {df_diabetes.shape}")

    logger.info(f"Checking column types:\n {check_column_type(df_diabetes)}")
    
    logger.info("Changing data types")

    df_diabetes['patient_nbr'] = df_diabetes['patient_nbr'].astype(str)
    df_diabetes['encounter_id'] = df_diabetes['encounter_id'].astype(str)
    df_diabetes['time_in_hospital'] = df_diabetes['time_in_hospital'].astype(int)
    df_diabetes['num_lab_procedures'] = df_diabetes['num_lab_procedures'].astype(int)
    df_diabetes['num_procedures'] = df_diabetes['num_procedures'].astype(int)
    df_diabetes['num_medications'] = df_diabetes['num_medications'].astype(int)
    df_diabetes['number_outpatient'] = df_diabetes['number_outpatient'].astype(int)
    df_diabetes['number_emergency'] = df_diabetes['number_emergency'].astype(int)
    df_diabetes['number_inpatient'] = df_diabetes['number_inpatient'].astype(int)
    df_diabetes['number_diagnoses'] = df_diabetes['number_diagnoses'].astype(int)

    df_diabetes['US'] = df_diabetes['US'].astype(int)
    df_diabetes['admission_type_id'] = df_diabetes['admission_type_id'].astype(str)
    df_diabetes['admission_source_id'] = df_diabetes['admission_source_id'].astype(str)
    df_diabetes['discharge_disposition_id'] = df_diabetes['discharge_disposition_id'].astype(str)

    # check duplicate rows
    duplicate_rows = df_diabetes[df_diabetes.duplicated(keep=False)]
 
    duplicate_rows.groupby(['encounter_id'])['patient_nbr'].count().sort_values(ascending=False)

    logger.info(f"Duplicate rows:\n {duplicate_rows.shape}")

    logger.info(f'Number of rows before removing duplicates: {df_diabetes.shape[0]}')
    df_diabetes.drop_duplicates(inplace=True)
    logger.info(f'Number of rows after removing duplicates: {df_diabetes.shape[0]}')
    

    duplicate_cols = equal_columns(df_diabetes)

    delete_cols = ['payer_code_2', 'glyburide-metformin_2', 'troglitazone_2',
               'metformin-pioglitazone_2', 'citoglipton']

    logger.info(f"Columns to be deleted: {delete_cols}")

    df_diabetes.drop(delete_cols, axis=1, inplace=True)

    # NaN mapping
    logger.info("Replacing '?'' values with np.nan")
    df_diabetes.replace('?', np.nan , inplace=True)

    # NaN mapping for gender
    logger.info("Replacing 'Unknown/Invalid' values with np.nan")
    df_diabetes['gender'] = df_diabetes['gender'].replace('Unknown/Invalid', np.nan)

    # NaN mapping for admission_type_id [8, 6, 5]
    logger.info("Replacing 'Not Available' values with np.nan")
    df_diabetes = df_diabetes.replace({'admission_type_id' : {'8' : np.nan, 
                                                              '6' : np.nan, 
                                                              '5' : np.nan}})

    # NaN mapping for admission_source_id [21, 20, 17, 15, 9]
    df_diabetes = df_diabetes.replace({'admission_source_id' : { '21' : np.nan,
                                                                 '20' : np.nan,
                                                                 '17' : np.nan, 
                                                                 '15' : np.nan, 
                                                                 '9' : np.nan}})

    # NaN mapping for discharge_disposition_id [18, 25, 26]
    df_diabetes = df_diabetes.replace({'discharge_disposition_id' : {'18' : np.nan, 
                                                                     '25' : np.nan, 
                                                                     '26' : np.nan}})

    # Expired patient mapping
    logger.info("Expired patient deletion")
    expired = ['11', '19', '20', '21']
    df_diabetes = df_diabetes[~df_diabetes['discharge_disposition_id'].isin(expired)]


    logger.info("Checking for unique values and nulls")

    logger.info(f"check_unique_nulls:\n {check_unique_nulls(df_diabetes)}")

    # drop columns with only one value and with null values >= 40%

    df_only_null = check_unique_nulls(df_diabetes)

    only = df_only_null[df_only_null['unique'] == 1]['column'].to_list()
    logger.info(f'cols with only one value: {only}')
    logger.info('-----------------------')

    nulls_40 = df_only_null[df_only_null['nullPercentage'] >= 40]['column'].to_list()
    logger.info(f'cols with null values >= 40%: {nulls_40}')
    del_col = only + nulls_40
    df_diabetes = df_diabetes.drop(del_col, axis=1)
    logger.info('-----------------------')
    logger.info('Columns successfully deleted')

    # drop null values

    logger.info(f'Number of rows before removing duplicates: {df_diabetes.shape[0]}')
    df_diabetes = df_diabetes.dropna()
    logger.info(f'Number of rows after removing duplicates: {df_diabetes.shape[0]}')

    logger.info('Data quality checks successfully completed')

    return df_diabetes
