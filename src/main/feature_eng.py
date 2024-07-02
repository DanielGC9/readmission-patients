""" 
This module is used for feature engineering
"""

import sys
import warnings
sys.path.append('.')
from src.main.utils.parameter_log import loggin_custom
from src.main.utils.utils_functions import (map_diagnosis,
                                            count_unique_diagnoses,
                                            count_medications,
                                            simplification_variables)

warnings.filterwarnings('ignore')
logger = loggin_custom()

def feature_eng(df_diabetes):
    """
    The `feature_eng` function performs feature engineering on a diabetes dataset, including modifying
    target variables, simplifying and creating new variables.
    
    Args:
      df_diabetes: The `feature_eng` function you provided seems to be performing various feature
    engineering tasks on a DataFrame containing diabetes data. Here's a breakdown of what the function
    does:
    
    Returns:
      The function `feature_eng` is returning the DataFrame `df_diabetes` after performing various
    feature engineering tasks such as modifying the target variable, simplifying variables, creating new
    variables like 'distinctDiagnoses', 'numberOfMedications', and 'precentageVisitsByAge'. The function
    logs information at different stages of feature engineering and returns the modified DataFrame
    `df_diabetes`.
    """

    logger.info("Starting Feature Engineering")

    # Modification of target
    logger.info("Modifying target")

    df_diabetes['readmitted'] = df_diabetes['readmitted'].map({'<30': 1, 'NO': 0, '>30': 0})
    target = df_diabetes['readmitted'].value_counts()
    target = target.reset_index(drop=False)
    target['%'] = round(target['count']/sum(target['count'])*100, 2)

    logger.info(f'Target:\n {target}')

    ### Simplification of variables ###

    logger.info("Simplifying variables")

    logger.info("Modifying change, gender and diabetesMed to 0 and 1")
    simplification_variables(df_diabetes, 'change')
    simplification_variables(df_diabetes, 'gender')
    simplification_variables(df_diabetes, 'diabetesMed')

    logger.info("Modifying age to 0-9")
    simplification_variables(df_diabetes, 'age')


    logger.info("Modifying medications to:\n 'No': 0, 'Steady': 1, 'Up': 1, 'Down': 1")
    medications = [
        'glyburide', 'glyburide-metformin','troglitazone',
        'metformin-pioglitazone', 'metformin-rosiglitazone',
        'rosiglitazone', 'chlorpropamide', 'tolazamide', 'miglitol',
        'glipizide', 'insulin', 'glipizide-metformin', 'glimepiride',
        'acarbose', 'acetohexamide', 'repaglinide', 'nateglinide',
        'glimepiride-pioglitazone', 'pioglitazone', 'tolbutamide', 'metformin']

    for med in medications:
        simplification_variables(df_diabetes, med)


    for col in ['diag_1', 'diag_2', 'diag_3']:
        logger.info(f"Modifying {col}")
        map_diagnosis(df_diabetes, col)

    ### New variables ###
        
    # Number of unique diagnoses
    logger.info("New variable: Number of unique diagnoses")
    df_diabetes['distinctDiagnoses'] = df_diabetes.apply(count_unique_diagnoses, axis=1)

    # Number of medications used
    logger.info("New variable: Number of medications used")
    df_diabetes['numberOfMedications'] = df_diabetes.apply(count_medications, axis=1)

    # Percentage of number of visits by age
    logger.info("New variable: Percentage of number of visits by age")

    df_diabetes['total_visits'] = df_diabetes['number_inpatient'] \
        + df_diabetes['number_emergency'] + df_diabetes['number_outpatient']
    
    group = df_diabetes.groupby('age')['total_visits'].mean()
    df_diabetes['precentageVisitsByAge'] = round(
        df_diabetes['total_visits']/df_diabetes['age'].map(group))

    logger.info("Feature Engineering Finished")

    return df_diabetes
