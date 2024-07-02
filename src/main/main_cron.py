""" 
Module for running the model
"""

import sys
import pandas as pd
sys.path.append('.')
from src.main.data_qa import data_qa
from src.main.graphics import graphics
from src.main.feature_eng import feature_eng
from src.main.model import model_catboost, model_lightgbm
from src.main.utils.parameter_log import loggin_custom
from src.main.utils.utils_functions import results_to_csv

logger = loggin_custom()

PATH = 'data'

logger.info("Starting the model")

logger.info("Reading the dataset")
data_df = pd.read_csv("data/diabetic_data.csv")

############################### DATA QUALITY ########################################

logger.info("Running the data quality checks")

data_clean = data_qa(data_df)

results_to_csv(data_clean,'data_clean', PATH)
############################# FEATURE ENGINEERING ###################################

data_clean_in = pd.read_csv("data/data_clean.csv")

logger.info("Running the feature engineering")
new_data = feature_eng(data_clean_in)

results_to_csv(new_data,'new_data', PATH)
############################# GRAPHIS ##############################################

data_new_in = pd.read_csv("data/new_data.csv")

# columns = [
#     'precentageVisitsByAge', 'race', 'age',
#     'numberOfMedications', 'categories_diag_1',
#     'number_inpatient', 'discharge_disposition_id']

test = ['precentageVisitsByAge']

logger.info("Running the graphics")
graphics(data_new_in, test)

#################################### MODEL #########################################

data_new_in = pd.read_csv("data/new_data.csv")

logger.info("Running the models")

model_catboost(data_new_in, train=False)

model_lightgbm(data_new_in, train=False)

logger.info("Finished the model")
