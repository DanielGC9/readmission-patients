""" 
This module contains the models
"""

import sys
import time
import pickle
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
sys.path.append('.')
from src.main.utils.parameter_log import loggin_custom
from src.main.utils.utils_functions import metrics_report


logger = loggin_custom()

def model_catboost(df_diabetes, train=False):
    """
    The code defines functions to train and save CatBoost and LightGBM models for a diabetes readmission
    prediction task.
    
    Args:
      df_diabetes: The code you provided defines two functions, `model_catboost` and `model_lightgbm`,
    for building CatBoost and LightGBM models, respectively. These functions take a DataFrame
    `df_diabetes` as input, which presumably contains data related to diabetes patients.
      train: The code you provided defines two functions, `model_catboost` and `model_lightgbm`, for
    training and creating models using CatBoost and LightGBM algorithms, respectively. These functions
    take a DataFrame `df_diabetes` as input and have a parameter `train` which is set to `. Defaults to
    False
    """

    features = ['race', 'gender', 'age','admission_type_id', 'discharge_disposition_id', 
                'admission_source_id', 'time_in_hospital', 'num_lab_procedures', 'num_procedures',
                'num_medications', 'number_outpatient', 'number_emergency',
                'number_inpatient', 'number_diagnoses', 'change', 'diabetesMed', 'glyburide',
                'glyburide-metformin', 'rosiglitazone','glipizide', 'insulin', 'glimepiride',
                'acarbose', 'repaglinide', 'nateglinide', 'pioglitazone', 'metformin', 

                'precentageVisitsByAge', 'distinctDiagnoses', 'numberOfMedications',
                'categories_diag_1', 'categories_diag_2', 'categories_diag_3', 'total_visits']

    df_result = df_diabetes.copy()

    X = df_result.loc[:, features]
    y = df_result['readmitted']

    cat_features = X.select_dtypes('object').columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=123)


    weight_0 = 1
    weight_1 = 72420/9397

    if not train:

        logger.info("Starting the creation of the model")

        clf_cat = CatBoostClassifier(subsample=0.8,
                                    n_estimators=150,
                                    max_depth=7,
                                    l2_leaf_reg=5,
                                    learning_rate=0.1,
                                    class_weights=[weight_0, weight_1],)

        clf_cat.fit(X_train, y_train,
                    cat_features=cat_features,
                    eval_set=(X_test, y_test),
                    verbose=False)

        logger.info(f'Best score:\n {clf_cat.get_best_score()}')

        metrics_report(clf_cat, X_test, y_test, X)

        filename = 'models/catboost_model.pkl'
        pickle.dump(clf_cat, open(filename, 'wb'))

        logger.info(f'Model saved in {filename}')

    else:
        logger.info("Starting the training of the model")

        filename = 'models/catboost_model.pkl'
        clf_cat = pickle.load(open(filename, 'rb'))

        start_time = time.time()

        grid = {'learning_rate': [i/10.0 for i in range(0, 5)],
                'max_depth': range(-1, 12, 2),
                'n_estimators': [100,200,300],
                'subsample': [i/10.0 for i in range(6, 10)],
                'l2_leaf_reg': [1, 3, 5]}


        clfCat = RandomizedSearchCV(estimator=clf_cat,
                                    param_distributions=grid,
                                    scoring='recall',
                                    n_iter=50,
                                    verbose=0)

        clfCat.fit(X_train, y_train)

        logger.info(f'The optimisation takes {(time.time()-start_time)/60.} minutes.')

        # Inspect the results
        logger.info(f"Best parameters:\n {clfCat.best_params_}")
        logger.info(f"Score:\n {clfCat.best_score_}")

        catBest = clfCat.best_params_

        clf_cat_train = CatBoostClassifier(**catBest,
                                    class_weights=[weight_0, weight_1])

        clf_cat_train.fit(X_train, y_train,
                    cat_features=cat_features,
                    eval_set=(X_test, y_test),
                    verbose=False)


        filename = 'models/catboost_model_train.pkl'
        pickle.dump(clf_cat_train, open(filename, 'wb'))

        logger.info(f'Model saved in {filename}')

        metrics_report(clf_cat_train, X_test, y_test, X)

        logger.info("End of the training of the model")



def model_lightgbm(df_diabetes, train=False):

    features = ['race', 'gender', 'age','admission_type_id', 'discharge_disposition_id', 
                'admission_source_id', 'time_in_hospital', 'num_lab_procedures', 'num_procedures',
                'num_medications', 'number_outpatient', 'number_emergency',
                'number_inpatient', 'number_diagnoses', 'change', 'diabetesMed', 'glyburide',
                'glyburide-metformin', 'rosiglitazone','glipizide', 'insulin', 'glimepiride',
                'acarbose', 'repaglinide', 'nateglinide', 'pioglitazone', 'metformin', 

                'precentageVisitsByAge', 'distinctDiagnoses', 'numberOfMedications',
                'categories_diag_1', 'categories_diag_2', 'categories_diag_3', 'total_visits']
    
    df_result = df_diabetes.copy()

    X = df_result.loc[:, features]
    y = df_result['readmitted']

    for c in X.columns:
        col_type = X[c].dtype
        if col_type == 'object' or col_type.name == 'category':
            X[c] = X[c].astype('category')

    cat_col = X.select_dtypes('category').columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3) #, random_state=123

    weight_0 = 1
    weight_1 = 72420/9397

    class_weights = {0: weight_0, 1: weight_1} 

    if not train:

        logger.info("Starting the creation of the model")

        best_params = {'boosting_type': 'gbdt',
                    'colsample_bytree': 1.0,
                    'importance_type': 'gain',
                    'learning_rate': 0.1625658410779909,
                    'max_depth': 7,
                    'min_child_samples': 1,
                    'n_estimators': 200,
                    'num_leaves': 250,
                    'reg_alpha': 2.0,
                    'reg_lambda': 2.0,
                    'subsample': 0.1,
                    'scale_pos_weight':9, 
                    'verbose': -1}

        model_lgbm = lgb.LGBMClassifier(categorical_feature=cat_col,
                                    objective='binary',
                                    class_weight=class_weights,
                                    **best_params)
        model_lgbm.fit(X_train, y_train)

        filename = 'models/lgbm_model.pkl'
        pickle.dump(model_lgbm, open(filename, 'wb'))

        logger.info(f'Model saved in {filename}')

        metrics_report(model_lgbm, X_test, y_test, X)

        logger.info("End of the creation of the model")


    else:

        logger.info("Starting the training of the model")

        filename = 'models/lgbm_model.pkl'
        model_lgb = pickle.load(open(filename, 'rb'))

        start_time = time.time()

        params_lgb = {'max_depth': range(-1, 12, 2),
                    'num_leaves': range(30, 150, 10),
                    'learning_rate': [i/10.0 for i in range(0, 5)],
                    'min_child_weight': range(1, 6, 2),
                    'n_estimators': [800, 850, 900, 950, 1000, 1050, 1100, 1150],
                    'subsample': [i/10.0 for i in range(6, 10)],
                    'colsample_bytree': [i/10.0 for i in range(5, 10)]
                    }

        modelclf_lgb = RandomizedSearchCV(estimator=model_lgb,
                                    param_distributions=params_lgb,
                                    scoring='recall',
                                    n_iter=50,
                                    verbose=0)

        modelclf_lgb.fit(X_train, y_train)

        logger.info(f'The optimisation takes {(time.time()-start_time)/60.} minutes.')

        logger.info(f"Best parameters: {modelclf_lgb.best_params_}")
        logger.info(f"Score: {modelclf_lgb.best_score_}")


        best_params = modelclf_lgb.best_params_

        clf_lgb_train = lgb.LGBMClassifier(categorical_feature = cat_col,
                                    objective = 'binary',
                                    class_weight=class_weights,
                                    verbose = -1,
                                    **best_params)

        clf_lgb_train.fit(X_train, y_train)

        filename = 'models/catboost_model_train.pkl'
        pickle.dump(clf_lgb_train, open(filename, 'wb'))
                    
        logger.info(f'Model saved in {filename}')

        metrics_report(clf_lgb_train, X_test, y_test, X)

        logger.info("End of the training of the model")
