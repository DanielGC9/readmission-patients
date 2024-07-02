""" 
This module contains utility functions
"""

import os
import sys
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay,
                             accuracy_score,
                             classification_report,
                             roc_auc_score
                             )
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append('.')
from src.main.utils.parameter_log import loggin_custom

logger = loggin_custom()

def results_to_csv(dataframe, name, path):
    """
    The function `results_to_csv` saves a DataFrame to a CSV file with a specified name and path.
    
    Args:
      dataframe: A pandas DataFrame containing the data that you want to write to a CSV file.
      name: The `name` parameter in the `results_to_csv` function is a string that represents the name
    of the CSV file that will be created. It is used to generate the file name by combining it with the
    provided path.
      path: The `path` parameter in the `results_to_csv` function represents the directory path where
    the CSV file will be saved. It is the location on your computer where you want to store the CSV file
    containing the results from the dataframe.
    """
    file_path = os.path.join(path, str(name) + ".csv")
    dataframe.to_csv(file_path, index=False, encoding="utf-8")

def check_column_type(df):
    """
    The function `check_column_type` takes a DataFrame as input and returns a DataFrame listing the
    unique data types present in each column.
    
    Args:
      df: It seems like you were about to provide the DataFrame `df` as input to the `check_column_type`
    function. If you provide me with the DataFrame `df`, I can help you check the types of data in each
    column. Please go ahead and provide the DataFrame so that we can proceed with
    
    Returns:
      The function `check_column_type` returns a DataFrame that contains information about the data
    types present in each column of the input DataFrame `df`. The DataFrame has two columns: 'column'
    which contains the column names, and 'types' which contains a list of unique data types present in
    each column.
    """
    df_list = []
    for col in df.columns:
        types = df[col].apply(type).unique()
        df_list.append([col, types])
    fin = pd.DataFrame(df_list, columns=['column', 'types'])

    return fin


def equal_columns(df):
    """
    The function `equal_columns` checks for and logs any columns in a DataFrame that have equal values.
    
    Args:
      df: It seems like you have provided the code snippet for a function called `equal_columns` that is
    intended to find and log columns in a DataFrame that have equal values. However, the DataFrame `df`
    that the function should operate on is missing from the context you provided.
    
    Returns:
      The function `equal_columns` returns a list of column names that have equal values in the input
    DataFrame `df`.
    """
    equal_cols = []
    cols = df.columns
    for i in range(len(cols)-1):
        for j in range(i+1,len(cols)):
            col1 = cols[i]
            col2 = cols[j]
            if df[col1].equals(df[col2]):
                equal_cols.append(col1)
                equal_cols.append(col2)
                logger.info(f'the columns {col1} and {col2} are equal')
    return equal_cols

# check for null values and unique values
def check_unique_nulls(df_diabetes):
    """
    The function `check_unique_nulls` takes a DataFrame as input, calculates the number of unique values
    and null values for each column, and returns a DataFrame with this information along with the
    percentage of null values.
    
    Args:
      df_diabetes: A DataFrame containing data related to diabetes. The function `check_unique_nulls`
    takes this DataFrame as input and returns a summary DataFrame with information about the uniqueness
    and null values in each column of the input DataFrame.
    
    Returns:
      The function `check_unique_nulls` returns a DataFrame containing information about each column in
    the input DataFrame `df_diabetes`. The DataFrame includes columns for the column name, number of
    unique values in the column, number of null values in the column, and the percentage of null values
    in the column relative to the total number of rows in the DataFrame.
    """
    columns = []
    unique = []
    nulls = []
    total = len(df_diabetes)

    for col in df_diabetes.columns:
        columns.append(col)
        unique.append(df_diabetes[col].nunique())
        nulls.append(df_diabetes[col].isnull().sum())

    df_nulls = pd.DataFrame({'column':columns, 'unique':unique, 'nulls':nulls}).sort_values(by='unique')
    df_nulls['nullPercentage'] = round(df_nulls['nulls']/total*100, 0)
    return df_nulls


def map_diagnosis(df, column):
    """
    The function `map_diagnosis` categorizes numeric diagnoses in a DataFrame column based on specified
    ranges and values.
    
    Args:
      df: The `df` parameter in the `map_diagnosis` function is expected to be a pandas DataFrame
    containing the data on which the diagnosis mapping will be performed. This DataFrame should have a
    column specified by the `column` parameter, which contains the diagnosis information that needs to
    be mapped to categories based on
      column: The `map_diagnosis` function you provided seems to be mapping diagnoses to specific
    categories based on ranges and values. However, the `column` parameter is missing. Could you please
    provide the name of the column in the DataFrame `df` that contains the diagnoses you want to map?
    """
    diagnosis = pd.to_numeric(df[column], errors='coerce')
    numeric_diagnosis = diagnosis[diagnosis.apply(type) == float]

    # Mapping of diagnoses based on the ranges and values ​​provided by the paper

    conditions = [
        numeric_diagnosis.between(390, 459) | (numeric_diagnosis == 785),
        numeric_diagnosis.between(460, 519) | (numeric_diagnosis == 786),
        numeric_diagnosis.between(520, 579) | (numeric_diagnosis == 787),
        numeric_diagnosis.between(250.00, 250.99),
        numeric_diagnosis.between(800, 999),
        numeric_diagnosis.between(710, 739),
        numeric_diagnosis.between(580, 629) | (numeric_diagnosis == 788),
        numeric_diagnosis.between(140, 239)
    ]

    # Categories
    categories = [
        'Circulatory',
        'Respiratory',
        'Digestive',
        'Diabetes',
        'Injury',
        'Musculoskeletal',
        'Genitourinary',
        'Neoplasms'
    ]

    new_col_name = f'categories_{column}'
    df[new_col_name] = 'Other'

    for condition, category in zip(conditions, categories):
        df.loc[df.index.isin(numeric_diagnosis[condition].index), new_col_name] = category

def simplification_variables(df, column):
    """
    The function `simplification_variables` takes a DataFrame and a column name as input, and simplifies
    the values in that column based on predefined mappings for specific columns like 'change', 'gender',
    'diabetesMed', 'age', and certain medications.
    
    Args:
      df: The `df` parameter in the `simplification_variables` function is expected to be a pandas
    DataFrame containing the dataset with various columns including the one specified by the `column`
    parameter. The function aims to simplify and transform specific columns within the DataFrame based
    on the logic provided in the function.
      column: The `column` parameter in the `simplification_variables` function represents the column
    name in the DataFrame `df` that you want to simplify or transform based on certain mappings. The
    function checks the value of `column` and applies specific mappings to that column to simplify the
    data. It handles different
    """
    medications = [
        'glyburide', 'glyburide-metformin',
        'rosiglitazone', 'glipizide', 'insulin', 'glimepiride',
        'acarbose', 'repaglinide', 'nateglinide',
        'pioglitazone', 'tolbutamide', 'metformin']

    if column == 'change':
        df['change'] = df['change'].map({'Ch': 1, 'No': 0})
    elif column == 'gender':
        df['gender'] = df['gender'].map({'Female': 0, 'Male': 1})
    elif column == 'diabetesMed':
        df['diabetesMed'] = df['diabetesMed'].map({'Yes': 1, 'No': 0})
    elif column == 'age':
        df['age'] = df['age'].map(
            {'[0-10)': 0, '[10-20)': 1, '[20-30)': 2, '[30-40)': 3,
             '[40-50)': 4, '[50-60)': 5, '[60-70)': 6, '[70-80)': 7,
             '[80-90)': 8, '[90-100)': 9})
    elif column in medications:
        df[column] = df[column].map({'No': 0, 'Steady': 1, 'Up': 1, 'Down': 1})
    else:
        pass


def count_unique_diagnoses(row):
    """
    The function `count_unique_diagnoses` takes a row of data and returns the count of unique diagnoses
    from three different categories.
    
    Args:
      row: A row from a dataset containing information about medical diagnoses. The row likely includes
    three columns: 'categories_diag_1', 'categories_diag_2', and 'categories_diag_3', each representing
    a different medical diagnosis category.
    
    Returns:
      The function `count_unique_diagnoses` returns the count of unique diagnoses from the input row. It
    extracts the diagnoses from three specific columns in the row, removes duplicates using a set, and
    then returns the count of unique diagnoses.
    """
    diagnoses = [row['categories_diag_1'], row['categories_diag_2'], row['categories_diag_3']]
    return len(set(diagnoses))


def count_medications(row):
    """
    The function `count_medications` calculates the total number of medications prescribed based on the
    input row data.
    
    Args:
      row: The `count_medications` function takes a dictionary `row` as input, which presumably contains
    information about different types of medications and their quantities. The function calculates the
    total number of medications by summing up the quantities of specific medications listed in the
    dictionary.
    
    Returns:
      The function `count_medications` returns the total count of medications based on the values in the
    input `row` dictionary for the specified medication keys.
    """
    medications = sum([row['glyburide'], row['glyburide-metformin'],
                       row['rosiglitazone'], row['glipizide'], row['insulin'],
                       row['glimepiride'], row['acarbose'], row['repaglinide'],
                       row['nateglinide'], row['pioglitazone'], row['metformin']])
    return medications


def graph_normalize_column(df, column, target_column):
    """
    The function `graph_normalize_column` normalizes the values in a specified column of a DataFrame
    based on another target column, and then visualizes the proportions using a bar plot.
    
    Args:
      df: The `df` parameter in the `graph_normalize_column` function is a pandas DataFrame that
    contains the data you want to visualize. It is the main input to the function and should include the
    columns specified in the `column` and `target_column` parameters.
      column: The `column` parameter in the `graph_normalize_column` function refers to the column in
    the DataFrame `df` that you want to group by for normalization. This column will be used as the
    x-axis in the resulting plot.
      target_column: The `target_column` parameter in the `graph_normalize_column` function refers to
    the column in the DataFrame `df` that contains the target values for which you want to normalize the
    proportions. This column will be used to calculate the proportions of each category within the
    specified `column`.
    """

    df = df.groupby(column)[target_column].value_counts().reset_index(drop=False)
    df_1 = df[df[target_column] == 1]
    df_2 = df[df[target_column] == 0]
    df_1['proportion'] = df_1['count'] / df_1['count'].sum()
    df_2['proportion'] = df_2['count'] / df_2['count'].sum()
    df = pd.concat([df_1, df_2], axis=0)


    plt.figure(figsize=(8, 4))
    ax = sns.barplot(data=df, x=column, y='proportion', hue=target_column, palette='magma')
    #ax = sns.lineplot(data=df, x='number_inpatient', y='proportion', hue='readmitted', palette='magma')

    plt.title(f'{column} VS. {target_column} (Normalized)', fontsize=16)
    plt.xlabel(column, fontsize=14)
    plt.ylabel('Proportion', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height:.2%}', (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='center', fontsize=12, color=p.get_facecolor(),
                    xytext=(0, 30),
                    textcoords='offset points',
                    rotation=90)
    _, y_max = ax.get_ylim()
    plt.legend(title=target_column, title_fontsize='13', fontsize='12')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.ylim(0, y_max*1.2)
    plt.savefig(f'data/images/{column}_VS_{target_column}_normalized.png')


def describe(df, columns):
    """
    The function `describe` takes a DataFrame and a list of columns, generates descriptive statistics
    and visualizations for each column, and saves the plots as images.
    
    Args:
      df: It looks like you were about to provide some information about the `df` parameter in the
    `describe` function, but the information is missing. Could you please provide the details about the
    `df` parameter so that I can assist you further with the code snippet you shared?
      columns: The `columns` parameter in the `describe` function is a list of column names that you
    want to describe and visualize in the DataFrame `df`.
    """

    for col in columns:
        try:
            logger.info(f'{col}:\n {pd.DataFrame(df[col].describe())}')
            f, ax = plt.subplots(1, 2, figsize=(8,4))

            ax[0] = sns.distplot(df[col], bins=10, color='blue',ax=ax[0])
            ax[0].set_title(f"Distribution of {col}")

            ax[1] = sns.boxplot(df[col])
            ax[1].set_title(f'Visualize outliers in {col} variable')

            plt.savefig(f'data/images/{col}.png')

            logger.info(f'{col}.png saved in data/images')

        except:
            logger.info(f'{col} not valid')


def metrics_report(model, X_test, y_test, X):
    """
    The function `metrics_report` generates various evaluation metrics and visualizations for a machine
    learning model's performance.
    
    Args:
      model: The `model` parameter is typically a trained machine learning model that you want to
    evaluate using the provided metrics. It could be a classifier like Random Forest, Logistic
    Regression, or any other model that has a `predict` method for making predictions.
      X_test: X_test is the feature matrix representing the independent variables of your test dataset.
    It is used to evaluate the performance of the machine learning model by making predictions on this
    data and comparing them with the actual target values (y_test).
      y_test: `y_test` is the actual target values from the test dataset. It is used to evaluate the
    performance of the model by comparing the predicted values with the actual values during testing.
      X: X is the input features used for training the model. It is a DataFrame containing the
    independent variables used to make predictions.
    """
    
    predicted_y = model.predict(X_test)
    accuracy = accuracy_score(y_test, predicted_y)
    logger.info("Accuracy test: %.2f%%" % (accuracy * 100.0))

    logger.info('\n Classification report \n', classification_report(y_test, predicted_y))

    logger.info('\n ROC AUC score: %.3f \n' % roc_auc_score(y_test, predicted_y))

    logger.info('\n Confusion matrix \n')
    
    disp = ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        cmap=plt.cm.Blues,
        display_labels=['not readmitted', 'readmitted'],
        normalize='true',
    )
    disp.ax_.set_title('Confusion matrix')
    plt.savefig('data/metrics/confusion_matrix.png')
    logger.info('\n Confusion matrix saved in data/metrics\n')

    plt.figure(figsize=(8, 4))
    # Feature importance
    feature_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values()
    feature_imp.plot(kind='barh', figsize=(8, 10),
                     title='Feature Importances', color='#087E8B')
    plt.savefig('data/metrics/feature_importance.png')
    logger.info('\n Feature importance saved in data/metrics\n')
