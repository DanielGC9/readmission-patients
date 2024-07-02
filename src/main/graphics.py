""" 
Module for graphics
"""

import sys
import warnings
sys.path.append('.')
from src.main.utils.parameter_log import loggin_custom
from src.main.utils.utils_functions import (graph_normalize_column,
                                            describe)

warnings.filterwarnings('ignore')
logger = loggin_custom()

def graphics(df, columns):
    """
    The function `graphics` generates graphics to compare columns to a target variable and describe the
    columns in a DataFrame.
    
    Args:
      df: The `df` parameter is typically used to represent a DataFrame in Python, which is a
    two-dimensional, size-mutable, potentially heterogeneous tabular data structure with labeled axes
    (rows and columns). It is commonly used in libraries such as Pandas for data manipulation and
    analysis.
      columns: The `columns` parameter in the `graphics` function is a list of column names from the
    DataFrame `df` that you want to create graphics for. These columns will be used to compare against
    the target variable 'readmitted' and to describe the variables in the DataFrame.
    """
    
    logger.info("Starting graphics")

    logger.info("Creating graphics to compare to target variable")
    for col in columns:
        try:
            graph_normalize_column(df, col, 'readmitted')

        except: 
            logger.info(f'{col} not valid')

    logger.info("Creating graphics to describe the variable")
    describe(df, columns)

    logger.info("Graphics created")
