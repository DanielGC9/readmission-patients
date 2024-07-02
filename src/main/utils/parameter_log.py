"""
Config module for the main module.
"""

import os
import logging
import logging.config

def loggin_custom():
    """
    It reads the logging configuration file and returns a logger object
    Returns:
      The logger object is being returned.
    """
    config_dir = os.path.abspath("config")

    logging.config.fileConfig(
        fname=os.path.join(config_dir, "logging.conf"), disable_existing_loggers=False
    )
    logger = logging.getLogger("develop")

    return logger