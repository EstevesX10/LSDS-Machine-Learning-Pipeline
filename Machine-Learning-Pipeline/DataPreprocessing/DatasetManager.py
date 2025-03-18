import numpy as np
import pandas as pd
import os
from pyspark.sql import SparkSession

# Custom Imports
from .CustomDecorators import (timeit)

class DatasetManager:
    def __init__(self, sparkSession: SparkSession, config:dict, pathsConfig:dict) -> None:
        """
        # Description
            -> Constructor Method of the DatasetManager which is responsible for preprocessing
            the data before training a regression model to predict the length of stay for a given patient.
        --------------------------------------------------------------------------------------------------
        # Params:
        ---------
            - sparkSession: SparkSession -> Apache Spark session.
            - config: dict -> Configuration Dictionary with important project settings.
            - pathsConfig: dict -> Configuration Dictionary with important paths used in the project.
        ----------
        # Returns:
        ----------
            - None, since we are only calling the constructor of the class
        """

        # Save the given parameters
        self.sparkSession: SparkSession = sparkSession
        self.config: dict = config
        self.pathsConfig: dict = pathsConfig

    @timeit
    def printConfig(self):
        # Testing the decorator
        # print(self.config)
        l = [i for i in range(100_000)]
        return np.max(l)

    @staticmethod
    def checkPath(path:str) -> bool:
        """
        # Description
            -> This static method allows to make sure that the nested
            directories of a given path exist and if not create them.
        -------------------------------------------------------------
        # Params:
        ---------
            - path : str -> A arbitrary path.
        ----------
        # Returns:
        ----------
            - A boolean value that returns if a file exists and if not creates the directory to accomudate it.
        """
        
        # Checks if the file exists
        if os.path.exists(path):
            return True
        else:
            # Make sure that the nested directories exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            return False