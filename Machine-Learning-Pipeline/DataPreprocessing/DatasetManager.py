import numpy as np
import pandas as pd
import os
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import (DataFrame)

# Custom Imports
from .CustomDecorators import (timeit, resourceProfiler)

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

        # Get the available filenames
        self.availableDatasets: list = list(self.pathsConfig['Datasets'].keys())
        self.availableDatasets.remove('CHARTEVENTS') # SEE IF THE DATASET IS NECESSARY

        # Create a Dictionary to store a copy of each dataframe
        self.dataframes: dict = {dataset:None for dataset in self.availableDatasets}
    
    # @timeit
    @resourceProfiler
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
        
    def loadDataFrame(self, filename:str) -> DataFrame:
        """
        # Description
            -> This method aims to load a .csv file into memory using pyspark.
        ----------------------------------------------------------------------
        # Params:
        ---------
            - filename: str -> Name of the dataframe to load.
        ----------
        # Returns:
        ----------
            - A pyspark DataFrame.
        """

        # Assert if the filename is valid
        if filename not in self.availableDatasets:
            raise ValueError(f"Please select one of the available Datasets: {self.availableDatasets}.")

        # Check if the dataframe has already been loaded
        if self.dataframes[filename] is not None:
            return self.dataframes[filename]

        # Load the dataframe
        df: DataFrame = self.sparkSession.read.csv(self.pathsConfig['Datasets'][filename], sep=',', header=True, inferSchema=True).drop("ROW_ID")

        # Check if it was previously computed
        if not self.dataframes[filename]:
            # Update the dictionary
            self.dataframes.update({filename:df})

        # Return the dataframe
        return df
    
    def loadAllDataFrames(self) -> None:
        """
        # Description
            -> This method loads all the available datasets using pyspark.
        ------------------------------------------------------------------
        # Params:
        ---------
            - None
        ----------
        # Returns:
        ----------
            - None, since we are only loading data
        """

        # Iterate through all the available datasets
        for dataset in self.availableDatasets:
            # Load the current dataframe
            self.loadDataFrame(filename=dataset)