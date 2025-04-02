from typing import (Tuple)
from pandas import DataFrame
import dask.dataframe as dd

class MyDataFrame:
    def __init__(self, config:dict, pathsConfig:dict):
        """
        # Description
            -> Constructor Method of the MyDataFrame which is responsible for preprocessing
            the data before training a regression model to predict the length of stay for a given patient.
        --------------------------------------------------------------------------------------------------
        # Params:
        ---------
            - config: dict -> Configuration Dictionary with important project settings.
            - pathsConfig: dict -> Configuration Dictionary with important paths used in the project.
        ----------
        # Returns:
        ----------
            - None, since we are only calling the constructor of the class
        """
        # Save the given parameters
        self.config: dict = config
        self.pathsConfig: dict = pathsConfig

        # Get the available filenames
        self.availableDatasets: list = list(self.pathsConfig['Datasets'].keys())
        self.availableDatasets.remove('CHARTEVENTS') # SEE IF THE DATASET IS NECESSARY

        # Create a Dictionary to store a copy of each dataframe
        self.dataframes: dict = {dataset:None for dataset in self.availableDatasets}
    
        # Save a variable for the main dataframe to use
        self.df = None

    @staticmethod
    def shape(dataframe:DataFrame) -> Tuple[int, int]:
        """
        # Description
            -> This method calculates the number of rows and columns on a DataFrame.
        ----------------------------------------------------------------------------
        # Params:
        ---------
            - dataframe: DataFrame (From a given framework) -> DataFrame in which to compute the rows and columns of.
        ----------
        # Returns:
        ----------
            - A tuple with the amount of rows and columns in the DataFrame.
        """
        # Return the number of rows and columns on the dataframe
        raise ValueError("[NOT IMPLEMENTED!]")

    def load(self, filename: str) -> dd.DataFrame:
        """
        Description:
            -> This method loads a CSV file.
        ------------------------------------
        Params:
            - filename: str -> The name of the dataset to load.
        ----------------------------------------------------------------------
        Returns:
            - A DataFrame.
        """
        # TO BE IMPLEMENTED
        raise ValueError("[NOT IMPLEMENTED!]")

    def loadAllData(self) -> None:
        """
        Description:
            -> This method loads all available datasets.
        ------------------------------------------------
        Params:
            - None
        ------------------------------------------------
        Returns:
            - None (it only loads the data).
        """
        # TO BE IMPLEMENTED
        raise ValueError("[NOT IMPLEMENTED!]")

    def join(self) -> dd.DataFrame:
        """
        Description:
            -> This method joins all loaded DataFrames based on common columns.
        -----------------------------------------------------------------------
        Params:
            - None
        --------------------------------------------------------
        Returns:
            - A Dask DataFrame with the merged information from the important tables.
        """
        # TO BE IMPLEMENTED
        raise ValueError("[NOT IMPLEMENTED!]")