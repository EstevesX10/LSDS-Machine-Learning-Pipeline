from typing import (Tuple)
import numpy as np
import pandas as pd
import os

# Custom Imports
from .CustomDecorators import (timeit, resourceProfiler)

"""
# TO BE RESOLVED AND UPDATED THIS FILE
"""

class DatasetManager:
    def __init__(self, config:dict, pathsConfig:dict) -> None:
        """
        # Description
            -> Constructor Method of the DatasetManager which is responsible for preprocessing
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

    # @timeit
    @resourceProfiler
    def testingDecoratorsFunction(self):
        # Testing the decorator
        # print(self.config)
        l = [i for i in range(100_000)]
        return np.max(l)