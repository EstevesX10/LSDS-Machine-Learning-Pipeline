from typing import (Tuple, List)
import dask.dataframe
import numpy as np
import pandas as pd
import dask
import os
from .MyDataFrame import (MyDataFrame)

class DaskDataframe(MyDataFrame):
    def __init__(self, config:dict, pathsConfig:dict) -> None:
        # Calling the super class constructor
        super().__init__(config, pathsConfig)
    
    @staticmethod
    def shape(dataframe:dask.dataframe) -> Tuple[int, int]:
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
        # Return the shape of the dask dataframe
        return (dataframe.shape[0].compute(), dataframe.shape[1])

    def load(self, filename: str) -> dask.dataframe:
        """
        Description:
            -> This method loads a CSV file using Dask.
        -----------------------------------------------
        Params:
            - filename: str -> The name of the dataset to load.
        -------------------------------------------------------
        Returns:
            - A DataFrame.
        """

        # Check if the filename is valid
        if filename not in self.availableDatasets:
            raise ValueError(f"Please select one of the available datasets: {self.availableDatasets}.")
        
        # Return the dataframe if it is already loaded
        if self.dataframes.get(filename) is not None:
            return self.dataframes[filename]
        
        # Load the CSV using Dask
        df = dask.dataframe.read_csv(self.pathsConfig['Datasets'][filename], sep=',', header=0, assume_missing=True, dtype={'ICD9_CODE': 'object'})
        
        # Remove the "ROW_ID" column if it exists
        if "ROW_ID" in df.columns:
            df = df.drop("ROW_ID", axis=1)
        
        # Update the dataframe on the current main dictionary
        self.dataframes[filename] = df

        # Return DataFrame
        return df
    
    def loadAllData(self) -> None:
        """
        Description:
            -> This method loads all available datasets using Dask.
        -----------------------------------------------------------
        Params:
            - None
        -----------------------------------------------------------
        Returns:
            - None (it only loads the data).
        """
        for dataset in self.availableDatasets:
            _ = self.load(filename=dataset)

    def join(self) -> dask.dataframe:
        """
        Description:
            -> This method joins all loaded DataFrames based on common columns using Dask.
        ----------------------------------------------------------------------------------
        Params:
            - None
        ------------
        Returns:
            - A Dask DataFrame with the merged information from the important tables.
        """

        # Check if all the data is available to perform the join / merge
        if any(value is None for value in self.dataframes.values()):
            raise ValueError("Not all DataFrames have been loaded!")
        
        # Retrieve DataFrames from the dictionary
        admissions_df = self.dataframes['ADMISSIONS']
        diagnosis_df = self.dataframes['DIAGNOSES_ICD']
        icuStays_df = self.dataframes['ICUSTAYS']
        patients_df = self.dataframes['PATIENTS']
        
        # Perform joins based on common columns (adjust join order and type as needed)
        df_joined = admissions_df.merge(diagnosis_df, on=["SUBJECT_ID", "HADM_ID"], how="outer")
        df_joined = df_joined.merge(icuStays_df, on=["SUBJECT_ID", "HADM_ID"], how="outer")
        df_joined = df_joined.merge(patients_df, on=["SUBJECT_ID"], how="outer")
        
        # Update the main dataframe inside this class
        self.df = df_joined

        # Return the joined dataframe
        return df_joined
    
    def getUselessColumns(self) -> List[str]:
        """
        # Description
            -> This method aims to fetch all the ID, time, date and
            flag related columns from the DataFrame to later remove them.
        -----------------------------------------------------------------
        # Params:
            - None
        
        # Returns:
            - A list with the names of the columns to remove.
        """

        # Check if the joined dataframe has already been computed
        if self.df is None:
            raise ValueError("The Joined DataFrame has yet to be computed!")

        # Define a list with the Keywords that exist on the columns' names
        keywords = ['ID', 'TIME', 'DOB', 'DOD', 'FLAG']

        # Fetch the columns that contain at least one of the keywords in their name
        columns = [col for col in self.df.columns if any(keyword in col for keyword in keywords)]

        # Return the fetched columns
        return columns

    def dropColumns(self, columns:list) -> None:
        """
        # Description
            -> This method focuses on droping a given set of columns in the dataframe.
        ------------------------------------------------------------------------------
        # Params:
            - columns: list -> List with the columns to drop
        
        # Returns:
            - None, since we are simply dropping rows on the dataframe.
        """

        # Check if the joined dataframe has already been computed
        if self.df is None:
            raise ValueError("The Joined DataFrame has yet to be computed!")

        # Drop the given columns
        self.df = self.df.drop(columns=columns)

    def dropNanRows(self, columns:list=None) -> None:
        """
        # Description
            -> This method aims to remove all the NaN enty rows in the main processed dataframe.
        ----------------------------------------------------------------------------------------
        # Params:
            - columns: list -> Possible subset of columns in which to consider the NaN values.
        
        # Returns:
            - None, since we are simply dropping rows with NaN values on the dataframe.
        """

        # Check if the joined dataframe has already been computed
        if self.df is None:
            raise ValueError("The Joined DataFrame has yet to be computed!")

        # Drop the NaN entries on the dataframe
        self.df = self.df.dropna(subset=columns, how='any')

    # ------------------------------------------------------------------------------------------------------------

    @staticmethod
    def getColumnMissingValues(df: dask.dataframe, column: str) -> dask.dataframe:
        """
        Description:
            -> This method returns the count of missing values for a given column in a Dask DataFrame.
        ---------------------------------------------------------
        Params:
            - df: Dask DataFrame -> The DataFrame to consider for the operation.
            - column: str -> The column name to evaluate.
        ---------------------------------------------------------
        Returns:
            - A Dask DataFrame containing the count of missing values for the specified column.
        """
        # Compute the count of missing (null) values in the column (use .compute() when needed)
        missing_count = df[column].isnull().sum().compute()

        # Create a small pandas DataFrame with the result and then convert it to a Dask DataFrame
        pdf = pd.DataFrame({f"[{column}] Missing Values": [missing_count]})
        return dask.dataframe.from_pandas(pdf, npartitions=1)

    @staticmethod
    def checkMissingValues(df: dask.dataframe) -> None:
        """
        Description:
            -> This method computes the count of missing values for each column in a Dask DataFrame
               and displays the percentage of missing values.
        --------------------------------------------------------
        Params:
            - df: Dask DataFrame -> The DataFrame to consider for the operation.
        --------------------------------------------------------
        Returns:
            - None (only displays the results).
        """
        # Get the total number of rows (this value is already known if the dataframe is loaded)
        totalCount = len(df)
        
        # Dictionary to hold missing value counts per column
        missing_dict = {}
        for c in df.columns:
            missing_dict[c] = df[c].isnull().sum().compute()
        
        # Create a pandas DataFrame for display
        pdf = pd.DataFrame(list(missing_dict.items()), columns=["Column", "MissingValuesCount"])
        pdf["MissingValuesPercentage(%)"] = round((pdf["MissingValuesCount"] / totalCount) * 100, 2)
        pdf = pdf.sort_values(by="MissingValuesPercentage(%)", ascending=False)
        print(pdf)

    @staticmethod
    def getColumnUniqueValues(df: dask.dataframe, column: str) -> dask.dataframe:
        """
        Description:
            -> This method returns the unique values and their counts for a specified column in a Dask DataFrame.
        ----------------------------------------------------
        Params:
            - df: Dask DataFrame -> The DataFrame to consider for the operation.
            - column: str -> The column name to evaluate.
        ----------------------------------------------------
        Returns:
            - A Dask DataFrame with the unique values and their corresponding counts (renamed as "Total Count").
        """
        # Use value_counts to obtain unique values and their counts
        unique_series = df[column].value_counts()

        # Convert the series into a DataFrame and reset the index
        uniqueValues = unique_series.rename("Total Count").to_frame().reset_index().rename(columns={"index": column})

        # Optionally, sort the DataFrame by the column values
        uniqueValues = uniqueValues.map_partitions(lambda pdf: pdf.sort_values(by=column))
        return uniqueValues

    @staticmethod
    def checkColumnsAmountUniqueValues(df: dask.dataframe) -> None:
        """
        Description:
            -> This method computes the number of unique values for each column in a Dask DataFrame.
        --------------------------------------------------------
        Params:
            - df: Dask DataFrame -> The DataFrame to consider for the operation.
        --------------------------------------------------------
        Returns:
            - None (only displays the results).
        """
        unique_counts = {}
        for c in df.columns:
            unique_counts[c] = df[c].nunique().compute()
        pdf = pd.DataFrame(list(unique_counts.items()), columns=["Column", "UniqueValuesCount"])
        print(pdf.set_index("Column"))