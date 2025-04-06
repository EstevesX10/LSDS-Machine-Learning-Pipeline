import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

class DatasetAnalyser:
    def __init__(self, df:pd.DataFrame) -> None:
        """
        # Description
            -> Constructor of the DatasetAnalyser Class which is responsible for
            analysing a given dataframe within a exploratory data analysis setting.
        ---------------------------------------------------------------------------
        # Params:
            - df: pd.DataFrame -> Dataframe to consider for the analysis.

        # Returns:
            - None, since we are simply creating a instance of a DatasetAnalyser class.
        """

        # Check if a dataframe was provided
        if df is None:
            raise ValueError('The dataframe was not provided.')

        # Save the dataframe
        self.df = df

    def computeMissingDataPercentages(self) -> pd.DataFrame:
        """
        # Description
            -> This method aims to create a dataframe showcasing the amount of missing
            data on each feature and the respective percentage within the dataset.
        ------------------------------------------------------------------------------
        # Params:
            - None
        
        # Returns:
            - Pandas DataFrame with the missing data analysis.
        """

        # Get the missing values on the dataframe
        missingData = self.df.isna().sum()

        # Compute the percentage of missing data
        missingPercentage = (missingData / len(self.df)) * 100

        # Create a dataframe to store all the information
        missing_df = pd.DataFrame({"Missing Values": missingData, "Percentage": missingPercentage})
        missing_df = missing_df.sort_values(by="Percentage", ascending=False)

        # Return the final dataframe
        return missing_df

    @staticmethod
    def pastelizeColor(c:tuple, weight:float=None) -> np.ndarray:
        """
        # Description
            -> Lightens the input color by mixing it with white, producing a pastel effect.
        -----------------------------------------------------------------------------------
        # Params:
            - c: tuple -> Original color.
            - weight: float -> Amount of white to mix (0 = full color, 1 = full white).
        
        # Returns:
            - New representation for the initial colors in a pastel spectrum.    
        """

        # Set a default weight
        weight = 0.5 if weight is None else weight

        # Initialize a array with the white color values to help create the pastel version of the given color
        white = np.array([1, 1, 1])

        # Returns a tuple with the values for the pastel version of the color provided
        return mcolors.to_rgba((np.array(mcolors.to_rgb(c)) * (1 - weight) + white * weight))

    def plotFeatureDistribution(self, feature:str=None, forceCategorical:bool=None, featureDecoder:dict=None) -> None:
        """
        # Description
            -> This function plots the distribution of a feature (column) in a dataset.
        -------------------------------------------------------------------------------
        # Params:
            - feature: str -> Feature of the dataset to plot.
            - forceCategorical: bool -> Forces a categorical analysis on a numerical feature.
            - featureDecoder: dict -> Dictionary with the conversion between the column value and its label [From Integer to String].
        
        # Returns:
            - None, since we are simply plotting the features distributions.
        """

        # Check if a feature was given
        if feature is None:
            raise ValueError('Missing a feature to Analyse.')

        # Check if the feature exists on the dataset
        if feature not in self.df.columns:
            raise ValueError(f"The feature '{feature}' is not present in the dataset.")

        # Set default value
        forceCategorical = False if forceCategorical is None else forceCategorical

        # Check the feature type
        if pd.api.types.is_numeric_dtype(self.df[feature]):
            # For numerical class-like features, we can treat them as categories
            if forceCategorical:
                # Create a figure
                plt.figure(figsize=(8, 5))

                # Get unique values and their counts
                valueCounts = self.df[feature].value_counts().sort_index()
                
                # Check if a feature Decoder was given and map the values if possible
                if featureDecoder is not None:
                    # Map the integer values to string labels
                    labels = valueCounts.index.map(lambda x: featureDecoder.get(x, x))
                    
                    # Tilt x-axis labels by 0 degrees and adjust the fontsize
                    plt.xticks(rotation=0, ha='center', fontsize=8)
                
                # Use numerical values as the class labels
                else:
                    labels = valueCounts.index

                # Create a color map from green to red
                cmap = plt.get_cmap('RdYlGn_r')
                colors = [self.pastelizeColor(cmap(i / (len(valueCounts) - 1))) for i in range(len(valueCounts))]

                # Plot the bars with gradient colors
                bars = plt.bar(labels.astype(str), valueCounts.values, color=colors, edgecolor='lightgrey', alpha=1.0, width=0.8, zorder=2)
                
                # Plot the grid behind the bars
                plt.grid(True, zorder=1)

                # Add text (value counts) to each bar at the center with a background color
                for i, bar in enumerate(bars):
                    yval = bar.get_height()
                    # Use a lighter color as the background for the text
                    lighterColor = self.pastelizeColor(colors[i], weight=0.2)
                    plt.text(bar.get_x() + bar.get_width() / 2,
                            yval / 2,
                            int(yval),
                            ha='center',
                            va='center',
                            fontsize=10,
                            color='black',
                            bbox=dict(facecolor=lighterColor, edgecolor='none', boxstyle='round,pad=0.3'))

                # Add title and labels
                plt.title(f'Distribution of {feature}')
                plt.xlabel(f'{feature} Labels', labelpad=20)
                plt.ylabel('Number of Samples')
                
                # Display the plot
                plt.show()
            
            # For numerical features, use a histogram
            else:
                # Create a figure
                plt.figure(figsize=(8, 5))

                # Plot the histogram with gradient colors
                plt.hist(self.df[feature], bins=30, color='#b0c4de', edgecolor='lightgrey', alpha=1.0, zorder=2)
                
                # Add title and labels
                plt.title(f'Distribution of {feature}')
                plt.xlabel(feature)
                plt.ylabel('Frequency')
                
                # Tilt x-axis labels by 0 degrees and adjust the fontsize
                plt.xticks(rotation=0, ha='center', fontsize=10)

                # Plot the grid behind the bars
                plt.grid(True, linestyle="--", alpha=0.7, zorder=1)
                
                # Display the plot
                plt.show()

        # For categorical features, use a bar plot
        elif pd.api.types.is_categorical_dtype(self.df[feature]) or self.df[feature].dtype == object:
                # Create a figure
                plt.figure(figsize=(8, 5))

                # Get unique values and their counts
                valueCounts = self.df[feature].value_counts().sort_index()
                
                # Create a color map
                cmap = plt.get_cmap('RdYlGn')
                colors = [self.pastelizeColor(cmap(i / (len(valueCounts) - 1))) for i in range(len(valueCounts))]

                # Plot the bars with gradient colors
                bars = plt.bar(valueCounts.index.astype(str), valueCounts.values, color=colors, edgecolor='lightgrey', alpha=1.0, width=0.8, zorder=2)
                
                # Plot the grid behind the bars
                plt.grid(True, zorder=1)

                # Add text (value counts) to each bar at the center with a background color
                for i, bar in enumerate(bars):
                    yval = bar.get_height()
                    # Use a lighter color as the background for the text
                    lighterColor = self.pastelizeColor(colors[i], weight=0.2)
                    plt.text(bar.get_x() + bar.get_width() / 2,
                            yval / 2,
                            int(yval),
                            ha='center',
                            va='center',
                            fontsize=10,
                            color='black',
                            bbox=dict(facecolor=lighterColor, edgecolor='none', boxstyle='round,pad=0.3'))

                # Add title and labels
                plt.title(f'Distribution of {feature}')
                plt.xlabel(f'{feature} Labels', labelpad=20)
                plt.ylabel('Number of Samples')
                
                # Tilt x-axis labels by 0 degrees and adjust the fontsize
                plt.xticks(rotation=0, ha='center', fontsize=8)

                # Display the plot
                plt.show()
        else:
            print(f"The feature '{feature}' is not supported for plotting.")

    def plotFeatureDistributions(self, features:list[str], forceCategorical:bool=False, featureDecoder:dict=None) -> None:
        """
        # Description
            -> Plots the distributions of multiple features in a grid layout (up to 4 subplots per row).
        ------------------------------------------------------------------------------------------------
        # Params:
            - features: list -> List with the features to analyse
            - forceCategorical: bool, optional -> Forces the treatment of numerical columns as categorcal ones (Columns in which was performed some sort of encoding).
            - featureDecoder: dict, optional -> Dictionary to convert the numerical values of forced categorical features into the previous values.
        
        # Returns:
            - None, since we are only plotting some data.
        """

        # Basic Validation
        if not features:
            raise ValueError("Failed to receive a list of features to analyse!")

        # Sort the features that exist in the DataFrame
        validFeatures = [f for f in features if f in self.df.columns]
        if not validFeatures:
            raise ValueError("None of the given features exist in the DataFrame!")

        # Get the number of valid features
        numberFeatures = len(validFeatures)

        # Define the layout for the grid-like plot
        cols = 4
        rows = (numberFeatures + cols - 1) // cols

        # Define a figure
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
        # fig.suptitle("Feature Distributions", fontsize=16, fontweight='bold', y=1.02)
        axes = axes.flatten()

        for idx, feature in enumerate(validFeatures):
            ax = axes[idx]

            # Checks if the current feature is categorical
            if forceCategorical or self.df[feature].dtype == object or pd.api.types.is_categorical_dtype(self.df[feature]):
                # Counts and sorts the feature frequencies
                valueCounts = self.df[feature].value_counts(dropna=False).sort_index()

                # Apply a possible decoder to the labels
                if featureDecoder is not None:
                    decodedIndex = [featureDecoder.get(v, v) for v in valueCounts.index]
                else:
                    decodedIndex = valueCounts.index.astype(str)

                cmap = plt.get_cmap("RdYlGn_r")
                numberValues = len(valueCounts)
                colors = [self.pastelizeColor(cmap(i / (numberValues - 1))) if numberValues > 1 else "lightblue" for i in range(numberValues)]

                bars = ax.bar(decodedIndex, valueCounts.values, color=colors, edgecolor="grey", alpha=1.0, width=0.8, zorder=2)
                ax.set_title(f"Distribution of {feature}")
                ax.set_xlabel(feature)
                ax.set_ylabel("Count")

                if numberValues > 5:
                    ax.tick_params(axis="x", rotation=45)
                else:
                    ax.tick_params(axis="x", rotation=0)
                ax.grid(True, axis="y", linestyle="--", alpha=0.7)

                # Insert the count of the label in the middle of the corresponding bar
                for i, bar in enumerate(bars):
                    yval = bar.get_height()
                    lighterColor = self.pastelizeColor(colors[i], weight=0.2)
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        yval / 2,
                        int(yval),
                        ha="center",
                        va="center",
                        fontsize=9,
                        bbox=dict(facecolor=lighterColor, edgecolor="none", boxstyle="round,pad=0.3"),
                    )
            else:
                # Numerical Features
                series_clean = self.df[feature].dropna()
                sns.histplot(series_clean, bins=30, kde=True, ax=ax, color='#b0c4de', edgecolor='grey', alpha=1.0)
                ax.lines[-1].set_color('#5072a7')
                ax.set_title(f"Distribution of {feature}")
                ax.set_xlabel(feature)
                ax.set_ylabel("Frequency")
                ax.grid(True, linestyle="--", alpha=0.7)

        # Remove subplots extras (se o número de features não for múltiplo de 4)
        for j in range(numberFeatures, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()