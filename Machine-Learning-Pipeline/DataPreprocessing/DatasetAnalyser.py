import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from pyspark.sql import (DataFrame)
from pyspark.sql import functions as F
from pyspark.sql.types import (NumericType)

class DatasetAnalyser:
    def __init__(self, df:DataFrame) -> None:
        """
        # Description
            -> Constructor of the DatasetAnalyser Class, adapted to handle a PySpark DataFrame
               for exploratory data analysis.
        ---------------------------------------------------------------------------
        # Params:
        ---------
            - df: pyspark.sql.DataFrame -> Spark DataFrame to consider for the analysis.
        ----------
        # Returns:
        ----------
            - None, since we are simply creating an instance of a DatasetAnalyser class.
        """

        if df is None:
            raise ValueError("The Spark DataFrame was not provided.")

        self.df = df

    @staticmethod
    def pastelizeColor(c, weight=None) -> np.ndarray:
        """
        # Description
            -> Lightens the input color by mixing it with white, producing a pastel effect.
        -----------------------------------------------------------------------------------
        # Params:
        ---------
            - c: tuple/str -> Original color in Matplotlib-acceptable format.
            - weight: float -> Amount of white to mix (0 = full color, 1 = full white).
        ----------
        # Returns:
        ----------
            - Numpy array with the pastelized RGB(A) values.
        """
        if weight is None:
            weight = 0.5

        white = np.array([1, 1, 1])
        return mcolors.to_rgba(
            np.array(mcolors.to_rgb(c)) * (1 - weight) + white * weight
        )

    def _is_numeric(self, feature:str) -> bool:
        """
        # Description
            -> Helper method to check if a given column is numeric based on the Spark schema.
        -------------------------------------------------------------------------------------
        # Params:
        ---------
            - feature: str -> Feature of the dataset to consider.
        ----------
        # Returns:
        ----------
            - Boolean value that determines whether or not the feature given is numeric.
        """

        columnType = [f.dataType for f in self.df.schema.fields if f.name == feature]
        if not columnType:
            raise ValueError(f"The feature '{feature}' is not present in the dataset.")
        return isinstance(columnType[0], NumericType)

    def plotFeatureDistribution(self, feature:str=None, forceCategorical:bool=False, featureDecoder:dict=None):
        """
        # Description
            -> Plots the distribution of a feature (column) in the Spark DataFrame.
        -------------------------------------------------------------------------------
        # Params:
        ---------
            - feature: str -> Feature of the dataset to plot.
            - forceCategorical: bool -> Forces a categorical analysis on a numeric column.
            - featureDecoder: dict -> Dictionary mapping numeric values to string labels.
                                     Useful if the numeric column is actually an encoded category.
        ----------
        # Returns:
        ----------
            - None, since we are simply plotting the feature's distribution.
        """
        if feature is None:
            raise ValueError("Missing a feature to analyse.")

        if feature not in self.df.columns:
            raise ValueError(f"The feature '{feature}' is not present in the dataset.")

        # Decide if the column is numeric
        is_numeric = self._is_numeric(feature)

        # If numeric but forced to be categorical, or if not numeric => treat as categorical
        if (is_numeric and forceCategorical) or (not is_numeric):
            # We treat it as categorical and build a bar chart.
            # Group by feature, count, collect local data.
            counts = (
                self.df.groupBy(feature)
                .count()
                .orderBy(feature)  # optional sorting by the feature
                .collect()
            )
            # counts -> [Row(feature=<value>, count=some_count), ...]

            # Extract x values (category labels) and y values (counts)
            x_vals = []
            y_vals = []
            for row in counts:
                val = row[feature]
                # If there's a decoder, convert the numeric to a label
                if featureDecoder and val in featureDecoder:
                    x_vals.append(str(featureDecoder[val]))
                else:
                    x_vals.append(str(val))
                y_vals.append(row["count"])

            # Create a figure
            
            if len(x_vals) > 10:
                plt.figure(figsize=(20, 12))
                plt.xticks(rotation=45, ha='right', fontsize=10)
            elif len(x_vals) > 5 :
                plt.figure(figsize=(8, 5))
                plt.xticks(rotation=45, ha='right', fontsize=10)
            else:
                plt.figure(figsize=(8, 5))
                plt.xticks(rotation=0, ha='center', fontsize=10)
            
            # Prepare color mapping
            cmap = plt.get_cmap('RdYlGn_r')
            if len(x_vals) > 1:
                colors = [self.pastelizeColor(cmap(i / (len(x_vals) - 1))) for i in range(len(x_vals))]
            else:
                # For a single unique value, just pick a color
                colors = ["lightblue"]

            bars = plt.bar(x_vals, y_vals, color=colors, edgecolor='lightgrey', alpha=1.0, width=0.8, zorder=2)

            # Add text (value counts) to each bar
            for i, bar in enumerate(bars):
                yval = bar.get_height()
                lighterColor = self.pastelizeColor(colors[i], weight=0.2)
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    yval / 2,
                    int(yval),
                    ha='center',
                    va='center',
                    fontsize=10,
                    color='black',
                    bbox=dict(facecolor=lighterColor, edgecolor='none', boxstyle='round,pad=0.3')
                )

            # Plot the grid behind the bars
            plt.grid(True, zorder=1, linestyle="--", alpha=0.7)

            # Add title and labels
            plt.title(f'Distribution of {feature}')
            plt.xlabel(f'{feature} Labels')
            plt.ylabel('Number of Samples')

            plt.show()

        else:
            # Numeric feature: we can do a histogram.
            # Collect the column values locally (careful with big data!).
            values = (
                self.df.select(feature)
                .na.drop()
                .rdd.flatMap(lambda row: row)
                .collect()
            )

            plt.figure(figsize=(8, 5))
            plt.hist(values, bins=30, color='#b0c4de', edgecolor='lightgrey', alpha=1.0, zorder=2)

            # Add title and labels
            plt.title(f'Distribution of {feature}')
            plt.xlabel(feature)
            plt.ylabel('Frequency')

            # Grid behind the bars
            plt.grid(True, linestyle="--", alpha=0.7, zorder=1)

            plt.show()

    def plotFeatureDistributions(self, features:list[str], forceCategorical:bool=False, featureDecoder:dict=None):
        """
        # Description
            -> Plots the distributions of multiple features in a grid layout (up to 4 subplots per row),
               adapted to a PySpark DataFrame.
        ------------------------------------------------------------------------------------------------
        # Params:
        ---------
            - features: list -> List with the features to analyse.
            - forceCategorical: bool, optional -> Forces the treatment of numeric columns as categorical.
            - featureDecoder: dict, optional -> Dictionary mapping numeric values to string labels
                                                (useful if the numeric column is actually an encoded category).
        ----------
        # Returns:
        ----------
            - None, since we are only plotting data.
        """
        if not features:
            raise ValueError("Failed to receive a list of features to analyse!")

        # Filter only valid features that exist in the DataFrame
        validFeatures = [f for f in features if f in self.df.columns]
        if not validFeatures:
            raise ValueError("None of the given features exist in the Spark DataFrame!")

        # Layout for subplots
        cols = 3
        numberFeatures = len(validFeatures)
        rows = (numberFeatures + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
        axes = axes.flatten()

        for idx, feature in enumerate(validFeatures):
            ax = axes[idx]

            # Determine if it's numeric in Spark
            is_numeric = self._is_numeric(feature)

            # If numeric but forced categorical, or if it's not numeric => treat as categorical
            if (is_numeric and forceCategorical) or (not is_numeric):
                counts = (
                    self.df.groupBy(feature)
                    .count()
                    .orderBy(feature)
                    .collect()
                )

                x_vals = []
                y_vals = []
                for row in counts:
                    val = row[feature]
                    if featureDecoder and val in featureDecoder:
                        x_vals.append(str(featureDecoder[val]))
                    else:
                        x_vals.append(str(val))
                    y_vals.append(row["count"])

                cmap = plt.get_cmap("RdYlGn_r")
                if len(x_vals) > 1:
                    colors = [self.pastelizeColor(cmap(i / (len(x_vals) - 1))) for i in range(len(x_vals))]
                else:
                    # Single unique value
                    colors = ["lightblue"]

                bars = ax.bar(x_vals, y_vals, color=colors, edgecolor="grey", alpha=1.0, width=0.8, zorder=2)
                ax.set_title(f"Distribution of {feature}")
                ax.set_xlabel(feature)
                ax.set_ylabel("Count")

                if len(x_vals) > 5:
                    ax.tick_params(axis="x", rotation=45)
                else:
                    ax.tick_params(axis="x", rotation=0)
                ax.grid(True, axis="y", linestyle="--", alpha=0.7)

                # Insert the count in the middle of each bar
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
                # Numeric histogram
                values = (
                    self.df.select(feature)
                    .na.drop()
                    .rdd.flatMap(lambda row: row)
                    .collect()
                )
                sns.histplot(values, bins=30, kde=True, ax=ax, color='#b0c4de', edgecolor='grey', alpha=1.0)
                
                # If a kernel density line was plotted, recolor it:
                if ax.lines:
                    ax.lines[-1].set_color('#5072a7')

                ax.set_title(f"Distribution of {feature}")
                ax.set_xlabel(feature)
                ax.set_ylabel("Frequency")
                ax.grid(True, linestyle="--", alpha=0.7)

        # Remove any unused subplots
        for j in range(numberFeatures, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()
