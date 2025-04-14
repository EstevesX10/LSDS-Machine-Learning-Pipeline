import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import seaborn as sns

class DatasetAnalyser:
    def __init__(self, df:pd.DataFrame) -> None:
        """
        # Description
            -> Constructor of the DataAnalyser Class which is responsible for
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
                # Get unique values and their counts
                valueCounts = self.df[feature].value_counts().sort_index()
                
                if len(valueCounts) > 20:
                    # Create a figure
                    plt.figure(figsize=(15, 9))
                    plt.tick_params(axis="x", rotation=90)
                elif len(valueCounts) > 5:
                    # Create a figure
                    plt.figure(figsize=(8, 5))
                    plt.tick_params(axis="x", rotation=90)
                else:
                    # Create a figure
                    plt.figure(figsize=(8, 5))
                    plt.tick_params(axis="x", rotation=0)
                plt.grid(True, axis="y", linestyle="--", alpha=0.7)

                # Create a color map
                cmap = plt.get_cmap('RdYlGn')
                colors = [self.pastelizeColor(cmap(i / (len(valueCounts) - 1))) for i in range(len(valueCounts))]

                # Plot the bars with gradient colors
                bars = plt.bar(valueCounts.index.astype(str), valueCounts.values, color=colors, edgecolor='lightgrey', alpha=1.0, width=0.8, zorder=2)

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
                # plt.xticks(rotation=0, ha='center', fontsize=8)

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
        cols = 2
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

                if numberValues > 20:
                    ax.tick_params(axis="x", rotation=90)
                elif numberValues > 8:
                    ax.tick_params(axis="x", rotation=90)
                elif numberValues > 5:
                    ax.tick_params(axis="x", rotation=65)
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

    def plotXyCharts(self, x:str, y:list[str], title:str, chartType:str='scatter') -> None:
        """
        # Description
            -> Plot multiple Y variables against a single X variable in a grid layout 
        with a maximum of 3 columns (or fewer if less than 3 Y variables are provided).
            In the bar chart (non-numeric) case, the colors for the bars are generated to 
        follow a fade effect using the 'RdYlGn' colormap, then pastelized. 
            Each bar gets a text label displaying its value, which is formatted so that if
        the number is integer-like it is shown as an integer and if it is decimal, 
        at most 2 decimal places are shown.
        ----------------------------------------------------------------------------------
        # Parameters:
            - x: str -> The feature to use for the X axis.
            - y: list[str] -> A list of features to plot on the Y axis.
            - title: str -> Title for the plot.
            - chartType: str, optional -> The type of plot to create ('scatter' or 'line') when both x and y are numeric. Default is 'scatter'.
        
        # Returns:
            - None. Displays the plots.
        """
        # Validate that the x-axis column exists.
        if x not in self.df.columns:
            raise ValueError(f"X axis feature '{x}' is not present in the DataFrame.")
        
        # Validate that each y-axis column exists.
        for y_ in y:
            if y_ not in self.df.columns:
                raise ValueError(f"Y axis feature '{y_}' is not present in the DataFrame.")
        
        # Determine layout: max columns is 3 (or less if fewer y variables).
        yTotalFeatures = len(y)
        ncols = min(3, yTotalFeatures)
        nrows = (yTotalFeatures + ncols - 1) // ncols  # Ceiling division for rows
        
        # Create figure and axes.
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
        if yTotalFeatures == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # Helper function to format numbers.
        def format_number(val):
            # If the value is integer-like, display as int; otherwise show 2 decimals.
            if abs(val - round(val)) < 1e-8:
                return f"{int(val)}"
            else:
                return f"{val:.2f}"

        # Loop over each y variable.
        for i, y_ in enumerate(y):
            ax = axes[i]
            
            # If both x and y are numeric, use scatter or line plot with a uniform pastel color.
            if pd.api.types.is_numeric_dtype(self.df[x]) and pd.api.types.is_numeric_dtype(self.df[y_]):
                pastelColor = self.pastelizeColor('skyblue', weight=0.5)
                if chartType == 'line':
                    ax.plot(self.df[x], self.df[y_], color=pastelColor, marker='o', linestyle='-', zorder=2)
                else:  # default to scatter plot.
                    ax.scatter(self.df[x], self.df[y_], alpha=0.7, color=pastelColor, zorder=2)
            else:
                # For non-numeric cases (or categorical grouping), create a bar plot.
                # Group by x and compute mean of y.
                grouped = self.df.groupby(x)[y_].mean().reset_index()
                # Number of distinct x categories
                n = len(grouped)
                cmap = plt.get_cmap("RdYlGn")
                # For a single bar, choose the middle of the colormap; else, generate a list of colors.
                if n > 1:
                    colors = [self.pastelizeColor(cmap(i / (n - 1))) for i in range(n)]
                else:
                    colors = [self.pastelizeColor(cmap(0.5))]
                
                bars = ax.bar(
                    grouped[x].astype(str), grouped[y_],
                    color=colors, edgecolor='grey', alpha=1.0, zorder=2
                )
                # Add text labels to each bar.
                for j, bar in enumerate(bars):
                    yval = bar.get_height()
                    # Use a slightly lighter version of the bar's color for the text background.
                    lighterColor = self.pastelizeColor(colors[j], weight=0.2)
                    labelText = format_number(yval)
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        yval / 2,
                        labelText,
                        ha='center',
                        va='center',
                        fontsize=10,
                        color='black',
                        bbox=dict(facecolor=lighterColor, edgecolor='none', boxstyle='round,pad=0.3'),
                        zorder=3
                    )
                # Adjust the x-axis tick labels.
                ax.set_xticklabels(grouped[x].astype(str), rotation=45, ha='right', fontsize=9)
            
            ax.set_xlabel(x)
            ax.set_ylabel(y_)
            ax.set_title(f'{y_} vs {x}')
            # Draw grid lines behind other elements.
            ax.grid(True, linestyle='--', alpha=0.7, zorder=1)
        
        # Remove any unused subplots.
        for j in range(yTotalFeatures, len(axes)):
            fig.delaxes(axes[j])
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    def plotPieChart(self, numericColumn:str, labelColumn:str, weight:float=0.5) -> None:
        """
        # Description
            -> Plots a pie chart where each wedge corresponds to the sum of a numeric column
        grouped by a specified label column. Each wedge is labeled with the (bold) 
        category label and the real numeric sum (no percentages).
            A pastel style is applied using the 'RdYlGn' colormap. Each label is displayed
        inside a rectangle with a lighter pastel background for readability.
        -------------------------------------------------------------------------------------
        # Parameters:
            - numericColumn : str -> The name of the numeric column in the DataFrame to sum.
            - labelColumn : str -> The column by which to group. Each distinct value in this column corresponds to one wedge in the pie chart.
            - weight : float, optional -> The weight used when converting the wedge color to a pastel color (0 = full color, 1 = full white). Defaults to 0.5.
        
        # Returns:
            - None
        """
        # 1. Validate columns.
        if numericColumn not in self.df.columns:
            raise ValueError(f"Numeric column '{numericColumn}' not found in the DataFrame.")
        if labelColumn not in self.df.columns:
            raise ValueError(f"Label column '{labelColumn}' not found in the DataFrame.")
        
        # 2. Group by labelColumn, summing the numericColumn.
        groupedSums = self.df.groupby(labelColumn)[numericColumn].sum()
        if groupedSums.empty:
            raise ValueError("No data available after grouping. Check your DataFrame or columns.")

        # 3. Prepare labels (the group names) and values (the sums).
        labels = groupedSums.index.astype(str).tolist()
        values = groupedSums.values
        
        n = len(values)
        
        # Generate colors from the 'RdYlGn' colormap and pastelize them.
        cmap = plt.get_cmap("RdYlGn")
        if n > 1:
            colors = [self.pastelizeColor(cmap(i / (n - 1)), weight=weight) for i in range(n)]
        else:
            # Only one wedge; pick the midpoint of colormap.
            colors = [self.pastelizeColor(cmap(0.5), weight=weight)]
        
        # Helper function for formatting numeric values.
        def format_number(val):
            # Display integers without decimals; otherwise, 2 decimals.
            if abs(val - round(val)) < 1e-8:
                return str(int(round(val)))
            else:
                return f"{val:.3f}"
        
        # Create a custom autopct function that puts the bold label and numeric sum in each wedge.
        def makeAutopct(labelsList, sumsList):
            def myAutopct(_pct):
                idx = myAutopct.index
                myAutopct.index += 1
                
                # Bold label (using MathText for a simple approach).
                bold_label = rf'$\bf{{{labelsList[idx]}}}$'
                
                # Real numeric sum (no percentages).
                val_str = format_number(sumsList[idx])
                
                # Display label on one line, numeric value on the next.
                return f"{bold_label}\n{val_str}"
            myAutopct.index = 0
            return myAutopct
        
        # Plot the pie chart, adjusting pctdistance so text sits closer to the center.
        fig, ax = plt.subplots(figsize=(6, 6))
        wedges, _, autotexts = ax.pie(
            values,
            labels=None,  # We'll place both label & sum inside the wedge using autopct.
            colors=colors,
            autopct=makeAutopct(labels, values),
            startangle=90,
            # pctdistance controls where inside the wedge the text is placed (lower means closer to center).
            pctdistance=0.6,
            textprops=dict(color="black", fontsize=9, ha='center', va='center'),
            wedgeprops=dict(edgecolor="white", linewidth=1.5)
        )
        
        # Add a pastel background behind each label, with reduced padding.
        for wedge, autotext in zip(wedges, autotexts):
            wedgeColor = wedge.get_facecolor()  # RGBA tuple.
            labelBackgroundColor = self.pastelizeColor(wedgeColor, weight=0.2)
            autotext.set_bbox(dict(
                facecolor=labelBackgroundColor,
                edgecolor='none',
                boxstyle='round,pad=0.2',
                alpha=0.8,
                clip_on=False  # Allows text box to slightly extend beyond wedge if needed.
            ))
        
        ax.axis('equal')  # Draw pie as a circle.
        ax.set_title(
            f"Distribution of {numericColumn} by {labelColumn}",
            fontsize=12,
            fontweight='bold',
        )
        
        plt.tight_layout()
        plt.show()

    def plotBoxPlot(self, x:str, y:str, hue:str, makeSmallPlot:bool=False) -> None:
        """
        # Description
            -> Creates a box plot using the specified features for the x-axis,
            y-axis, and a grouping variable (hue). Uses a custom pastel palette
            generated from the "RdYlGn" colormap for the hue groups.
        -----------------------------------------------------------------------
        # Parameters:
        - x: str -> The feature to use for the X axis.
        - y: str -> The feature to use for the Y axis.
        - hue: str -> The feature to use for grouping the data.
        - makeSmallPlot: bool -> Determines whether or not to use a bigger figure for the plot.
        
        # Returns:
            - None. Displays the plots.
        """

        # Making sure the given features are within the dataframe
        if x not in self.df.columns:
            raise ValueError(f"Invalid {x} feature given! Please make sure to use one of the following {list(self.df.columns)}")
        if y not in self.df.columns:
            raise ValueError(f"Invalid {y} feature given! Please make sure to use one of the following {list(self.df.columns)}")
        if hue not in self.df.columns:
            raise ValueError(f"Invalid {hue} feature given! Please make sure to use one of the following {list(self.df.columns)}")

        # Set a value for the figsize
        if makeSmallPlot:
            figuresize = (6, 5)
        else:
            figuresize = (10, 6)
        
        # Set the seaborn style.
        sns.set(style="whitegrid")
        plt.figure(figsize=figuresize)
        
        # Generate a custom pastel palette for the hue groups.
        # Get the unique hue values (drop nans and sort for consistent ordering).
        unique_vals = sorted(self.df[hue].dropna().unique())
        n = len(unique_vals)
        cmap = plt.get_cmap("RdYlGn")
        palette = {}
        for i, val in enumerate(unique_vals):
            # Use the colormap to generate a base color.
            base_color = cmap(i / (n - 1)) if n > 1 else cmap(0.5)
            # Pastelize the base color.
            pastel_color = self.pastelizeColor(base_color, weight=0.5)
            palette[val] = pastel_color
        
        # Create the box plot using the custom palette.
        ax = sns.boxplot(x=x, y=y, hue=hue, data=self.df, palette=palette)
        
        # Set title, labels, and legend.
        if y == 'AVG_LOS':
            plt.ylabel("Average Length of Stay (Days)", fontsize=12)
            plt.title(f"Average Length of Stay by {x} and {hue}", fontsize=14, fontweight='bold', pad=20)
        else:
            plt.ylabel(y, fontsize=12)
            plt.title(f"{y} by {x} and {hue}", fontsize=14, fontweight='bold', pad=20)

        plt.xlabel(x, fontsize=12)
        plt.legend(title=hue, fontsize=10, title_fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5, zorder=2)
        plt.tight_layout()
        plt.show()

    def plotTargetOutliers(self)-> None:
        """
        # Description
            -> This method was developed exclusively to
        analyse the outliers within the LOS target class.
        -------------------------------------------------
        # Params:
            - None.
        
        # Returns:
            - None, beacause we are only plotting data.
        """

        # Check if there is a LOS column in the loaded dataframe
        if 'LOS' not in self.df.columns:
            raise ValueError("Not possible to plot the Target Feature outliers as there is no column in the loaded dataframe with the Column \'LOS\'!")

        # Assuming 'self.df' contains a pre-computed LOS column in days
        los = self.df['LOS']

        # Compute quartiles and IQR
        q1 = los.quantile(0.25)
        q3 = los.quantile(0.75)
        iqr = q3 - q1
        lowerBound = q1 - 1.5 * iqr
        upperBound = q3 + 1.5 * iqr

        # Identify outliers
        outliers = self.df[(self.df['LOS'] < lowerBound) | (self.df['LOS'] > upperBound)]
        numberOutliers = len(outliers)

        # Plot the boxplot using Seaborn with a pastel aesthetic.
        sns.set(style="whitegrid")
        plt.figure(figsize=(6, 5))
        
        # Define a initial color
        initialColor = '#4a9b65'

        # Use your pastel color function to get a light color for the box.
        pastelColor = self.pastelizeColor(initialColor, weight=0.7)
        ax = sns.boxplot(x=los, color=pastelColor, zorder=2)
        plt.title("Boxplot of Length of Stay (LOS)", fontsize=14, fontweight='bold', pad=20)
        plt.xlabel("LOS (Days)")
        
        # Add a custom legend item with the outlier count.
        legend_patch = Patch(facecolor=pastelColor, edgecolor=initialColor, label=f'Outliers: {numberOutliers}')
        ax.legend(handles=[legend_patch], loc='upper right', title='Total Outliers')

        # Draw grid lines behind the boxplot
        plt.grid(True, linestyle='--', alpha=0.7, zorder=1)
        plt.tight_layout()
        plt.show()