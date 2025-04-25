from functools import (cached_property)
import bigframes.pandas as bpd
import numpy as np
import pandas as pd

class BigQueryLoader:
    """
    # Description
        -> A class for loading and processing data
        from the MIMIC dataset using BigFrames.
    """
    def __init__(self, vital_patterns: dict[str, list[str] | str], dataset: str = "MIMIC"):
        """
        # Description:
            -> Constructor of the MIMICDataLoader class, which initializes patterns used
               to match items in the MIMIC D_ITEMS table, and sets a default dataset name.
        ----------------------------------------------------------------------------------
        # Params:
            - vital_patterns: dict[str, list[str] | str]
                -> Dictionary mapping a user-friendly vital name (e.g. 'heart_rate') to one or
                more SQL LIKE patterns (e.g. ['%heart rate%', '%hr%']).
            - dataset: str
                -> The name of the BigQuery dataset containing MIMIC tables. Defaults to 'MIMIC'.
        ------------
        # Returns:
            - None
        """
        # Store the user-provided dictionary of patterns (SQL LIKE expressions)
        self._vital_patterns = vital_patterns

        # Store the dataset name, e.g. "MIMIC" (default)
        self.dataset = dataset
        
        # Internal cache for matched items DataFrame - Starts as None until first load
        self._matched_vitals_df = None

    @property
    def vital_patterns(self) -> dict[str, list[str] | str]:
        """
        # Description:
            -> Property to access the current dictionary of patterns for matching item IDs.
        ----------------------------------------------------------------------------------
        # Params:
            - None.
        ------------
        # Returns:
            - dict[str, list[str] | str]: The dictionary mapping vital names to LIKE patterns.
        """
        return self._vital_patterns

    @vital_patterns.setter
    def vital_patterns(self, new_patterns: dict[str, list[str] | str]) -> None:
        """
        # Description:
            -> Setter for the vital_patterns property, used to update the dictionary of
               user-friendly vital labels and patterns, and reset the cached matched items.
        ----------------------------------------------------------------------------------
        # Params:
            - new_patterns: dict[str, list[str] | str]
                -> The updated dictionary of vital names to SQL LIKE patterns.
        ------------
        # Returns:
            - None.
        """
        # Update the dictionary of patterns and invalidate any previously cached items
        self._vital_patterns = new_patterns
        self._matched_vitals_df = None

    def load_demo(self) -> bpd.DataFrame:
        """
        # Description:
            -> Loads key tables (ICUSTAYS, ADMISSIONS, PATIENTS) from BigQuery,
               merges them, computes AGE and LOS_HOURS, and filters out unrealistic ages.
        ----------------------------------------------------------------------------------
        # Params:
            - None.
        ------------
        # Returns:
            - bpd.DataFrame: A BigFrames DataFrame with demographic info including
              ICUSTAY_ID, HADM_ID, SUBJECT_ID, GENDER, ETHNICITY, AGE, LOS, LOS_HOURS.
        """
        # Read ICUSTAYS from BigQuery via BigFrames
        icu_stays = bpd.read_gbq(f"{self.dataset}.ICUSTAYS")

        # Read ADMISSIONS from BigQuery
        admissions = bpd.read_gbq(f"{self.dataset}.ADMISSIONS")

        # Read PATIENTS from BigQuery
        patients = bpd.read_gbq(f"{self.dataset}.PATIENTS")

        # Merge all three tables to get a combined DataFrame with relevant columns
        df = (
            icu_stays
            .merge(admissions, on=["HADM_ID", "SUBJECT_ID"])
            .merge(patients, on="SUBJECT_ID")
            # Assign new columns:
            .assign(
                # Approximate age = year of INTIME minus year of DOB
                AGE=lambda df: df.INTIME.dt.year - df.DOB.dt.year,
            )
            # Filter out rows where AGE is 120 or more (likely data artifact)
            .query("AGE < 120")
            # Retain only columns of interest
            [
                [
                    "ICUSTAY_ID",
                    "HADM_ID",
                    "SUBJECT_ID",
                    "GENDER",
                    "ETHNICITY",
                    "AGE",
                    "LOS"
                ]
            ]
        )

        # Define the conditions to perform ethnicity relabelling
        eth = df["ETHNICITY"].str.upper()
        conds = [
            eth.str.contains("WHITE") & ~eth.str.contains("HISPANIC"),
            eth.str.contains("BLACK"),
            eth.str.contains("ASIAN"),
            eth.str.contains("HISPANIC") | eth.str.contains("LATINO"),
            eth.str.contains("AMERICAN INDIAN") | eth.str.contains("ALASKA NATIVE"),
            eth.str.contains("NATIVE HAWAIIAN") | eth.str.contains("PACIFIC ISLANDER"),
            eth.str.contains("UNKNOWN") | eth.str.contains("UNABLE") | eth.str.contains("DECLINED"),
        ]
        choices = [
            "White/European",
            "Black/African American",
            "Asian",
            "Hispanic/Latino",
            "Native American",
            "Pacific Islander",
            "Unknown/Not Provided",
        ]

        # Apply the ethnicity relabeling
        df["ETHNICITY_GROUP"] = np.select(conds, choices, default="Other")

        # Select the important columns
        demographics_df = df[[
            "ICUSTAY_ID", "HADM_ID", "SUBJECT_ID",
            "GENDER", "ETHNICITY_GROUP",
            "AGE", "LOS"
        ]]

        return demographics_df

    @property
    def matched_vitals_df(self) -> pd.DataFrame:
        """
        # Description:
            -> Creates a DataFrame of item IDs and labels in D_ITEMS that match the
               user-defined patterns (SQL LIKE clauses). Then assigns a user-friendly
               'VITAL_NAME' to each item.
        ----------------------------------------------------------------------------------
        # Params:
            - None.
        ------------
        # Returns:
            - pd.DataFrame with columns: ITEMID, LABEL, VITAL_NAME.
              Cached after the first load for performance.
        """
        # If we have already matched items, return the cached DataFrame
        if self._matched_vitals_df is not None:
            return self._matched_vitals_df

        # Convert any single strings in the patterns to lists for consistency
        normalized_patterns = {
            vital_name: [p] if isinstance(p, str) else p
            for vital_name, p in self.vital_patterns.items()
        }

        # Flatten out a list of (vital_name, pattern) tuples
        flat_patterns = []
        for vital_name, patterns in normalized_patterns.items():
            for pattern in patterns:
                flat_patterns.append((vital_name, pattern))

        # Build a WHERE clause with OR for each pattern:
        # e.g. LOWER(LABEL) LIKE LOWER('%heart rate%') OR LOWER(LABEL) LIKE LOWER('%hr%') ...
        where_clauses = " OR ".join(
            [f"LOWER(LABEL) LIKE LOWER('{pattern}')" for _, pattern in flat_patterns]
        )

        query = f"""
            SELECT ITEMID, LABEL
            FROM `{self.dataset}.D_ITEMS`
            WHERE {where_clauses}
        """
        # Read from BigQuery into a Pandas DataFrame
        matched_items = bpd.read_gbq(query).to_pandas()

        # If no rows matched, raise an error
        if matched_items.empty:
            raise ValueError("No ITEMIDs matched any of the provided patterns.")

        # A helper function to assign a user-friendly name based on the pattern match
        def assign_vital_name(label: str) -> str:
            label_lower = label.lower()
            # Check each (friendly_name, list_of_patterns) pair
            for vital_name, patterns in normalized_patterns.items():
                # If the pattern (minus wildcards) is in the label, we consider it a match
                for pattern in patterns:
                    if pattern.strip("%").lower() in label_lower:
                        return vital_name
            return "unknown"

        # Apply the assignment to each row, resulting in a new column 'VITAL_NAME'
        matched_items["VITAL_NAME"] = matched_items["LABEL"].apply(assign_vital_name)

        # Cache the result to avoid repeating the query if asked again
        self._matched_vitals_df = matched_items
        return matched_items

    def pivot_vitals_24h(self) -> bpd.DataFrame:
        """
        # Description:
            -> Aggregates vital measurements from CHARTEVENTS in the first 24 hours of
               an ICU stay. Each vital is pivoted with stats like mean, min, max, std, count.
        ----------------------------------------------------------------------------------
        # Params:
            - None.
        ------------
        # Returns:
            - bpd.DataFrame: One row per ICUSTAY_ID, columns of the form <vital>_<stat>.
        """
        # Retrieve the DataFrame of matched items (ITEMID, LABEL, VITAL_NAME)
        matched_vitals = self.matched_vitals_df

        # Deduplicate so each (ITEMID, VITAL_NAME) pair is unique
        item_map_df = matched_vitals[["ITEMID", "VITAL_NAME"]].drop_duplicates()

        # Convert item_map_df to a string of (ITEMID, VITAL_NAME) tuples for inline usage
        item_map_struct_str = ",\n".join(
            f"({row.ITEMID}, '{row.VITAL_NAME}')"
            for row in item_map_df.itertuples(index=False)
        )

        # We want to compute these stats for each vital sign
        stats = ["mean", "min", "max", "std", "count"]
        agg_expressions = []

        # For each stat, build the appropriate SQL expression (AVG, MIN, MAX, STDDEV, COUNT)
        for stat in stats:
            if stat == "mean":
                agg_func = "AVG"
            elif stat == "std":
                agg_func = "STDDEV"
            else:
                agg_func = stat.upper()

            # For each distinct vital name, create a CASE expression that picks rows for that vital
            for vital_name in item_map_df["VITAL_NAME"].unique():
                expr = (
                    f"{agg_func}(CASE WHEN VITAL_NAME = '{vital_name}' "
                    f"THEN VALUENUM ELSE NULL END) AS `{vital_name}_{stat}`"
                )
                agg_expressions.append(expr)

        # Join the expressions with commas and line breaks for readability in SQL
        pivot_cols = ",\n    ".join(agg_expressions)

        # Define the Query
        query = (
            f"""
            WITH item_map AS (
                SELECT ITEMID, VITAL_NAME
                FROM UNNEST([
                    STRUCT<ITEMID INT64, VITAL_NAME STRING>
                    {item_map_struct_str}
                ])
            ),
            ce_filtered AS (
                SELECT 
                    ce.ICUSTAY_ID,
                    ce.VALUENUM,
                    im.VITAL_NAME,
                    TIMESTAMP_DIFF(ce.CHARTTIME, ic.INTIME, SECOND) / 3600.0 AS HOURS_FROM_INTIME
                FROM `{self.dataset}.CHARTEVENTS` ce
                JOIN item_map im ON ce.ITEMID = im.ITEMID
                JOIN `{self.dataset}.ICUSTAYS` ic ON ce.ICUSTAY_ID = ic.ICUSTAY_ID
                WHERE ce.VALUENUM IS NOT NULL
            ),
            ce_first24h AS (
                -- Restrict to the first 24 hours of the ICU stay
                SELECT *
                FROM ce_filtered
                WHERE HOURS_FROM_INTIME <= 24
            )
            SELECT
                ICUSTAY_ID,
                {pivot_cols}
            FROM ce_first24h
            GROUP BY ICUSTAY_ID
            """
        )

        # The final pivoted DataFrame, one row per ICUSTAY_ID, columns for each vital+stat
        return bpd.read_gbq(query)

    def extract_ventilation_flag(self) -> bpd.DataFrame:
        """
        # Description:
            -> Identifies ICU stays with a 'ventilator' event in the first 24 hours.
               Yields a binary on_vent=1 or 0 for each ICU stay.
        ----------------------------------------------------------------------------------
        # Params:
            - None.
        ------------
        # Returns:
            - bpd.DataFrame: [ICUSTAY_ID, on_vent].
              on_vent=1 if a ventilator-related item is found, else 0.
        """
        # Define the Query
        query = (
            f"""
            WITH vent_items AS (
                -- Grab all items from D_ITEMS whose label mentions 'ventilator'
                SELECT ITEMID
                FROM `{self.dataset}.D_ITEMS`
                WHERE LOWER(LABEL) LIKE '%ventilator%'
            ),
            ce_first24h_vent AS (
                SELECT 
                    ce.ICUSTAY_ID,
                    TIMESTAMP_DIFF(ce.CHARTTIME, ic.INTIME, SECOND)/3600.0 AS HOURS_FROM_INTIME
                FROM `{self.dataset}.CHARTEVENTS` ce
                JOIN `{self.dataset}.ICUSTAYS` ic USING(ICUSTAY_ID)
                JOIN vent_items vi ON ce.ITEMID = vi.ITEMID
                WHERE ce.VALUENUM IS NOT NULL
            ),
            flagged AS (
                -- If an ICU stay has at least one row in the first 24 hours, mark on_vent=1
                SELECT ICUSTAY_ID, 1 AS on_vent
                FROM ce_first24h_vent
                WHERE HOURS_FROM_INTIME <= 24
                GROUP BY ICUSTAY_ID
            )
            SELECT 
                ic.ICUSTAY_ID,
                IFNULL(f.on_vent, 0) AS on_vent
            FROM `{self.dataset}.ICUSTAYS` ic
            LEFT JOIN flagged f USING(ICUSTAY_ID)
            """
        )

        # Return a DataFrame with two columns: ICUSTAY_ID, on_vent (0 or 1)
        return bpd.read_gbq(query)
