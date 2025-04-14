from functools import cached_property
import bigframes.pandas as bpd
import pandas as pd


class BigQueryLoader:
    def __init__(
        self, vital_patterns: dict[str, list[str] | str], dataset: str = "MIMIC"
    ):
        self._vital_patterns = vital_patterns
        self.dataset = dataset

        self._matched_items = None

    @property
    def vital_patterns(self):
        return self._vital_patterns

    @vital_patterns.setter
    def vital_patterns(self, value):
        self._vital_patterns = value
        self._matched_items = None

    def get_demo_df(self) -> bpd.DataFrame:
        # Load key tables
        icustays = bpd.read_gbq(f"{self.dataset}.ICUSTAYS")
        admissions = bpd.read_gbq(f"{self.dataset}.ADMISSIONS")
        patients = bpd.read_gbq(f"{self.dataset}.PATIENTS")

        # Join to get demographics
        icu_demo = (
            icustays.merge(admissions, on=["HADM_ID", "SUBJECT_ID"])
            .merge(patients, on="SUBJECT_ID")
            .assign(
                AGE=lambda df: df.INTIME.dt.year - df.DOB.dt.year,
                LOS_HOURS=lambda df: df.LOS * 24,
            )
            .query("AGE < 120")[  # remove outliers
                [
                    "ICUSTAY_ID",
                    "HADM_ID",
                    "SUBJECT_ID",
                    "GENDER",
                    "ETHNICITY",
                    "AGE",
                    "LOS",
                    "LOS_HOURS",
                ]
            ]
        )
        return icu_demo

    @property
    def matched_items(self) -> pd.DataFrame:
        """
        Matches ITEMIDs from D_ITEMS using a dict of {friendly_name: list of SQL LIKE patterns}.

        Returns:
        - DataFrame with: ITEMID, LABEL, VITAL_NAME (your friendly label)
        """

        if self._matched_items is not None:
            return self._matched_items

        # Normalize to dict[str, list[str]]
        normalized_patterns = {
            name: [p] if isinstance(p, str) else p
            for name, p in self.vital_patterns.items()
        }

        # Flatten to (pattern, label) pairs
        all_patterns = []
        for vital_name, patterns in normalized_patterns.items():
            for pattern in patterns:
                all_patterns.append((vital_name, pattern))

        # Build WHERE clause
        conditions = " OR ".join(
            [f"LOWER(LABEL) LIKE LOWER('{pattern}')" for _, pattern in all_patterns]
        )

        query = f"""
            SELECT ITEMID, LABEL
            FROM `{self.dataset}.D_ITEMS`
            WHERE {conditions}
        """
        matched = bpd.read_gbq(query).to_pandas()

        if matched.empty:
            raise ValueError("No ITEMIDs matched any of the provided patterns.")

        # Assign VITAL_NAME by checking which friendly name's pattern matched
        def assign_vital_name(label: str) -> str:
            label_lower = label.lower()
            for vital_name, patterns in normalized_patterns.items():
                for pattern in patterns:
                    if pattern.strip("%").lower() in label_lower:
                        return vital_name
            return "unknown"

        matched["VITAL_NAME"] = matched["LABEL"].apply(assign_vital_name)

        self._matched_items = matched
        return matched

    def summarize_vitals(self) -> bpd.DataFrame:
        """
        Aggregates and pivots multiple vitals into a wide-format BigFrames DataFrame.

        Returns:
        - A BigFrames DataFrame with one row per ICUSTAY_ID and one column per vital_stat
        """

        items_df = self.matched_items
        item_lookup = items_df[["ITEMID", "VITAL_NAME"]].drop_duplicates()
        item_lookup_str = ",\n".join(
            f"({row.ITEMID}, '{row.VITAL_NAME}')"
            for row in item_lookup.itertuples(index=False)
        )

        # Build pivot columns
        stats = ["mean", "min", "max", "std", "count"]
        select_exprs = []
        for stat in stats:
            agg_func = (
                "AVG" if stat == "mean" else "STDDEV" if stat == "std" else stat.upper()
            )
            for vital in item_lookup["VITAL_NAME"].unique():
                expr = f"""{agg_func}(CASE WHEN VITAL_NAME = '{vital}' THEN VALUENUM ELSE NULL END) AS `{vital}_{stat}`"""
                select_exprs.append(expr)

        pivot_sql = ",\n    ".join(select_exprs)

        query = f"""
        WITH item_map AS (
            SELECT ITEMID, VITAL_NAME
            FROM UNNEST([
                STRUCT<ITEMID INT64, VITAL_NAME STRING>
                {item_lookup_str}
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
        ce_24hr AS (
            SELECT *
            FROM ce_filtered
            WHERE HOURS_FROM_INTIME <= 24
        )
        SELECT
            ICUSTAY_ID,
            {pivot_sql}
        FROM ce_24hr
        GROUP BY ICUSTAY_ID
        """

        return bpd.read_gbq(query)

    def extract_vent_flag(self) -> bpd.DataFrame:
        """
        Returns a BigFrames DataFrame with one row per ICU stay and a binary on_vent flag
        indicating presence of a ventilation-related event in the first 24 hours.
        """
        query = f"""
        WITH vent_items AS (
            SELECT ITEMID
            FROM `{self.dataset}.D_ITEMS`
            WHERE LOWER(LABEL) LIKE '%ventilator%'
        ),
        ce_24hr AS (
            SELECT 
                ce.ICUSTAY_ID,
                TIMESTAMP_DIFF(ce.CHARTTIME, ic.INTIME, SECOND)/3600.0 AS HOURS_FROM_INTIME
            FROM `{self.dataset}.CHARTEVENTS` ce
            JOIN `{self.dataset}.ICUSTAYS` ic USING(ICUSTAY_ID)
            JOIN vent_items vi ON ce.ITEMID = vi.ITEMID
            WHERE ce.VALUENUM IS NOT NULL
        ),
        flagged AS (
            SELECT ICUSTAY_ID, 1 AS on_vent
            FROM ce_24hr
            WHERE HOURS_FROM_INTIME <= 24
            GROUP BY ICUSTAY_ID
        )
        SELECT 
            ic.ICUSTAY_ID,
            IFNULL(f.on_vent, 0) AS on_vent
        FROM `{self.dataset}.ICUSTAYS` ic
        LEFT JOIN flagged f USING(ICUSTAY_ID)
        """

        return bpd.read_gbq(query)
