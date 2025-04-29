"""
# ------------------------------------------------------------------------- #
| Configuration regarding important constants/variables used in the Project |
# ------------------------------------------------------------------------- #
"""

def loadConfig() -> dict:
    """
    # Description
        -> This function focuses on creating a dictionary with important
        settings to be used throughout the study.
    --------------------------------------------------------------------
    # Params:
        - None

    # Retuns:
        - A Dictionary with important configuration settings.
    """

    return {
        # Constants
        'seed':14,
        'targetFeature':'LOS'
    }

"""
# ---------------------------------------------------------------- #
| Configuration regarding important file paths used in the Project |
# ---------------------------------------------------------------- #
"""

def createDatasetsPaths() -> dict:
    """
    # Description
        -> This function focuses on creating a dictionary with important
        dataset's file paths to be used throughout the project.
    --------------------------------------------------------------------
    # Params:
        - None

    # Retuns:
        - A Dictionary with important file paths.
    """

    return {
        'ADMISSIONS':'./Datasets/ADMISSIONS.csv',
        'CHARTEVENTS':'./Datasets/CHARTEVENTS.csv',
        'DIAGNOSES_ICD':'./Datasets/DIAGNOSES_ICD.csv',
        'ICUSTAYS':'./Datasets/ICUSTAYS.csv',
        'PATIENTS':'./Datasets/PATIENTS.csv',
    }

def loadPathsConfig() -> dict:
    """
    # Description
        -> This function focuses on creating a dictionary with important
        file paths to be used throughout the study.
    --------------------------------------------------------------------
    # Params:
        - None

    # Retuns:
        - A Dictionary with important file paths.
    """

    return {
        'Datasets':createDatasetsPaths()
    }

"""
# ------------------------------------------------ #
| Configuration of the Queries used in the Project |
# ------------------------------------------------ #
"""

"""
# -------------------------- #
| Illnesses Analysis Queries |
# -------------------------- #
"""

def loadMostCommonIllnessesQuery() -> str:
    """
    # Description
        -> This function load the query used
    to analyse the most common illnesses on the dataset.
    ----------------------------------------------------
    # Params:
        - None.
    
    # Returns:
        - The query developed.
    """

    # Return the Query
    return (
        """
        SELECT 
            descriptionDiagnoses.SHORT_TITLE AS ILLNESS,
            COUNT(*) AS TOTAL_CASES
        FROM `MIMIC.DIAGNOSES_ICD` AS diagnoses
            JOIN `MIMIC.D_ICD_DIAGNOSES` AS descriptionDiagnoses
                ON diagnoses.ICD9_CODE = descriptionDiagnoses.ICD9_CODE
        GROUP BY descriptionDiagnoses.SHORT_TITLE
        ORDER BY TOTAL_CASES DESC;
        """
    )

def loadLeastFrequentCoOccuringDiagnosisQuery() -> str:
    """
    # Description
        -> This function load the query used to analyse the
        least frequent Co-Occuring Diagnosis on the dataset.
    --------------------------------------------------------
    # Params:
        - None.
    
    # Returns:
        - The query developed.
    """

    # Return the query
    return (
        """
        SELECT 
            diagA.SHORT_TITLE AS ILLNESS_A,
            diagB.SHORT_TITLE AS ILLNESS_B,
            COUNT(*) AS TOTAL_CO_OCCURENCIES
        FROM `MIMIC.DIAGNOSES_ICD` AS a
        JOIN `MIMIC.DIAGNOSES_ICD` AS b
            ON a.HADM_ID = b.HADM_ID
            AND a.ICD9_CODE < b.ICD9_CODE
        JOIN `MIMIC.D_ICD_DIAGNOSES` AS diagA
            ON a.ICD9_CODE = diagA.ICD9_CODE
        JOIN `MIMIC.D_ICD_DIAGNOSES` AS diagB
            ON b.ICD9_CODE = diagB.ICD9_CODE
        GROUP BY diagA.SHORT_TITLE, diagB.SHORT_TITLE
        ORDER BY TOTAL_CO_OCCURENCIES ASC;
        """
    )

def loadMostFrequentCoOccuringDiagnosisQuery() -> str:
    """
    # Description
        -> This function load the query used to analyse the
        most frequent Co-Occuring Diagnosis on the dataset.
    -------------------------------------------------------
    # Params:
        - None.
    
    # Returns:
        - The query developed.
    """

    # Return the query
    return (
        """
        SELECT 
            diagA.SHORT_TITLE AS ILLNESS_A,
            diagB.SHORT_TITLE AS ILLNESS_B,
            COUNT(*) AS TOTAL_CO_OCCURENCIES
        FROM `MIMIC.DIAGNOSES_ICD` AS a
        JOIN `MIMIC.DIAGNOSES_ICD` AS b
            ON a.HADM_ID = b.HADM_ID
            AND a.ICD9_CODE < b.ICD9_CODE
        JOIN `MIMIC.D_ICD_DIAGNOSES` AS diagA
            ON a.ICD9_CODE = diagA.ICD9_CODE
        JOIN `MIMIC.D_ICD_DIAGNOSES` AS diagB
            ON b.ICD9_CODE = diagB.ICD9_CODE
        GROUP BY diagA.SHORT_TITLE, diagB.SHORT_TITLE
        ORDER BY TOTAL_CO_OCCURENCIES DESC;
        """
    )

def loadHighLengthOfStayIllnessesQuery() -> str:
    """
    # Description
        -> This function load the query used to analyse the
    illnesses with higher lengths of stay on the dataset.
    ---------------------------------------------------------
    # Params:
        - None.
    
    # Returns:
        - The query developed.
    """

    # Return the Query
    return (
        """
        WITH illness_stats AS (
            SELECT
                d.icd9_code,
                COUNT(*)                   AS TOTAL_CASES,
                AVG(s.los)                 AS AVG_LOS,
                APPROX_QUANTILES(s.los, 100)[OFFSET(50)] 
                                            AS median_los
            FROM `MIMIC.DIAGNOSES_ICD`   AS d
            JOIN `MIMIC.ICUSTAYS`        AS s
                ON d.subject_id = s.subject_id
            AND d.hadm_id    = s.hadm_id
            GROUP BY d.icd9_code
            HAVING COUNT(*) >= 50         
        )
        SELECT
            diag.short_title   AS illness,
            stats.TOTAL_CASES,
            ROUND(stats.AVG_LOS, 2)      AS AVG_LOS_DAYS,
            ROUND(stats.median_los, 2)   AS MEDIAN_LOS_DAYS
        FROM illness_stats AS stats
            JOIN `MIMIC.D_ICD_DIAGNOSES`   AS diag
                ON stats.icd9_code = diag.icd9_code
        ORDER BY stats.AVG_LOS DESC
        LIMIT 20;
        """
    )

def loadDeadliestIllnessesQuery() -> str:
    """
    # Description
        -> This function load the query used
    to analyse the deadliest illnesses on the dataset.
    ----------------------------------------------------
    # Params:
        - None.
    
    # Returns:
        - The query developed.
    """

    # Return the Query
    return (
        """
        SELECT 
            descriptionDiagnoses.SHORT_TITLE AS ILLNESS,
            COUNT(*) AS TOTAL_DEATHS
        FROM `MIMIC.DIAGNOSES_ICD` AS diagnoses
            JOIN `MIMIC.PATIENTS` AS patients
                ON diagnoses.SUBJECT_ID = patients.SUBJECT_ID
            JOIN `MIMIC.D_ICD_DIAGNOSES` AS descriptionDiagnoses
                ON diagnoses.ICD9_CODE = descriptionDiagnoses.ICD9_CODE
        WHERE patients.EXPIRE_FLAG = 1
        GROUP BY descriptionDiagnoses.SHORT_TITLE
        ORDER BY TOTAL_DEATHS DESC;
        """
    )

def loadIllnessesAnalysisQueries() -> dict:
    """
    # Description
        -> This function aims to load all the queries
    regarding the analysis of the illnesses on the dataset.
    -------------------------------------------------------
    # Params:
        - None.

    # Returns:
        - A dictionary with the developed queries.
    """

    # Return the dictionary
    return {
        'Most-Common-Illnesses':loadMostCommonIllnessesQuery(),
        'Least-Frequent-Co-Occuring-Diagnoses':loadLeastFrequentCoOccuringDiagnosisQuery(),
        'Most-Frequent-Co-Occuring-Diagnoses':loadMostFrequentCoOccuringDiagnosisQuery(),
        'High-Length-Of-Stay-Illnesses':loadHighLengthOfStayIllnessesQuery(),
        'Deadliest-Illnesses':loadDeadliestIllnessesQuery()
    }

"""
# ---------------------- #
| Mortality Rate Queries |
# ---------------------- #
"""
def loadGenderMortalityRatesQuery() -> str:
    """
    # Description
        -> This function loads the query used to analyse
    the mortality rates over the patients based on their gender.
    ------------------------------------------------------------
    # Params:
        - None.
    
    # Returns:
        - A string with the query used to develop the analysis.
    """

    # Return the Query used
    return (
        """
        WITH TOTAL_PATIENTS_GENDER AS (
            SELECT 
                GENDER,
                COUNT(*) AS TOTAL_PATIENTS
            FROM `MIMIC.PATIENTS`
            GROUP BY GENDER
        ),
        OVERALL_DEATHS AS (
            SELECT COUNT(*) AS OVERALL_DEATHS
            FROM `MIMIC.PATIENTS`
            WHERE EXPIRE_FLAG = 1
        )
        SELECT 
            p.GENDER,
            COUNT(*) AS TOTAL_DEATHS,
            t.TOTAL_PATIENTS,
            SAFE_DIVIDE(COUNT(*), t.TOTAL_PATIENTS) AS GENDER_MORTALITY_RATIO, -- local mortality rate within the gender
            SAFE_DIVIDE(COUNT(*), od.OVERALL_DEATHS) AS PROPORTIONAL_MORTALITY_RATIO  -- proportional mortality ratio (share of overall deaths)
        FROM `MIMIC.PATIENTS` AS p
        JOIN TOTAL_PATIENTS_GENDER AS t 
            ON p.GENDER = t.GENDER
        CROSS JOIN OVERALL_DEATHS AS od
        WHERE p.EXPIRE_FLAG = 1
        GROUP BY p.GENDER, t.TOTAL_PATIENTS, od.OVERALL_DEATHS
        ORDER BY PROPORTIONAL_MORTALITY_RATIO DESC;
        """
    )

def loadMaritalStatusMortalityRatesQuery() -> str:
    """
    # Description
        -> This function loads the query used to analyse the mortality
        rates over the patients based on their marital status.
    ------------------------------------------------------------------
    # Params:
        - None.
    
    # Returns:
        - A string with the query used to develop the analysis.
    """

    # Return the Query used
    return (
        """
        WITH MARITAL_STATUS_TABLE AS (
            SELECT
                SUBJECT_ID,
                MARITAL_STATUS
            FROM `MIMIC.ADMISSIONS`
            GROUP BY SUBJECT_ID, MARITAL_STATUS
        ),
        -- Get one record per patient so they are not counted multiple times if they have more than one admission.
        PATIENTS_MARITAL AS (
            SELECT DISTINCT
                SUBJECT_ID,
                MARITAL_STATUS
            FROM MARITAL_STATUS_TABLE
        ),
        -- Total number of patients per marital status.
        TOTAL_PATIENTS_MARITAL AS (
            SELECT 
                MARITAL_STATUS,
                COUNT(*) AS TOTAL_PATIENTS
            FROM PATIENTS_MARITAL
            GROUP BY MARITAL_STATUS
        ),
        -- Overall deaths in the dataset.
        OVERALL_DEATHS AS (
            SELECT COUNT(*) AS OVERALL_DEATHS
            FROM `MIMIC.PATIENTS`
            WHERE EXPIRE_FLAG = 1
        ),
        -- Total deaths per marital status.
        MARITAL_DEATHS AS (
            SELECT 
                pm.MARITAL_STATUS,
                COUNT(*) AS TOTAL_DEATHS
            FROM `MIMIC.PATIENTS` AS p
            JOIN PATIENTS_MARITAL AS pm
                ON p.SUBJECT_ID = pm.SUBJECT_ID
            WHERE p.EXPIRE_FLAG = 1
            GROUP BY pm.MARITAL_STATUS
        )
        SELECT
            t.MARITAL_STATUS,
            d.TOTAL_DEATHS,
            t.TOTAL_PATIENTS,
            SAFE_DIVIDE(d.TOTAL_DEATHS, t.TOTAL_PATIENTS) AS MARITAL_STATUS_MORTALITY_RATE,
            SAFE_DIVIDE(d.TOTAL_DEATHS, o.OVERALL_DEATHS) AS PROPORTIONAL_MORTALITY_RATIO
        FROM TOTAL_PATIENTS_MARITAL AS t
        JOIN MARITAL_DEATHS AS d
            ON t.MARITAL_STATUS = d.MARITAL_STATUS
        CROSS JOIN OVERALL_DEATHS AS o
        ORDER BY PROPORTIONAL_MORTALITY_RATIO DESC;
        """
    )

def loadEthnicityMortalityRatesQuery() -> str:
    """
    # Description
        -> This function loads the query used to analyse
    the mortality rates over the patients based on their ethnicity.
    ------------------------------------------------------------
    # Params:
        - None.
    
    # Returns:
        - A string with the query used to develop the analysis.
    """

    # Return the Query used
    return (
        """
        WITH ETHNICITIES_TABLE AS (
            SELECT
                SUBJECT_ID,
                CASE 
                WHEN UPPER(ETHNICITY) LIKE '%WHITE%' 
                    AND UPPER(ETHNICITY) NOT LIKE '%HISPANIC%' 
                    THEN 'White/European'
                WHEN UPPER(ETHNICITY) LIKE '%BLACK%' 
                    THEN 'Black/African American'
                WHEN UPPER(ETHNICITY) LIKE '%ASIAN%' 
                    THEN 'Asian'
                WHEN UPPER(ETHNICITY) LIKE '%HISPANIC%' 
                    OR UPPER(ETHNICITY) LIKE '%LATINO%' 
                    THEN 'Hispanic/Latino'
                WHEN UPPER(ETHNICITY) LIKE '%AMERICAN INDIAN%' 
                    OR UPPER(ETHNICITY) LIKE '%ALASKA NATIVE%' 
                    THEN 'Native American'
                WHEN UPPER(ETHNICITY) LIKE '%NATIVE HAWAIIAN%' 
                    OR UPPER(ETHNICITY) LIKE '%PACIFIC ISLANDER%' 
                    THEN 'Pacific Islander'
                WHEN UPPER(ETHNICITY) LIKE '%UNKNOWN%'
                    OR UPPER(ETHNICITY) LIKE '%UNABLE%'
                    OR UPPER(ETHNICITY) LIKE '%DECLINED%'
                    THEN 'Unknown/Not Provided'
                ELSE 'Other'
                END AS ETHNICITY_GROUP
            FROM `MIMIC.ADMISSIONS`
            GROUP BY SUBJECT_ID, ETHNICITY_GROUP
        ),
        -- Get one record per patient to avoid counting the same person multiple times
        PATIENTS_ETHNICITIES AS (
            SELECT DISTINCT
                SUBJECT_ID,
                ETHNICITY_GROUP
            FROM ETHNICITIES_TABLE
        ),
        TOTAL_PATIENTS_ETHNICITIES AS (
            SELECT 
                ETHNICITY_GROUP,
                COUNT(*) AS TOTAL_PATIENTS
            FROM PATIENTS_ETHNICITIES
            GROUP BY ETHNICITY_GROUP
        ),
        OVERALL_DEATHS AS (
            SELECT COUNT(*) AS OVERALL_DEATHS
            FROM `MIMIC.PATIENTS`
            WHERE EXPIRE_FLAG = 1
        ),
        ETHNICITY_DEATHS AS (
            SELECT 
                pe.ETHNICITY_GROUP,
                COUNT(*) AS TOTAL_DEATHS
            FROM `MIMIC.PATIENTS` AS patients
            JOIN PATIENTS_ETHNICITIES AS pe
                ON patients.SUBJECT_ID = pe.SUBJECT_ID
            WHERE patients.EXPIRE_FLAG = 1
            GROUP BY pe.ETHNICITY_GROUP
        )
        SELECT
            t.ETHNICITY_GROUP,
            d.TOTAL_DEATHS,
            t.TOTAL_PATIENTS,
            d.TOTAL_DEATHS / t.TOTAL_PATIENTS AS ETHNICITY_MORTALITY_RATE,
            d.TOTAL_DEATHS / o.OVERALL_DEATHS AS PROPORTIONAL_MORTALITY_RATIO
        FROM TOTAL_PATIENTS_ETHNICITIES AS t
        JOIN ETHNICITY_DEATHS AS d
            ON t.ETHNICITY_GROUP = d.ETHNICITY_GROUP
        CROSS JOIN OVERALL_DEATHS AS o
        ORDER BY ETHNICITY_MORTALITY_RATE DESC;
        """
    )

def loadCareUnitsMortalityRatesQuery() -> str:
    """
    # Description
        -> This function loads the query used to analyse the mortality
        rates over the patients based on their initial care unit.
    ------------------------------------------------------------------
    # Params:
        - None.
    
    # Returns:
        - A string with the query used to develop the analysis.
    """

    # Return the Query used
    return (
        """
        WITH OVERALL_DEATHS_TABLE AS (
            SELECT COUNT(*) AS OVERALL_DEATHS
            FROM `MIMIC.PATIENTS`
            WHERE EXPIRE_FLAG = 1
        ),
        CAREUNIT_DEATHS AS (
            SELECT 
                icu.FIRST_CAREUNIT,
                COUNT(*) AS TOTAL_DEATHS
            FROM `MIMIC.PATIENTS` AS patients
            JOIN `MIMIC.ICUSTAYS` AS icu
                ON patients.SUBJECT_ID = icu.SUBJECT_ID
            WHERE patients.EXPIRE_FLAG = 1
            GROUP BY icu.FIRST_CAREUNIT
        ),
        CAREUNIT_PATIENTS AS (
            SELECT 
                icu.FIRST_CAREUNIT,
                COUNT(*) AS TOTAL_PATIENTS
            FROM `MIMIC.PATIENTS` AS patients
            JOIN `MIMIC.ICUSTAYS` AS icu
                ON patients.SUBJECT_ID = icu.SUBJECT_ID
            GROUP BY icu.FIRST_CAREUNIT
        )
        SELECT 
            cp.FIRST_CAREUNIT,
            cd.TOTAL_DEATHS,
            cp.TOTAL_PATIENTS,
            cd.TOTAL_DEATHS / cp.TOTAL_PATIENTS AS CARE_UNIT_MORTALITY_RATE,
            cd.TOTAL_DEATHS / od.OVERALL_DEATHS AS PROPORTIONAL_MORTALITY_RATIO
            FROM CAREUNIT_PATIENTS AS cp
        JOIN CAREUNIT_DEATHS AS cd
            ON cp.FIRST_CAREUNIT = cd.FIRST_CAREUNIT
        CROSS JOIN OVERALL_DEATHS_TABLE AS od
        ORDER BY CARE_UNIT_MORTALITY_RATE DESC;
        """
    )

def loadMortalityRatesQueries() -> dict:
    """
    # Description
        -> This function loads the queries regarding the analysis of
        the mortality rates over multiple groups of interest.
    ----------------------------------------------------------------
    # Params:
        - None.
    
    # Returns:
        - A dictionary with the queries used for each group of interest.
    """

    return {
        'Gender':loadGenderMortalityRatesQuery(),
        'Marital-Status':loadMaritalStatusMortalityRatesQuery(),
        'Ethnicity':loadEthnicityMortalityRatesQuery(),
        'Care-Units':loadCareUnitsMortalityRatesQuery()
    }

"""
# ------------------------------- #
| Length of Stay Analysis Queries |
# ------------------------------- #
"""

def loadDemographicCorrelationsLOSQuery() -> str:
    """
    # Description
        -> This function loads the query responsible for computing
        the demographic correlations regarding the patient's LOS.
    --------------------------------------------------------------
    # Params:
        - None.
    
    # Returns:
        - A string with the query used to develop the analysis.
    """
    # Return the Query used
    return (
        """
        WITH patient_admissions AS (
            SELECT 
                p.SUBJECT_ID,
                p.GENDER,
                p.DOB,
                a.ETHNICITY,
                -- Apply the relabeling for ethnicity:
                CASE 
                WHEN UPPER(a.ETHNICITY) LIKE '%WHITE%' 
                    AND UPPER(a.ETHNICITY) NOT LIKE '%HISPANIC%' 
                    THEN 'White/European'
                WHEN UPPER(a.ETHNICITY) LIKE '%BLACK%' 
                    THEN 'Black/African American'
                WHEN UPPER(a.ETHNICITY) LIKE '%ASIAN%' 
                    THEN 'Asian'
                WHEN UPPER(a.ETHNICITY) LIKE '%HISPANIC%' 
                    OR UPPER(a.ETHNICITY) LIKE '%LATINO%' 
                    THEN 'Hispanic/Latino'
                WHEN UPPER(a.ETHNICITY) LIKE '%AMERICAN INDIAN%' 
                    OR UPPER(a.ETHNICITY) LIKE '%ALASKA NATIVE%' 
                    THEN 'Native American'
                WHEN UPPER(a.ETHNICITY) LIKE '%NATIVE HAWAIIAN%' 
                    OR UPPER(a.ETHNICITY) LIKE '%PACIFIC ISLANDER%' 
                    THEN 'Pacific Islander'
                WHEN UPPER(a.ETHNICITY) LIKE '%UNKNOWN%'
                    OR UPPER(a.ETHNICITY) LIKE '%UNABLE%'
                    OR UPPER(a.ETHNICITY) LIKE '%DECLINED%'
                    THEN 'Unknown/Not Provided'
                ELSE 'Other'
                END AS ETHNICITY_GROUP,
                a.ADMITTIME,
                a.DISCHTIME,
                TIMESTAMP_DIFF(a.DISCHTIME, a.ADMITTIME, DAY) AS LOS,
                DATE_DIFF(DATE(a.ADMITTIME), DATE(p.DOB), YEAR) AS AGE_AT_ADMISSION
            FROM `MIMIC.PATIENTS` p
            JOIN `MIMIC.ADMISSIONS` a
                ON p.SUBJECT_ID = a.SUBJECT_ID
            WHERE p.DOB IS NOT NULL
                AND a.ADMITTIME IS NOT NULL 
                AND a.DISCHTIME IS NOT NULL
        )
        SELECT 
            GENDER,
            ETHNICITY_GROUP,
            CASE 
                WHEN AGE_AT_ADMISSION < 30 THEN '<30'
                WHEN AGE_AT_ADMISSION BETWEEN 30 AND 50 THEN '30-50'
                WHEN AGE_AT_ADMISSION BETWEEN 51 AND 70 THEN '51-70'
                ELSE '>70'
            END AS AGE_GROUP,
            AVG(LOS) AS AVG_LOS,
            COUNT(*) AS ADMISSION_COUNT
        FROM patient_admissions
        GROUP BY GENDER, ETHNICITY_GROUP, AGE_GROUP
        ORDER BY AVG_LOS DESC;
        """
    )

def loadOutlierAnalysisLOSQuery() -> str:
    """
    # Description
        -> This function loads the query responsible for aiding the
        outlier analysis of the the patients length of stay (LOS).
    --------------------------------------------------------------
    # Params:
        - None.
    
    # Returns:
        - A string with the query used to develop the analysis.
    """
    # Return the Query Used
    return (
        """
        WITH LOS_DATA AS (
            SELECT 
                TIMESTAMP_DIFF(DISCHTIME, ADMITTIME, DAY) AS LOS
            FROM `MIMIC.ADMISSIONS`
            WHERE ADMITTIME IS NOT NULL AND DISCHTIME IS NOT NULL
        ),
        quartiles AS (
            SELECT
                APPROX_QUANTILES(LOS, 4)[OFFSET(1)] AS q1,
                APPROX_QUANTILES(LOS, 4)[OFFSET(3)] AS q3
            FROM LOS_DATA
        ),
        outlier_bounds AS (
            SELECT
                q1,
                q3,
                (q3 - q1) AS iqr,
                q1 - 1.5 * (q3 - q1) AS lowerBound,
                q3 + 1.5 * (q3 - q1) AS upperBound
            FROM quartiles
        )
        SELECT 
            LOS,
            CASE 
                WHEN LOS < lowerBound THEN 'Low Outlier'
                WHEN LOS > upperBound THEN 'High Outlier'
                ELSE 'Normal'
            END AS OUTLIER_FLAG
        FROM LOS_DATA, outlier_bounds;
        """
    )

def loadLengthOfStayAnalysisQueries() -> dict:
    """
    # Description
        -> This function loads the queries responsible to help us
        analyse the length of stay of the patients in the dataset.
    --------------------------------------------------------------
    # Params:
        - None.
    
    # Returns:
        - A structured dictionary with the developed queries.
    """

    return {
        'Demographic-Correlations':loadDemographicCorrelationsLOSQuery(),
        'Outlier-Analysis':loadOutlierAnalysisLOSQuery()
    }

"""
# -------------------------------------- #
| Exploratory Data Analysis Query Loader |
# -------------------------------------- #
"""

def loadExploratoryDataAnalysisQueries() -> dict:
    """
    # Description
        -> This function aims to load all the queries used
        through the exploratory data analysis stage of the project.
    ---------------------------------------------------------------
    # Params:
        - None.
    
    # Returns:
        - A dictionary with the queries developed.
    """

    return {
        'Illnesses-Analysis':loadIllnessesAnalysisQueries(),
        'Mortality-Rates':loadMortalityRatesQueries(),
        'Length-Of-Stay-Analysis':loadLengthOfStayAnalysisQueries()
    }

"""
# ------------------------------- #
| Data Preprocessing Query Loader |
# ------------------------------- #
"""
# [TODO] If needed
def loadDataPreprocessingQueries() -> dict:
    """
    # Description
        -> This function aims to load all the queries
    regarding the data preprocessing phase of the project.
    ------------------------------------------------------
    # Params:
        - None.
    
    # Returns:
        - A structured dictionary with the developed queries.
    """

    return {}

"""
# ----------------- #
| Main Query Loader |
# ----------------- #
"""

def loadQueries() -> dict:
    """
    # Description
        -> This function aims to create a dictionary to store all
        the important queries used throughout the project.
    --------------------------------------------------------------
    # Params:
        - None.
    
    # Returns:
        - A dictionary with the developed queries.
    """

    return {
        'Exploratory-Data-Analysis':loadExploratoryDataAnalysisQueries(),
        'Data-Preprocessing':loadDataPreprocessingQueries()
    }