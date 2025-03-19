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