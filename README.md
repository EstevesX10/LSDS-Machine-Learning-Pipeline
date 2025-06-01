<div align="center">

# LSDS | Machine Learning Pipeline

## ICU Patient Length of Stay Prediction

</div>

<p align="center" width="100%">
    <img src="./Machine-Learning-Pipeline/Assets/Robot-Hand.png" width="55%" height="55%" />
</p>

<div align="center">
    <a>
        <img src="https://img.shields.io/badge/Made%20with-Python-9eb0f2?style=for-the-badge&logo=Python&logoColor=9eb0f2">
    </a>
    <a>
        <img src="https://img.shields.io/badge/Made%20with-Jupyter-9eb0f2?style=for-the-badge&logo=Jupyter&logoColor=9eb0f2">
    </a>
    <a>
        <img src="https://img.shields.io/badge/Made%20with-Apache Spark-9eb0f2?style=for-the-badge&logo=apachespark&logoColor=9eb0f2">
    </a>
</div>

<br/>

<div align="center">
    <a href="https://github.com/EstevesX10/_REPO_NAME_/blob/main/LICENSE">
        <img src="https://img.shields.io/github/license/EstevesX10/_REPO_NAME_?style=flat&logo=gitbook&logoColor=9eb0f2&label=License&color=9eb0f2">
    </a>
    <a href="">
        <img src="https://img.shields.io/github/repo-size/EstevesX10/_REPO_NAME_?style=flat&logo=googlecloudstorage&logoColor=9eb0f2&logoSize=auto&label=Repository%20Size&color=9eb0f2">
    </a>
    <a href="">
        <img src="https://img.shields.io/github/stars/EstevesX10/_REPO_NAME_?style=flat&logo=adafruit&logoColor=9eb0f2&logoSize=auto&label=Stars&color=9eb0f2">
    </a>
    <a href="https://github.com/EstevesX10/_REPO_NAME_/blob/main/DEPENDENCIES.md">
        <img src="https://img.shields.io/badge/Dependencies-DEPENDENCIES.md-white?style=flat&logo=anaconda&logoColor=9eb0f2&logoSize=auto&color=9eb0f2"> 
    </a>
</div>

## Project Background

In **critical care**, being able to accurately predict how long a patient will stay in the intensive care unit is essential for managing resources effectively and **improving patient outcomes**. This challenge is complex because many clinical factors are involved, including vital signs, laboratory results, and the treatments administered. That is why close **collaboration between clinical expertise and advanced data analytics** is necessary to develop a reliable solution.

## Project Development

### Dependencies & Execution

This project was developed using a `Notebook`. Therefore if you're looking forward to test it out yourself, keep in mind to either use a **[Anaconda Distribution](https://www.anaconda.com/)** or a 3rd party software that helps you inspect and execute it.

Therefore, for more informations regarding the **Virtual Environment** used in Anaconda, consider checking the [DEPENDENCIES.md](https://github.com/EstevesX10/_REPO_NAME_/blob/main/DEPENDENCIES.md) file.

### Project Overview

Our focus is to build a **complete machine learning pipeline** that handles everything from **preparing the data** (from the MIMIC-III dataset) and **creating features** to training and validating the model. By including **profiling** throughout the entire process, we aim to continuously monitor **key performance measures** like how long tasks take and how much computing power is used. This is especially important given the large amount of data we are working with. Therefore, this approach allows us to **fine-tune the machine learning algorithm** we select, ultimately delivering a **reliable model** to predict the length of stay in the intensive care unit.

### MIMIC-III Dataset

The [MIMIC-III](https://physionet.org/content/mimiciii/1.4/) Dataset contains **multiple tables** from which we are going to select the ones that provide a **greater value** to our project and further analyse them to better gain a better understanding of the available data. Furthermore, its important to notice that we have uploaded the tables from the dataset onto the google cloud BigQuery Service to help us properly manage these high volumes of data, as this dataset contains a wide range of tables covering billing, procedures, laboratory results, microbiology, clinical notes and ICU events.

Since our project focuses on predicting ICU length of stay, MIMIC provides a robust platform to explore complex clinical patterns and build predictive models that can help optimize patient care and resource allocation, we chose the tables that offer the most relevant structured information:

1. **Patient demographics and admission context**  
   **PATIENTS** with age, sex and date of birth for baseline characteristics  
   **ADMISSIONS** with admission and discharge times, insurance type, marital status, discharge location and in-hospital mortality indicator

2. **ICU episode definitions**  
   **ICUSTAYS** with ICU entry and exit timestamps and unit transfers for ground truth length of stay

3. **High volume clinical measurements**  
   **CHARTEVENTS** for vital signs, ventilator settings and infusion rates as dynamic predictors

4. **Diagnosis coding**  
   **DIAGNOSES_ICD** for assigned ICD-9 codes per hospitalization  
   **D_ICD_DIAGNOSES** for human-readable code titles to enable grouping and interpretation

5. **Measurement metadata**  
   **D_ITEMS** for item identifiers, units and categories that define every measurement captured in the ICU event tables

For more details regarding the tables within the database, feel free to see the official `MIMIC-III` documentation [here](https://mimic.mit.edu/docs/iii/tables/chartevents/).

## Project Results and Conclusions

Here are the results from all the previously discussed frameworks, organized in the table below:

| Regressor |         Framework          | Total Fits | Total Time | Time Per Fit | MSE Hold-out | $R^{2}$ |
| :-------: | :------------------------: | ---------: | ---------: | -----------: | -----------: | ------: |
|   Lasso   |   Scikit-learn + Pandas    |         10 |     5.12 s |       0.51 s |        79.72 |   0.032 |
|   Lasso   |   Dask (single machine)    |         10 |    60.54 s |       6.05 s |        83.12 |   0.158 |
|   Lasso   |        Modin + Ray         |         10 |    51.98 s |       5.20 s |        79.72 |   0.032 |
|   Lasso   |          Pyspark           |         10 |    88.42 s |       8.44 s |        75.43 |   0.142 |
|   Lasso   | Dask (cluster of machines) |         10 |    40.61 s |       2.44 s |        83.12 |   0.158 |

Among all **tested frameworks**, **Scikit-learn combined with Pandas** delivered the best overall **efficiency**. It completed the entire training and cross-validation process in just over five seconds, with consistently low time per fit and **no signs of performance inflation**. Although its $R^2$ was relatively low at 0.032, this reflects the proper use of group-aware cross-validation, making it a trustworthy benchmark for generalization in subject-aware datasets. For workloads that fit comfortably in memory, this approach remains the most robust and **reliable option**.

**Dask** showed mixed results depending on how it was deployed. When running on a **single machine**, it incurred significant overhead, taking over 60 seconds to complete the same task, largely due to **inefficiencies with GroupKFold** and **repeated data materialization**. However, when scaled **across a cluster**, Dask cut total runtime by a third, showing that it can benefit from **distributed execution**. As a result, the $R^2$ score rose to 0.158.

**PySpark and Modin** each demonstrated **unique limitations**. PySpark was the _slowest_ among all, with more than 88 seconds total runtime, and lacks native support for grouped cross-validation. Its higher $R^2$ is misleading due to the absence of grouping during training, which **introduces potential data leakage**. Modin with Ray achieved better performance than PySpark but still lagged behind Dask cluster in speed. Like Scikit-learn, it respected groups during evaluation, which kept its metrics more conservative and meaningful. Overall, for subject-based modeling tasks, frameworks that support or emulate **group-aware validation are essential**. While distributed tools like Dask and PySpark are **promising for larger-scale problems**, they **require careful handling** of data partitioning and evaluation logic to ensure that speed gains do not come at the cost of **valid model assessment**.

## Authorship

- **Authors** &#8594; [Gonçalo Esteves](https://github.com/EstevesX10) and [Pedro Afonseca](https://github.com/PsuperX)
- **Course** &#8594; Large Scale Data Science [[CC3047](https://sigarra.up.pt/fcup/en/UCURR_GERAL.FICHA_UC_VIEW?pv_ocorrencia_id=546537)]
- **University** &#8594; Faculty of Sciences, University of Porto

<div align="right">
<sub>

<!-- <sup></sup> -->

`README.md by Gonçalo Esteves`
</sub>

</div>
