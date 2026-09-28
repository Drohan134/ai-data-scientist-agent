# AI Data Scientist Report

> This report was generated deterministically from the analysis results. Set AI_DATA_SCIENTIST_LLM_REPORT=1 to request an LLM-written report.

## 1. Executive Summary

The dataset contains **9551 rows** and **21 columns**.
The calculated data quality score is **94.99/100**.
The machine learning analysis evaluated multiple models. The reported best-model result was **Gradient Boosting**.
The critic stage identified limitations that should be considered when interpreting the machine learning results.

## 2. Dataset Overview

- Rows: 9551
- Columns: 21
- Columns:
  - `Restaurant ID`
  - `Restaurant Name`
  - `Country Code`
  - `City`
  - `Address`
  - `Locality`
  - `Locality Verbose`
  - `Longitude`
  - `Latitude`
  - `Cuisines`
  - `Average Cost for two`
  - `Currency`
  - `Has Table booking`
  - `Has Online delivery`
  - `Is delivering now`
  - `Switch to order menu`
  - `Price range`
  - `Aggregate rating`
  - `Rating color`
  - `Rating text`
  - `Votes`

Missing values identified during profiling:
- `Cuisines`: 9

## 3. Data Quality

- Duplicate rows: 0
- Constant columns: Switch to order menu
- High-cardinality columns: Restaurant Name, Address
- Binary columns: Has Table booking, Has Online delivery, Is delivering now
- Potential target columns: Has Table booking, Has Online delivery, Is delivering now, Rating color, Rating text

## 4. Data Cleaning

- Rows: 9551 → 9551
- Columns: 21 → 20

## 5. Exploratory Data Analysis

Strong observed correlations:
- `Country Code` ↔ `Longitude`: -0.698
- `Price range` ↔ `Aggregate rating`: 0.438
- `Restaurant ID` ↔ `Aggregate rating`: -0.326
- `Aggregate rating` ↔ `Votes`: 0.314
- `Price range` ↔ `Votes`: 0.309
- `Country Code` ↔ `Aggregate rating`: 0.282
- `Country Code` ↔ `Price range`: 0.243
- `Restaurant ID` ↔ `Longitude`: -0.226
- `Latitude` ↔ `Price range`: -0.167
- `Country Code` ↔ `Votes`: 0.155
- `Restaurant ID` ↔ `Country Code`: 0.148
- `Restaurant ID` ↔ `Votes`: -0.147
- `Restaurant ID` ↔ `Price range`: -0.135
- `Longitude` ↔ `Aggregate rating`: -0.117
- `Latitude` ↔ `Average Cost for two`: -0.111
- `Longitude` ↔ `Votes`: -0.085
- `Longitude` ↔ `Price range`: -0.079
- `Average Cost for two` ↔ `Price range`: 0.075
- `Average Cost for two` ↔ `Votes`: 0.068
- `Restaurant ID` ↔ `Latitude`: -0.052
- `Average Cost for two` ↔ `Aggregate rating`: 0.052
- `Longitude` ↔ `Average Cost for two`: 0.046
- `Country Code` ↔ `Average Cost for two`: 0.043
- `Longitude` ↔ `Latitude`: 0.043
- `Latitude` ↔ `Votes`: -0.023
- `Country Code` ↔ `Latitude`: 0.02
- `Restaurant ID` ↔ `Average Cost for two`: -0.002
- `Latitude` ↔ `Aggregate rating`: 0.001

These are observed statistical relationships and should not be interpreted as evidence of causation.

Detected outliers:
- `Country Code`: 899
- `Longitude`: 1953
- `Latitude`: 1982
- `Average Cost for two`: 853
- `Price range`: 586
- `Aggregate rating`: 2148
- `Votes`: 1126

## 6. Machine Learning Analysis

- Target column: `Has Table booking`
- Problem type: `classification`
- Training samples: 7640
- Test samples: 1911
- Removed identifier columns: Address

### Model Results

**Logistic Regression**
- accuracy: 0.9006
- precision: 0.8884
- recall: 0.9006
- f1_score: 0.8794
- roc_auc: 0.9392

**Random Forest**
- accuracy: 0.9215
- precision: 0.9146
- recall: 0.9215
- f1_score: 0.9147
- roc_auc: 0.9644

**Gradient Boosting**
- accuracy: 0.9383
- precision: 0.9392
- recall: 0.9383
- f1_score: 0.9387
- roc_auc: 0.9696

Reported best model: **Gradient Boosting**

## 7. Validation and Critic Findings

### Findings
- Missing values detected before cleaning: {'Cuisines': {'count': 9, 'percentage': 0.09}}
- No duplicate rows detected.
- Data cleaning successfully removed remaining missing values.
- Dataset contains 9551 rows.
- Training set contains 7640 samples.
- Test set contains 1911 samples.
- Evaluated 3 machine learning models.
- Best model by F1 score: Gradient Boosting.
- Potential outliers detected in: Country Code: 899, Longitude: 1953, Latitude: 1982, Average Cost for two: 853, Price range: 586, Aggregate rating: 2148, Votes: 1126

### Critic Recommendation

No major validation issues detected.

## 8. Key Findings

1. Missing data was identified during the profiling stage.
2. Several strong statistical relationships were identified during EDA.
3. Multiple machine learning models were evaluated.

## 9. Recommendations

- Treat the reported model metrics as evaluation results on the available dataset rather than proof of generalization.
- Consider additional validation data or cross-validation before drawing stronger conclusions from model performance.
- Investigate the strong feature relationships further to understand whether they reflect meaningful domain patterns or dataset construction.
- Review the critic warnings before using the model for real-world decision making.

## 10. Conclusion

The AI Data Scientist pipeline completed profiling, data quality analysis, cleaning, EDA, visualization, machine learning analysis, and validation. The results should be interpreted in the context of the dataset characteristics and the limitations identified by the critic stage.