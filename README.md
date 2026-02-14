# Temporal Dataset Shift Detection in Public-Sector Crime Data

Master's Capstone Project — Empirical analysis of how machine learning model performance degrades over time as underlying data distributions change, using FBI crime datasets.

---

## Motivation

Machine learning models deployed on real-world data don't stay accurate forever. As data distributions evolve due to policy changes, societal shifts, or reporting methodology updates, model predictions can silently degrade. This project investigates whether temporal dataset shift can be detected *before* significant model performance loss occurs, using publicly available FBI crime data as a testbed.

## Research Questions

| # | Question |
|---|----------|
| **RQ1** | How can temporal dataset shift be statistically detected in public-sector time-series data? |
| **RQ2** | Which detection methods provide the earliest warning signals prior to model degradation? |
| **RQ3** | What is the relationship between shift metrics and ML model performance loss? |
| **RQ4** | Can shift indicators predict impending model failure? |

## Datasets

| Dataset | Records | Timespan | Source |
|---------|---------|----------|--------|
| **Crime Type (CT)** | ~272,000 | 2012–2024 (13 years) | [FBI Crime Data Explorer](https://cde.ucr.cjis.gov/) |
| **LEOKA** (Law Enforcement Officers Killed & Assaulted) | ~362,000 (assault) + assignment activity files | 1995–2024 (30 years) | [FBI Crime Data Explorer](https://cde.ucr.cjis.gov/) |

## Methods

**Shift Detection**
- Kolmogorov-Smirnov (KS) tests with Benjamini-Hochberg FDR correction
- Population Stability Index (PSI)
- Wasserstein distance
- Chi-square tests with Cramér's V for categorical features
- ADF/KPSS stationarity testing
- CUSUM structural break detection

**Predictive Modeling**
- Random Forest, Gradient Boosting, Logistic Regression
- Voting Ensemble with bootstrap resampling
- Leave-One-Out and Repeated Stratified K-Fold cross-validation
- Cohen's d effect sizes and bootstrap confidence intervals

## Key Findings

- **CT dataset**: Statistically significant distributional shifts detected across multiple years, with measurable correlation between shift magnitude and model accuracy degradation.
- **LEOKA dataset**: Remarkably stable over 30 years — models maintained consistent performance across decades. This temporal stability is itself a substantive finding about law enforcement assault reporting patterns.
- Weapon data in the CT dataset is ~92% missing, requiring careful interpretation of weapon-related analyses.

## Repository Structure

```
├── README.md
├── CT_2013_2024/
│   ├── CT_2013_2024.csv              # Raw dataset
│   ├── CT_2013_2024_cleaned.csv      # Cleaned dataset
│   ├── CT_Analysis.ipynb             # Full analysis notebook
│   └── CT_Analysis_Summary.csv       # Summary results table
│
└── LEOKA_1995_2024/
    ├── LEOKA_ASSAULT_TIME_WEAPON_INJURY_1995_2024.csv
    ├── LEOKA_ASSIGNMENT_ACTIVITY_*.csv  # Multiple files by year range
    ├── LEOKA_ASSAULT_cleaned.csv
    ├── LEOKA_ASSIGNMENT_ACTIVITY_combined_cleaned.csv
    ├── LEOKA_Analysis.ipynb             # Full analysis notebook
    └── LEOKA_Analysis_Summary.csv       # Summary results table
```

## Requirements

Python 3.8+ with the following packages:

```
pandas
numpy
scikit-learn
scipy
statsmodels
matplotlib
seaborn
```

Each notebook is self-contained — open and run top to bottom.

## References

- Gama et al. (2014). A survey on concept drift adaptation. *ACM Computing Surveys*.
- Lu et al. (2019). Learning under concept drift: A review. *IEEE TKDE*.
- Webb et al. (2016). Characterizing concept drift. *Data Mining and Knowledge Discovery*.
- Rabanser et al. (2019). Failing loudly: An empirical study of methods for detecting dataset shift. *NeurIPS*.
