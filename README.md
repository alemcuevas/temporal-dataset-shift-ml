# Temporal Dataset Shift Detection in Public-Sector ML

Master's capstone project analyzing temporal dataset shift in public-sector crime data.

## Research Questions

1. How can temporal dataset shift be statistically detected in public-sector time-series data?
2. Which detection methods provide the earliest warning signals prior to model degradation?
3. What is the relationship between shift metrics and ML model performance loss?
4. Can shift indicators predict impending model failure?

## Datasets

- **CT (Crime Type) 2013-2024**: Crime classification data
- **LEOKA 1995-2024**: Law Enforcement Officers Killed and Assaulted data

## Methods

- Kolmogorov-Smirnov Test
- Population Stability Index (PSI)
- Wasserstein Distance
- Random Forest / Gradient Boosting / Ensemble models

## Structure

```
├── CT_2013_2024/
│   ├── CT_2013_2024.csv
│   └── CT_Analysis.ipynb
├── LEOKA_1995_2024/
│   ├── LEOKA_*.csv
│   └── LEOKA_Analysis.ipynb
```

## Requirements

- Python 3.8+
- pandas, numpy, matplotlib, seaborn
- scikit-learn, scipy

## Usage

Open the Jupyter notebooks in each folder and run all cells.
