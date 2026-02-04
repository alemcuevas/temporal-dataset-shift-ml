# temporal-dataset-shift-ml

Capstone project for my Master's degree. Looking at how ML models degrade over time when the underlying data distribution changes — specifically using FBI crime datasets.

## what this is about

ML models don't stay accurate forever. Data drifts, patterns change, and suddenly your model is making predictions based on outdated assumptions. This project explores whether we can detect that shift *before* the model starts failing.

I'm using two FBI datasets:
- **Crime Type data (2013-2024)** — ~270k records of crime classifications
- **LEOKA (1995-2024)** — Law enforcement officer assault data spanning 30 years

## the questions I'm trying to answer

1. Can we statistically detect when a dataset has shifted enough to matter?
2. Do some detection methods give us earlier warnings than others?
3. Is there a relationship between how much the data shifted and how much accuracy drops?
4. Can we actually predict model failure before it happens?

## approach

For shift detection I'm comparing KS tests, PSI, and Wasserstein distance. Then using RF, gradient boosting, and some ensemble stuff to see if shift metrics can predict when a model is about to tank.

The LEOKA data turned out to be surprisingly stable over 30 years — which is actually an interesting finding on its own.

## repo structure

```
CT_2013_2024/
  CT_2013_2024.csv
  CT_Analysis.ipynb

LEOKA_1995_2024/
  LEOKA_*.csv (multiple files by year range)
  LEOKA_Analysis.ipynb
```

## running it

Need Python 3.8+ with the usual suspects: pandas, numpy, sklearn, scipy, matplotlib, seaborn.

Just open the notebooks and run them. Each one is self-contained.
