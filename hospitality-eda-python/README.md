# hospitality-eda-python

Exploratory Data Analysis (EDA) and visualization of hotel booking data using Python (pandas, NumPy, matplotlib, seaborn, SciPy). Covers data cleaning, descriptive statistics, outlier detection, correlation analysis, hypothesis tests, and 9 publication-style charts.

## Project structure

```
hospitality-eda-python/
├── generate_data.py      # builds the simulated dataset (seeded, reproducible)
├── eda.py                # full EDA pipeline: clean -> stats -> outliers -> plots
├── requirements.txt
├── data/hotel_bookings.csv
├── figures/              # generated charts (PNG)
└── outputs/              # summary tables, cleaned data, run log
```

## Quick start

```bash
git clone https://github.com/<your-username>/hospitality-eda-python.git
cd hospitality-eda-python
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python generate_data.py   # creates data/hotel_bookings.csv
python eda.py             # prints findings, saves figures/ and outputs/
```

## Dataset

Simulated hotel bookings (6,000 records, 2023-2024, City + Resort hotel) with seasonality, lead-time-driven cancellations and deliberately injected missing values, duplicates and outliers. To use a real dataset (e.g. the Kaggle *Hotel Booking Demand* data), replace `data/hotel_bookings.csv` and adjust column names in `eda.py`.

## Key findings

- Demand and ADR peak in June-August; January-March is the trough.
- Overall cancellation rate is 21%, rising from 11.5% (0-7 day lead) to 37.5% (180+ days).
- Group (34%) and Online TA (26%) bookings cancel most; Corporate (8%) least.
- Guests with special requests cancel less (26% with none vs 11% with 3).
- ADR and lead time are heavily right-skewed; outliers were capped at the 99th percentile.

## Libraries

| Library | Used for |
|---|---|
| pandas / NumPy | loading, cleaning, feature engineering, aggregation |
| matplotlib / seaborn | histograms, box plots, scatter, time series, heatmaps |
| SciPy | z-scores, t-test, chi-square, ANOVA |
