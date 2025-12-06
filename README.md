environment:
  python: 3.10
  operating_system: ubuntu-latest
  dependencies:
    - pandas
    - numpy
    - matplotlib
    - scikit-learn
    - statsmodels
    - pmdarima
    - prophet

entry_point:
  script: src/forecast.py
  input_data: data/Month_Value_1.csv
  output:
    - outputs/forecast_comparison.png
    - outputs/metrics.csv

workflow:
  steps:
    - load_time_series_data
    - train_test_split (time-based)
    - auto_sarima_modeling
    - prophet_modeling
    - forecast_generation
    - model_evaluation
    - visualization
