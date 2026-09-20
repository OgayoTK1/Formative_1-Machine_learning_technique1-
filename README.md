# Mobile Network Traffic Forecasting

Comparative analysis of three sequential models (LSTM, Temporal Convolutional Network, Transformer) for one-step-ahead Internet traffic forecasting, using the Milan telecommunications activity dataset. Full methodology, results, and discussion are in `reports/final_report.pdf`; the complete executed pipeline is in `notebooks/01_data_inspection.ipynb`.

**Research question:** How do different sequential models compare for one-step-ahead mobile network traffic forecasting, and how does their performance vary across geographical areas with different traffic characteristics?

**Headline finding:** Model ranking is not stable across areas, and the instability tracks each area's measured volatility. LSTM achieved the strongest results across all three metrics on the steadiest of the three evaluated squares, while TCN achieved the strongest results on the square with the sharpest weekday/weekend split. On the most volatile square, simple persistence achieved the lowest MAE and MAPE. Full results in Section 6 of the report.

## 1. Project Overview

This project investigates whether sequential deep learning architectures with genuinely different mechanisms (recurrence, dilated convolution, self-attention) forecast mobile Internet traffic differently depending on the traffic character of the geographical area, rather than assuming one architecture is universally best. Three squares from the Milan dataset were selected specifically because they differ in total volume, volatility, and weekday/weekend behavior, and the three models were tuned iteratively on the highest-traffic square before the resulting configurations were evaluated on all three.

## 2. Dataset Source

Milan telecommunications activity dataset (SMS, call, and Internet usage, 10-minute resolution, approximately 10,000 geographical squares), from:

G. Barlacchi et al., "A multi-source dataset of urban life in the city of Milan and the Province of Trentino," *Scientific Data*, vol. 2, no. 150055, 2015.

Harvard Dataverse: DOI [10.7910/DVN/EGZHFV](https://doi.org/10.7910/DVN/EGZHFV)

The dataset is not redistributed in this repository (19.38 GB across 62 daily files). `src/data/download_dataverse.py` downloads it programmatically via the Dataverse API; see Section 8 below.

## 3. Dataset Placement

After download, raw files must sit at:

```
data/raw/sms-call-internet-mi-2013-11-01.txt
data/raw/sms-call-internet-mi-2013-11-02.txt
...
data/raw/sms-call-internet-mi-2014-01-01.txt
```

`data/`, `data/raw/`, `data/interim/`, and `data/processed/*.parquet` are excluded from version control via `.gitignore`, since raw and derived data volumes are far too large to commit and are fully reproducible from the steps below.

## 4. Environment Setup

This project was built and run in Google Colab, with Google Drive as persistent storage (Colab's local filesystem does not survive session restarts). To reproduce:

1. Open a new Colab notebook, or use `notebooks/01_data_inspection.ipynb` directly.
2. Mount Google Drive:
   ```python
   from google.colab import drive
drive.mount('/content/drive')
   ```
3. Clone or copy this repository's contents into `/content/drive/MyDrive/mobile-network-forecasting/`.

Running outside Colab (a local machine or another cloud environment) is possible: replace `PROJECT_ROOT` in each script with the local repository path, and skip the Drive-mount step. The code itself has no Colab-specific dependency beyond that path.

## 5. Python Version

Python 3.13.15 (the exact version used throughout this project, captured in `configs/environment_snapshot.json`). Python 3.10+ should work; the code uses no version-specific syntax beyond standard type hints.

## 6. Dependency Installation

```bash
pip install -r requirements.txt
```

Pinned versions (`requirements.txt`) match what was actually used and verified:

```
numpy==2.1.3
pandas==2.2.3
scikit-learn==1.6.1
torch==2.11.0
matplotlib==3.10.0
statsmodels==0.15.0
requests==2.32.4
pyarrow==23.0.1
psutil>=5.9
PyYAML>=6.0
```

## 7. Configuration

All split boundaries, evaluation settings, and the three models' final hyperparameters are centralized in `configs/config.yaml`, rather than scattered across scripts. This is the single source of truth for reproducing the final results; scripts that hardcode these values (from the iterative tuning stage, preserved for the experiment history) are noted as such in `experiments/experiment_log.csv`.

The captured hardware and software environment for this project's actual runs is in `configs/environment_snapshot.json`: 12.67 GB RAM, 2 vCPUs (Intel Xeon @ 2.20 GHz), no GPU, Google Colab.

## 8. How to Run Preprocessing

**Step 1, verify the dataset metadata (no download yet):**

```python
import sys
sys.path.insert(0, 'src/data')
from download_dataverse import get_file_listing
listing = get_file_listing('configs/dataverse_raw_metadata.json')
```

If `configs/dataverse_raw_metadata.json` does not exist yet, query the Dataverse API directly first (see `notebooks/01_data_inspection.ipynb`, Section 2, for the exact metadata request).

**Step 2, download all 62 raw files (resumable, skips already-complete files):**

```python
from download_dataverse import download_all
download_all(listing, out_dir='data/raw', log_path='experiments/download_log.csv')
```

This downloads 19.38 GB total; expect it to take a while depending on network conditions. Already-downloaded, correctly-sized files are skipped automatically if interrupted and re-run.

**Step 3, aggregate all 62 files into compact per-day Parquet files:**

```python
import sys
sys.path.insert(0, 'src/preprocessing')
from aggregate_daily import run_pipeline
run_pipeline(
    raw_dir='data/raw',
    interim_dir='data/interim',
    processed_dir='data/processed',
)
```

This reads each raw file once, keeps only the needed columns with narrow dtypes, aggregates Internet traffic across country code, and writes one Parquet file per day to `data/interim/`, plus `data/processed/total_traffic_per_square.parquet` (total traffic per square across the full period, used to identify the three evaluated squares) and `data/processed/daily_processing_summary.csv` (the memory and timing evidence reported in Section 3 of the report).

## 9. How to Reproduce the EDA

Sections 4 and 4.3 of `notebooks/01_data_inspection.ipynb` reproduce the full exploratory analysis: total traffic distribution, the five-square first-two-week comparison, weekday/weekend and hourly breakdowns, and the ACF/PACF/ADF analysis on square 5161. All figures are saved to `figures/`.

## 10. How to Train Models

The three models' shared training harness is `src/training.py` (function `train_and_evaluate`), used during iterative tuning with early stopping against the validation split. Model classes are in `src/models/lstm_model.py`, `src/models/tcn_model.py`, and `src/models/transformer_model.py`.

Example, training the final LSTM configuration:

```python
import sys
sys.path.insert(0, 'src')
sys.path.insert(0, 'src/preprocessing')
sys.path.insert(0, 'src/models')
from training import train_and_evaluate
from lstm_model import LSTMForecaster

model = LSTMForecaster(hidden_size=64, num_layers=1, dropout=0.0)
result = train_and_evaluate(
    model, square_id=5161, seq_len=24,
    project_root='.', experiment_id='EXP-LSTM-CUSTOM', model_name='LSTM',
    learning_rate=3e-3,
)
```

## 11. How to Run Experiments

The full staged-tuning experiment history (14 experiments across the three models, with the reasoning behind each parameter change) is in `experiments/experiment_log.csv`. To reproduce or extend it, call `train_and_evaluate` with different hyperparameters as shown above; each call appends a new row to the log automatically.

## 12. How to Evaluate Models

Final evaluation (early-stopped training on the original training split, one-shot evaluation on the untouched December 16-22 test period) uses `src/final_evaluation.py`:

```python
import sys
sys.path.insert(0, 'src')
from final_evaluation import train_final_and_test
from lstm_model import LSTMForecaster

model = LSTMForecaster(hidden_size=64, num_layers=1, dropout=0.0)
result = train_final_and_test(
    model, square_id=5161, seq_len=24,
    project_root='.', model_name='LSTM', learning_rate=3e-3,
)
```

This trains with early stopping on train/validation, then evaluates once on the test period, saving predictions to `results/predictions/{model}_{square}.csv` and returning the metrics dict.

To reproduce the full required comparison (all 3 models x 3 squares), see `notebooks/01_data_inspection.ipynb`, Section 8.

## 13. Where Results Are Saved

| Artifact | Location |
|---|---|
| Experiment log (14 tuning experiments) | `experiments/experiment_log.csv` |
| Download log | `experiments/download_log.csv` |
| Per-file memory/timing summary | `data/processed/daily_processing_summary.csv` |
| Total traffic per square | `data/processed/total_traffic_per_square.parquet` |
| Validation-period baseline metrics | `results/tables/baseline_validation_metrics.csv` |
| Test-period baseline metrics | `results/tables/baseline_test_metrics.csv` |
| Final test metrics, all 9 combinations | `results/tables/final_test_metrics.csv` |
| Per-square performance tables | `results/tables/performance_table_square_{id}.csv` |
| Raw predictions, all 9 combinations | `results/predictions/{model}_{square}.csv` |
| Figures | `figures/` (EDA and analysis plots), `figures/individual/` (per-model-per-square prediction plots, if regenerated with `plt.show()`) |
| Environment snapshot | `configs/environment_snapshot.json` |

## 14. Hardware Used

Google Colab, standard runtime: 12.67 GB RAM, 2 vCPUs (Intel Xeon @ 2.20 GHz), no GPU. Captured programmatically at project start in `configs/environment_snapshot.json`, and re-verified at points where the runtime changed unexpectedly (documented in the report, Section 5.5, and in the notebook, Section 8).

## 15. Expected Computational Requirements

Running the full pipeline from scratch, on comparable hardware:

| Stage | Approximate time |
|---|---|
| Download all 62 raw files (19.38 GB) | Depends on network; budget 15-30 minutes |
| Aggregation pipeline (all 62 files) | ~20 minutes |
| LSTM tuning (7 experiments) | ~5 minutes total |
| TCN tuning (4 experiments) | ~9 minutes total |
| Transformer tuning (3 experiments) | ~10 minutes total |
| Final evaluation (9 model/square combinations) | ~20-30 minutes |

Peak memory usage stays under 1.5 GB throughout, well within a 12.67 GB environment; the binding constraint is CPU-only training time for the TCN and Transformer, not memory.

## Repository Structure

```
mobile-network-forecasting/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   ├── config.yaml
│   ├── environment_snapshot.json
│   └── dataverse_raw_metadata.json
├── data/                          (gitignored: raw, interim, processed)
├── notebooks/
│   └── 01_data_inspection.ipynb   (full executed pipeline, with markdown narrative)
├── src/
│   ├── data/
│   │   └── download_dataverse.py
│   ├── preprocessing/
│   │   ├── sequence_pipeline.py
│   │   └── aggregate_daily.py
│   ├── models/
│   │   ├── lstm_model.py
│   │   ├── tcn_model.py
│   │   └── transformer_model.py
│   ├── training.py
│   └── final_evaluation.py
├── experiments/
│   ├── experiment_log.csv
│   ├── download_log.csv
│   └── exp001_lstm_baseline.py    (historical: Experiment 1's original script)
├── results/
│   ├── tables/
│   └── predictions/
├── figures/
│   └── individual/
├── tests/
│   └── test_pipeline.py
└── reports/
    ├── final_report.md
    └── final_report.pdf
```

## Links

Demo video: 
https://www.youtube.com/watch?v=yw-OpzfU_ew
