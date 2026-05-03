# Thesis ML Repository  
**Bachelor’s Thesis:** *Detection of Bots in Old School RuneScape Using Artificial Intelligence and Hiscore Data*  
**Author:** Me  

## Repository Purpose
This repository contains all ML used for my bachelor’s thesis research.  

> [!Warning]
> The input files are tailored to the output format of the [osrs-hiscores-scrape](https://github.com/NotADucc/osrs-hiscores-scrape) package. As a result, the scripts may not be compatible with arbitrary datasets without preprocessing.

# Requirements 
- Python 3.12.x or greater [Download here](https://www.python.org/downloads/) (3.11.x might be fine but not sure, change setup file if you run 3.11.x) 
- Certain py packages, run the command at [Get started](#Getstarted).

# Get started
```console
pip install -r requirements.txt -e .
# or
python -m pip install -r requirements.txt -e .
```

# Main Features
- Machine learning pipeline for anomaly-based bot detection in OSRS hiscore data  
- Extensive feature engineering on player progression, combat, skill, and bossing behavior  
- Support for multiple unsupervised anomaly detection models (e.g., Isolation Forest, LOF, One-Class SVM, DBSCAN)  
- Automated hyperparameter evaluation with summary metrics and ranking-based scoring  
- Configurable model experimentation via parameter grids (`settings.json`)  

# Misc Features
- UI to view model results
- Labeling utility for mapping known usernames to dataset records  
- Efficiency metric computation (EHP/EHB) between time snapshots  
- Correlation analysis notebook for feature redundancy inspection  
- DBSCAN k-distance analysis notebook for clustering parameter tuning  

  
# Usage

## run_models.py

Runs anomaly detection models on a hiscore dataset with feature engineering and optional automated evaluation.

```console
py .\scripts\run_models.py --in-file input.txt --out-dir results --settings-file .\settings\settings.json
```
| Argument           | Required | Description |
|-------------------|----------|-------------|
| `--in-file`        | Yes      | Path to the hiscore data |
| `--out-dir`        | Yes      | Directory where results will be written |
| `--settings-file`  | Yes      | Path to the JSON file containing model parameters and combinations |
| `--mode`      | Yes      | Output mode: `full` (per-model CSV output) or `summary` (aggregated evaluation results) |

## apply_labeling.py

Applies label scoring to a JSON dataset based on a provided username list.

```console
py .\scripts\apply_labeling.py --in-file input.txt --users-file users.txt --score 1
```
| Argument      | Required | Description                                |
| ------------- | -------- | ------------------------------------------ |
| `--in-file`  | Yes      | Path to the input JSON file. Each line must be a JSON object containing a record.username field. |
| `--users-file`  | Yes      | Path to the users file  |
| `--score`  | Yes      | Numeric label assigned to matching users |


## calc_metrics.py

Computes efficiency metrics between two hiscore snapshots (e.g., before/after).  
Supports both **EHP (Efficient Hours Played)** and **EHB (Efficient Hours Bossed)** calculations depending on the selected hiscore category.

```console
py .\scripts\calc_metrics.py --before-file before.txt --after-file after.txt --out-file output.txt --hs-type zuk
```
| Argument         | Required | Description |
|------------------|----------|-------------|
| `--before-file`  | Yes      | Path to the older snapshot file |
| `--after-file`   | Yes      | Path to the newer snapshot file |
| `--out-file`     | Yes      | Path to the output file |
| `--hs-type`      | Yes      | Hiscore category used to compute efficiency (determines EHP vs EHB) |

## ui.py

```console
py .\scripts\ui.py 
```

### Configuration

To customize the MongoDB connection and database name, create a `.env` file in the root of the repository.

The following variables can be defined:

```env
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB=ml_models_db
```

If no `.env` file is provided, the application will fall back to default values (a local MongoDB instance).


## feature_correlation.ipynb

Analyzes correlations between engineered features to identify redundancy and highly correlated variables in the dataset.
It produces a correlation matrix and optional heatmap visualization of feature dependencies.

To customize the in file, create a `.env` file in the root of the repository.

The following variables can be defined:

```env
NOTEBOOK_IN_FILE=url
```

If no `.env` file is provided, the fallback variable is `BACKUP_IN_FILE` which can be set in the notebook.

## dbscan_distance_plot.ipynb

Analyzes feature space density to help tune DBSCAN hyperparameters, specifically the `eps` value.

To customize the in file, create a `.env` file in the root of the repository.

The following variables can be defined:

```env
NOTEBOOK_IN_FILE=url
```

If no `.env` file is provided, the fallback variable is `BACKUP_IN_FILE` which can be set in the notebook.

# Logging
Several log messages and progressbar is used to report progress.
