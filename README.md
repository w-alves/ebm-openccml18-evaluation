# Machine Learning Models Comparative Study

**Federal University of Pernambuco (UFPE)** - Center of Informatics  
**Course**: Machine Learning (AM-POS-2025-1)  
**Master's Degree in Computer Science**

## Overview

This project presents a comprehensive comparative analysis of state-of-the-art machine learning models for classification tasks. The study evaluates 6 different AutoML and traditional ML algorithms across 30 diverse datasets from the OpenML-CC18 benchmark suite.

## 📊 Models Evaluated

- **AutoGluon**: Amazon's AutoML framework
- **AutoSklearn**: Automated machine learning with scikit-learn
- **CatBoost**: Gradient boosting with categorical features support
- **EBM (Explainable Boosting Machine)**: Interpretable ML model from Microsoft
- **LightGBM**: Microsoft's gradient boosting framework
- **XGBoost**: Extreme gradient boosting

## 🎯 Key Findings

### Performance Summary
Based on statistical analysis using the Friedman test with post-hoc Nemenyi test:

**Accuracy Rankings** (lower is better):
1. **CatBoost** (2.450) - Best overall accuracy
2. **EBM** (2.900)
3. **AutoSklearn** (3.217)
4. **LightGBM** (3.700)
5. **XGBoost** (4.250)
6. **AutoGluon** (4.483)

**AUC-OVO Rankings**:
1. **CatBoost** (2.350) - Best multi-class performance
2. **EBM** (2.650)
3. **AutoSklearn** (3.017)

**Cross-Entropy Loss Rankings** (tied for best):
1. **CatBoost** (2.400) - Best calibration
1. **EBM** (2.400) - Best calibration

### Statistical Significance
- All performance differences are statistically significant (p < 0.05)
- CatBoost and EBM consistently outperform other models
- AutoSklearn shows competitive results despite being fully automated

## 📁 Project Structure

```
mestrado2/
├── cloud/              # Google Cloud Platform deployment scripts
│   ├── cloud_experiment_runner.py
│   ├── cloud_job_launcher.py
│   ├── gcp_config.py
│   └── deploy_gcp.sh
├── config/             # Configuration files
│   ├── logging.conf
│   └── requirements_*.txt
├── data/               # Dataset storage and processing
│   ├── raw/           # Original datasets
│   ├── split/         # Train/test splits
│   └── processed/     # Preprocessed data
├── docker/            # Containerization
│   ├── Dockerfile
│   └── Dockerfile.py38
├── docs/              # Academic documentation (LaTeX)
├── models/            # Model implementations
│   ├── base_model.py
│   ├── autogluon_model.py
│   ├── autosklearn_model.py
│   ├── catboost_model.py
│   ├── ebm_model.py
│   ├── lgbm_model.py
│   └── xgboost_model.py
├── results/           # Experimental results
│   └── raw/          # Raw JSON results per model/dataset
├── scripts/           # Data processing and analysis scripts
│   ├── fetch_datasets.py
│   ├── preprocess.py
│   ├── split_data.py
│   ├── run_analysis.py
│   └── stat_analysis.py
├── src/               # Core source code
│   ├── experiment_tracker.py
│   ├── monitor_experiments.py
│   └── check_missing_experiments.py
└── stat_results/      # Statistical analysis outputs
    ├── critical_difference_*.png
    ├── performance_*.csv
    └── results.txt
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+ or 3.11
- Required packages (see `config/requirements_*.txt`)

### Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r config/requirements_analysis.txt
   ```

### Running Experiments

1. **Fetch datasets**:
   ```bash
   python scripts/fetch_datasets.py
   ```

2. **Preprocess data**:
   ```bash
   python scripts/preprocess.py
   ```

3. **Split datasets**:
   ```bash
   python scripts/split_data.py
   ```

4. **Run experiments**:
   ```bash
   python scripts/run_analysis.py
   ```

5. **Generate statistical analysis**:
   ```bash
   python scripts/stat_analysis.py
   ```

## 📈 Experimental Design

### Datasets
- **30 datasets** from OpenML-CC18 benchmark suite
- Ranging from 150 to 1,519 samples
- 2-11 classes (binary and multi-class problems)
- 4-856 features

### Evaluation Protocol
- **Stratified train/test split** (70/30)
- **5-fold cross-validation** for hyperparameter tuning
- **Metrics**: Accuracy, AUC-OVO, Cross-entropy loss
- **Time measurements**: Tuning, training, prediction times

### Statistical Analysis
- **Friedman test** for overall significance
- **Nemenyi post-hoc test** for pairwise comparisons
- **Critical difference diagrams** for visualization
- **95% confidence intervals** for performance estimates

## 🔧 Cloud Deployment

The project supports deployment on Google Cloud Platform:

1. Configure GCP credentials
2. Run deployment script:
   ```bash
   bash cloud/deploy_gcp.sh
   ```

See `cloud/README_GCP.md` for detailed instructions.

## 📊 Results Visualization

The project generates comprehensive visualizations:
- Critical difference diagrams
- Performance heatmaps
- Box plots for metric distributions
- Overall ranking charts

## 📝 Academic Context

This study is part of the Machine Learning course (AM-POS-2025-1) at the Federal University of Pernambuco's Center of Informatics. The research follows rigorous statistical methodology for fair comparison of ML algorithms.

### Key Contributions
1. Comprehensive evaluation of modern AutoML systems
2. Statistical significance testing with proper multiple comparison correction
3. Analysis of both performance and computational efficiency
4. Reproducible experimental pipeline with cloud deployment

## 🔍 Future Work

- Extend to regression problems
- Include deep learning models
- Analyze feature importance across models
- Meta-learning for algorithm selection

## 👥 Contributors

This project is developed as part of the Master's program in Computer Science at UFPE.

---

For detailed results and analysis, see the `stat_results/` directory and the generated plots in `plots/`. 