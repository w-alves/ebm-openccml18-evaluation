# Experimental Workflow - Milestones 3 & 4

This document explains how to execute the automated experiment pipeline and statistical analysis.

## Prerequisites

Ensure your environment is set up with all required dependencies:
```bash
# Activate your environment (if using conda/virtualenv)
# conda activate your_env_name

# Verify required packages are installed
pip list | grep -E "(scikit-posthocs|scipy|matplotlib|pandas|numpy)"
```

## Milestone 3: Automated Experiment Execution

### Overview
The `run_experiments.py` script automatically:
1. Discovers all datasets in `data/split/`
2. Instantiates all available models
3. For each model-dataset combination:
   - Tunes hyperparameters using 5-fold CV
   - Trains the model with best parameters
   - Tests on hold-out data
   - Evaluates performance metrics
4. Saves results to `results/raw/`

### Execution
```bash
python run_experiments.py
```

### Expected Output
- **Log file**: `experiments.log` with detailed progress
- **Results**: Individual JSON files in `results/raw/` with format `{model}_{dataset}.json`
- **Metrics calculated**: 
  - Mean AUC (One-vs-One for multiclass)
  - Mean Accuracy (ACC)
  - Mean Cross-Entropy (CE)  
  - Mean time (seconds): tune + train + predict

### Example Output Structure
```
results/raw/
├── EBM_6332_cylinder-bands.json
├── LGBM_6332_cylinder-bands.json
├── CatBoost_6332_cylinder-bands.json
├── XGBoost_6332_cylinder-bands.json
├── AutoSklearn_6332_cylinder-bands.json
├── AutoGluon_6332_cylinder-bands.json
├── EBM_23381_dresses-sales.json
└── ... (all model-dataset combinations)
```

## Milestone 4: Statistical Comparison (Demšar Protocol)

### Overview
The `stat_analysis.py` script implements the Demšar protocol:
1. Loads all experiment results from `results/raw/`
2. Ranks models per dataset for each metric
3. Performs Friedman test to detect significant differences
4. Conducts Nemenyi post-hoc test for pairwise comparisons
5. Generates Critical Difference (CD) diagrams
6. Creates summary tables

### Execution
```bash
python stat_analysis.py
```

**Note**: Must be run AFTER `run_experiments.py` completes successfully.

### Expected Output

#### Files Generated
1. **`results/summary.csv`**: Comprehensive summary with:
   - Model performance means and standard deviations
   - Average ranks per metric
   - Best/worst ranks

2. **`results/friedman_results.csv`**: Friedman test statistics:
   - Test statistics and p-values
   - Significance indicators
   - Dataset and model counts

3. **`plots/cd_diagram_accuracy.png`**: Critical Difference diagram for accuracy
4. **`plots/cd_diagram_auc_ovo.png`**: Critical Difference diagram for AUC (One-vs-One)
5. **`plots/cd_diagram_cross_entropy.png`**: Critical Difference diagram for cross-entropy
6. **`plots/cd_diagram_total_time.png`**: Critical Difference diagram for total time

#### Console Output
- Friedman test results for each metric
- Best and worst performing models per metric
- Significance indicators

## Understanding the Results

### Critical Difference Diagrams
- **X-axis**: Average rank (lower = better for accuracy/AUC, higher = better for CE/time)
- **Blue bars**: Models not significantly different from the best
- **Red bars**: Models significantly worse than the best
- **Red dashed line**: Critical difference threshold

### Statistical Interpretation
1. **Friedman test p < 0.05**: Significant differences exist between models
2. **Nemenyi post-hoc**: Identifies which specific model pairs differ significantly
3. **Critical Difference**: Models within CD of the best are not significantly different

## Troubleshooting

### Common Issues

1. **No results found**:
   - Ensure `run_experiments.py` completed successfully
   - Check `results/raw/` directory exists and contains JSON files

2. **Missing models in analysis**:
   - Some models may have failed during experiments
   - Check `experiments.log` for error messages

3. **Incomplete data for metrics**:
   - Models without successful runs on all datasets are excluded
   - Review individual result files for errors

4. **Memory issues with large experiments**:
   - AutoGluon and AutoSklearn can be memory-intensive
   - Consider running on smaller dataset subsets for testing

### Customization

#### Modifying Evaluation Metrics
Edit the `metrics` list in `stat_analysis.py`:
```python
metrics = ['accuracy', 'auc_ovo', 'cross_entropy', 'total_time']
```

#### Changing CV Folds
Modify the `cv_folds` parameter in `run_experiments.py`:
```python
model.tune(X_train, y_train, cv_folds=3)  # Default: 5
```

#### Adding Models
1. Implement new model class inheriting from `BaseModel`
2. Add import to `run_experiments.py`
3. Include in `get_model_instances()` function

## Expected Runtime

- **Small datasets (3 datasets × 6 models)**: ~30-60 minutes
- **Runtime varies significantly by model**:
  - EBM, LGBM, CatBoost, XGBoost: Fast (1-5 min per dataset)
  - AutoSklearn: Medium (5-15 min per dataset)  
  - AutoGluon: Slow (10-30 min per dataset)

## Verification

After completion, verify results:
```bash
# Check number of result files
ls results/raw/*.json | wc -l

# Check summary files exist
ls results/summary.csv results/friedman_results.csv

# Check CD diagrams exist  
ls plots/cd_diagram_*.png
```

Expected: 18 JSON files (3 datasets × 6 models) + summary files + 4 CD diagrams 