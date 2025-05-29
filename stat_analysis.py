#!/usr/bin/env python3
"""
Statistical Analysis Script 
Milestone 4: Statistical comparison using Demšar protocol with Friedman test and Nemenyi post-hoc
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_all_results():
    """Load all experiment results from results/raw/ directory"""
    results_dir = Path('results/raw')
    all_results = []
    
    if not results_dir.exists():
        logger.error(f"Results directory {results_dir} does not exist!")
        return pd.DataFrame()
    
    json_files = list(results_dir.glob('*.json'))
    logger.info(f"Found {len(json_files)} result files")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                result = json.load(f)
            
            # Skip failed experiments
            if 'error' in result:
                logger.warning(f"Skipping failed experiment: {json_file}")
                continue
                
            all_results.append(result)
            
        except Exception as e:
            logger.error(f"Error loading {json_file}: {str(e)}")
    
    if not all_results:
        logger.error("No valid results found!")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_results)
    logger.info(f"Loaded {len(df)} successful experiments")
    return df

def prepare_data_for_analysis(df):
    """Prepare data in the format needed for statistical analysis"""
    # Metrics to analyze
    metrics = ['accuracy', 'auc_ovo', 'cross_entropy', 'total_time']
    
    # Create pivot tables for each metric
    prepared_data = {}
    
    for metric in metrics:
        # Create pivot table: rows=datasets, columns=models, values=metric
        pivot = df.pivot_table(
            index='dataset_name', 
            columns='model_name', 
            values=metric, 
            aggfunc='mean'  # In case of duplicates
        )
        
        # Remove rows/columns with any NaN values
        pivot = pivot.dropna(axis=0, how='any').dropna(axis=1, how='any')
        
        if pivot.empty:
            logger.warning(f"No complete data for metric: {metric}")
            continue
            
        prepared_data[metric] = pivot
        logger.info(f"Prepared data for {metric}: {pivot.shape[0]} datasets x {pivot.shape[1]} models")
    
    return prepared_data

def rank_models_per_dataset(data_dict):
    """Rank models for each dataset and metric"""
    rankings = {}
    
    for metric, pivot in data_dict.items():
        # For accuracy and AUC: higher is better (rank 1 = best)
        # For cross_entropy and time: lower is better (rank 1 = best)
        ascending = metric in ['cross_entropy', 'total_time']
        
        # Rank models per dataset (row-wise ranking)
        ranks = pivot.rank(axis=1, ascending=ascending, method='average')
        rankings[metric] = ranks
        
        logger.info(f"Computed rankings for {metric}")
        logger.info(f"Average ranks:\n{ranks.mean().sort_values()}")
    
    return rankings

def friedman_test(rankings_dict):
    """Perform Friedman test for each metric"""
    friedman_results = {}
    
    for metric, ranks in rankings_dict.items():
        # Prepare data for Friedman test (each column is a treatment/model)
        rank_data = [ranks[col].values for col in ranks.columns]
        
        try:
            # Perform Friedman test
            statistic, p_value = friedmanchisquare(*rank_data)
            
            # Calculate degrees of freedom
            k = len(ranks.columns)  # number of models
            df = k - 1
            
            # Critical value at alpha=0.05
            critical_value = stats.chi2.ppf(0.95, df)
            
            friedman_results[metric] = {
                'statistic': statistic,
                'p_value': p_value,
                'degrees_freedom': df,
                'critical_value': critical_value,
                'significant': p_value < 0.05,
                'models': list(ranks.columns),
                'n_datasets': len(ranks),
                'n_models': len(ranks.columns)
            }
            
            logger.info(f"Friedman test for {metric}:")
            logger.info(f"  Statistic: {statistic:.4f}, p-value: {p_value:.6f}")
            logger.info(f"  Significant: {p_value < 0.05}")
            
        except Exception as e:
            logger.error(f"Error in Friedman test for {metric}: {str(e)}")
            
    return friedman_results

def nemenyi_posthoc(rankings_dict, friedman_results):
    """Perform Nemenyi post-hoc test for metrics with significant Friedman test"""
    nemenyi_results = {}
    
    for metric, ranks in rankings_dict.items():
        if metric not in friedman_results or not friedman_results[metric]['significant']:
            logger.info(f"Skipping Nemenyi test for {metric} (Friedman not significant)")
            continue
            
        try:
            # Perform Nemenyi post-hoc test
            # scikit_posthocs expects data as (n_observations, n_groups)
            data_for_nemenyi = ranks.values
            
            # Perform pairwise Nemenyi test
            nemenyi_matrix = sp.posthoc_nemenyi_friedman(data_for_nemenyi)
            nemenyi_matrix.index = ranks.columns
            nemenyi_matrix.columns = ranks.columns
            
            nemenyi_results[metric] = {
                'pairwise_pvalues': nemenyi_matrix,
                'significant_pairs': []
            }
            
            # Find significantly different pairs (p < 0.05)
            for i, model1 in enumerate(ranks.columns):
                for j, model2 in enumerate(ranks.columns):
                    if i < j:  # Avoid duplicates
                        p_val = nemenyi_matrix.loc[model1, model2]
                        if p_val < 0.05:
                            nemenyi_results[metric]['significant_pairs'].append((model1, model2, p_val))
            
            logger.info(f"Nemenyi post-hoc test for {metric}:")
            logger.info(f"  Found {len(nemenyi_results[metric]['significant_pairs'])} significant pairs")
            
        except Exception as e:
            logger.error(f"Error in Nemenyi test for {metric}: {str(e)}")
    
    return nemenyi_results

def calculate_critical_difference(n_datasets, n_models, alpha=0.05):
    """Calculate critical difference for Nemenyi test"""
    # Nemenyi critical difference formula
    # CD = q_alpha * sqrt(k(k+1)/(6N))
    # where q_alpha is the studentized range statistic
    
    # For alpha=0.05, approximate q values for different k
    q_values = {
        2: 1.960, 3: 2.344, 4: 2.569, 5: 2.728, 6: 2.850,
        7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164
    }
    
    if n_models in q_values:
        q_alpha = q_values[n_models]
    else:
        # Approximation for larger k
        q_alpha = 2.0 + 0.3 * np.log(n_models)
        logger.warning(f"Using approximated q value for {n_models} models: {q_alpha}")
    
    cd = q_alpha * np.sqrt((n_models * (n_models + 1)) / (6 * n_datasets))
    return cd

def create_critical_difference_diagram(rankings_dict, output_dir):
    """Create Critical Difference diagram for each metric"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for metric, ranks in rankings_dict.items():
        try:
            # Calculate average ranks
            avg_ranks = ranks.mean().sort_values()
            n_datasets = len(ranks)
            n_models = len(ranks.columns)
            
            # Calculate critical difference
            cd = calculate_critical_difference(n_datasets, n_models)
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot model ranks
            y_pos = np.arange(len(avg_ranks))
            bars = ax.barh(y_pos, avg_ranks.values, alpha=0.7)
            
            # Customize the plot
            ax.set_yticks(y_pos)
            ax.set_yticklabels(avg_ranks.index, fontsize=10)
            ax.set_xlabel('Average Rank', fontsize=12)
            ax.set_title(f'Critical Difference Diagram - {metric.upper()}\n'
                        f'CD = {cd:.3f} (α=0.05, {n_datasets} datasets, {n_models} models)', 
                        fontsize=14, fontweight='bold')
            
            # Add rank values on bars
            for i, (bar, rank) in enumerate(zip(bars, avg_ranks.values)):
                ax.text(rank + 0.05, bar.get_y() + bar.get_height()/2, 
                       f'{rank:.3f}', va='center', fontsize=9)
            
            # Add critical difference line
            best_rank = avg_ranks.iloc[0]
            cd_threshold = best_rank + cd
            ax.axvline(cd_threshold, color='red', linestyle='--', alpha=0.8, 
                      label=f'CD threshold = {cd_threshold:.3f}')
            
            # Highlight significantly different models
            for i, rank in enumerate(avg_ranks.values):
                if rank - best_rank > cd:
                    bars[i].set_color('lightcoral')
                else:
                    bars[i].set_color('lightblue')
            
            # Add legend
            ax.legend()
            
            # Adjust layout and save
            plt.tight_layout()
            
            # Save the plot
            plot_path = output_dir / f'cd_diagram_{metric}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Critical difference diagram saved: {plot_path}")
            
        except Exception as e:
            logger.error(f"Error creating CD diagram for {metric}: {str(e)}")

def create_summary_table(df, rankings_dict, friedman_results, output_dir):
    """Create summary table with metric averages and ranks"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary_data = []
    
    for metric in rankings_dict.keys():
        pivot = df.pivot_table(
            index='dataset_name', 
            columns='model_name', 
            values=metric, 
            aggfunc='mean'
        ).dropna(axis=0, how='any').dropna(axis=1, how='any')
        
        ranks = rankings_dict[metric]
        
        for model in pivot.columns:
            summary_data.append({
                'metric': metric,
                'model': model,
                'mean_value': pivot[model].mean(),
                'std_value': pivot[model].std(),
                'mean_rank': ranks[model].mean(),
                'std_rank': ranks[model].std(),
                'best_rank': ranks[model].min(),
                'worst_rank': ranks[model].max()
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Add Friedman test results
    friedman_summary = []
    for metric, results in friedman_results.items():
        friedman_summary.append({
            'metric': metric,
            'friedman_statistic': results['statistic'],
            'friedman_pvalue': results['p_value'],
            'friedman_significant': results['significant'],
            'n_datasets': results['n_datasets'],
            'n_models': results['n_models']
        })
    
    friedman_df = pd.DataFrame(friedman_summary)
    
    # Save summary tables
    summary_path = output_dir / 'summary.csv'
    friedman_path = output_dir / 'friedman_results.csv'
    
    summary_df.to_csv(summary_path, index=False)
    friedman_df.to_csv(friedman_path, index=False)
    
    logger.info(f"Summary table saved: {summary_path}")
    logger.info(f"Friedman results saved: {friedman_path}")
    
    return summary_df, friedman_df

def main():
    """Main execution function"""
    logger.info("Starting statistical analysis...")
    
    # Load all experimental results
    df = load_all_results()
    if df.empty:
        logger.error("No results to analyze!")
        return
    
    # Prepare data for analysis
    prepared_data = prepare_data_for_analysis(df)
    if not prepared_data:
        logger.error("Failed to prepare data for analysis!")
        return
    
    # Rank models per dataset for each metric
    rankings = rank_models_per_dataset(prepared_data)
    
    # Perform Friedman test
    friedman_results = friedman_test(rankings)
    
    # Perform Nemenyi post-hoc test
    nemenyi_results = nemenyi_posthoc(rankings, friedman_results)
    
    # Create output directories
    results_dir = Path('results')
    plots_dir = Path('plots')
    
    # Create summary tables
    summary_df, friedman_df = create_summary_table(df, rankings, friedman_results, results_dir)
    
    # Create Critical Difference diagrams
    create_critical_difference_diagram(rankings, plots_dir)
    
    # Print summary
    logger.info("="*60)
    logger.info("STATISTICAL ANALYSIS COMPLETED")
    logger.info("="*60)
    
    for metric, results in friedman_results.items():
        logger.info(f"\n{metric.upper()}:")
        logger.info(f"  Friedman test: χ² = {results['statistic']:.4f}, p = {results['p_value']:.6f}")
        logger.info(f"  Significant differences: {'Yes' if results['significant'] else 'No'}")
        
        if metric in rankings:
            avg_ranks = rankings[metric].mean().sort_values()
            logger.info(f"  Best model: {avg_ranks.index[0]} (rank = {avg_ranks.iloc[0]:.3f})")
            logger.info(f"  Worst model: {avg_ranks.index[-1]} (rank = {avg_ranks.iloc[-1]:.3f})")
    
    logger.info(f"\nFiles generated:")
    logger.info(f"  - results/summary.csv")
    logger.info(f"  - results/friedman_results.csv")
    logger.info(f"  - plots/cd_diagram_*.png")

if __name__ == "__main__":
    main() 