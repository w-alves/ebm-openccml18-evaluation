import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.stats import friedmanchisquare, t
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)

# Show all rows
pd.set_option('display.max_rows', None)

# Optional: Make sure the output doesn't get truncated horizontally
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)  # For older pandas versions

# Try to import aeon for critical difference plots
try:
    from aeon.visualisation import plot_critical_difference
    aeon_available = True
except ImportError:
    aeon_available = False
    print("⚠️  aeon not available - critical difference plots will be skipped")

from gcp_storage import GCPStorageManager

class ExperimentAnalyzer:
    """Analyzes ML experiment results with statistical testing"""
    
    def __init__(self, storage_manager: GCPStorageManager):
        self.storage_manager = storage_manager
        self.results_data = {}
        self.performance_matrix = {}
        self.metrics = ['accuracy', 'auc_ovo', 'cross_entropy', 'tuning_time', 
                       'training_time', 'prediction_time', 'total_time']
        self.results_dir = Path("stat_results")
        self.results_dir.mkdir(exist_ok=True)
        
    def download_results(self, results_prefix: str = "results/raw/") -> None:
        """Download all result files from GCP bucket"""
        print(f"📥 Downloading results from gs://{self.storage_manager.config.bucket_name}/{results_prefix}")
        
        # List all result files
        result_files = self.storage_manager.list_files(results_prefix)
        json_files = [f for f in result_files if f.endswith('.json')]
        
        print(f"🔍 Found {len(json_files)} result files")
        
        # Download and parse each file
        for file_path in json_files:
            try:
                result_data = self.storage_manager.download_json(file_path)
                
                model_name = result_data['model_name']
                dataset_name = result_data['dataset_name']
                
                # Create nested structure
                if model_name not in self.results_data:
                    self.results_data[model_name] = {}
                
                self.results_data[model_name][dataset_name] = result_data
                
            except Exception as e:
                print(f"❌ Error processing {file_path}: {e}")
        
        print(f"✅ Successfully downloaded results for {len(self.results_data)} models")
        
    def extract_best_performance(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Extract best performance for each model+dataset combination"""
        print("🔍 Extracting best performance from cross-validation results...")
        
        processed_results = {}
        
        for model_name, datasets in self.results_data.items():
            processed_results[model_name] = {}
            
            for dataset_name, result_data in datasets.items():
                # Get cross-validation results
                cv_results = result_data.get('cross_validation_results', [])
                
                if not cv_results:
                    print(f"⚠️  No CV results for {model_name} on {dataset_name}")
                    continue
                
                # Find best parameter combination based on mean accuracy
                best_combination = None
                best_accuracy = -1
                
                for combination in cv_results:
                    mean_acc = combination.get('mean_accuracy', 0)
                    if mean_acc > best_accuracy:
                        best_accuracy = mean_acc
                        best_combination = combination
                
                if best_combination:
                    # Extract mean values for all metrics
                    performance = {}
                    for metric in self.metrics:
                        if metric == 'accuracy':
                            performance[metric] = best_combination.get('mean_accuracy', 0)
                        elif metric == 'auc_ovo':
                            performance[metric] = best_combination.get('mean_auc_ovo', 0)
                        elif metric == 'cross_entropy':
                            performance[metric] = best_combination.get('mean_cross_entropy', 0)
                        elif metric == 'tuning_time':
                            performance[metric] = best_combination.get('tuning_time', 0)
                        elif metric == 'training_time':
                            performance[metric] = best_combination.get('mean_training_time', 0)
                        elif metric == 'prediction_time':
                            performance[metric] = best_combination.get('mean_prediction_time', 0)
                        elif metric == 'total_time':
                            performance[metric] = best_combination.get('mean_total_time', 0)
                    
                    processed_results[model_name][dataset_name] = performance
        
        return processed_results
    
    def perform_data_sanity_check(self, processed_results: Dict) -> None:
        """Perform sanity checks on the downloaded data"""
        print("\n🔍 Performing data sanity checks...")
        
        total_combinations = 0
        nan_issues = {}
        missing_metrics = {}
        
        for model_name, datasets in processed_results.items():
            for dataset_name, performance in datasets.items():
                total_combinations += 1
                
                # Check each metric for NaN values
                for metric in self.metrics:
                    if metric not in performance:
                        if metric not in missing_metrics:
                            missing_metrics[metric] = []
                        missing_metrics[metric].append(f"{model_name}+{dataset_name}")
                    elif pd.isna(performance[metric]) or performance[metric] is None:
                        if metric not in nan_issues:
                            nan_issues[metric] = []
                        nan_issues[metric].append(f"{model_name}+{dataset_name}")
        
        print(f"📊 Total model+dataset combinations: {total_combinations}")
        
        # Report missing metrics
        if missing_metrics:
            print("\n⚠️  MISSING METRICS DETECTED:")
            for metric, combinations in missing_metrics.items():
                print(f"  {metric}: {len(combinations)} combinations missing")
                for combo in combinations[:5]:  # Show first 5
                    print(f"    - {combo}")
                if len(combinations) > 5:
                    print(f"    - ... and {len(combinations) - 5} more")
        
        # Report NaN values
        if nan_issues:
            print("\n❌ NaN VALUES DETECTED:")
            for metric, combinations in nan_issues.items():
                print(f"  {metric}: {len(combinations)} combinations have NaN values")
                for combo in combinations[:5]:  # Show first 5
                    print(f"    - {combo}")
                if len(combinations) > 5:
                    print(f"    - ... and {len(combinations) - 5} more")
        
        # Data completeness summary
        print(f"\n📈 DATA COMPLETENESS SUMMARY:")
        for metric in self.metrics:
            valid_count = 0
            for model_name, datasets in processed_results.items():
                for dataset_name, performance in datasets.items():
                    if metric in performance and not pd.isna(performance[metric]) and performance[metric] is not None:
                        valid_count += 1
            
            completeness = (valid_count / total_combinations) * 100 if total_combinations > 0 else 0
            status = "✅" if completeness >= 90 else "⚠️" if completeness >= 70 else "❌"
            print(f"  {metric}: {valid_count}/{total_combinations} ({completeness:.1f}%) {status}")
        
        # Overall assessment
        if not nan_issues and not missing_metrics:
            print("\n✅ All data looks good! No NaN values or missing metrics detected.")
        else:
            print(f"\n⚠️  Data quality issues detected. Consider investigating the problematic combinations.")
            
        # Save sanity check report
        sanity_report = {
            'total_combinations': total_combinations,
            'missing_metrics': missing_metrics,
            'nan_issues': nan_issues,
            'completeness': {}
        }
        
        for metric in self.metrics:
            valid_count = 0
            for model_name, datasets in processed_results.items():
                for dataset_name, performance in datasets.items():
                    if metric in performance and not pd.isna(performance[metric]) and performance[metric] is not None:
                        valid_count += 1
            sanity_report['completeness'][metric] = {
                'valid_count': valid_count,
                'total_count': total_combinations,
                'percentage': (valid_count / total_combinations) * 100 if total_combinations > 0 else 0
            }
        
        # Save detailed sanity check report
        sanity_file = self.results_dir / "data_sanity_check_report.json"
        with open(sanity_file, 'w') as f:
            json.dump(sanity_report, f, indent=2)
        print(f"📄 Detailed sanity check report saved to: {sanity_file}")
    
    def create_performance_matrices(self, processed_results: Dict) -> None:
        """Create performance matrices for each metric"""
        print("📊 Creating performance matrices...")
        
        # Get all unique models and datasets
        all_models = list(processed_results.keys())
        all_datasets = set()
        
        for model_data in processed_results.values():
            all_datasets.update(model_data.keys())
        
        all_datasets = sorted(list(all_datasets))
        
        print(f"📈 Found {len(all_models)} models and {len(all_datasets)} datasets")
        
        # Create matrix for each metric
        for metric in self.metrics:
            matrix_data = []
            
            for dataset in all_datasets:
                row = []
                for model in all_models:
                    if dataset in processed_results[model]:
                        value = processed_results[model][dataset][metric]
                        row.append(value)
                    else:
                        row.append(np.nan)  # Missing data
                matrix_data.append(row)
            
            # Create DataFrame
            df = pd.DataFrame(matrix_data, columns=all_models, index=all_datasets)
            
            # Remove rows/columns with too many missing values
            df = df.dropna(thresh=len(all_models) * 0.5)  # Keep rows with at least 50% data
            df = df.dropna(axis=1, thresh=len(df) * 0.5)  # Keep columns with at least 50% data
            
            self.performance_matrix[metric] = df
        
        print("✅ Performance matrices created")
    
    def calculate_confidence_interval(self, data: pd.Series, confidence: float = 0.95) -> Tuple[float, float, float]:
        """Calculate mean and 95% confidence interval for a data series"""
        n = len(data.dropna())
        if n < 2:
            return data.mean(), np.nan, np.nan
        
        mean = data.mean()
        std_err = data.std() / np.sqrt(n)
        
        # Use t-distribution for small samples, normal for large
        if n < 30:
            t_val = t.ppf((1 + confidence) / 2, n - 1)
        else:
            t_val = 1.96  # z-score for 95% CI
        
        margin_error = t_val * std_err
        ci_lower = mean - margin_error
        ci_upper = mean + margin_error
        
        return mean, ci_lower, ci_upper
    
    def perform_statistical_analysis(self) -> None:
        """Perform statistical analysis for each metric"""
        print("\n" + "="*80)
        print("STATISTICAL ANALYSIS OF ML EXPERIMENT RESULTS")
        print("="*80)
        
        # Metrics that should be minimized (lower is better)
        minimize_metrics = ['cross_entropy', 'tuning_time', 'training_time', 'prediction_time', 'total_time']
        
        for metric in self.metrics:
            if metric not in self.performance_matrix:
                continue
                
            df_performance = self.performance_matrix[metric]
            
            # Skip if insufficient data
            if df_performance.empty or df_performance.shape[1] < 2:
                print(f"⚠️  Insufficient data for {metric} analysis")
                continue
            
            print(f"\n{'='*60}")
            print(f"STATISTICAL ANALYSIS FOR {metric.upper()}")
            print(f"{'='*60}")
            
            # Remove rows with any NaN values for statistical tests
            df_clean = df_performance.dropna()
            
            if df_clean.empty:
                print(f"⚠️  No complete data for {metric}")
                continue
            
            # Create ranking matrix
            ascending = metric in minimize_metrics
            df_rank = df_clean.rank(axis=1, method='average', ascending=ascending)
            
            print(f"\nPerformance Summary for {metric}:")
            summary_stats = df_clean.describe()
            print(summary_stats)
            
            # Add confidence intervals to summary
            print(f"\nPerformance Summary with 95% CI for {metric}:")
            for model in df_clean.columns:
                mean_val, ci_lower, ci_upper = self.calculate_confidence_interval(df_clean[model])
                print(f"  {model}: {mean_val:.4f} [95% CI: {ci_lower:.4f} - {ci_upper:.4f}]")
            
            print(f"\nRanking Summary for {metric} (average ranks):")
            mean_ranks = df_rank.mean().sort_values()
            for model_name, rank in mean_ranks.items():
                print(f"  {model_name}: {rank:.3f}")
            
            # Friedman Test (if we have enough data)
            if len(df_clean) >= 3 and len(df_clean.columns) >= 3:
                try:
                    friedman_stat, p_value = friedmanchisquare(*df_rank.T.values)
                    print(f"\nFriedman test results:")
                    print(f"  Statistic: {friedman_stat:.4f}")
                    print(f"  p-value: {p_value:.6f}")
                    
                    if p_value < 0.05:
                        print(f"  Result: Significant differences detected (p < 0.05)")
                        
                        # Critical Difference Plot
                        if aeon_available:
                            try:
                                plt.figure(figsize=(16, 12))
                                
                                plot_critical_difference(
                                    df_clean.values,
                                    df_clean.columns.tolist(),
                                    lower_better=ascending,
                                    test='nemenyi',
                                    correction='none',
                                    alpha=0.05,
                                )
                                
                                # Get the current axes and adjust formatting
                                ax = plt.gca()
                                
                                # Adjust font size and rotation of x-axis labels
                                for label in ax.get_xticklabels():
                                    label.set_fontsize(14)
                                    label.set_rotation(45)
                                    label.set_horizontalalignment('right')
                                
                                # Increase padding between labels and axis
                                ax.tick_params(axis='x', which='major', pad=20)
                                
                                # Adjust margins to provide more space for labels
                                plt.subplots_adjust(bottom=0.35)
                                
                                # Optionally adjust y-axis label font size
                                ax.tick_params(axis='y', labelsize=12)
                                
                                plt.title(f'Critical Difference Diagram - {metric.upper()}', fontsize=16, pad=20)
                                
                                # Save the plot
                                plot_filename = self.results_dir / f'critical_difference_{metric}.png'
                                plt.savefig(plot_filename, format="png", bbox_inches="tight", dpi=500)
                                #plt.show()
                                
                                print(f"  Critical difference plot saved as: {plot_filename}")
                                
                            except Exception as e:
                                print(f"  Error creating critical difference plot: {e}")
                        else:
                            print(f"  Skipping critical difference plot (aeon not available)")
                            
                    else:
                        print(f"  Result: No significant differences detected (p >= 0.05)")
                        
                except Exception as e:
                    print(f"Error during Friedman test: {e}")
            else:
                print(f"⚠️  Insufficient data for Friedman test (need at least 3 datasets and 3 models)")
            
            # Save detailed results
            performance_file = self.results_dir / f"performance_scores_{metric}.csv"
            ranking_file = self.results_dir / f"performance_ranks_{metric}.csv"
            df_clean.to_csv(performance_file)
            df_rank.to_csv(ranking_file)
            print(f"  Performance data saved to: {performance_file}")
            print(f"  Ranking data saved to: {ranking_file}")
    
    def create_visualizations(self) -> None:
        """Create various visualizations"""
        print("\n📊 Creating visualizations...")
        
        # Performance heatmaps
        for metric in self.metrics:
            if metric not in self.performance_matrix:
                continue
                
            df = self.performance_matrix[metric]
            if df.empty:
                continue
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(df, annot=True, fmt='.4f', cbar_kws={'label': metric})
            plt.title(f'Performance Heatmap - {metric.upper()}')
            plt.xlabel('Models')
            plt.ylabel('Datasets')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()
            heatmap_file = self.results_dir / f'heatmap_{metric}.png'
            plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
            # plt.show()
        
        # Boxplots comparing models
        for metric in ['accuracy', 'auc_ovo', 'cross_entropy']:
            if metric not in self.performance_matrix:
                continue
                
            df = self.performance_matrix[metric]
            if df.empty:
                continue
            
            plt.figure(figsize=(12, 6))
            df.boxplot()
            plt.title(f'Model Performance Distribution - {metric.upper()}')
            plt.xlabel('Models')
            plt.ylabel(metric.replace('_', ' ').title())
            plt.xticks(rotation=45)
            plt.tight_layout()
            boxplot_file = self.results_dir / f'boxplot_{metric}.png'
            plt.savefig(boxplot_file, dpi=300, bbox_inches='tight')
            # plt.show()
    
    def generate_summary_report(self) -> None:
        """Generate comprehensive summary report"""
        print(f"\n{'='*60}")
        print("OVERALL PERFORMANCE SUMMARY")
        print(f"{'='*60}")
        
        # Create summary table with mean performance and ranks across all metrics
        summary_performance = {}
        summary_ranks = {}
        
        minimize_metrics = ['cross_entropy', 'tuning_time', 'training_time', 'prediction_time', 'total_time']
        
        for metric in self.metrics:
            if metric not in self.performance_matrix:
                continue
                
            df = self.performance_matrix[metric].dropna()
            if df.empty:
                continue
            
            # Get mean performance for each model
            mean_performance = df.mean()
            summary_performance[metric] = mean_performance.to_dict()
            
            # Calculate ranks
            ascending = metric in minimize_metrics
            ranks = mean_performance.rank(ascending=ascending)
            summary_ranks[metric] = ranks.to_dict()
        
        # Create summary tables
        if summary_performance:
            performance_summary_df = pd.DataFrame(summary_performance).T
            ranks_summary_df = pd.DataFrame(summary_ranks).T
            
            # Create confidence interval summary
            ci_summary = {}
            for metric in self.metrics:
                if metric not in self.performance_matrix:
                    continue
                df = self.performance_matrix[metric].dropna()
                if df.empty:
                    continue
                    
                ci_summary[metric] = {}
                for model in df.columns:
                    mean_val, ci_lower, ci_upper = self.calculate_confidence_interval(df[model])
                    ci_summary[metric][model] = f"{mean_val:.4f} [{ci_lower:.4f}-{ci_upper:.4f}]"
            
            ci_summary_df = pd.DataFrame(ci_summary).T
            
            print("\nMean Performance Across All Metrics:")
            print(performance_summary_df.round(4))
            
            print("\nMean Performance with 95% CI Across All Metrics:")
            print(ci_summary_df)
            
            print("\nAverage Rankings Across All Metrics:")
            print(ranks_summary_df.round(2))
            
            # Calculate overall average rank
            overall_ranks = ranks_summary_df.mean().sort_values()
            print("\nOverall Average Ranking (lower is better):")
            for model_name, rank in overall_ranks.items():
                print(f"  {model_name}: {rank:.2f}")
            
            # Save summary results
            performance_file = self.results_dir / "performance_summary_all_metrics.csv"
            ci_file = self.results_dir / "performance_summary_with_ci.csv"
            ranks_file = self.results_dir / "ranks_summary_all_metrics.csv"
            overall_ranks_file = self.results_dir / "overall_average_ranks.csv"
            
            performance_summary_df.to_csv(performance_file)
            ci_summary_df.to_csv(ci_file)
            ranks_summary_df.to_csv(ranks_file)
            overall_ranks.to_csv(overall_ranks_file, header=['Average_Rank'])
            
            print(f"\nSummary files saved:")
            print(f"  {performance_file}")
            print(f"  {ci_file}")
            print(f"  {ranks_file}")
            print(f"  {overall_ranks_file}")
            
            # Create overall ranking visualization
            plt.figure(figsize=(10, 6))
            overall_ranks.plot(kind='bar')
            plt.title('Overall Average Ranking Across All Metrics')
            plt.xlabel('Models')
            plt.ylabel('Average Rank (lower is better)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            ranking_plot_file = self.results_dir / 'overall_ranking.png'
            plt.savefig(ranking_plot_file, dpi=300, bbox_inches='tight')
            #plt.show()
        
        # Dataset-wise analysis
        print(f"\n{'='*60}")
        print("DATASET-WISE ANALYSIS")
        print(f"{'='*60}")
        
        if 'accuracy' in self.performance_matrix:
            df_acc = self.performance_matrix['accuracy']
            if not df_acc.empty:
                # Best model per dataset
                best_per_dataset = df_acc.idxmax(axis=1)
                print("\nBest model per dataset (based on accuracy):")
                for dataset, best_model in best_per_dataset.items():
                    accuracy = df_acc.loc[dataset, best_model]
                    print(f"  {dataset}: {best_model} ({accuracy:.4f})")
                
                # Win counts
                win_counts = best_per_dataset.value_counts()
                print(f"\nWin counts across datasets:")
                for model, wins in win_counts.items():
                    print(f"  {model}: {wins} wins")
                
                # Save dataset analysis
                best_dataset_file = self.results_dir / "best_model_per_dataset.csv"
                win_counts_file = self.results_dir / "model_win_counts.csv"
                best_per_dataset.to_csv(best_dataset_file, header=['Best_Model'])
                win_counts.to_csv(win_counts_file, header=['Win_Count'])

def main():
    """Main execution function"""
    print("🚀 Starting Statistical Analysis of ML Experiment Results")
    
    # Initialize storage manager
    storage_manager = GCPStorageManager()
    
    # Initialize analyzer
    analyzer = ExperimentAnalyzer(storage_manager)
    
    # Download results
    analyzer.download_results()
    
    if not analyzer.results_data:
        print("❌ No results found! Please check the bucket and results path.")
        return
    
    # Extract best performance
    processed_results = analyzer.extract_best_performance()
    
    # Perform data sanity check
    analyzer.perform_data_sanity_check(processed_results)
    
    # Create performance matrices
    analyzer.create_performance_matrices(processed_results)
    
    # Perform statistical analysis
    analyzer.perform_statistical_analysis()
    
    # Create visualizations
    analyzer.create_visualizations()
    
    # Generate summary report
    analyzer.generate_summary_report()
    
    print("\n✅ Statistical analysis completed!")
    print("📁 Check the 'stat_results' folder for all generated CSV files and PNG plots.")

if __name__ == "__main__":
    main()
