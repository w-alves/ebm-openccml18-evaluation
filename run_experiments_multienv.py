#!/usr/bin/env python3
"""
Multi-Environment Experiment Execution Script
Coordinates ML training across different conda environments to handle dependency conflicts
"""

import os
import sys
import json
import pickle
import logging
import asyncio
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import time

# Configure logging
def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('experiments_multienv.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_environment_config():
    """Load environment configuration from JSON file"""
    config_path = Path('envs/environment_config.json')
    if not config_path.exists():
        raise FileNotFoundError(f"Environment config not found at {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config['environments']

def get_available_datasets():
    """Get list of available datasets from split directory"""
    try:
        import pandas as pd
        split_dir = Path('data/split')
        datasets_summary = pd.read_csv(split_dir / 'datasets_summary.csv')
        
        available_datasets = []
        for _, row in datasets_summary.iterrows():
            dataset_name = f"dataset_{row['dataset_id']}_{row['name']}"
            train_path = split_dir / f"{dataset_name}_train.pkl"
            test_path = split_dir / f"{dataset_name}_test.pkl"
            
            if train_path.exists() and test_path.exists():
                available_datasets.append({
                    'name': dataset_name,
                    'train_path': str(train_path),
                    'test_path': str(test_path),
                    'n_features': row['n_features'],
                    'n_classes': row['n_classes']
                })
        
        return available_datasets
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return []

def create_experiment_script(model_name, dataset_info, env_name, script_path):
    """Create a standalone experiment script for a specific model/dataset combination"""
    
    # Get the absolute path to the project directory
    project_dir = os.getcwd()
    
    script_content = f'''#!/usr/bin/env python3
"""
Standalone experiment script for {model_name} on {dataset_info['name']}
Environment: {env_name}
"""

import os
import sys
import json
import pickle
import logging
from datetime import datetime
from pathlib import Path

# Add project directory to path to find models
project_dir = "{project_dir}"
sys.path.insert(0, project_dir)

def setup_logging():
    """Setup logging for this experiment"""
    # Ensure logs directory exists
    log_dir = Path(project_dir) / "logs"
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - {model_name} - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / '{model_name}_{dataset_info['name']}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_dataset(dataset_path):
    """Load dataset from pickle file"""
    try:
        with open(dataset_path, 'rb') as f:
            data = pickle.load(f)
        
        if '_train.pkl' in str(dataset_path):
            return data['X_train'], data['y_train']
        elif '_test.pkl' in str(dataset_path):
            return data['X_test'], data['y_test']
        else:
            if 'X_train' in data and 'y_train' in data:
                return data['X_train'], data['y_train']
            elif 'X' in data and 'y' in data:
                return data['X'], data['y']
            else:
                return None, None
    except Exception as e:
        return None, None

def main():
    # Change to project directory
    os.chdir(project_dir)
    
    logger = setup_logging()
    logger.info(f"Starting experiment: {model_name} on {dataset_info['name']}")
    logger.info(f"Working directory: {{os.getcwd()}}")
    logger.info(f"Python path: {{sys.path[:3]}}")
    
    try:
        # Import model class
        if "{model_name}" == "AutoSklearn":
            from models.autosklearn_model import AutoSklearnModel
            model = AutoSklearnModel()
        elif "{model_name}" == "AutoGluon":
            from models.autogluon_model import AutoGluonModel
            model = AutoGluonModel()
        elif "{model_name}" == "XGBoost":
            from models.xgboost_model import XGBoostModel
            model = XGBoostModel()
        elif "{model_name}" == "LightGBM":
            from models.lgbm_model import LGBMModel
            model = LGBMModel()
        elif "{model_name}" == "CatBoost":
            from models.catboost_model import CatBoostModel
            model = CatBoostModel()
        elif "{model_name}" == "EBM":
            from models.ebm_model import EBMModel
            model = EBMModel()
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        logger.info(f"Successfully imported and initialized {{type(model).__name__}}")
        
        # Load data
        X_train, y_train = load_dataset("{dataset_info['train_path']}")
        X_test, y_test = load_dataset("{dataset_info['test_path']}")
        
        if X_train is None or X_test is None:
            raise ValueError("Failed to load dataset")
        
        logger.info(f"Data loaded - Train: {{len(y_train)}}, Test: {{len(y_test)}}")
        
        # Run experiment pipeline
        logger.info("Step 1: Tuning hyperparameters...")
        model.tune(X_train, y_train, cv_folds=5)
        
        logger.info("Step 2: Training model...")
        model.fit(X_train, y_train)
        
        logger.info("Step 3: Evaluating model...")
        results = model.evaluate(X_test, y_test)
        
        # Add metadata
        results.update({{
            'model_name': "{model_name}",
            'dataset_name': "{dataset_info['name']}",
            'dataset_n_features': {dataset_info['n_features']},
            'dataset_n_classes': {dataset_info['n_classes']},
            'train_samples': len(y_train),
            'test_samples': len(y_test),
            'experiment_timestamp': datetime.now().isoformat(),
            'environment': "{env_name}"
        }})
        
        # Save results
        results_dir = Path(project_dir) / 'results' / 'raw'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        clean_dataset_name = "{dataset_info['name']}".replace('dataset_', '')
        filename = f"{model_name}_{{clean_dataset_name}}.json"
        filepath = results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Experiment completed successfully!")
        logger.info(f"Results - ACC: {{results['accuracy']:.4f}}, AUC: {{results['auc_ovo']:.4f}}, CE: {{results['cross_entropy']:.4f}}, Time: {{results['total_time']:.2f}}s")
        logger.info(f"Results saved to: {{filepath}}")
        
        return results
        
    except Exception as e:
        error_msg = f"Error in experiment {model_name} on {dataset_info['name']}: {{str(e)}}"
        logger.error(error_msg)
        logger.error(f"Exception details: {{type(e).__name__}}: {{e}}")
        
        # Save error result
        error_result = {{
            'model_name': "{model_name}",
            'dataset_name': "{dataset_info['name']}",
            'error': str(e),
            'experiment_timestamp': datetime.now().isoformat(),
            'environment': "{env_name}"
        }}
        
        results_dir = Path(project_dir) / 'results' / 'raw'
        results_dir.mkdir(parents=True, exist_ok=True)
        clean_dataset_name = "{dataset_info['name']}".replace('dataset_', '')
        filename = f"{model_name}_{{clean_dataset_name}}_ERROR.json"
        filepath = results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(error_result, f, indent=2, default=str)
        
        raise

if __name__ == "__main__":
    main()
'''
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    os.chmod(script_path, 0o755)

async def run_experiment_in_environment(model_name, dataset_info, env_config, temp_dir):
    """Run a single experiment in its designated environment using async subprocess"""
    logger = logging.getLogger(__name__)
    
    env_name = env_config['name']
    
    logger.info(f"Preparing experiment: {model_name} on {dataset_info['name']} in {env_name}")
    
    # Create experiment script
    script_name = f"exp_{model_name}_{dataset_info['name'].replace('dataset_', '')}.py"
    script_path = temp_dir / script_name
    
    create_experiment_script(model_name, dataset_info, env_name, script_path)
    
    # Create conda execution script
    conda_script = temp_dir / f"run_{script_name}.sh"
    
    # Get current working directory
    current_dir = os.getcwd()
    
    conda_script_content = f'''#!/bin/bash
set -e

echo "🔌 Activating environment {env_name}..."
eval "$(conda shell.bash hook)"
conda activate {env_name}

echo "🚀 Running experiment: {model_name} on {dataset_info['name']}"
echo "📂 Project directory: {current_dir}"
echo "🐍 Python environment: $(which python)"

# Run the experiment script
python "{script_path}"

echo "✅ Experiment completed"
conda deactivate
'''
    
    with open(conda_script, 'w') as f:
        f.write(conda_script_content)
    
    os.chmod(conda_script, 0o755)
    
    # Execute the experiment asynchronously
    try:
        logger.info(f"Executing: {model_name} on {dataset_info['name']} in {env_name}")
        
        start_time = time.time()
        
        # Create subprocess
        process = await asyncio.create_subprocess_shell(
            f'bash "{conda_script}"',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=current_dir
        )
        
        # Wait for completion with timeout
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), 
                timeout=3600  # 1 hour timeout
            )
            end_time = time.time()
            
            stdout_text = stdout.decode('utf-8') if stdout else ""
            stderr_text = stderr.decode('utf-8') if stderr else ""
            
            if process.returncode == 0:
                logger.info(f"SUCCESS: {model_name} on {dataset_info['name']} ({end_time - start_time:.1f}s)")
                return {
                    'status': 'success',
                    'model_name': model_name,
                    'dataset_name': dataset_info['name'],
                    'environment': env_name,
                    'execution_time': end_time - start_time,
                    'stdout': stdout_text,
                    'stderr': stderr_text
                }
            else:
                logger.error(f"FAILED: {model_name} on {dataset_info['name']} (exit code: {process.returncode})")
                logger.error(f"STDERR: {stderr_text}")
                return {
                    'status': 'failed',
                    'model_name': model_name,
                    'dataset_name': dataset_info['name'],
                    'environment': env_name,
                    'execution_time': end_time - start_time,
                    'error': stderr_text,
                    'exit_code': process.returncode
                }
                
        except asyncio.TimeoutError:
            # Kill the process if it times out
            try:
                process.kill()
                await process.wait()
            except:
                pass
            
            end_time = time.time()
            logger.error(f"TIMEOUT: {model_name} on {dataset_info['name']} (1 hour limit)")
            return {
                'status': 'timeout',
                'model_name': model_name,
                'dataset_name': dataset_info['name'],
                'environment': env_name,
                'execution_time': end_time - start_time,
                'error': 'Experiment timed out after 1 hour'
            }
            
    except Exception as e:
        logger.error(f"ERROR: {model_name} on {dataset_info['name']}: {str(e)}")
        return {
            'status': 'error',
            'model_name': model_name,
            'dataset_name': dataset_info['name'],
            'environment': env_name,
            'error': str(e)
        }

async def main():
    """Main execution function with async multi-environment coordination"""
    logger = setup_logging()
    logger.info("Starting Async Multi-Environment ML Experiment Coordination...")
    
    # Load configurations
    try:
        env_configs = load_environment_config()
        datasets = get_available_datasets()
    except Exception as e:
        logger.error(f"Failed to load configurations: {e}")
        return
    
    # Create model-environment mapping
    model_env_mapping = {}
    for env_key, env_config in env_configs.items():
        for model in env_config['models']:
            model_env_mapping[model] = env_config
    
    logger.info(f"Environment configuration loaded:")
    for model, env_config in model_env_mapping.items():
        logger.info(f"  {model} → {env_config['name']} (Python {env_config['python_version']})")
    
    logger.info(f"Found {len(datasets)} datasets")
    
    # Prepare experiment queue
    experiment_queue = []
    for dataset_info in datasets:
        for model_name, env_config in model_env_mapping.items():
            experiment_queue.append({
                'model_name': model_name,
                'dataset_info': dataset_info,
                'env_config': env_config
            })
    
    total_experiments = len(experiment_queue)
    logger.info(f"Total experiments to run: {total_experiments}")
    
    # Create temporary directory for experiment scripts
    temp_dir = Path(tempfile.mkdtemp(prefix="ml_experiments_"))
    logger.info(f"Using temporary directory: {temp_dir}")
    
    try:
        # Create semaphore to limit concurrent experiments (avoid resource conflicts)
        max_concurrent = min(8, len(experiment_queue))  # Limit to 8 parallel environments
        logger.info(f"Using {max_concurrent} concurrent experiments")
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def run_with_semaphore(experiment):
            """Run experiment with semaphore to limit concurrency"""
            async with semaphore:
                return await run_experiment_in_environment(
                    experiment['model_name'],
                    experiment['dataset_info'],
                    experiment['env_config'],
                    temp_dir
                )
        
        # Execute all experiments concurrently with limited parallelism
        logger.info("Starting concurrent experiment execution...")
        start_time = time.time()
        
        # Create tasks for all experiments
        tasks = [run_with_semaphore(exp) for exp in experiment_queue]
        
        # Track progress
        completed_experiments = 0
        failed_experiments = 0
        results_summary = []
        
        # Process completed experiments as they finish
        for completed_task in asyncio.as_completed(tasks):
            result = await completed_task
            results_summary.append(result)
            
            if result['status'] == 'success':
                completed_experiments += 1
                logger.info(f"Progress: {completed_experiments + failed_experiments}/{total_experiments} - SUCCESS: {result['model_name']} on {result['dataset_name']} ({result['execution_time']:.1f}s)")
            else:
                failed_experiments += 1
                logger.warning(f"Progress: {completed_experiments + failed_experiments}/{total_experiments} - FAILED: {result['model_name']} on {result['dataset_name']}")
        
        total_execution_time = time.time() - start_time
        
        # Save execution summary
        summary_path = Path('results/execution_summary.json')
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        
        execution_summary = {
            'total_experiments': total_experiments,
            'successful': completed_experiments,
            'failed': failed_experiments,
            'success_rate': completed_experiments / total_experiments * 100,
            'total_execution_time': total_execution_time,
            'max_concurrent': max_concurrent,
            'execution_timestamp': datetime.now().isoformat(),
            'experiments': results_summary
        }
        
        with open(summary_path, 'w') as f:
            json.dump(execution_summary, f, indent=2, default=str)
        
        # Final summary
        logger.info("="*60)
        logger.info("ASYNC MULTI-ENVIRONMENT EXPERIMENT EXECUTION COMPLETED")
        logger.info(f"Total experiments: {total_experiments}")
        logger.info(f"Successful: {completed_experiments}")
        logger.info(f"Failed: {failed_experiments}")
        logger.info(f"Success rate: {completed_experiments/total_experiments*100:.1f}%")
        logger.info(f"Total execution time: {total_execution_time:.1f}s")
        logger.info(f"Average time per experiment: {total_execution_time/total_experiments:.1f}s")
        logger.info(f"Results saved in: results/raw/")
        logger.info(f"Execution summary: {summary_path}")
        logger.info("="*60)
        
    finally:
        # Cleanup temporary directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")

if __name__ == "__main__":
    asyncio.run(main()) 