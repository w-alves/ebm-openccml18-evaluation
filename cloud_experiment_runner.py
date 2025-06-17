#!/usr/bin/env python3
"""
Cloud Experiment Runner - Runs individual model+dataset experiments in Cloud Run
"""

import os
import sys
import json
import pickle
import logging
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Import our GCP modules
from gcp_config import get_gcp_config, ExperimentStatus
from gcp_storage import GCPStorageManager
from experiment_tracker import ExperimentTracker

def setup_logging(job_id: str) -> logging.Logger:
    """Setup logging for this experiment"""
    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s - {job_id} - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def download_and_setup_code(storage: GCPStorageManager) -> None:
    """Download code and models from Cloud Storage"""
    print("📥 Downloading code from Cloud Storage...")
    
    # Download models
    storage.download_directory('code/models', 'models')
    
    # Download requirements and config files
    code_files = ['requirements_python311.txt', 'logging.conf']
    for file_name in code_files:
        try:
            storage.download_file(f'code/{file_name}', file_name)
        except Exception as e:
            print(f"⚠️ Could not download {file_name}: {e}")

def load_dataset_from_cloud(storage: GCPStorageManager, dataset_name: str) -> tuple:
    """Load dataset from Cloud Storage"""
    print(f"📊 Loading dataset: {dataset_name}")
    
    train_path = f'datasets/{dataset_name}_train.pkl'
    test_path = f'datasets/{dataset_name}_test.pkl'
    
    # Download to temporary files
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp_train:
        storage.download_file(train_path, temp_train.name)
        with open(temp_train.name, 'rb') as f:
            train_data = pickle.load(f)
        os.unlink(temp_train.name)
    
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp_test:
        storage.download_file(test_path, temp_test.name)
        with open(temp_test.name, 'rb') as f:
            test_data = pickle.load(f)
        os.unlink(temp_test.name)
    
    # Extract X, y
    if 'X_train' in train_data and 'y_train' in train_data:
        X_train, y_train = train_data['X_train'], train_data['y_train']
    elif 'X' in train_data and 'y' in train_data:
        X_train, y_train = train_data['X'], train_data['y']
    else:
        raise ValueError(f"Invalid train data format for {dataset_name}")
    
    if 'X_test' in test_data and 'y_test' in test_data:
        X_test, y_test = test_data['X_test'], test_data['y_test']
    elif 'X' in test_data and 'y' in test_data:
        X_test, y_test = test_data['X'], test_data['y']
    else:
        raise ValueError(f"Invalid test data format for {dataset_name}")
    
    print(f"✅ Dataset loaded - Train: {len(y_train)}, Test: {len(y_test)}")
    return X_train, y_train, X_test, y_test

def create_model(model_name: str):
    """Create and return model instance"""
    print(f"🤖 Creating model: {model_name}")
    
    if model_name == "AutoSklearn":
        from models.autosklearn_model import AutoSklearnModel
        return AutoSklearnModel()
    elif model_name == "AutoGluon":
        from models.autogluon_model import AutoGluonModel
        return AutoGluonModel()
    elif model_name == "XGBoost":
        from models.xgboost_model import XGBoostModel
        return XGBoostModel()
    elif model_name == "LightGBM":
        from models.lgbm_model import LGBMModel
        return LGBMModel()
    elif model_name == "CatBoost":
        from models.catboost_model import CatBoostModel
        return CatBoostModel()
    elif model_name == "EBM":
        from models.ebm_model import EBMModel
        return EBMModel()
    else:
        raise ValueError(f"Unknown model: {model_name}")

def run_single_experiment(job_id: str, model_name: str, dataset_name: str) -> Dict[str, Any]:
    """Run a single model+dataset experiment"""
    logger = setup_logging(job_id)
    config = get_gcp_config()
    storage = GCPStorageManager(config)
    tracker = ExperimentTracker(config)
    
    try:
        # Update status to running
        tracker.update_job_status(job_id, ExperimentStatus.RUNNING)
        
        logger.info(f"🚀 Starting experiment: {model_name} on {dataset_name}")
        
        # Download code and setup environment
        download_and_setup_code(storage)
        
        # Load dataset
        X_train, y_train, X_test, y_test = load_dataset_from_cloud(storage, dataset_name)
        
        # Create model
        model = create_model(model_name)
        logger.info(f"✅ Model created: {type(model).__name__}")
        
        # Run experiment pipeline
        logger.info("🔧 Step 1: Tuning hyperparameters...")
        model.tune(X_train, y_train, cv_folds=5)
        
        logger.info("🏋️ Step 2: Training model...")
        model.fit(X_train, y_train)
        
        logger.info("📊 Step 3: Evaluating model...")
        results = model.evaluate(X_test, y_test)
        
        # Get detailed cross-validation results
        cv_results = model.get_cv_results()
        logger.info(f"📋 Captured CV results for {len(cv_results)} parameter combinations")
        
        # Add metadata
        results.update({
            'model_name': model_name,
            'dataset_name': dataset_name,
            'train_samples': len(y_train),
            'test_samples': len(y_test),
            'experiment_timestamp': datetime.now().isoformat(),
            'job_id': job_id,
            'cross_validation_results': cv_results  # Add detailed CV data
        })
        
        # Save results to Cloud Storage
        clean_dataset_name = dataset_name.replace('dataset_', '')
        results_path = f'results/raw/{model_name}_{clean_dataset_name}.json'
        storage.upload_json(results, results_path)
        
        # Update status to completed
        tracker.update_job_status(
            job_id, 
            ExperimentStatus.COMPLETED,
            results_path=results_path
        )
        
        logger.info(f"🎉 Experiment completed successfully!")
        logger.info(f"📋 Results summary: {results}")
        
        return results
        
    except Exception as e:
        error_msg = f"Experiment failed: {str(e)}"
        logger.error(error_msg)
        
        # Update status to failed
        tracker.update_job_status(
            job_id,
            ExperimentStatus.FAILED,
            error_message=error_msg
        )
        
        # Re-raise to exit with error code
        raise

def main():
    """Main entry point for Cloud Run job"""
    # Get job parameters from environment variables
    job_id = os.getenv('JOB_ID')
    model_name = os.getenv('MODEL_NAME')
    dataset_name = os.getenv('DATASET_NAME')
    
    if not all([job_id, model_name, dataset_name]):
        print("❌ Missing required environment variables: JOB_ID, MODEL_NAME, DATASET_NAME")
        sys.exit(1)
    
    print(f"🚀 Cloud Experiment Runner")
    print(f"Job ID: {job_id}")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print("=" * 50)
    
    try:
        # Import here to avoid import issues during container build
        from gcp_config import ExperimentStatus
        
        run_single_experiment(job_id, model_name, dataset_name)
        print(f"✅ Experiment completed successfully!")
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 