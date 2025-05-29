#!/usr/bin/env python3
"""
Data Splitting Script
Applies 70/30 train-test split to all preprocessed datasets
"""

import pickle
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pandas as pd
from preprocess import DatasetPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Fixed random seed for reproducibility
RANDOM_SEED = 42

def split_dataset(X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=True):
    """
    Split a single dataset into train and test sets
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Proportion of test set
        random_state: Random seed
        stratify: Whether to stratify split based on target
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    try:
        # Use stratification if classes are balanced enough
        stratify_y = y if stratify else None
        
        # Check if stratification is possible
        if stratify_y is not None:
            unique_classes, counts = np.unique(y, return_counts=True)
            min_class_count = min(counts)
            
            # Don't stratify if any class has fewer than 2 samples
            if min_class_count < 2:
                logger.warning("Some classes have < 2 samples. Disabling stratification.")
                stratify_y = None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=stratify_y
        )
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        logger.error(f"Error in splitting: {str(e)}")
        logger.info("Attempting split without stratification...")
        
        # Fallback: split without stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=None
        )
        
        return X_train, X_test, y_train, y_test

def split_all_datasets():
    """Split all preprocessed datasets"""
    processed_data_dir = Path("data/processed")
    split_data_dir = Path("data/split")
    split_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all processed dataset files
    dataset_files = list(processed_data_dir.glob("dataset_*.pkl"))
    
    if not dataset_files:
        logger.error("No processed dataset files found. Run preprocess.py first.")
        return
    
    logger.info(f"Found {len(dataset_files)} datasets to split")
    
    split_log = []
    
    for dataset_file in tqdm(dataset_files, desc="Splitting datasets"):
        try:
            # Load processed dataset
            with open(dataset_file, 'rb') as f:
                dataset_info = pickle.load(f)
            
            dataset_name = dataset_info['name']
            dataset_id = dataset_info['dataset_id']
            
            # Check if preprocessing was completed
            if not dataset_info.get('preprocessing_complete', False):
                logger.warning(f"Skipping {dataset_name}: preprocessing not completed")
                continue
            
            # Extract processed data
            X = dataset_info['X_processed']
            y = dataset_info['y_processed']
            
            logger.info(f"Splitting {dataset_name} (ID: {dataset_id})")
            logger.info(f"Dataset shape: {X.shape}, Classes: {len(np.unique(y))}")
            
            # Split the data
            X_train, X_test, y_train, y_test = split_dataset(X, y)
            
            # Update dataset info with splits
            dataset_info.update({
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'train_size': X_train.shape[0],
                'test_size': X_test.shape[0],
                'split_ratio': f"{X_train.shape[0]}/{X_test.shape[0]}",
                'random_seed': RANDOM_SEED,
                'splitting_complete': True
            })
            
            # Save split dataset
            split_file = split_data_dir / dataset_file.name
            with open(split_file, 'wb') as f:
                pickle.dump(dataset_info, f)
            
            # Also save individual split files for easy access
            base_name = f"dataset_{dataset_id}_{dataset_name.replace(' ', '_')}"
            
            # Save train split
            train_data = {
                'X_train': X_train,
                'y_train': y_train,
                'dataset_info': {
                    'name': dataset_name,
                    'dataset_id': dataset_id,
                    'n_features': X_train.shape[1],
                    'n_samples': X_train.shape[0],
                    'n_classes': len(np.unique(y_train))
                }
            }
            with open(split_data_dir / f"{base_name}_train.pkl", 'wb') as f:
                pickle.dump(train_data, f)
            
            # Save test split
            test_data = {
                'X_test': X_test,
                'y_test': y_test,
                'dataset_info': {
                    'name': dataset_name,
                    'dataset_id': dataset_id,
                    'n_features': X_test.shape[1],
                    'n_samples': X_test.shape[0],
                    'n_classes': len(np.unique(y_test))
                }
            }
            with open(split_data_dir / f"{base_name}_test.pkl", 'wb') as f:
                pickle.dump(test_data, f)
            
            # Log split information
            train_class_dist = pd.Series(y_train).value_counts().sort_index()
            test_class_dist = pd.Series(y_test).value_counts().sort_index()
            
            split_log.append({
                'dataset_id': dataset_id,
                'name': dataset_name,
                'total_samples': X.shape[0],
                'train_samples': X_train.shape[0],
                'test_samples': X_test.shape[0],
                'n_features': X.shape[1],
                'n_classes': len(np.unique(y)),
                'train_class_distribution': train_class_dist.to_dict(),
                'test_class_distribution': test_class_dist.to_dict(),
                'status': 'success'
            })
            
            logger.info(f"✅ Successfully split {dataset_name}")
            logger.info(f"   Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
            
        except Exception as e:
            logger.error(f"Failed to split {dataset_file.name}: {str(e)}")
            import traceback
            logger.error(f"Error details: {traceback.format_exc()}")
            split_log.append({
                'dataset_id': dataset_file.stem,
                'name': 'unknown',
                'status': 'failed',
                'error': str(e)
            })
    
    # Save split log
    log_df = pd.DataFrame(split_log)
    log_df.to_csv(split_data_dir / "splitting_log.csv", index=False)
    
    # Generate summary
    successful = log_df[log_df['status'] == 'success']
    logger.info(f"\nData splitting completed:")
    logger.info(f"✅ Successfully split: {len(successful)}/{len(dataset_files)} datasets")
    
    if len(successful) > 0:
        avg_train_size = successful['train_samples'].mean()
        avg_test_size = successful['test_samples'].mean()
        logger.info(f"Average train size: {avg_train_size:.0f} samples")
        logger.info(f"Average test size: {avg_test_size:.0f} samples")
        logger.info(f"Average train/test ratio: {avg_train_size/avg_test_size:.2f}")
    
    # Create dataset summary for experiments
    if len(successful) > 0:
        summary = successful[['dataset_id', 'name', 'train_samples', 'test_samples', 'n_features', 'n_classes']].copy()
        summary.to_csv(split_data_dir / "datasets_summary.csv", index=False)
        logger.info(f"Dataset summary saved to: {split_data_dir / 'datasets_summary.csv'}")

def main():
    """Main function"""
    split_all_datasets()

if __name__ == "__main__":
    main() 