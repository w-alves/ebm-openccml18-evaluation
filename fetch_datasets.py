#!/usr/bin/env python3
"""
Dataset Acquisition Script for OpenML-CC18 Datasets
Fetches the 30 smallest classification datasets from OpenML-CC18 benchmark suite
"""

import os
import pandas as pd
import openml
import pickle
from pathlib import Path
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_openml_cc18_smallest_datasets(n_datasets: int = 30) -> list:
    """
    Deterministically fetch the smallest datasets from OpenML-CC18 benchmark suite
    
    Args:
        n_datasets: Number of smallest datasets to return
        
    Returns:
        list: Dataset IDs sorted by number of instances (smallest first)
    """
    logger.info("Fetching OpenML-CC18 benchmark suite information...")
    
    # Get the OpenML-CC18 benchmark suite
    suite = openml.study.get_suite(99)  # OpenML-CC18 suite ID
    dataset_ids = suite.data
    
    logger.info(f"Found {len(dataset_ids)} datasets in OpenML-CC18 suite")
    
    # Fetch metadata for all datasets to get their sizes
    dataset_info = []
    
    for dataset_id in tqdm(dataset_ids, desc="Fetching dataset metadata"):
        try:
            dataset = openml.datasets.get_dataset(dataset_id, download_data=False)
            dataset_info.append({
                'dataset_id': dataset_id,
                'name': dataset.name,
                'n_instances': dataset.qualities.get('NumberOfInstances', float('inf')),
                'n_features': dataset.qualities.get('NumberOfFeatures', 0),
                'n_classes': dataset.qualities.get('NumberOfClasses', 0)
            })
        except Exception as e:
            logger.warning(f"Failed to get metadata for dataset {dataset_id}: {str(e)}")
            continue
    
    # Sort by number of instances and take the smallest n_datasets
    dataset_info.sort(key=lambda x: x['n_instances'])
    smallest_datasets = dataset_info[:n_datasets]
    
    logger.info(f"Selected {len(smallest_datasets)} smallest datasets:")
    for i, ds in enumerate(smallest_datasets[:10]):  # Show first 10
        logger.info(f"  {i+1}. {ds['name']} (ID: {ds['dataset_id']}) - "
                   f"{ds['n_instances']} instances, {ds['n_features']} features")
    if len(smallest_datasets) > 10:
        logger.info(f"  ... and {len(smallest_datasets) - 10} more")
    
    return [ds['dataset_id'] for ds in smallest_datasets]

def fetch_openml_dataset(dataset_id: int, data_dir: Path) -> bool:
    """
    Fetch a single dataset from OpenML and save it as pickle
    
    Args:
        dataset_id: OpenML dataset ID
        data_dir: Directory to save the dataset
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Fetch dataset
        dataset = openml.datasets.get_dataset(dataset_id)
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            dataset_format="dataframe", target=dataset.default_target_attribute
        )
        
        # Create dataset info
        dataset_info = {
            'name': dataset.name,
            'dataset_id': dataset_id,
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'n_classes': len(y.unique()) if hasattr(y, 'unique') else len(set(y)),
            'categorical_indicator': categorical_indicator,
            'attribute_names': attribute_names,
            'target_name': dataset.default_target_attribute,
            'description': dataset.description[:500] if dataset.description else "",
            'X': X,
            'y': y
        }
        
        # Save dataset
        dataset_path = data_dir / f"dataset_{dataset_id}_{dataset.name.replace(' ', '_')}.pkl"
        with open(dataset_path, 'wb') as f:
            pickle.dump(dataset_info, f)
            
        logger.info(f"Successfully fetched dataset {dataset_id}: {dataset.name} "
                   f"({X.shape[0]} samples, {X.shape[1]} features)")
        return True
        
    except Exception as e:
        logger.error(f"Failed to fetch dataset {dataset_id}: {str(e)}")
        return False

def main():
    """Main function to fetch all datasets"""
    # Create data directory
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting dataset acquisition from OpenML-CC18 benchmark suite")
    
    # Get the 30 smallest datasets deterministically
    smallest_dataset_ids = get_openml_cc18_smallest_datasets(3)
    
    logger.info(f"Fetching {len(smallest_dataset_ids)} smallest datasets...")
    
    successful_downloads = 0
    failed_downloads = []
    
    # Fetch each dataset with progress bar
    for dataset_id in tqdm(smallest_dataset_ids, desc="Downloading datasets"):
        success = fetch_openml_dataset(dataset_id, data_dir)
        if success:
            successful_downloads += 1
        else:
            failed_downloads.append(dataset_id)
    
    # Summary
    logger.info(f"\nDataset acquisition completed:")
    logger.info(f"✅ Successfully downloaded: {successful_downloads}/{len(smallest_dataset_ids)} datasets")
    
    if failed_downloads:
        logger.warning(f"❌ Failed downloads: {failed_downloads}")
    
    # Create manifest file
    manifest_path = data_dir / "datasets_manifest.txt"
    with open(manifest_path, 'w') as f:
        f.write("OpenML-CC18 Datasets (30 smallest by instance count)\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total datasets: {len(smallest_dataset_ids)}\n")
        f.write(f"Successfully downloaded: {successful_downloads}\n")
        f.write(f"Failed downloads: {len(failed_downloads)}\n\n")
        
        f.write("Selected dataset IDs (sorted by size):\n")
        for dataset_id in smallest_dataset_ids:
            f.write(f"- {dataset_id}\n")
        
        if failed_downloads:
            f.write("\nFailed dataset IDs:\n")
            for dataset_id in failed_downloads:
                f.write(f"- {dataset_id}\n")
    
    logger.info(f"Dataset manifest saved to: {manifest_path}")

if __name__ == "__main__":
    main() 