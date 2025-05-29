#!/usr/bin/env python3
"""
Data Preprocessing Script
Handles missing values, categorical encoding, and feature standardization
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import logging
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetPreprocessor:
    """Comprehensive dataset preprocessor for OpenML datasets"""
    
    def __init__(self, use_onehot=True, max_categories=10):
        """
        Initialize preprocessor
        
        Args:
            use_onehot: Whether to use one-hot encoding for categorical features
            max_categories: Maximum number of categories for one-hot encoding
        """
        self.use_onehot = use_onehot
        self.max_categories = max_categories
        self.label_encoder = LabelEncoder()
        self.preprocessor = None
        self.feature_names = None
        
    def detect_feature_types(self, X, categorical_indicator=None):
        """
        Detect numerical and categorical features
        
        Args:
            X: Feature matrix
            categorical_indicator: OpenML categorical indicator
            
        Returns:
            tuple: (numerical_features, categorical_features)
        """
        if categorical_indicator is not None:
            # Use OpenML categorical indicator
            categorical_features = [i for i, is_cat in enumerate(categorical_indicator) if is_cat]
            numerical_features = [i for i, is_cat in enumerate(categorical_indicator) if not is_cat]
        else:
            # Detect automatically
            categorical_features = []
            numerical_features = []
            
            for i, col in enumerate(X.columns):
                if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                    categorical_features.append(i)
                elif X[col].nunique() <= 10 and X[col].dtype in ['int64', 'int32']:
                    # Likely categorical if few unique values
                    categorical_features.append(i)
                else:
                    numerical_features.append(i)
        
        return numerical_features, categorical_features
    
    def preprocess_dataset(self, X, y, categorical_indicator=None, dataset_name=""):
        """
        Preprocess a single dataset
        
        Args:
            X: Feature matrix
            y: Target vector
            categorical_indicator: OpenML categorical indicator
            dataset_name: Name of the dataset for logging
            
        Returns:
            tuple: (X_processed, y_processed, feature_names)
        """
        logger.info(f"Preprocessing dataset: {dataset_name}")
        logger.info(f"Original shape: {X.shape}")
        
        # Ensure X is DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Ensure y is Series
        if not isinstance(y, (pd.Series, np.ndarray)):
            y = pd.Series(y)
        
        # Handle missing values in target
        if y.isnull().any():
            logger.warning(f"Removing {y.isnull().sum()} samples with missing target values")
            mask = ~y.isnull()
            X = X[mask]
            y = y[mask]
        
        # Detect feature types
        numerical_features, categorical_features = self.detect_feature_types(X, categorical_indicator)
        
        logger.info(f"Detected {len(numerical_features)} numerical and {len(categorical_features)} categorical features")
        
        # Create preprocessing pipeline
        transformers = []
        
        # Numerical features: impute + scale
        if numerical_features:
            numerical_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            transformers.append(('num', numerical_transformer, numerical_features))
        
        # Categorical features: impute + encode
        if categorical_features:
            # Check cardinality for each categorical feature
            for feat_idx in categorical_features:
                col_name = X.columns[feat_idx]
                cardinality = X[col_name].nunique()
                
                if self.use_onehot and cardinality <= self.max_categories:
                    # Use one-hot encoding for low cardinality
                    cat_transformer = Pipeline([
                        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
                    ])
                else:
                    # Use ordinal encoding for high cardinality
                    cat_transformer = Pipeline([
                        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
                    ])
                
                transformers.append((f'cat_{feat_idx}', cat_transformer, [feat_idx]))
        
        # Create column transformer
        if transformers:
            self.preprocessor = ColumnTransformer(
                transformers=transformers,
                remainder='drop'
            )
            
            # Fit and transform
            X_processed = self.preprocessor.fit_transform(X)
            
            # Generate feature names
            self.feature_names = self._generate_feature_names(X, transformers)
        else:
            # No preprocessing needed
            X_processed = X.values
            self.feature_names = list(X.columns)
        
        # Encode target variable
        y_processed = self.label_encoder.fit_transform(y)
        
        logger.info(f"Processed shape: {X_processed.shape}")
        logger.info(f"Number of classes: {len(np.unique(y_processed))}")
        
        return X_processed, y_processed, self.feature_names
    
    def _generate_feature_names(self, X, transformers):
        """Generate feature names after preprocessing"""
        feature_names = []
        
        for name, transformer, columns in transformers:
            if isinstance(columns, list):
                if 'onehot' in str(transformer).lower():
                    # One-hot encoded features
                    for col_idx in columns:
                        col_name = X.columns[col_idx]
                        unique_vals = X[col_name].dropna().unique()
                        if len(unique_vals) > 1:
                            feature_names.extend([f"{col_name}_{val}" for val in unique_vals[1:]])
                else:
                    # Other transformations
                    feature_names.extend([X.columns[col_idx] for col_idx in columns])
            else:
                # Multiple columns (numerical features)
                feature_names.extend([X.columns[col_idx] for col_idx in columns])
        
        return feature_names

def process_all_datasets():
    """Process all downloaded datasets"""
    raw_data_dir = Path("data/raw")
    processed_data_dir = Path("data/processed")
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all dataset files
    dataset_files = list(raw_data_dir.glob("dataset_*.pkl"))
    
    if not dataset_files:
        logger.error("No dataset files found in data/raw/. Run fetch_datasets.py first.")
        return
    
    logger.info(f"Found {len(dataset_files)} datasets to process")
    
    processing_log = []
    
    for dataset_file in tqdm(dataset_files, desc="Processing datasets"):
        try:
            # Load dataset
            with open(dataset_file, 'rb') as f:
                dataset_info = pickle.load(f)
            
            # Extract data
            X = dataset_info['X']
            y = dataset_info['y']
            categorical_indicator = dataset_info.get('categorical_indicator', None)
            dataset_name = dataset_info['name']
            
            # Preprocess
            preprocessor = DatasetPreprocessor()
            X_processed, y_processed, feature_names = preprocessor.preprocess_dataset(
                X, y, categorical_indicator, dataset_name
            )
            
            # Update dataset info
            dataset_info.update({
                'X_processed': X_processed,
                'y_processed': y_processed,
                'feature_names_processed': feature_names,
                'preprocessor': preprocessor,
                'n_features_processed': X_processed.shape[1],
                'preprocessing_complete': True
            })
            
            # Save processed dataset
            processed_file = processed_data_dir / dataset_file.name
            with open(processed_file, 'wb') as f:
                pickle.dump(dataset_info, f)
            
            processing_log.append({
                'dataset_id': dataset_info['dataset_id'],
                'name': dataset_name,
                'original_shape': dataset_info['X'].shape,
                'processed_shape': X_processed.shape,
                'n_classes': len(np.unique(y_processed)),
                'status': 'success'
            })
            
        except Exception as e:
            logger.error(f"Failed to process {dataset_file.name}: {str(e)}")
            logger.error(f"Full traceback:", exc_info=True)
            processing_log.append({
                'dataset_id': dataset_file.stem,
                'name': 'unknown',
                'status': 'failed',
                'error': str(e)
            })
    
    # Save processing log
    log_df = pd.DataFrame(processing_log)
    log_df.to_csv(processed_data_dir / "preprocessing_log.csv", index=False)
    
    # Summary
    successful = log_df[log_df['status'] == 'success']
    logger.info(f"\nPreprocessing completed:")
    logger.info(f"✅ Successfully processed: {len(successful)}/{len(dataset_files)} datasets")
    
    if len(successful) > 0:
        logger.info(f"Average features after preprocessing: {successful['processed_shape'].apply(lambda x: x[1]).mean():.1f}")

def main():
    """Main function"""
    process_all_datasets()

if __name__ == "__main__":
    from sklearn.pipeline import Pipeline
    main() 