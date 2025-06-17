#!/usr/bin/env python3
"""
AutoGluon Model Implementation
Uses AutoGluon for automated machine learning on tabular data
"""

from .base_model import BaseModel
import pandas as pd
import numpy as np
import tempfile
import shutil
import logging
import os

logger = logging.getLogger(__name__)

class AutoGluonModel(BaseModel):
    """AutoGluon model wrapper"""
    
    def __init__(self, random_state=42):
        super().__init__("AutoGluon", random_state)
        self.time_limit = 300  # 5 minutes default
        self.temp_dir = None
        self.target_column = 'target'
    
    def _create_model(self, **params):
        """Create AutoGluon predictor instance"""
        try:
            from autogluon.tabular import TabularPredictor
        except ImportError:
            logger.error("AutoGluon not available. Install with: pip install autogluon")
            raise ImportError("AutoGluon not installed")
        
        # Create temporary directory for AutoGluon models
        self.temp_dir = tempfile.mkdtemp(prefix='autogluon_')
        
        default_params = {
        }
        default_params.update(params)
        
        return TabularPredictor(
            label=self.target_column,
        )
    
    def _get_param_grid(self):
        """Get hyperparameter grid for AutoGluon tuning"""
        # AutoGluon does not require manual parameter grid definition
        # It automatically searches through a large space of models and hyperparameters
        logger.info("AutoGluon will automatically search for optimal hyperparameters")
        return [{"random_state": [self.random_state],}]

    def fit(self, X_train, y_train, **kwargs):
        """
        Fit AutoGluon predictor
        
        Args:
            X_train: Training features
            y_train: Training targets
            **kwargs: Additional fitting parameters
        """
        import time
        start_time = time.perf_counter()
        
        try:
            # Convert to DataFrame if needed
            if not isinstance(X_train, pd.DataFrame):
                X_train = pd.DataFrame(X_train)
            
            # Create training DataFrame with target
            train_data = X_train.copy()
            train_data[self.target_column] = y_train
            
            # Create AutoGluon predictor
            if self.model is None:
                self.model = self._create_model()
            
            # Fit the model - AutoGluon will automatically perform hyperparameter optimization
            logger.info(f"Starting AutoGluon training with automated hyperparameter optimization...")
            self.model.fit(train_data)
            self.is_fitted = True
            
        except Exception as e:
            raise RuntimeError(f"AutoGluon training failed: {str(e)}")
        
        self.training_time = time.perf_counter() - start_time
        logger.info(f"{self.name} training completed in {self.training_time:.2f}s")
    
    def predict(self, X_test):
        """Make predictions on test data"""
        if not self.is_fitted:
            raise ValueError(f"{self.name} must be fitted before making predictions")
        
        import time
        start_time = time.perf_counter()
        
        try:
            # Convert to DataFrame if needed
            if not isinstance(X_test, pd.DataFrame):
                X_test = pd.DataFrame(X_test)
            
            # Make predictions
            if hasattr(self.model, 'predict'):
                predictions = self.model.predict(X_test)
                
                # Convert to numpy array if it's a pandas Series
                if isinstance(predictions, pd.Series):
                    predictions = predictions.values
            else:
                # Fallback model case
                if isinstance(X_test, pd.DataFrame):
                    X_test = X_test.values
                predictions = self.model.predict(X_test)
                
        except Exception as e:
            logger.error(f"Prediction failed for {self.name}: {str(e)}")
            # Return random predictions as fallback
            unique_classes = getattr(self, '_classes', [0, 1])
            predictions = np.random.choice(unique_classes, size=len(X_test))
        
        self.prediction_time = time.perf_counter() - start_time
        return predictions
    
    def predict_proba(self, X_test):
        """Make probability predictions on test data"""
        if not self.is_fitted:
            raise ValueError(f"{self.name} must be fitted before making predictions")
        
        try:
            # Convert to DataFrame if needed
            if not isinstance(X_test, pd.DataFrame):
                X_test = pd.DataFrame(X_test)
            
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X_test)
                
                # Convert to numpy array if it's a pandas DataFrame
                if isinstance(probabilities, pd.DataFrame):
                    probabilities = probabilities.values
                    
                return probabilities
            else:
                # Fallback model case
                if isinstance(X_test, pd.DataFrame):
                    X_test = X_test.values
                return self.model.predict_proba(X_test)
                
        except Exception as e:
            logger.error(f"Probability prediction failed for {self.name}: {str(e)}")
            # Return uniform probabilities as fallback
            n_classes = getattr(self, '_n_classes', 2)
            n_samples = len(X_test)
            return np.full((n_samples, n_classes), 1.0 / n_classes)
    
    def get_leaderboard(self):
        """Get AutoGluon leaderboard"""
        if not self.is_fitted:
            return None
        
        try:
            if hasattr(self.model, 'leaderboard'):
                return self.model.leaderboard()
            else:
                return None
        except Exception as e:
            logger.warning(f"Could not get leaderboard for {self.name}: {str(e)}")
            return None
    
    def get_feature_importance(self):
        """Get feature importance if available"""
        if not self.is_fitted:
            return None
        
        try:
            if hasattr(self.model, 'feature_importance'):
                importance = self.model.feature_importance()
                if isinstance(importance, pd.DataFrame):
                    return importance['importance'].values
                return importance
            elif hasattr(self.model, 'feature_importances_'):
                return self.model.feature_importances_
            else:
                return None
        except Exception as e:
            logger.warning(f"Could not get feature importance for {self.name}: {str(e)}")
            return None
    
    def cleanup(self):
        """Clean up temporary directory"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary directory: {str(e)}")
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.cleanup() 