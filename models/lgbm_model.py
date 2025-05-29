#!/usr/bin/env python3
"""
LightGBM Model Implementation
Uses LightGBM for gradient boosting with optimized hyperparameter tuning
"""

from .base_model import BaseModel
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import itertools
import logging
import numpy as np

logger = logging.getLogger(__name__)

class LGBMModel(BaseModel):
    """LightGBM model wrapper"""
    
    def __init__(self, random_state=42):
        super().__init__("LightGBM", random_state)
        self.num_classes = None
    
    def _create_model(self, **params):
        """Create LightGBM classifier instance"""
        # Set objective and metric based on number of classes
        if self.num_classes is not None:
            if self.num_classes == 2:
                # Binary classification
                default_params = {
                    'objective': 'binary',
                    'metric': 'binary_logloss',
                    'boosting_type': 'gbdt',
                    'random_state': self.random_state,
                    'verbosity': -1,
                    'force_row_wise': True
                }
                logger.info(f"Configuring LGBM for binary classification ({self.num_classes} classes)")
            else:
                # Multiclass classification
                default_params = {
                    'objective': 'multiclass',
                    'metric': 'multi_logloss',
                    'boosting_type': 'gbdt',
                    'random_state': self.random_state,
                    'verbosity': -1,
                    'force_row_wise': True,
                    'num_class': self.num_classes
                }
                logger.info(f"Configuring LGBM for multiclass classification ({self.num_classes} classes)")
        else:
            # Default to multiclass if num_classes not set yet (for tuning phase)
            default_params = {
                'objective': 'multiclass',
                'metric': 'multi_logloss',
                'boosting_type': 'gbdt',
                'random_state': self.random_state,
                'verbosity': -1,
                'force_row_wise': True
            }
            logger.warning("Number of classes not determined yet, using multiclass default")
        
        default_params.update(params)
        
        return lgb.LGBMClassifier(**default_params)
    
    def _get_param_grid(self):
        """Get hyperparameter grid for LightGBM tuning"""
        # Simplified parameter grid with max 20 combinations (2x2x2x2x2 = 16)
        param_grid = {
            'n_estimators': [200, 500],           # 2 values
            'max_depth': [5, 7],                  # 2 values
            'num_leaves': [31, 63],               # 2 values
            'reg_alpha': [0.0, 0.1]              # 2 values
        }
        
        # Fixed parameters for all combinations
        fixed_params = {
            'feature_fraction': 0.9,
            'bagging_fraction': 0.9,
            'bagging_freq': 5,
            'min_child_samples': 10,
            'min_child_weight': 1e-3,
            'reg_lambda': 0.1
        }
        
        # Add class-specific parameters
        if self.num_classes is not None:
            if self.num_classes == 2:
                # Binary classification specific parameters
                fixed_params.update({
                    'objective': 'binary',
                    'metric': 'binary_logloss'
                })
            else:
                # Multiclass classification specific parameters
                fixed_params.update({
                    'objective': 'multiclass',
                    'metric': 'multi_logloss',
                    'num_class': self.num_classes
                })
        
        # Generate all combinations
        param_combinations = []
        keys = list(param_grid.keys())
        for values in itertools.product(*param_grid.values()):
            param_dict = dict(zip(keys, values))
            param_dict.update(fixed_params)
            param_combinations.append(param_dict)
        
        logger.info(f"Generated {len(param_combinations)} LightGBM parameter combinations")
        return param_combinations
    
    def tune(self, X_train, y_train, cv_folds=5, **kwargs):
        """
        Tune hyperparameters using cross-validation
        Override to ensure num_classes is set before tuning
        
        Args:
            X_train: Training features
            y_train: Training targets
            cv_folds: Number of CV folds
            **kwargs: Additional tuning parameters
        """
        # Determine number of classes from training data before tuning
        self.num_classes = len(np.unique(y_train))
        logger.info(f"Detected {self.num_classes} classes before tuning")
        
        # Call parent tune method
        super().tune(X_train, y_train, cv_folds, **kwargs)
    
    def fit(self, X_train, y_train, **kwargs):
        """
        Fit LightGBM with early stopping if validation data provided
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            **kwargs: Additional fitting parameters
        """
        # Determine number of classes from training data
        self.num_classes = len(np.unique(y_train))
        logger.info(f"Detected {self.num_classes} classes for classification")
        
        # Extract validation data if provided
        X_val = kwargs.pop('X_val', None)
        y_val = kwargs.pop('y_val', None)
        
        fit_params = {}
        
        # Add early stopping if validation data is provided
        if X_val is not None and y_val is not None:
            fit_params['eval_set'] = [(X_val, y_val)]
            # Set appropriate eval metric based on number of classes
            if self.num_classes == 2:
                fit_params['eval_metric'] = 'binary_logloss'
            else:
                fit_params['eval_metric'] = 'multi_logloss'
            fit_params['callbacks'] = [lgb.early_stopping(50, verbose=False)]
        
        # Update with any additional parameters
        fit_params.update(kwargs)
        
        # Call parent fit method
        super().fit(X_train, y_train, **fit_params)
    
    def get_feature_importance(self):
        """Get feature importance from LightGBM model"""
        if not self.is_fitted:
            return None
        
        return self.model.feature_importances_ 