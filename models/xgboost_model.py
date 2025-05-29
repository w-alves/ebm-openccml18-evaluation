#!/usr/bin/env python3
"""
XGBoost Model Implementation
Uses XGBoost for gradient boosting with optimized hyperparameter tuning
"""

from .base_model import BaseModel
import xgboost as xgb
import itertools
import logging

logger = logging.getLogger(__name__)

class XGBoostModel(BaseModel):
    """XGBoost model wrapper"""
    
    def __init__(self, random_state=42):
        super().__init__("XGBoost", random_state)
    
    def _create_model(self, **params):
        """Create XGBoost classifier instance"""
        default_params = {
            'random_state': self.random_state,
            'verbosity': 0,
            'use_label_encoder': False,
            'eval_metric': 'mlogloss'
        }
        default_params.update(params)
        
        return xgb.XGBClassifier(**default_params)
    
    def _get_param_grid(self):
        """Get hyperparameter grid for XGBoost tuning"""
        # Simplified parameter grid with max 20 combinations (2x2x2x2 = 16)
        param_grid = {
            'n_estimators': [200, 500],           # 2 values
            'max_depth': [4, 6],                  # 2 values  
            'subsample': [0.8, 0.9],              # 2 values
            'reg_alpha': [0, 0.1]                 # 2 values
        }
        
        # Fixed parameters for all combinations
        fixed_params = {
            'colsample_bytree': 0.9,
            'colsample_bylevel': 0.9,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_lambda': 0.1
        }
        
        # Generate all combinations
        param_combinations = []
        keys = list(param_grid.keys())
        for values in itertools.product(*param_grid.values()):
            param_dict = dict(zip(keys, values))
            param_dict.update(fixed_params)
            param_combinations.append(param_dict)
        
        logger.info(f"Generated {len(param_combinations)} XGBoost parameter combinations")
        return param_combinations
    
    def fit(self, X_train, y_train, **kwargs):
        """
        Fit XGBoost with early stopping if validation data provided
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            **kwargs: Additional fitting parameters
        """
        # Extract validation data if provided
        X_val = kwargs.pop('X_val', None)
        y_val = kwargs.pop('y_val', None)
        
        fit_params = {}
        
        # Add early stopping if validation data is provided
        if X_val is not None and y_val is not None:
            fit_params['eval_set'] = [(X_val, y_val)]
            fit_params['early_stopping_rounds'] = 50
            fit_params['verbose'] = False
        
        # Update with any additional parameters
        fit_params.update(kwargs)
        
        # Call parent fit method
        super().fit(X_train, y_train, **fit_params)
    
    def get_feature_importance(self):
        """Get feature importance from XGBoost model"""
        if not self.is_fitted:
            return None
        
        return self.model.feature_importances_ 