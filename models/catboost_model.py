#!/usr/bin/env python3
"""
CatBoost Model Implementation
Uses CatBoost for gradient boosting with categorical feature support
"""

from .base_model import BaseModel
from catboost import CatBoostClassifier
import itertools
import logging

logger = logging.getLogger(__name__)

class CatBoostModel(BaseModel):
    """CatBoost model wrapper"""
    
    def __init__(self, random_state=42):
        super().__init__("CatBoost", random_state)
    
    def _create_model(self, **params):
        """Create CatBoost classifier instance"""
        default_params = {
            'random_state': self.random_state,
            'verbose': False,
            'allow_writing_files': False,
            'thread_count': -1
        }
        default_params.update(params)
        
        return CatBoostClassifier(**default_params)
    
    def _get_param_grid(self):
        """Get hyperparameter grid for CatBoost tuning"""
        # Simplified parameter grid with max 20 combinations (2x2x2x2 = 16)
        param_grid = {
            'iterations': [200, 500],             # 2 values
            'depth': [4, 6],                      # 2 values
            'l2_leaf_reg': [3, 5],                # 2 values
            'bootstrap_type': ['Bayesian', 'Bernoulli']  # 2 values
        }
        

        fixed_params = {
            'border_count': 64,
            'random_strength': 10,
            'grow_policy': 'SymmetricTree'
        }
        
        # Generate all combinations
        param_combinations = []
        keys = list(param_grid.keys())
        for values in itertools.product(*param_grid.values()):
            param_dict = dict(zip(keys, values))
            param_dict.update(fixed_params)
            param_combinations.append(param_dict)
        
        logger.info(f"Generated {len(param_combinations)} CatBoost parameter combinations")
        return param_combinations
    
    def fit(self, X_train, y_train, **kwargs):
        """
        Fit CatBoost with early stopping if validation data provided
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            cat_features: Categorical feature indices (optional)
            **kwargs: Additional fitting parameters
        """
        # Extract validation data if provided
        X_val = kwargs.pop('X_val', None)
        y_val = kwargs.pop('y_val', None)
        cat_features = kwargs.pop('cat_features', None)
        
        fit_params = {}
        
        # Add early stopping if validation data is provided
        if X_val is not None and y_val is not None:
            fit_params['eval_set'] = [(X_val, y_val)]
            fit_params['early_stopping_rounds'] = 50
            fit_params['verbose'] = False
        
        # Add categorical features if provided
        if cat_features is not None:
            fit_params['cat_features'] = cat_features
        
        # Update with any additional parameters
        fit_params.update(kwargs)
        
        # Call parent fit method
        super().fit(X_train, y_train, **fit_params)
    
    def get_feature_importance(self):
        """Get feature importance from CatBoost model"""
        if not self.is_fitted:
            return None
        
        return self.model.get_feature_importance() 