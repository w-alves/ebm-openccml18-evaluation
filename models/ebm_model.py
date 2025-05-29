#!/usr/bin/env python3
"""
Explainable Boosting Machine (EBM) Model Implementation
Uses the interpret library for EBM classifier
"""

from .base_model import BaseModel
from interpret.glassbox import ExplainableBoostingClassifier
import itertools
import logging

logger = logging.getLogger(__name__)

class EBMModel(BaseModel):
    """Explainable Boosting Machine model wrapper"""
    
    def __init__(self, random_state=42):
        super().__init__("EBM", random_state)
    
    def _create_model(self, **params):
        """Create EBM classifier instance"""
        return ExplainableBoostingClassifier(
            random_state=self.random_state,
            **params
        )
    
    def _get_param_grid(self):
        """Get hyperparameter grid for EBM tuning"""
        # Simplified parameter grid with max 20 combinations (3x3x2 = 18)
        param_grid = {
            'max_bins': [32, 64, 128],            # 3 values
            'interactions': [3, 5, 10],           # 3 values
            'learning_rate': [0.05, 0.1]         # 2 values
        }
        
        # Fixed parameters for all combinations
        fixed_params = {
            'max_interaction_bins': 32,
            'outer_bags': 16,
            'inner_bags': 0,
            'min_samples_leaf': 2,
            'max_leaves': 3
        }
        
        # Generate all combinations
        param_combinations = []
        keys = list(param_grid.keys())
        for values in itertools.product(*param_grid.values()):
            param_dict = dict(zip(keys, values))
            param_dict.update(fixed_params)
            param_combinations.append(param_dict)
        
        logger.info(f"Generated {len(param_combinations)} EBM parameter combinations")
        return param_combinations
    
    def get_global_explanation(self):
        """
        Get global explanation from EBM model
        
        Returns:
            dict: Global explanation information
        """
        if not self.is_fitted:
            return None
        
        try:
            explanation = self.model.explain_global()
            return {
                'explanation_type': 'global',
                'feature_names': explanation.data()['names'],
                'feature_scores': explanation.data()['scores'],
                'explanation_object': explanation
            }
        except Exception as e:
            logger.warning(f"Could not get global explanation: {str(e)}")
            return None
    
    def get_local_explanation(self, X_sample):
        """
        Get local explanation for a specific sample
        
        Args:
            X_sample: Single sample to explain
            
        Returns:
            dict: Local explanation information
        """
        if not self.is_fitted:
            return None
        
        try:
            explanation = self.model.explain_local(X_sample)
            return {
                'explanation_type': 'local',
                'feature_names': explanation.data()['names'],
                'feature_scores': explanation.data()['scores'],
                'explanation_object': explanation
            }
        except Exception as e:
            logger.warning(f"Could not get local explanation: {str(e)}")
            return None
    
    def get_feature_importance(self):
        """Get feature importance from EBM model"""
        if not self.is_fitted:
            return None
        
        try:
            # EBM provides feature importance through global explanation
            global_exp = self.get_global_explanation()
            if global_exp:
                return global_exp['feature_scores']
            else:
                return None
        except Exception as e:
            logger.warning(f"Could not get feature importance: {str(e)}")
            return None 