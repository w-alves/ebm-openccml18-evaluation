#!/usr/bin/env python3
"""
Auto-sklearn Model Implementation
Uses Auto-sklearn 2.0 for automated machine learning
"""

from .base_model import BaseModel
import autosklearn.classification
import logging
import numpy as np

logger = logging.getLogger(__name__)

class AutoSklearnModel(BaseModel):
    """Auto-sklearn model wrapper"""
    
    def __init__(self, random_state=42):
        super().__init__("Auto-sklearn", random_state)
    
    def _create_model(self, **params):
        """Create Auto-sklearn classifier instance"""
        default_params = {
            'seed': self.random_state
            ,'time_left_for_this_task': 120  # 5 minutes default
            , 'per_run_time_limit': 30  # 30 seconds per model
            , 'n_jobs': -1  # Use all available cores
        }
        default_params.update(params)
        
        return autosklearn.classification.AutoSklearnClassifier(**default_params)
    
    def _get_param_grid(self):
        """Get hyperparameter grid for Auto-sklearn tuning"""
        # Auto-sklearn does not require manual parameter grid definition
        # It automatically searches through a large space of models and hyperparameters
        logger.info("Auto-sklearn will automatically search for optimal hyperparameters")
        return [{}]
     
    def fit(self, X_train, y_train, **kwargs):
        """
        Fit Auto-sklearn classifier
        
        Args:
            X_train: Training features
            y_train: Training targets
            **kwargs: Additional fitting parameters
        """
        import time
        start_time = time.perf_counter()
        
        if self.model is None:
            self.model = self._create_model()
        
        try:
            # Auto-sklearn fit - it will automatically perform hyperparameter optimization
            logger.info(f"Starting Auto-sklearn training with automated hyperparameter optimization...")
            self.model.fit(X_train, y_train)
            self.is_fitted = True
            
            # Try to refit with best found models to create final ensemble
            try:
                self.model.refit(X_train, y_train)
            except Exception as e:
                logger.warning(f"Refit failed for {self.name}: {str(e)}")
                
        except Exception as e:
            raise RuntimeError(f"Auto-sklearn fitting failed: {str(e)}")
    
    def predict(self, X_test):
        """Make predictions on test data"""
        if not self.is_fitted:
            raise ValueError(f"{self.name} must be fitted before making predictions")
        
        import time
        start_time = time.perf_counter()
        
        try:
            predictions = self.model.predict(X_test)
        except Exception as e:
            logger.error(f"Prediction failed for {self.name}: {str(e)}")
            # Return random predictions as fallback
            n_classes = len(np.unique(self.model.classes_)) if hasattr(self.model, 'classes_') else 2
            predictions = np.random.choice(n_classes, size=len(X_test))
        
        self.prediction_time = time.perf_counter() - start_time
        return predictions
    
    def predict_proba(self, X_test):
        """Make probability predictions on test data"""
        if not self.is_fitted:
            raise ValueError(f"{self.name} must be fitted before making predictions")
        
        try:
            return self.model.predict_proba(X_test)
        except Exception as e:
            logger.error(f"Probability prediction failed for {self.name}: {str(e)}")
            # Return uniform probabilities as fallback
            n_classes = len(np.unique(self.model.classes_)) if hasattr(self.model, 'classes_') else 2
            n_samples = len(X_test)
            return np.full((n_samples, n_classes), 1.0 / n_classes)
    
    def get_model_info(self):
        """Get information about the selected models"""
        if not self.is_fitted:
            return None
        
        try:
            # Get statistics about the AutoML run
            stats = self.model.sprint_statistics()
            return {
                'statistics': stats,
                'models': str(self.model.show_models()),
                'leaderboard': self.model.leaderboard()
            }
        except Exception as e:
            logger.warning(f"Could not get model info for {self.name}: {str(e)}")
            return None
    
    def get_feature_importance(self):
        """Get feature importance if available"""
        if not self.is_fitted:
            return None
        
        try:
            # Auto-sklearn doesn't directly provide feature importance
            # Try to get it from the underlying models if possible
            if hasattr(self.model, 'feature_importances_'):
                return self.model.feature_importances_
            else:
                return None
        except Exception as e:
            logger.warning(f"Could not get feature importance for {self.name}: {str(e)}")
            return None 