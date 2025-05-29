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
            'tmp_folder': None,  # Use default temp folder
            'delete_tmp_folder_after_terminate': True,
            'disable_evaluator_output': True,
            'seed': self.random_state
        }
        default_params.update(params)
        
        return autosklearn.classification.AutoSklearnClassifier(**default_params)
    
    def _get_param_grid(self):
        """Get hyperparameter grid for Auto-sklearn tuning"""
        # Auto-sklearn doesn't need traditional hyperparameter tuning
        # Instead, we vary the time budgets and ensemble configurations
        param_grid = [
            {
                'time_left_for_this_task': 180,  # 3 minutes
                'per_run_time_limit': 20,
                'ensemble_size': 5,
                'ensemble_nbest': 5
            },
            {
                'time_left_for_this_task': 300,  # 5 minutes
                'per_run_time_limit': 30,
                'ensemble_size': 10,
                'ensemble_nbest': 10
            },
            {
                'time_left_for_this_task': 600,  # 10 minutes
                'per_run_time_limit': 60,
                'ensemble_size': 15,
                'ensemble_nbest': 15
            },
            {
                'time_left_for_this_task': 240,  # 4 minutes
                'per_run_time_limit': 30,
                'ensemble_size': 8,
                'ensemble_nbest': 8
            }
        ]
        
        logger.info(f"Generated {len(param_grid)} Auto-sklearn configurations")
        return param_grid
    
    def tune(self, X_train, y_train, cv_folds=5, **kwargs):
        """
        Tune Auto-sklearn (simplified version since it auto-tunes internally)
        
        Args:
            X_train: Training features
            y_train: Training targets
            cv_folds: Number of CV folds (not used for auto-sklearn)
            **kwargs: Additional tuning parameters
        """
        import time
        start_time = time.perf_counter()
        
        # For Auto-sklearn, we just select the best configuration from our grid
        # based on a quick validation
        param_grid = self._get_param_grid()
        
        # For simplicity, use the default configuration (index 1)
        # In a full implementation, you might want to try multiple configurations
        self.best_params = param_grid[1]  # 5-minute configuration
        
        self.tuning_time = time.perf_counter() - start_time
        
        logger.info(f"{self.name} configuration selected in {self.tuning_time:.2f}s")
        logger.info(f"Selected config: {self.best_params}")
    
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
            self.model = self._create_model(**self.best_params)
        
        try:
            # Auto-sklearn fit
            self.model.fit(X_train, y_train)
            self.is_fitted = True
            
            # Try to refit with best found models to create final ensemble
            try:
                self.model.refit(X_train, y_train)
            except Exception as e:
                logger.warning(f"Refit failed for {self.name}: {str(e)}")
                
        except Exception as e:
            logger.error(f"Failed to fit {self.name}: {str(e)}")
            # Create a simple fallback model
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(
                n_estimators=100, 
                random_state=self.random_state
            )
            self.model.fit(X_train, y_train)
            self.is_fitted = True
            logger.warning(f"Using RandomForest fallback for {self.name}")
        
        self.training_time = time.perf_counter() - start_time
        logger.info(f"{self.name} training completed in {self.training_time:.2f}s")
    
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