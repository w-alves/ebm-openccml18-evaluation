#!/usr/bin/env python3
"""
Base Model Class
Defines the interface for all machine learning models
"""

import time
import numpy as np
from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelBinarizer
import logging

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """Abstract base class for all machine learning models"""
    
    def __init__(self, name: str, random_state: int = 42):
        """
        Initialize base model
        
        Args:
            name: Name of the model
            random_state: Random seed for reproducibility
        """
        self.name = name
        self.random_state = random_state
        self.model = None
        self.is_fitted = False
        self.tuning_time = 0.0
        self.training_time = 0.0
        self.prediction_time = 0.0
        self.best_params = {}
        
    @abstractmethod
    def _create_model(self, **params):
        """Create the underlying model instance"""
        pass
    
    @abstractmethod
    def _get_param_grid(self):
        """Get hyperparameter grid for tuning"""
        pass
    
    def fit(self, X_train, y_train, **kwargs):
        """
        Fit the model to training data
        
        Args:
            X_train: Training features
            y_train: Training targets
            **kwargs: Additional fitting parameters
        """
        start_time = time.perf_counter()
        
        if self.model is None:
            self.model = self._create_model(**self.best_params)
        
        self.model.fit(X_train, y_train, **kwargs)
        self.is_fitted = True
        
        self.training_time = time.perf_counter() - start_time
        logger.info(f"{self.name} training completed in {self.training_time:.2f}s")
    
    def predict(self, X_test):
        """
        Make predictions on test data
        
        Args:
            X_test: Test features
            
        Returns:
            np.array: Predicted class labels
        """
        if not self.is_fitted:
            raise ValueError(f"{self.name} must be fitted before making predictions")
        
        start_time = time.perf_counter()
        predictions = self.model.predict(X_test)
        self.prediction_time = time.perf_counter() - start_time
        
        return predictions
    
    def predict_proba(self, X_test):
        """
        Make probability predictions on test data
        
        Args:
            X_test: Test features
            
        Returns:
            np.array: Predicted class probabilities
        """
        if not self.is_fitted:
            raise ValueError(f"{self.name} must be fitted before making predictions")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_test)
        else:
            # Fallback for models without predict_proba
            logger.warning(f"{self.name} doesn't support predict_proba, using decision_function")
            if hasattr(self.model, 'decision_function'):
                scores = self.model.decision_function(X_test)
                # Convert to probabilities using softmax
                exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
                return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            else:
                raise ValueError(f"{self.name} doesn't support probability predictions")
    
    def tune(self, X_train, y_train, cv_folds=5, **kwargs):
        """
        Tune hyperparameters using cross-validation
        
        Args:
            X_train: Training features
            y_train: Training targets
            cv_folds: Number of CV folds
            **kwargs: Additional tuning parameters
        """
        start_time = time.perf_counter()
        
        param_grid = self._get_param_grid()
        best_score = -np.inf
        best_params = {}
        
        # Stratified K-Fold cross-validation
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        logger.info(f"Tuning {self.name} with {len(param_grid)} parameter combinations...")
        
        for i, params in enumerate(param_grid):
            cv_scores = []
            
            for train_idx, val_idx in skf.split(X_train, y_train):
                X_train_cv = X_train[train_idx]
                X_val_cv = X_train[val_idx]
                y_train_cv = y_train[train_idx]
                y_val_cv = y_train[val_idx]
                
                # Create and fit model with current parameters
                model = self._create_model(**params)
                model.fit(X_train_cv, y_train_cv)
                
                # Evaluate on validation set
                y_pred = model.predict(X_val_cv)
                score = accuracy_score(y_val_cv, y_pred)
                cv_scores.append(score)
            
            mean_score = np.mean(cv_scores)
            
            if mean_score > best_score:
                best_score = mean_score
                best_params = params
            
            logger.debug(f"Params {i+1}/{len(param_grid)}: {params} -> CV Score: {mean_score:.4f}")
        
        self.best_params = best_params
        self.tuning_time = time.perf_counter() - start_time
        
        logger.info(f"{self.name} tuning completed in {self.tuning_time:.2f}s")
        logger.info(f"Best CV score: {best_score:.4f}")
        logger.info(f"Best params: {best_params}")
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance on test data
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError(f"{self.name} must be fitted before evaluation")
        
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Calculate AUC (One-vs-One for multiclass)
        try:
            y_proba = self.predict_proba(X_test)
            n_classes = len(np.unique(y_test))
            
            if n_classes == 2:
                # Binary classification
                auc = roc_auc_score(y_test, y_proba[:, 1])
            else:
                # Multiclass classification (One-vs-One)
                auc = roc_auc_score(y_test, y_proba, multi_class='ovo', average='macro')
        except Exception as e:
            logger.warning(f"Could not calculate AUC for {self.name}: {str(e)}")
            auc = np.nan
        
        # Calculate cross-entropy loss
        try:
            y_proba = self.predict_proba(X_test)
            cross_entropy = log_loss(y_test, y_proba)
        except Exception as e:
            logger.warning(f"Could not calculate cross-entropy for {self.name}: {str(e)}")
            cross_entropy = np.nan
        
        # Total time
        total_time = self.tuning_time + self.training_time + self.prediction_time
        
        results = {
            'model_name': self.name,
            'accuracy': accuracy,
            'auc_ovo': auc,
            'cross_entropy': cross_entropy,
            'tuning_time': self.tuning_time,
            'training_time': self.training_time,
            'prediction_time': self.prediction_time,
            'total_time': total_time,
            'best_params': self.best_params
        }
        
        logger.info(f"{self.name} evaluation results:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  AUC (OvO): {auc:.4f}")
        logger.info(f"  Cross-entropy: {cross_entropy:.4f}")
        logger.info(f"  Total time: {total_time:.2f}s")
        
        return results
    
    def get_feature_importance(self):
        """
        Get feature importance if available
        
        Returns:
            np.array or None: Feature importance scores
        """
        if not self.is_fitted:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # For linear models, use absolute coefficients
            return np.abs(self.model.coef_).flatten()
        else:
            return None
    
    def reset(self):
        """Reset the model to unfitted state"""
        self.model = None
        self.is_fitted = False
        self.tuning_time = 0.0
        self.training_time = 0.0
        self.prediction_time = 0.0
        self.best_params = {} 