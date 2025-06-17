#!/usr/bin/env python3
"""
Base Model Class
Defines the interface for all machine learning models
"""

import time
import os
import numpy as np
from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelBinarizer
import logging

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """Abstract base class for all machine learning models"""
    
    def __init__(self, name: str, random_state: int = 42, target_column: str = None, log_file: str = None):
        """
        Initialize base model
        
        Args:
            name: Name of the model
            random_state: Random seed for reproducibility
            target_column: Name of target column (required for AutoGluon models)
            log_file: Path to log file for saving logs to storage (optional)
        """
        self.name = name
        self.random_state = random_state
        self.target_column = target_column
        self.model = None
        self.is_fitted = False
        self.tuning_time = 0.0
        self.training_time = 0.0
        self.prediction_time = 0.0
        self.best_params = {}
        
        # Setup file logging if log_file is provided
        if log_file:
            self._setup_file_logging(log_file)
        
    def _setup_file_logging(self, log_file: str):
        """
        Setup file logging to save logs to storage
        
        Args:
            log_file: Path to the log file
        """
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # Create file handler
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.DEBUG)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
        
        # Also ensure console handler exists
        if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # Set logger level
        logger.setLevel(logging.DEBUG)
        
        logger.info(f"File logging initialized for {self.name}. Logs will be saved to: {log_file}")
        
    @abstractmethod
    def _create_model(self, **params):
        """Create the underlying model instance"""
        pass
    
    @abstractmethod
    def _get_param_grid(self):
        """Get hyperparameter grid for tuning"""
        pass
    
    def _is_autogluon_model(self):
        """Check if this is an AutoGluon model"""
        return 'autogluon' in self.name.lower() or hasattr(self, '_is_autogluon')
    
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
        
        # Store detailed CV results
        self.cv_results = []
        
        # Stratified K-Fold cross-validation
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        logger.info(f"Tuning model {self.name} with {len(param_grid)} parameter combinations...")
        
        for i, params in enumerate(param_grid):
            logger.info(f"Testing parameter combination {i + 1}/{len(param_grid)}: {params}")
            cv_scores = []
            param_cv_results = {
                'param_combination': i + 1,
                'params': params,
                'folds': []
            }
            
            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
                logger.info(f"Processing fold {fold_idx + 1}/{cv_folds} for params {params}")
                fold_start_time = time.perf_counter()
                
                X_train_cv = X_train[train_idx]
                X_val_cv = X_train[val_idx]
                y_train_cv = y_train[train_idx]
                y_val_cv = y_train[val_idx]
                
                # Convert to DataFrame if necessary
                import pandas as pd
                if isinstance(X_train_cv, np.ndarray):
                    X_train_cv = pd.DataFrame(X_train_cv)
                    X_val_cv = pd.DataFrame(X_val_cv)
                if isinstance(y_train_cv, np.ndarray):
                    y_train_cv = pd.DataFrame(y_train_cv)
                    y_val_cv = pd.DataFrame(y_val_cv)

                # Create and fit model with current parameters
                model = self._create_model(**params)
                
                # Time training
                train_start = time.perf_counter()
                
                # Handle AutoGluon models differently
                if self._is_autogluon_model():
                    logger.info(f"Fitting AutoGluon model with params: {params}")  
                    # For AutoGluon, combine X and y into single DataFrame
                    if self.target_column is None:
                        raise ValueError("target_column must be specified for AutoGluon models")
                    
                    # Combine features and target into single DataFrame
                    if isinstance(X_train_cv, np.ndarray):
                        import pandas as pd
                        X_train_cv = pd.DataFrame(X_train_cv)
                    
                    train_data = X_train_cv.copy()
                    if isinstance(y_train_cv, (pd.DataFrame, pd.Series)):
                        train_data[self.target_column] = y_train_cv.values if hasattr(y_train_cv, 'values') else y_train_cv
                    else:
                        train_data[self.target_column] = y_train_cv
                    
                    model.fit(train_data)
                else:
                    # Standard sklearn-style fit
                    model.fit(X_train_cv, y_train_cv)
                
                training_time = time.perf_counter() - train_start
                
                # Time predictions
                pred_start = time.perf_counter()
                
                if self._is_autogluon_model():
                    # For AutoGluon, pass DataFrame directly
                    if isinstance(X_val_cv, np.ndarray):
                        import pandas as pd
                        X_val_cv = pd.DataFrame(X_val_cv)
                    y_pred = model.predict(X_val_cv)
                else:
                    # Standard sklearn-style predict
                    y_pred = model.predict(X_val_cv)
                
                prediction_time = time.perf_counter() - pred_start
                
                # Calculate accuracy
                accuracy = accuracy_score(y_val_cv, y_pred)
                cv_scores.append(accuracy)
                
                # Calculate AUC (One-vs-One for multiclass)
                try:
                    if hasattr(model, 'predict_proba'):
                        if self._is_autogluon_model():
                            if isinstance(X_val_cv, np.ndarray):
                                import pandas as pd
                                X_val_cv = pd.DataFrame(X_val_cv)
                            y_proba = model.predict_proba(X_val_cv)
                            y_proba = y_proba.to_numpy()
                        else:
                            y_proba = model.predict_proba(X_val_cv)

                        y_proba = np.squeeze(y_proba)  # Handles extra dimensions if present
                        n_classes = len(np.unique(y_val_cv))

                        try:
                            if n_classes == 2:
                                auc_ovo = roc_auc_score(y_val_cv, y_proba[:, 1])
                            else:
                                auc_ovo = roc_auc_score(y_val_cv, y_proba, multi_class='ovo', average='macro')
                        except Exception as e1:
                            print("1st attempt failed:", e1)

                            try:
                                # Attempt with reshaping
                                y_proba = y_proba.reshape((y_proba.shape[0], -1))
                                if n_classes == 2:
                                    auc_ovo = roc_auc_score(y_val_cv, y_proba[:, 1])
                                else:
                                    auc_ovo = roc_auc_score(y_val_cv, y_proba, multi_class='ovo', average='macro')
                            except Exception as e2:
                                print("2nd attempt (reshape) failed:", e2)

                                try:
                                    # Attempt with one-hot encoding of y_val_cv
                                    from sklearn.preprocessing import label_binarize
                                    y_val_bin = label_binarize(y_val_cv, classes=np.unique(y_val_cv))
                                    auc_ovo = roc_auc_score(y_val_bin, y_proba, average='macro', multi_class='ovo')
                                except Exception as e3:
                                    print("3rd attempt (label_binarize) failed:", e3)

                                    try:
                                        # Attempt with different multi_class mode
                                        auc_ovo = roc_auc_score(y_val_cv, y_proba, multi_class='ovr', average='macro')
                                    except Exception as e4:
                                        print("4th attempt (ovr) failed:", e4)
                    else:
                        auc_ovo = -999
                except Exception as e:
                    import traceback
                    error = f"Could not calculate AUC for fold {fold_idx + 1}:\n{traceback.format_exc()}"
                    logger.debug(error)
                    print(error)
                    logger.info(error)
                    logger.warning(error)
                    auc_ovo = np.nan
                
                # Calculate cross-entropy loss
                try:
                    if hasattr(model, 'predict_proba'):
                        if self._is_autogluon_model():
                            # For AutoGluon, ensure X_val_cv is DataFrame
                            if isinstance(X_val_cv, np.ndarray):
                                import pandas as pd
                                X_val_cv = pd.DataFrame(X_val_cv)
                            y_proba = model.predict_proba(X_val_cv)
                        else:
                            y_proba = model.predict_proba(X_val_cv)
                        cross_entropy = log_loss(y_val_cv, y_proba)
                    else:
                        cross_entropy = np.nan
                except Exception as e:
                    logger.debug(f"Could not calculate cross-entropy for fold {fold_idx + 1}: {str(e)}")
                    cross_entropy = np.nan
                
                # Total time for this fold
                total_time = time.perf_counter() - fold_start_time
                
                # Store fold results
                fold_result = {
                    'fold': fold_idx + 1,
                    'accuracy': accuracy,
                    'auc_ovo': auc_ovo,
                    'cross_entropy': cross_entropy,
                    'tuning_time': 0.0,  # Individual fold doesn't have tuning time
                    'training_time': training_time,
                    'prediction_time': prediction_time,
                    'total_time': total_time,
                    'train_samples': len(y_train_cv),
                    'val_samples': len(y_val_cv)
                }
                
                param_cv_results['folds'].append(fold_result)
            
            # Calculate mean metrics across folds
            mean_score = np.mean(cv_scores)
            
            # Add summary for this parameter combination
            param_cv_results['mean_accuracy'] = mean_score
            param_cv_results['std_accuracy'] = np.std(cv_scores)
            
            # Calculate mean of other metrics across folds
            fold_metrics = ['auc_ovo', 'cross_entropy', 'training_time', 'prediction_time', 'total_time']
            for metric in fold_metrics:
                values = [fold[metric] for fold in param_cv_results['folds'] if not np.isnan(fold[metric])]
                if values:
                    param_cv_results[f'mean_{metric}'] = np.mean(values)
                    param_cv_results[f'std_{metric}'] = np.std(values)
                else:
                    param_cv_results[f'mean_{metric}'] = np.nan
                    param_cv_results[f'std_{metric}'] = np.nan
            
            self.cv_results.append(param_cv_results)
            
            if mean_score > best_score:
                best_score = mean_score
                best_params = params
            
            logger.debug(f"Params {i+1}/{len(param_grid)}: {params} -> CV Score: {mean_score:.4f} (±{np.std(cv_scores):.4f})")
        
        self.best_params = best_params
        self.tuning_time = time.perf_counter() - start_time
        
        # Add tuning time to CV results
        for param_result in self.cv_results:
            # Distribute tuning time proportionally across parameter combinations
            param_result['tuning_time'] = self.tuning_time / len(param_grid)
            # Update total time for each fold to include tuning time
            for fold in param_result['folds']:
                fold['tuning_time'] = param_result['tuning_time'] / cv_folds
                fold['total_time'] += fold['tuning_time']
        
        logger.info(f"{self.name} tuning completed in {self.tuning_time:.2f}s")
        logger.info(f"Best CV score: {best_score:.4f}")
        logger.info(f"Best params: {best_params}")
    
    def get_cv_results(self):
        """
        Get detailed cross-validation results
        
        Returns:
            list: List of CV results for each parameter combination
        """
        return getattr(self, 'cv_results', []) or getattr(self, 'cv_results_', [])
    
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
        self.cv_results = []  # Clear CV results as well 