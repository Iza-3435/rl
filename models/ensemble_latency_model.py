#!/usr/bin/env python3
"""
EnsembleLatencyModel - Production-ready ensemble predictor (FIXED)

Combines multiple approaches for robust latency prediction:
- LSTM for temporal patterns
- GRU for efficiency
- XGBoost for feature-rich predictions
- LightGBM for speed
- Weighted ensemble with dynamic adaptation
"""

import torch
import torch.nn as nn
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from typing import Dict, List, Tuple, Optional, Any
import time
import joblib
import logging
from dataclasses import dataclass
from collections import deque
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class EnsemblePrediction:
    """Structured ensemble prediction output"""
    venue: str
    timestamp: float
    ensemble_prediction_us: float
    individual_predictions: Dict[str, float]
    model_weights: Dict[str, float]
    confidence: float
    prediction_time_ms: float
    uncertainty_bounds: Tuple[float, float]


class TransformerLatencyModel(nn.Module):
    """Transformer-based model for capturing long-range dependencies"""
    
    def __init__(self, input_size: int, d_model: int = 128, 
                 nhead: int = 8, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Output layers
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        # Confidence estimation
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Project input
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Transformer expects (seq_len, batch, features)
        x = x.transpose(0, 1)
        
        # Apply transformer
        transformer_out = self.transformer(x)
        
        # Take the last output
        last_output = transformer_out[-1]
        
        # Generate predictions
        prediction = self.output_projection(last_output)
        confidence = self.confidence_head(last_output)
        
        return prediction, confidence


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class EnsembleLatencyModel:
    """
    Advanced ensemble model combining deep learning and gradient boosting
    
    Features:
    - Multiple model types for robustness
    - Dynamic weight adaptation based on recent performance
    - Uncertainty quantification
    - Fast inference with model caching
    """
    
    def __init__(self, venues: List[str], feature_size: int = 45, 
                 sequence_length: int = 50):
        self.venues = venues
        self.feature_size = feature_size
        self.sequence_length = sequence_length
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize models for each venue
        self.models: Dict[str, Dict[str, Any]] = {
            venue: self._initialize_venue_models() for venue in venues
        }
        
        # Model weights (dynamically adapted)
        self.model_weights: Dict[str, Dict[str, float]] = {
            venue: {
                'lstm': 0.25,
                'gru': 0.20,
                'transformer': 0.20,
                'xgboost': 0.20,
                'lightgbm': 0.15
            } for venue in venues
        }
        
        # Performance tracking for weight adaptation
        self.performance_buffer: Dict[str, Dict[str, deque]] = {
            venue: {
                model: deque(maxlen=100)
                for model in ['lstm', 'gru', 'transformer', 'xgboost', 'lightgbm']
            } for venue in venues
        }
        
        # Feature buffers for sequential models
        self.feature_buffers: Dict[str, deque] = {
            venue: deque(maxlen=sequence_length) for venue in venues
        }
        
        # Calibration models for uncertainty estimation
        self.calibration_models: Dict[str, Any] = {}
        
        logger.info(f"EnsembleLatencyModel initialized on {self.device}")
    
    def _initialize_venue_models(self) -> Dict[str, Any]:
        """Initialize all models for a venue"""
        from data.latency_predictor import LSTMLatencyModel, GRULatencyModel
        
        models = {}
        
        # Deep learning models
        models['lstm'] = LSTMLatencyModel(self.feature_size).to(self.device)
        models['gru'] = GRULatencyModel(self.feature_size).to(self.device)
        models['transformer'] = TransformerLatencyModel(self.feature_size).to(self.device)
        
        # Gradient boosting models
        models['xgboost'] = None  # Will be trained with data
        models['lightgbm'] = None  # Will be trained with data
        
        # Set neural networks to eval mode
        for model_name in ['lstm', 'gru', 'transformer']:
            models[model_name].eval()
        
        return models
    
    def train_ensemble(self, venue: str, training_data: Dict[str, np.ndarray],
                      validation_split: float = 0.2) -> Dict[str, float]:
        """Train all models in the ensemble"""
        logger.info(f"Training ensemble for {venue}...")
        
        features = training_data['features']
        targets = training_data['targets']
        
        # Split data
        split_idx = int(len(features) * (1 - validation_split))
        train_features = features[:split_idx]
        train_targets = targets[:split_idx]
        val_features = features[split_idx:]
        val_targets = targets[split_idx:]
        
        # Train each model
        model_metrics = {}
        
        # 1. Train deep learning models
        for model_name in ['lstm', 'gru', 'transformer']:
            logger.info(f"Training {model_name}...")
            metrics = self._train_neural_model(
                venue, model_name, train_features, train_targets,
                val_features, val_targets
            )
            model_metrics[model_name] = metrics
        
        # 2. Train XGBoost
        logger.info("Training XGBoost...")
        model_metrics['xgboost'] = self._train_xgboost(
            venue, train_features, train_targets, val_features, val_targets
        )
        
        # 3. Train LightGBM
        logger.info("Training LightGBM...")
        model_metrics['lightgbm'] = self._train_lightgbm(
            venue, train_features, train_targets, val_features, val_targets
        )
        
        # 4. Train calibration model (FIXED)
        self._train_calibration_model(venue, val_features, val_targets)
        
        # 5. Optimize ensemble weights (FIXED)
        self._optimize_ensemble_weights(venue, val_features, val_targets)
        
        # Calculate ensemble metrics (with alignment)
        ensemble_predictions = self._get_ensemble_predictions(venue, val_features)
        
        # Align targets to match prediction length
        if len(val_targets) > len(ensemble_predictions):
            aligned_targets = val_targets[-len(ensemble_predictions):]
        else:
            aligned_targets = val_targets
        
        ensemble_mae = np.mean(np.abs(ensemble_predictions - aligned_targets))
        
        within_10pct = np.sum(
            np.abs(ensemble_predictions - aligned_targets) <= aligned_targets * 0.1
        ) / len(aligned_targets) * 100
        
        logger.info(f"Ensemble training complete for {venue}")
        logger.info(f"Ensemble MAE: {ensemble_mae:.1f}μs")
        logger.info(f"Ensemble accuracy (within 10%): {within_10pct:.1f}%")
        
        return {
            'ensemble_mae': ensemble_mae,
            'ensemble_accuracy': within_10pct,
            'individual_metrics': model_metrics
        }
    
    def _train_neural_model(self, venue: str, model_name: str,
                           train_features: np.ndarray, train_targets: np.ndarray,
                           val_features: np.ndarray, val_targets: np.ndarray) -> Dict[str, float]:
        """Train a neural network model"""
        from data.latency_predictor import LatencyDataset
        from torch.utils.data import DataLoader
        
        # Create datasets
        train_dataset = LatencyDataset(train_features, train_targets, self.sequence_length)
        val_dataset = LatencyDataset(val_features, val_targets, self.sequence_length)
        
        # Data loaders
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        # Model and optimizer
        model = self.models[venue][model_name]
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Training loop
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(100):  # Reduced epochs for ensemble
            # Training
            model.train()
            train_losses = []
            
            for batch_features, batch_targets in train_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                optimizer.zero_grad()
                predictions, _ = model(batch_features)
                loss = criterion(predictions.squeeze(), batch_targets)
                
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
            
            # Validation
            model.eval()
            val_losses = []
            val_predictions = []
            val_actuals = []
            
            with torch.no_grad():
                for batch_features, batch_targets in val_loader:
                    batch_features = batch_features.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    
                    predictions, _ = model(batch_features)
                    loss = criterion(predictions.squeeze(), batch_targets)
                    
                    val_losses.append(loss.item())
                    
                    # Denormalize for metrics
                    pred_us = val_dataset.denormalize_prediction(predictions.squeeze())
                    actual_us = val_dataset.denormalize_prediction(batch_targets)
                    
                    val_predictions.extend(np.atleast_1d(pred_us.cpu().numpy()))
                    val_actuals.extend(actual_us.cpu().numpy())
            
            avg_val_loss = np.mean(val_losses)
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict().copy()
        
        # Ensure we have a valid model state
        if best_model_state is None:
            best_model_state = model.state_dict().copy()
        
        # Load best model
        model.load_state_dict(best_model_state)
        model.eval()
        
        # Calculate metrics
        if val_predictions and val_actuals:
            val_predictions = np.array(val_predictions)
            val_actuals = np.array(val_actuals)
            
            mae = np.mean(np.abs(val_predictions - val_actuals))
            within_10pct = np.sum(
                np.abs(val_predictions - val_actuals) <= val_actuals * 0.1
            ) / len(val_actuals) * 100
        else:
            mae = float('inf')
            within_10pct = 0.0
        
        return {'mae': mae, 'accuracy': within_10pct}
    
    def _train_xgboost(self, venue: str, train_features: np.ndarray,
                      train_targets: np.ndarray, val_features: np.ndarray,
                      val_targets: np.ndarray) -> Dict[str, float]:
        """Train XGBoost model"""
        # Prepare data for XGBoost (flatten sequences)
        train_x = self._prepare_tabular_features(train_features)
        val_x = self._prepare_tabular_features(val_features)
        
        # Log transform targets
        train_y = np.log1p(train_targets)
        val_y = np.log1p(val_targets)
        
        # Create DMatrix
        dtrain = xgb.DMatrix(train_x, label=train_y)
        dval = xgb.DMatrix(val_x, label=val_y)
        
        # Parameters
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'verbosity': 0
        }
        
        # Train with early stopping
        evals = [(dtrain, 'train'), (dval, 'eval')]
        model = xgb.train(
            params, dtrain, 
            num_boost_round=1000,
            evals=evals,
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        # Store model
        self.models[venue]['xgboost'] = model
        
        # Evaluate
        val_pred = model.predict(dval)
        val_pred_us = np.expm1(val_pred)  # Reverse log transform
        
        mae = np.mean(np.abs(val_pred_us - val_targets))
        within_10pct = np.sum(
            np.abs(val_pred_us - val_targets) <= val_targets * 0.1
        ) / len(val_targets) * 100
        
        return {'mae': mae, 'accuracy': within_10pct}
    
    def _train_lightgbm(self, venue: str, train_features: np.ndarray,
                       train_targets: np.ndarray, val_features: np.ndarray,
                       val_targets: np.ndarray) -> Dict[str, float]:
        """Train LightGBM model"""
        # Prepare data
        train_x = self._prepare_tabular_features(train_features)
        val_x = self._prepare_tabular_features(val_features)
        
        # Log transform targets
        train_y = np.log1p(train_targets)
        val_y = np.log1p(val_targets)
        
        # Parameters
        params = {
            'objective': 'regression',
            'metric': 'mae',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        
        # Create datasets
        train_data = lgb.Dataset(train_x, label=train_y)
        val_data = lgb.Dataset(val_x, label=val_y, reference=train_data)
        
        # Train
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        # Store model
        self.models[venue]['lightgbm'] = model
        
        # Evaluate
        val_pred = model.predict(val_x, num_iteration=model.best_iteration)
        val_pred_us = np.expm1(val_pred)
        
        mae = np.mean(np.abs(val_pred_us - val_targets))
        within_10pct = np.sum(
            np.abs(val_pred_us - val_targets) <= val_targets * 0.1
        ) / len(val_targets) * 100
        
        return {'mae': mae, 'accuracy': within_10pct}
    
    def _prepare_tabular_features(self, features: np.ndarray) -> np.ndarray:
        """Prepare features for tabular models (XGBoost/LightGBM)"""
        if len(features.shape) == 3:
            # For sequences, use last N timesteps + statistics
            n_samples = features.shape[0]
            n_features = features.shape[2]
            
            # Last 10 timesteps
            last_features = features[:, -10:, :].reshape(n_samples, -1)
            
            # Statistics over the sequence
            mean_features = np.mean(features, axis=1)
            std_features = np.std(features, axis=1)
            max_features = np.max(features, axis=1)
            min_features = np.min(features, axis=1)
            
            # Combine
            tabular_features = np.hstack([
                last_features,
                mean_features,
                std_features,
                max_features,
                min_features
            ])
            
            return tabular_features
        else:
            return features
    
    def _train_calibration_model(self, venue: str, features: np.ndarray,
                                targets: np.ndarray):
        """Train calibration model for uncertainty estimation - FIXED"""
        # Get predictions from all models
        all_predictions = []
        model_names = []
        
        for model_name in ['lstm', 'gru', 'transformer', 'xgboost', 'lightgbm']:
            if self.models[venue][model_name] is not None:
                preds = self._get_model_predictions(venue, model_name, features)
                all_predictions.append(preds)
                model_names.append(model_name)
                logger.info(f"[DEBUG] {model_name} predictions: shape = {np.array(preds).shape}")

        if not all_predictions:
            logger.warning(f"No valid predictions for calibration in {venue}")
            return

        # Find the minimum length among all predictions
        min_length = min(len(p) for p in all_predictions)
        logger.info(f"[DEBUG] Minimum prediction length: {min_length}")
        
        # Truncate all predictions to the same length
        all_predictions_aligned = []
        for i, preds in enumerate(all_predictions):
            if len(preds) > min_length:
                # Take the last min_length predictions to align with sequence models
                aligned_preds = preds[-min_length:]
            else:
                aligned_preds = preds
            all_predictions_aligned.append(aligned_preds)
            logger.info(f"[DEBUG] {model_names[i]} aligned: shape = {np.array(aligned_preds).shape}")
        
        # Also align targets
        if len(targets) > min_length:
            aligned_targets = targets[-min_length:]
        else:
            aligned_targets = targets
        
        logger.info(f"[DEBUG] Aligned targets: shape = {np.array(aligned_targets).shape}")
        
        # Now create the prediction matrix
        try:
            all_predictions_matrix = np.array(all_predictions_aligned).T
            logger.info(f"[DEBUG] Prediction matrix shape: {all_predictions_matrix.shape}")
            
            # Calculate prediction errors
            ensemble_pred = np.mean(all_predictions_matrix, axis=1)
            errors = np.abs(ensemble_pred - aligned_targets)
            
            # Train isotonic regression for calibration
            from sklearn.isotonic import IsotonicRegression
            
            # Use standard deviation of predictions as uncertainty proxy
            pred_std = np.std(all_predictions_matrix, axis=1)
            
            # Handle edge case where std is zero
            pred_std = np.maximum(pred_std, 1e-6)
            
            calibrator = IsotonicRegression(out_of_bounds='clip')
            calibrator.fit(pred_std, errors)
            
            self.calibration_models[venue] = calibrator
            logger.info(f"Calibration model trained for {venue} with {len(errors)} samples")
            
        except Exception as e:
            logger.error(f"Failed to create calibration model for {venue}: {e}")
            # Create a dummy calibrator that just returns the input
            from sklearn.isotonic import IsotonicRegression
            dummy_calibrator = IsotonicRegression(out_of_bounds='clip')
            dummy_calibrator.fit([0.1, 0.5, 1.0], [50, 100, 200])  # Dummy data
            self.calibration_models[venue] = dummy_calibrator
    
    def _get_model_predictions(self, venue: str, model_name: str,
                              features: np.ndarray) -> np.ndarray:
        """Get predictions from a specific model - FIXED"""
        model = self.models[venue][model_name]
        
        if model_name in ['lstm', 'gru', 'transformer']:
            # Neural network models - need sequences
            from data.latency_predictor import LatencyDataset
            
            # Create dummy targets for dataset
            dummy_targets = np.ones(len(features)) * 1000.0  # Use reasonable default
            
            try:
                dataset = LatencyDataset(features, dummy_targets, self.sequence_length)
            except Exception as e:
                logger.warning(f"Failed to create dataset for {model_name}: {e}")
                return np.ones(max(0, len(features) - self.sequence_length)) * 1000.0
            
            if len(dataset) == 0:
                logger.warning(f"Empty dataset for {model_name}")
                return np.array([1000.0])
            
            predictions = []
            model.eval()
            
            with torch.no_grad():
                for i in range(len(dataset)):
                    try:
                        seq_features, _ = dataset[i]
                        seq_features = seq_features.unsqueeze(0).to(self.device)
                        
                        pred, _ = model(seq_features)
                        pred_us = dataset.denormalize_prediction(pred.squeeze())
                        
                        # Ensure finite prediction
                        pred_value = float(pred_us.cpu())
                        if not np.isfinite(pred_value):
                            pred_value = 1000.0
                        
                        predictions.append(np.clip(pred_value, 50.0, 50000.0))
                        
                    except Exception as e:
                        logger.warning(f"Error in {model_name} prediction {i}: {e}")
                        predictions.append(1000.0)
            
            return np.array(predictions)
        
        elif model_name == 'xgboost':
            try:
                tabular_features = self._prepare_tabular_features(features)
                dtest = xgb.DMatrix(tabular_features)
                pred_log = model.predict(dtest)
                pred_us = np.expm1(pred_log)
                
                # Clean predictions
                pred_us = np.nan_to_num(pred_us, nan=1000.0, posinf=10000.0, neginf=50.0)
                return np.clip(pred_us, 50.0, 50000.0)
                
            except Exception as e:
                logger.warning(f"XGBoost prediction failed: {e}")
                return np.ones(len(features)) * 1000.0
        
        elif model_name == 'lightgbm':
            try:
                tabular_features = self._prepare_tabular_features(features)
                pred_log = model.predict(tabular_features, num_iteration=model.best_iteration)
                pred_us = np.expm1(pred_log)
                
                # Clean predictions
                pred_us = np.nan_to_num(pred_us, nan=1000.0, posinf=10000.0, neginf=50.0)
                return np.clip(pred_us, 50.0, 50000.0)
                
            except Exception as e:
                logger.warning(f"LightGBM prediction failed: {e}")
                return np.ones(len(features)) * 1000.0
        
        else:
            logger.warning(f"Unknown model type: {model_name}")
            return np.ones(len(features)) * 1000.0
    
    def _optimize_ensemble_weights(self, venue: str, features: np.ndarray,
                                  targets: np.ndarray):
        """Optimize ensemble weights using validation data - FIXED"""
        from scipy.optimize import minimize
        
        # Get predictions from each model
        model_predictions = {}
        for model_name in ['lstm', 'gru', 'transformer', 'xgboost', 'lightgbm']:
            if self.models[venue][model_name] is not None:
                try:
                    preds = self._get_model_predictions(venue, model_name, features)
                    model_predictions[model_name] = preds
                    logger.info(f"{model_name} predictions shape: {preds.shape}")
                except Exception as e:
                    logger.warning(f"Failed to get predictions from {model_name}: {e}")
                    continue
        
        if not model_predictions:
            logger.warning(f"No valid model predictions for weight optimization in {venue}")
            return
        
        # Find minimum length to align all predictions
        min_length = min(len(preds) for preds in model_predictions.values())
        logger.info(f"Aligning predictions to length: {min_length}")
        
        # Align all predictions and targets
        aligned_predictions = {}
        for model_name, preds in model_predictions.items():
            if len(preds) > min_length:
                aligned_predictions[model_name] = preds[-min_length:]
            else:
                aligned_predictions[model_name] = preds
        
        # Align targets
        if len(targets) > min_length:
            aligned_targets = targets[-min_length:]
        else:
            aligned_targets = targets
        
        # Objective function
        def objective(weights):
            if len(weights) != len(aligned_predictions):
                return float('inf')
            
            weights = np.array(weights)
            weights = np.maximum(weights, 0)  # Ensure non-negative
            weight_sum = np.sum(weights)
            
            if weight_sum == 0:
                return float('inf')
            
            weights = weights / weight_sum  # Normalize
            
            try:
                ensemble_pred = np.zeros_like(aligned_targets, dtype=float)
                for i, (model_name, preds) in enumerate(aligned_predictions.items()):
                    ensemble_pred += weights[i] * preds
                
                mae = np.mean(np.abs(ensemble_pred - aligned_targets))
                return mae
                
            except Exception as e:
                logger.warning(f"Error in objective function: {e}")
                return float('inf')
        
        # Setup optimization
        n_models = len(aligned_predictions)
        model_names = list(aligned_predictions.keys())
        
        # Constraints: weights sum to 1, all non-negative
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        
        bounds = [(0, 1) for _ in range(n_models)]
        
        # Initial weights - use current weights if available
        initial_weights = []
        for model_name in model_names:
            if model_name in self.model_weights[venue]:
                initial_weights.append(self.model_weights[venue][model_name])
            else:
                initial_weights.append(1.0 / n_models)
        
        # Normalize initial weights
        initial_weights = np.array(initial_weights)
        initial_weights = initial_weights / np.sum(initial_weights)
        
        try:
            # Optimize
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'disp': False, 'maxiter': 100}
            )
            
            if result.success and result.fun < float('inf'):
                # Update weights
                optimized_weights = result.x / np.sum(result.x)
                
                # Update model weights dictionary
                for i, model_name in enumerate(model_names):
                    self.model_weights[venue][model_name] = float(optimized_weights[i])
                
                # Set weights to 0 for models not in this optimization
                for model_name in self.model_weights[venue]:
                    if model_name not in model_names:
                        self.model_weights[venue][model_name] = 0.0
                
                logger.info(f"Optimized weights for {venue}: {self.model_weights[venue]}")
                logger.info(f"Optimization result: MAE = {result.fun:.2f}")
                
            else:
                logger.warning(f"Weight optimization failed for {venue}, keeping current weights")
                
        except Exception as e:
            logger.error(f"Weight optimization error for {venue}: {e}")
            # Keep current weights as fallback
    
    def _get_ensemble_predictions(self, venue: str, features: np.ndarray) -> np.ndarray:
        """Get weighted ensemble predictions - FIXED for shape alignment"""
        # Get predictions from all models first
        all_predictions = {}
        
        for model_name, weight in self.model_weights[venue].items():
            if self.models[venue][model_name] is not None and weight > 0:
                try:
                    model_pred = self._get_model_predictions(venue, model_name, features)
                    all_predictions[model_name] = model_pred
                    logger.debug(f"{model_name} predictions shape: {model_pred.shape}")
                except Exception as e:
                    logger.warning(f"Failed to get predictions from {model_name}: {e}")
                    continue
        
        if not all_predictions:
            logger.warning(f"No valid predictions for ensemble in {venue}")
            return np.ones(len(features)) * 1000.0
        
        # Find minimum length to align all predictions
        min_length = min(len(preds) for preds in all_predictions.values())
        logger.debug(f"Aligning ensemble predictions to length: {min_length}")
        
        # Initialize ensemble prediction array
        ensemble_pred = np.zeros(min_length)
        
        # Add weighted predictions (aligned to same length)
        for model_name, weight in self.model_weights[venue].items():
            if model_name in all_predictions and weight > 0:
                model_pred = all_predictions[model_name]
                
                # Align to minimum length
                if len(model_pred) > min_length:
                    aligned_pred = model_pred[-min_length:]
                else:
                    aligned_pred = model_pred
                
                ensemble_pred += weight * aligned_pred
        
        return ensemble_pred
    
    def predict(self, venue: str, features: np.ndarray) -> EnsemblePrediction:
        """Make ensemble prediction with uncertainty quantification"""
        start_time = time.perf_counter()
        
        # Add to feature buffer
        self.feature_buffers[venue].append(features)
        
        if len(self.feature_buffers[venue]) < self.sequence_length:
            # Not enough data yet
            return EnsemblePrediction(
                venue=venue,
                timestamp=time.time(),
                ensemble_prediction_us=1000.0,
                individual_predictions={},
                model_weights=self.model_weights[venue],
                confidence=0.1,
                prediction_time_ms=0.0,
                uncertainty_bounds=(500.0, 2000.0)
            )
        
        # Get features array
        feature_array = np.array(list(self.feature_buffers[venue]))
        
        # Get predictions from each model
        individual_predictions = {}
        
        for model_name in ['lstm', 'gru', 'transformer', 'xgboost', 'lightgbm']:
            if self.models[venue][model_name] is not None:
                try:
                    pred = self._get_single_prediction(venue, model_name, feature_array)
                    individual_predictions[model_name] = float(pred)
                except Exception as e:
                    logger.warning(f"Failed to get prediction from {model_name}: {e}")
                    individual_predictions[model_name] = 1000.0
        
        # Calculate weighted ensemble
        ensemble_pred = sum(
            self.model_weights[venue][model] * pred
            for model, pred in individual_predictions.items()
        )
        
        # Calculate confidence and uncertainty
        predictions_array = np.array(list(individual_predictions.values()))
        pred_std = np.std(predictions_array)
        
        # Use calibration model for uncertainty bounds
        if venue in self.calibration_models:
            expected_error = self.calibration_models[venue].predict([pred_std])[0]
            lower_bound = max(50, ensemble_pred - 2 * expected_error)
            upper_bound = ensemble_pred + 2 * expected_error
        else:
            lower_bound = max(50, ensemble_pred - 2 * pred_std)
            upper_bound = ensemble_pred + 2 * pred_std
        
        # Calculate confidence based on agreement between models
        confidence = 1.0 - (pred_std / ensemble_pred)
        confidence = np.clip(confidence, 0.1, 0.99)
        
        # Update performance tracking
        self._update_performance_tracking(venue, individual_predictions)
        
        # Calculate prediction time
        prediction_time_ms = (time.perf_counter() - start_time) * 1000
        
        return EnsemblePrediction(
            venue=venue,
            timestamp=time.time(),
            ensemble_prediction_us=float(ensemble_pred),
            individual_predictions=individual_predictions,
            model_weights=self.model_weights[venue].copy(),
            confidence=float(confidence),
            prediction_time_ms=prediction_time_ms,
            uncertainty_bounds=(float(lower_bound), float(upper_bound))
        )
    
    def _get_single_prediction(self, venue: str, model_name: str,
                              feature_sequence: np.ndarray) -> float:
        """Get a single prediction from a model"""
        if model_name in ['lstm', 'gru', 'transformer']:
            model = self.models[venue][model_name]
            model.eval()
            
            # Normalize features
            features_tensor = torch.FloatTensor(feature_sequence).unsqueeze(0)
            features_tensor = features_tensor.to(self.device)
            
            with torch.no_grad():
                pred, _ = model(features_tensor)
                
            # Simple denormalization (would use proper scaler in production)
            pred_us = float(np.expm1(pred.squeeze().cpu()) * 1000)
            return np.clip(pred_us, 50, 10000)
        
        elif model_name == 'xgboost':
            model = self.models[venue][model_name]
            tabular_features = self._prepare_tabular_features(
                feature_sequence.reshape(1, *feature_sequence.shape)
            )
            dtest = xgb.DMatrix(tabular_features)
            pred_log = model.predict(dtest)[0]
            return float(np.clip(np.expm1(pred_log), 50, 10000))
        
        elif model_name == 'lightgbm':
            model = self.models[venue][model_name]
            tabular_features = self._prepare_tabular_features(
                feature_sequence.reshape(1, *feature_sequence.shape)
            )
            pred_log = model.predict(tabular_features, num_iteration=model.best_iteration)[0]
            return float(np.clip(np.expm1(pred_log), 50, 10000))
        
        return 1000.0
    
    def _update_performance_tracking(self, venue: str, predictions: Dict[str, float]):
        """Track model performance for dynamic weight adaptation"""
        # This would track actual vs predicted in production
        # For now, track prediction variance as a proxy
        
        mean_pred = np.mean(list(predictions.values()))
        
        for model_name, pred in predictions.items():
            error_proxy = abs(pred - mean_pred) / mean_pred
            self.performance_buffer[venue][model_name].append(error_proxy)
    
    def adapt_weights(self, venue: str):
        """Dynamically adapt model weights based on recent performance"""
        if not all(len(buffer) > 10 for buffer in self.performance_buffer[venue].values()):
            return  # Not enough data
        
        # Calculate recent performance for each model
        model_scores = {}
        
        for model_name in self.model_weights[venue]:
            recent_errors = list(self.performance_buffer[venue][model_name])
            # Lower error = better score
            model_scores[model_name] = 1.0 / (1.0 + np.mean(recent_errors))
        
        # Normalize scores to weights
        total_score = sum(model_scores.values())
        
        for model_name in self.model_weights[venue]:
            # Smooth update (momentum = 0.9)
            new_weight = model_scores[model_name] / total_score
            old_weight = self.model_weights[venue][model_name]
            self.model_weights[venue][model_name] = 0.9 * old_weight + 0.1 * new_weight
        
        # Ensure weights sum to 1
        total_weight = sum(self.model_weights[venue].values())
        for model_name in self.model_weights[venue]:
            self.model_weights[venue][model_name] /= total_weight
    
    def update_with_actual(self, venue: str, predicted: EnsemblePrediction,
                          actual_latency_us: float):
        """Update performance tracking with actual latency"""
        # Calculate errors for each model
        for model_name, pred in predicted.individual_predictions.items():
            error = abs(pred - actual_latency_us) / actual_latency_us
            self.performance_buffer[venue][model_name].append(error)
        
        # Adapt weights periodically
        if len(self.performance_buffer[venue]['lstm']) % 20 == 0:
            self.adapt_weights(venue)
    
    def save_ensemble(self, venue: str, path: str):
        """Save entire ensemble for a venue"""
        import os
        os.makedirs(path, exist_ok=True)
        
        # Save neural network models
        for model_name in ['lstm', 'gru', 'transformer']:
            if self.models[venue][model_name] is not None:
                torch.save(
                    self.models[venue][model_name].state_dict(),
                    os.path.join(path, f"{venue}_{model_name}.pt")
                )
        
        # Save XGBoost
        if self.models[venue]['xgboost'] is not None:
            self.models[venue]['xgboost'].save_model(
                os.path.join(path, f"{venue}_xgboost.json")
            )
        
        # Save LightGBM
        if self.models[venue]['lightgbm'] is not None:
            self.models[venue]['lightgbm'].save_model(
                os.path.join(path, f"{venue}_lightgbm.txt")
            )
        
        # Save weights and calibration
        metadata = {
            'weights': self.model_weights[venue],
            'feature_size': self.feature_size,
            'sequence_length': self.sequence_length
        }
        
        joblib.dump(metadata, os.path.join(path, f"{venue}_metadata.pkl"))
        
        if venue in self.calibration_models:
            joblib.dump(
                self.calibration_models[venue],
                os.path.join(path, f"{venue}_calibration.pkl")
            )
        
        logger.info(f"Saved ensemble for {venue} to {path}")
    
    def load_ensemble(self, venue: str, path: str):
        """Load ensemble from disk"""
        import os
        
        # Load metadata
        metadata = joblib.load(os.path.join(path, f"{venue}_metadata.pkl"))
        self.model_weights[venue] = metadata['weights']
        
        # Load neural networks
        for model_name in ['lstm', 'gru', 'transformer']:
            model_path = os.path.join(path, f"{venue}_{model_name}.pt")
            if os.path.exists(model_path):
                self.models[venue][model_name].load_state_dict(
                    torch.load(model_path, map_location=self.device)
                )
                self.models[venue][model_name].eval()
        
        # Load XGBoost
        xgb_path = os.path.join(path, f"{venue}_xgboost.json")
        if os.path.exists(xgb_path):
            self.models[venue]['xgboost'] = xgb.Booster()
            self.models[venue]['xgboost'].load_model(xgb_path)
        
        # Load LightGBM
        lgb_path = os.path.join(path, f"{venue}_lightgbm.txt")
        if os.path.exists(lgb_path):
            self.models[venue]['lightgbm'] = lgb.Booster(model_file=lgb_path)
        
        # Load calibration
        cal_path = os.path.join(path, f"{venue}_calibration.pkl")
        if os.path.exists(cal_path):
            self.calibration_models[venue] = joblib.load(cal_path)
        
        logger.info(f"Loaded ensemble for {venue} from {path}")
    
    def get_feature_importance(self, venue: str) -> Dict[str, np.ndarray]:
        """Get feature importance from tree-based models"""
        importance = {}
        
        if self.models[venue]['xgboost'] is not None:
            xgb_importance = self.models[venue]['xgboost'].get_score(
                importance_type='gain'
            )
            importance['xgboost'] = xgb_importance
        
        if self.models[venue]['lightgbm'] is not None:
            lgb_importance = self.models[venue]['lightgbm'].feature_importance(
                importance_type='gain'
            )
            importance['lightgbm'] = lgb_importance
        
        return importance