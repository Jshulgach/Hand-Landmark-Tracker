import os
import pickle
import json
import torch
import logging
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from handtrack.ml import EMGRegressor

available_models = {
    'EMGRegressor': EMGRegressor
}

def setup_logger():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


setup_logger()


class ModelManager:
    def __init__(self, root_dir, label=None, model=None, input_dim=None, output_dim=None, verbose=False):
        self.root_dir = root_dir
        self.label = label
        self.model = model
        self.scalar = None
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.eval_metrics = {}
        self.verbose = verbose

        self.model_dir = os.path.join(root_dir, 'model')
        os.makedirs(self.model_dir, exist_ok=True)
        self.scalar_path = os.path.join(self.model_dir, f"{label}_scalar.pkl" if label else "scalar.pkl")
        self.model_path = os.path.join(self.model_dir, f"{label}_emg_regressor.pth" if label else "emg_regressor.pth")
        self.metrics_path = os.path.join(self.model_dir, f'{label}_metrics.json')
        self.model_exists = os.path.exists(self.model_path) and os.path.exists(self.scalar_path) and self.model is not None

        if model is not None:
            self.set_model(model)

    def train(self, X, y, num_epochs=3000, early_stop_patience=5, learning_rate=1e-3, val_interval=20, save_metrics=True):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        if self.verbose:
            print(f"Training set shape: {X_train.shape}, {y_train.shape}")
            print(f"Testing set shape: {X_test.shape}, {y_test.shape}")

        scalar = StandardScaler()
        X_train = scalar.fit_transform(X_train)
        X_test = scalar.transform(X_test)

        if self.model is None:
            raise ValueError("Model must be set before training. Use set_model() to set the model class.")

        model = self.model
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.MSELoss()

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

        if self.verbose:
            print("Starting training...")

        loss_curve = []
        best_val_loss = np.inf
        epochs_no_improve = 0
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            pred = model(X_train_tensor)
            loss = criterion(pred, y_train_tensor)
            loss.backward()
            optimizer.step()
            loss_curve.append(loss.item())

            if (epoch + 1) % val_interval == 0:
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_test_tensor)
                    val_loss = criterion(val_outputs, y_test_tensor).item()
                logging.info(
                    f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {loss.item():.8f} | Val Loss: {val_loss:.8f}")

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= early_stop_patience:
                        logging.info("Early stopping triggered.")
                        break

        # Save model and scalar
        print(f" Training complete. Saving model to {self.model_path}")
        torch.save(model.state_dict(), self.model_path)
        with open(self.scalar_path, 'wb') as f:
            pickle.dump(scalar, f)

        # Evaluate
        model.eval()
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
        with torch.no_grad():
            y_pred = model(X_test_tensor).numpy()
            y_true = y_test_tensor.numpy()

        self.eval_metrics = {
            'mse': float(mean_squared_error(y_true, y_pred)),
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'r2': float(r2_score(y_true, y_pred))
        }

        if save_metrics:
            with open(self.metrics_path, 'w') as f:
                json.dump(self.eval_metrics, f, indent=2)

        return model, scalar

    def load_weights(self, model_path=None):
        if model_path is None and self.model_path is None:
            raise ValueError("Weights path must be provided or set in the ModelManager.")
        if model_path is None:
            model_path = self.model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Weights file not found at {model_path}")
        if self.model is None:
            raise ValueError("Model must be set before loading weights. Use set_model() to set the model class.")
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def load_scalar(self, scalar=None):

        if scalar is not None:
            self.scalar = scalar
            if self.verbose:
                logging.info("Scaler set directly.")
            return
        if not os.path.exists(self.scalar_path):
            raise FileNotFoundError(f"Scaler file not found: {self.scalar_path}")
        with open(self.scalar_path, 'rb') as f:
            self.scalar = pickle.load(f)
        if self.verbose:
            logging.info(f"Scaler loaded from {self.scalar_path}")

    def save_scalar(self, scalar):
        with open(self.scalar_path, 'wb') as f:
            pickle.dump(scalar, f)
        if self.verbose:
            logging.info(f"Scaler saved to {self.scalar_path}")

    def set_model(self, model_class, input_dim=None, output_dim=None):
        """
        Set the model class and optionally input/output dimensions.
        If input/output dimensions are not provided, they will be taken from the model class.

        Parameters:
            model_class (class): The model class to set.
            input_dim (int, optional): Input dimension for the model.
            output_dim (int, optional): Output dimension for the model.

        """
        if model_class is None:
            raise ValueError("Model class must be provided.")

        self.model = model_class

    def load_model(self, model, weights=None, scalar=None):
        if not model:
            raise ValueError("Model must be provided to load.")

        self.set_model(model)
        if self.verbose:
            logging.info(f"Model set to {self.model.__class__.__name__}")

        if weights is not None:
            self.load_weights(weights)
        elif os.path.exists(self.model_path):
            self.load_weights()

        if scalar is not None:
            self.load_scalar(scalar)
        elif os.path.exists(self.scalar_path):
            self.load_scalar()

        self.model_exists = True

    def load_model_weights(self):
        """
        Load the model and scalar from disk.
        If the model is not set, it will raise an error.

        """

        self.load_weights()
        self.load_scalar()

        if self.verbose:
            logging.info(f"Model loaded from {self.model_path} and scalar from {self.scalar_path}")
        # print ut the model shape, input and output dimensions
        if self.verbose:
            logging.info(f"Model: {self.model}, Input Dim: {self.model.input_dim}, Output Dim: {self.model.output_dim}")

        # Optionally load evaluation metrics if available
        if os.path.exists(self.metrics_path):
            with open(self.metrics_path, 'r') as f:
                self.eval_metrics = json.load(f)

        #return self.model, self.scalar

    def predict(self, X):
        """
        Predict using the loaded model and scalar.
        If scalar is not yet loaded, attempt to load it from disk.

        Parameters:
            X (np.ndarray): Input data to predict.

        Returns:
            np.ndarray: Predicted values.
        """
        if self.model is None:
            raise ValueError("Model must be loaded before prediction.")

        if self.scalar is None:
            if os.path.exists(self.scalar_path):
                with open(self.scalar_path, 'rb') as f:
                    self.scalar = pickle.load(f)
                if self.verbose:
                    logging.info(f"Scaler loaded from {self.scalar_path}")
            else:
                raise ValueError("Scaler is not loaded and could not be found.")

        X_scaled = self.scalar.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad():
            predictions = self.model(X_tensor).numpy()

        return predictions

    def cross_validate(self, X, y, k=5, num_epochs=3000, early_stop_patience=5, learning_rate=1e-3, val_interval=20):
        """
        Perform k-fold cross-validation on the model.

        Parameters:
            X (np.ndarray): Input features.
            y (np.ndarray): Target values.
            k (int): Number of folds for cross-validation.
            num_epochs (int): Number of epochs for training.
            early_stop_patience (int): Patience for early stopping.
            learning_rate (float): Learning rate for the optimizer.
            val_interval (int): Validation interval.

        Returns:
            dict: Cross-validation metrics.
        """

        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        metrics_list = []
        loss_curves = []
        best_val_loss = float('inf')
        best_fold_index = -1
        best_model_state = None
        best_scalar = None

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"\n--- Fold {fold + 1}/{k} ---")
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            scalar = StandardScaler()
            X_train_scaled = scalar.fit_transform(X_train)
            X_val_scaled = scalar.transform(X_val)

            model = EMGRegressor(input_dim=X.shape[1], output_dim=y.shape[1])
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            criterion = torch.nn.MSELoss()

            X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
            X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

            loss_curve = []
            no_improve = 0
            best_fold_val_loss = float('inf')

            for epoch in range(num_epochs):
                model.train()
                optimizer.zero_grad()
                pred = model(X_train_tensor)
                loss = criterion(pred, y_train_tensor)
                loss.backward()
                optimizer.step()
                loss_curve.append(loss.item())

                if (epoch + 1) % val_interval == 0:
                    model.eval()
                    with torch.no_grad():
                        val_pred = model(X_val_tensor)
                        val_loss = criterion(val_pred, y_val_tensor).item()
                        print(f"Epoch {epoch + 1} - Train Loss: {loss.item():.6f}, Val Loss: {val_loss:.6f}")

                    if val_loss < best_fold_val_loss:
                        best_fold_val_loss = val_loss
                        no_improve = 0
                    else:
                        no_improve += 1
                        if no_improve >= early_stop_patience:
                            print("Early stopping")
                            break

            # Evaluate final model for this fold
            model.eval()
            with torch.no_grad():
                y_pred = model(X_val_tensor).numpy()
                y_true = y_val_tensor.numpy()
            metrics = {
                'fold': fold,
                'mse': float(mean_squared_error(y_true, y_pred)),
                'mae': float(mean_absolute_error(y_true, y_pred)),
                'r2': float(r2_score(y_true, y_pred))
            }
            metrics_list.append(metrics)
            loss_curves.append(loss_curve)

            if best_fold_val_loss < best_val_loss:
                best_val_loss = best_fold_val_loss
                best_model_state = model.state_dict()
                best_scalar = scalar
                best_fold_index = fold

        # Save best model and scalar
        self.model = EMGRegressor(input_dim=X.shape[1], output_dim=y.shape[1])
        self.model.load_state_dict(best_model_state)
        torch.save(self.model.state_dict(), self.model_path)
        with open(self.scalar_path, 'wb') as f:
            pickle.dump(best_scalar, f)

        average_metrics = {
            'mse': float(np.mean([m['mse'] for m in metrics_list])),
            'mae': float(np.mean([m['mae'] for m in metrics_list])),
            'r2': float(np.mean([m['r2'] for m in metrics_list]))
        }
        with open(self.metrics_path.replace('.json', '_kfold.json'), 'w') as f:
            json.dump({'folds': metrics_list, 'average': average_metrics, 'loss_curves': loss_curves}, f, indent=2)

        print("\nK-fold cross-validation complete. Average metrics:")
        print(average_metrics)
        print(f"\nBest model came from fold {best_fold_index + 1}")
        return self.model, best_scalar