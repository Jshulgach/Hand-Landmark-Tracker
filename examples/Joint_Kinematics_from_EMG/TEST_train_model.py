import os
import numpy as np
import argparse

from handtrack.ml import ModelManager, EMGRegressor
from handtrack.io import load_yaml_config


def train_model(cfg):

    root_dir = cfg['root_dir']
    label = cfg.get('label', '')
    kfold = cfg.get('kfold', False)
    verbose = cfg.get('verbose', False)
    overwrite = cfg.get('overwrite', False)

    data_path = os.path.join(root_dir, f"{label}_training_dataset.npz" if label else "training_dataset.npz")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found at {data_path}")

    if verbose:
        print(f"Loading dataset from {data_path}")
    data = np.load(data_path)
    X, y = data['features'], data['labels']
    print(f"Data shape - X: {X.shape}, y: {y.shape}")

    # Use model manager
    manager = ModelManager(root_dir=root_dir,
                           label=label,
                           model=EMGRegressor(input_dim=X.shape[1], output_dim=y.shape[1]),
                           verbose=verbose)
    if kfold:
        print("Running k-fold cross-validation...")
        model, scaler = manager.cross_validate(X, y)
    elif not manager.model_exists or overwrite:
        print("Training new model...")
        model, scaler = manager.train(X, y)
    else:
        print("Model already exists, loading existing model and scaler.")
        model, scaler = manager.load_model()

    return model, scaler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare EMG-predicted angles with MP joint angles.")
    parser.add_argument('--config_file', type=str, default=None, help='Path to config file with dataset paths and variables')
    parser.add_argument('--root_dir',   type=str, default='',      help='Root directory')
    parser.add_argument('--label',      type=str, default='',      help='Label used for model and data')
    parser.add_argument('--kfold',      action='store_true',       help='Use k-fold cross-validation')
    parser.add_argument('--overwrite',  action='store_true',       help='Train new model')
    parser.add_argument('--verbose',    action='store_true',       help='Verbose debugging output')
    args = parser.parse_args()

    #config = load_yaml_config(args.config_file)
    #if config is None:
    #    config = {}
    config = {}

    # Allow command-line override of config values
    config['root_dir'] = args.root_dir or config.get('root_dir', '')
    config['label'] = args.label or config.get('label', '')
    config['kfold'] = args.kfold or config.get('kfold', False)
    config['overwrite'] = args.overwrite or config.get('overwrite', False)
    config['verbose'] = args.verbose or config.get('verbose', False)

    # Run training
    model, scalar = train_model(config)
    print(f" Model {model}, Scalar {scalar}")
