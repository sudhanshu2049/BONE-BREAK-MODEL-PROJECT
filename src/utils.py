"""
Utility Module for Bone Fracture Classification

This module provides:
- Helper functions for plotting and visualization
- Model saving and loading utilities
- Logging and configuration management
- General utility functions
"""

import os
import json
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import tensorflow as tf
from pathlib import Path


class Utils:
    """Utility class with helper functions for the project."""

    def __init__(self, base_dir="outputs"):
        """
        Initialize Utils.

        Args:
            base_dir: Base directory for outputs
        """
        self.base_dir = Path(base_dir)
        self.models_dir = self.base_dir / "models"
        self.plots_dir = self.base_dir / "plots"
        self.logs_dir = self.base_dir / "logs"

        # Create directories
        for dir_path in [self.models_dir, self.plots_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def save_model(self, model, model_name, save_format='h5'):
        """
        Save trained model to disk.

        Args:
            model: Trained Keras model
            model_name: Name for the saved model
            save_format: Format to save ('h5' or 'tf')
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if save_format == 'h5':
            filename = f"{model_name}_{timestamp}.h5"
            filepath = self.models_dir / filename
            model.save(filepath)
        elif save_format == 'tf':
            dirname = f"{model_name}_{timestamp}"
            filepath = self.models_dir / dirname
            model.save(filepath)
        else:
            raise ValueError("save_format must be 'h5' or 'tf'")

        print(f"✅ Model saved to {filepath}")
        return str(filepath)

    def load_model(self, model_path):
        """
        Load saved model from disk.

        Args:
            model_path: Path to saved model

        Returns:
            Loaded Keras model
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            model = tf.keras.models.load_model(model_path)
            print(f"✅ Model loaded from {model_path}")
            return model
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None

    def save_history(self, history, model_name):
        """
        Save training history to JSON file.

        Args:
            history: Keras History object
            model_name: Name of the model
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_history_{timestamp}.json"
        filepath = self.logs_dir / filename

        # Convert history to dictionary
        history_dict = {}
        for key, values in history.history.items():
            history_dict[key] = [float(v) for v in values]

        try:
            with open(filepath, 'w') as f:
                json.dump(history_dict, f, indent=4)
            print(f"✅ Training history saved to {filepath}")
        except Exception as e:
            print(f"❌ Error saving history: {e}")

    def load_history(self, history_path):
        """
        Load training history from JSON file.

        Args:
            history_path: Path to history file

        Returns:
            History dictionary
        """
        if not os.path.exists(history_path):
            raise FileNotFoundError(f"History file not found: {history_path}")

        try:
            with open(history_path, 'r') as f:
                history = json.load(f)
            print(f"✅ History loaded from {history_path}")
            return history
        except Exception as e:
            print(f"❌ Error loading history: {e}")
            return None

    def plot_training_history(self, history, model_name, save_path=None):
        """
        Plot training and validation metrics.

        Args:
            history: Keras History object or dictionary
            model_name: Name of the model
            save_path: Path to save plot (optional)
        """
        # Handle both History object and dictionary
        if hasattr(history, 'history'):
            hist = history.history
        else:
            hist = history

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Accuracy
        ax1.plot(hist.get('accuracy', []), label='Training Accuracy')
        ax1.plot(hist.get('val_accuracy', []), label='Validation Accuracy')
        ax1.set_title(f'{model_name} - Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Loss
        ax2.plot(hist.get('loss', []), label='Training Loss')
        ax2.plot(hist.get('val_loss', []), label='Validation Loss')
        ax2.set_title(f'{model_name} - Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Precision (if available)
        if 'precision' in hist:
            ax3.plot(hist.get('precision', []), label='Training Precision')
            ax3.plot(hist.get('val_precision', []), label='Validation Precision')
            ax3.set_title(f'{model_name} - Precision')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Precision')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        # Recall (if available)
        if 'recall' in hist:
            ax4.plot(hist.get('recall', []), label='Training Recall')
            ax4.plot(hist.get('val_recall', []), label='Validation Recall')
            ax4.set_title(f'{model_name} - Recall')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Recall')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Training history plot saved to {save_path}")

        plt.show()

    def plot_multiple_histories(self, histories_dict, metric='accuracy', save_path=None):
        """
        Plot comparison of training histories for multiple models.

        Args:
            histories_dict: Dictionary of model_name -> history
            metric: Metric to plot ('accuracy', 'loss', 'precision', 'recall')
            save_path: Path to save plot (optional)
        """
        plt.figure(figsize=(12, 8))

        colors = plt.cm.tab10(np.linspace(0, 1, len(histories_dict)))

        for i, (model_name, history) in enumerate(histories_dict.items()):
            # Handle both History object and dictionary
            if hasattr(history, 'history'):
                hist = history.history
            else:
                hist = history

            train_metric = hist.get(metric, [])
            val_metric = hist.get(f'val_{metric}', [])

            if train_metric:
                plt.plot(train_metric, label=f'{model_name} (Train)',
                        color=colors[i], linestyle='-', alpha=0.7)
            if val_metric:
                plt.plot(val_metric, label=f'{model_name} (Val)',
                        color=colors[i], linestyle='--', alpha=0.9)

        plt.title(f'Model Comparison - {metric.capitalize()}')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Model comparison plot saved to {save_path}")

        plt.show()

    def create_model_summary_table(self, models_dict, save_path=None):
        """
        Create a summary table of model architectures.

        Args:
            models_dict: Dictionary of model_name -> model
            save_path: Path to save table (optional)
        """
        import pandas as pd

        summary_data = []

        for model_name, model in models_dict.items():
            try:
                # Get model summary info
                total_params = model.count_params()
                trainable_params = sum([layer.count_params() for layer in model.layers if layer.trainable])
                non_trainable_params = total_params - trainable_params

                # Get input shape
                input_shape = model.input_shape if hasattr(model, 'input_shape') else 'Unknown'

                # Get output shape
                output_shape = model.output_shape if hasattr(model, 'output_shape') else 'Unknown'

                summary_data.append({
                    'Model': model_name,
                    'Total Parameters': f"{total_params:,}",
                    'Trainable Parameters': f"{trainable_params:,}",
                    'Non-trainable Parameters': f"{non_trainable_params:,}",
                    'Input Shape': str(input_shape),
                    'Output Shape': str(output_shape)
                })

            except Exception as e:
                print(f"Error getting summary for {model_name}: {e}")
                continue

        df = pd.DataFrame(summary_data)

        if save_path:
            df.to_csv(save_path, index=False)
            print(f"✅ Model summary table saved to {save_path}")

        return df

    def setup_logging(self, log_level='INFO'):
        """
        Setup logging configuration.

        Args:
            log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        """
        import logging

        # Create logger
        logger = logging.getLogger('bone_fracture_classification')
        logger.setLevel(getattr(logging, log_level))

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level))

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(console_handler)

        return logger

    def save_config(self, config_dict, filename="config.json"):
        """
        Save configuration dictionary to JSON file.

        Args:
            config_dict: Configuration dictionary
            filename: Output filename
        """
        filepath = self.logs_dir / filename

        try:
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=4)
            print(f"✅ Configuration saved to {filepath}")
        except Exception as e:
            print(f"❌ Error saving config: {e}")

    def load_config(self, filename="config.json"):
        """
        Load configuration from JSON file.

        Args:
            filename: Configuration filename

        Returns:
            Configuration dictionary
        """
        filepath = self.logs_dir / filename

        if not os.path.exists(filepath):
            print(f"⚠️  Config file not found: {filepath}")
            return {}

        try:
            with open(filepath, 'r') as f:
                config = json.load(f)
            print(f"✅ Configuration loaded from {filepath}")
            return config
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            return {}

    def get_system_info(self):
        """
        Get system and environment information.

        Returns:
            Dictionary with system information
        """
        import platform
        import psutil

        try:
            info = {
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'tensorflow_version': tf.__version__,
                'cpu_count': os.cpu_count(),
                'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
                'gpu_available': len(tf.config.list_physical_devices('GPU')) > 0
            }

            if info['gpu_available']:
                gpu_devices = tf.config.list_physical_devices('GPU')
                info['gpu_devices'] = [device.name for device in gpu_devices]

        except ImportError:
            # psutil not available
            info = {
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'tensorflow_version': tf.__version__,
                'cpu_count': os.cpu_count(),
                'memory_gb': 'Unknown',
                'gpu_available': len(tf.config.list_physical_devices('GPU')) > 0
            }

        return info

    def print_system_info(self):
        """Print system information."""
        info = self.get_system_info()

        print(f"\n{'='*50}")
        print("SYSTEM INFORMATION")
        print(f"{'='*50}")
        print(f"Platform: {info['platform']}")
        print(f"Python Version: {info['python_version']}")
        print(f"TensorFlow Version: {info['tensorflow_version']}")
        print(f"CPU Cores: {info['cpu_count']}")
        print(f"Memory: {info['memory_gb']} GB")
        print(f"GPU Available: {info['gpu_available']}")
        if info['gpu_available'] and 'gpu_devices' in info:
            print(f"GPU Devices: {info['gpu_devices']}")
        print(f"{'='*50}\n")

    def create_experiment_folder(self, experiment_name=None):
        """
        Create a new experiment folder with timestamp.

        Args:
            experiment_name: Name for the experiment (optional)

        Returns:
            Path to experiment folder
        """
        if experiment_name is None:
            experiment_name = "experiment"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_folder = self.base_dir / f"{experiment_name}_{timestamp}"

        exp_folder.mkdir(parents=True, exist_ok=True)

        # Create subfolders
        (exp_folder / "models").mkdir(exist_ok=True)
        (exp_folder / "plots").mkdir(exist_ok=True)
        (exp_folder / "metrics").mkdir(exist_ok=True)
        (exp_folder / "logs").mkdir(exist_ok=True)

        print(f"✅ Experiment folder created: {exp_folder}")
        return exp_folder

    def cleanup_temp_files(self, temp_dir="temp"):
        """
        Clean up temporary files.

        Args:
            temp_dir: Temporary directory to clean
        """
        temp_path = Path(temp_dir)

        if temp_path.exists():
            import shutil
            try:
                shutil.rmtree(temp_path)
                print(f"✅ Temporary files cleaned up: {temp_path}")
            except Exception as e:
                print(f"❌ Error cleaning temp files: {e}")
        else:
            print(f"ℹ️  Temp directory not found: {temp_path}")


# Example usage
if __name__ == "__main__":
    utils = Utils()

    # Print system info
    utils.print_system_info()

    # Create experiment folder
    exp_folder = utils.create_experiment_folder("bone_fracture_test")

    # Example config
    config = {
        'batch_size': 32,
        'epochs': 50,
        'learning_rate': 0.001,
        'image_size': (224, 224),
        'num_classes': 10
    }

    # Save config
    utils.save_config(config)

    print("✅ Utils module ready!")