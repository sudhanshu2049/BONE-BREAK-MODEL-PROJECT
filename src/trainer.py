"""
Trainer Module for Bone Fracture Classification

This module handles:
- Training multiple models
- Early stopping and callbacks
- Training history tracking
- Model compilation
"""

import tensorflow as tf
from tensorflow.keras import callbacks
import matplotlib.pyplot as plt
import os


class ModelTrainer:
    """Handles training of multiple models with proper callbacks and monitoring."""

    def __init__(self, models_dir="models", logs_dir="outputs/logs"):
        """
        Initialize ModelTrainer.

        Args:
            models_dir: Directory to save trained models
            logs_dir: Directory for training logs
        """
        self.models_dir = models_dir
        self.logs_dir = logs_dir

        # Create directories if they don't exist
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)

    def compile_model(self, model, model_name):
        """
        Compile model with standard settings.

        Args:
            model: Keras model to compile
            model_name: Name for logging

        Returns:
            Compiled model
        """
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        print(f"✓ {model_name} compiled - {model.count_params():,} parameters")
        return model

    def create_callbacks(self, model_name, patience=3):
        """
        Create training callbacks for a model.

        Args:
            model_name: Name of the model
            patience: Early stopping patience

        Returns:
            List of callbacks
        """
        callbacks_list = []

        # Early stopping
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks_list.append(early_stopping)

        # Model checkpoint
        checkpoint_path = os.path.join(self.models_dir, f"{model_name.lower().replace(' ', '_')}_best.keras")
        model_checkpoint = callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        callbacks_list.append(model_checkpoint)

        # TensorBoard logging
        log_dir = os.path.join(self.logs_dir, model_name.lower().replace(' ', '_'))
        tensorboard = callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1
        )
        callbacks_list.append(tensorboard)

        return callbacks_list

    def train_model(self, model, model_name, train_dataset, val_dataset, epochs=10):
        """
        Train a single model.

        Args:
            model: Keras model to train
            model_name: Name of the model
            train_dataset: Training dataset
            val_dataset: Validation dataset
            epochs: Maximum number of epochs

        Returns:
            Trained model and training history
        """
        print(f"\n{'='*60}")
        print(f"🚀 Training {model_name}")
        print(f"{'='*60}")

        # Compile model
        model = self.compile_model(model, model_name)

        # Create callbacks
        training_callbacks = self.create_callbacks(model_name)

        # Train the model
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=training_callbacks,
            verbose=1
        )

        print(f"✅ {model_name} training completed!")
        print(f"   Final training accuracy: {history.history['accuracy'][-1]:.4f}")
        print(f"   Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
        print(f"   Best validation accuracy: {max(history.history['val_accuracy']):.4f}")

        return model, history

    def train_all_models(self, models_dict, train_dataset, val_dataset, epochs=10):
        """
        Train all models in the dictionary.

        Args:
            models_dict: Dictionary of model names and models
            train_dataset: Training dataset
            val_dataset: Validation dataset
            epochs: Maximum number of epochs

        Returns:
            Dictionary of trained models and their histories
        """
        trained_models = {}
        training_histories = {}

        print(f"🎯 Starting training for {len(models_dict)} models...")
        print(f"Training configuration: {epochs} epochs max, early stopping enabled")

        for model_name, model in models_dict.items():
            try:
                trained_model, history = self.train_model(
                    model, model_name, train_dataset, val_dataset, epochs
                )

                trained_models[model_name] = trained_model
                training_histories[model_name] = history

            except Exception as e:
                print(f"❌ Error training {model_name}: {e}")
                continue

        print(f"\n✅ Training completed for {len(trained_models)}/{len(models_dict)} models")

        # Print summary
        self.print_training_summary(training_histories)

        return trained_models, training_histories

    def print_training_summary(self, histories):
        """Print training summary for all models."""
        print(f"\n📊 Training Summary:")
        print("-" * 70)

        for model_name, history in histories.items():
            final_train_acc = history.history['accuracy'][-1]
            final_val_acc = history.history['val_accuracy'][-1]
            best_val_acc = max(history.history['val_accuracy'])
            epochs_trained = len(history.history['accuracy'])

            print(f"{model_name:15s} | Train: {final_train_acc:.4f} | Val: {final_val_acc:.4f} | Best: {best_val_acc:.4f} | Epochs: {epochs_trained}")

    def plot_training_history(self, histories, save_path=None):
        """
        Plot training history for all models.

        Args:
            histories: Dictionary of training histories
            save_path: Path to save the plot (optional)
        """
        if not histories:
            print("❌ No training histories to plot")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        colors = ['blue', 'red', 'green', 'orange', 'purple']

        for i, (model_name, history) in enumerate(histories.items()):
            color = colors[i % len(colors)]
            epochs = range(1, len(history.history['accuracy']) + 1)

            # Training accuracy
            ax1.plot(epochs, history.history['accuracy'], color=color,
                    label=f'{model_name} (Train)', linestyle='-')
            # Validation accuracy
            ax1.plot(epochs, history.history['val_accuracy'], color=color,
                    label=f'{model_name} (Val)', linestyle='--')

            # Training loss
            ax2.plot(epochs, history.history['loss'], color=color,
                    label=f'{model_name} (Train)', linestyle='-')
            # Validation loss
            ax2.plot(epochs, history.history['val_loss'], color=color,
                    label=f'{model_name} (Val)', linestyle='--')

        # Configure plots
        ax1.set_title('Model Accuracy Comparison')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)

        ax2.set_title('Model Loss Comparison')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)

        # Final accuracy comparison
        model_names = list(histories.keys())
        final_accuracies = [histories[name].history['val_accuracy'][-1] for name in model_names]

        bars = ax3.bar(model_names, final_accuracies, color=colors[:len(model_names)])
        ax3.set_title('Final Validation Accuracy')
        ax3.set_xlabel('Model')
        ax3.set_ylabel('Accuracy')
        ax3.set_ylim(0, 1)
        ax3.tick_params(axis='x', rotation=45)

        # Add value labels
        for bar, acc in zip(bars, final_accuracies):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Plot saved to {save_path}")

        plt.show()


# Example usage
if __name__ == "__main__":
    from data_loader import DataLoader
    from model_builder import ModelBuilder

    # Initialize components
    loader = DataLoader()
    builder = ModelBuilder()
    trainer = ModelTrainer()

    # Load and preprocess data
    train_ds, val_ds, class_names, num_classes = loader.load_dataset()
    aug = loader.create_data_augmentation()
    train_ds = loader.preprocess_dataset(train_ds, aug)
    val_ds = loader.preprocess_dataset(val_ds)

    # Build models
    models = builder.build_all_models(num_classes)

    # Train all models
    trained_models, histories = trainer.train_all_models(models, train_ds, val_ds)

    # Plot training history
    trainer.plot_training_history(histories)