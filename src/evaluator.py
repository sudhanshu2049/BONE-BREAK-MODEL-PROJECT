"""
Evaluator Module for Bone Fracture Classification

This module handles:
- Model evaluation on validation/test sets
- Performance metrics calculation
- Confusion matrix generation
- Accuracy comparison across models
- Results visualization and saving
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import os
import json


class ModelEvaluator:
    """Evaluates trained models and generates performance metrics."""

    def __init__(self, metrics_dir="outputs/metrics", plots_dir="outputs/plots"):
        """
        Initialize ModelEvaluator.

        Args:
            metrics_dir: Directory to save metrics
            plots_dir: Directory to save plots
        """
        self.metrics_dir = metrics_dir
        self.plots_dir = plots_dir

        # Create directories
        os.makedirs(metrics_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)

    def evaluate_model(self, model, model_name, dataset, class_names):
        """
        Evaluate a single model on a dataset.

        Args:
            model: Trained Keras model
            model_name: Name of the model
            dataset: Dataset to evaluate on
            class_names: List of class names

        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\n🔍 Evaluating {model_name}...")

        # Get predictions and true labels
        true_labels, predicted_labels, predicted_probs = self.get_predictions_and_labels(
            model, dataset, class_names
        )

        # Calculate metrics
        loss, accuracy = model.evaluate(dataset, verbose=0)

        # Generate classification report
        report = classification_report(
            true_labels, predicted_labels,
            target_names=class_names,
            digits=4,
            output_dict=True
        )

        # Calculate confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels)

        # Store results
        results = {
            'model_name': model_name,
            'accuracy': float(accuracy),
            'loss': float(loss),
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'true_labels': true_labels.tolist(),
            'predicted_labels': predicted_labels.tolist()
        }

        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Loss: {loss:.4f}")

        return results

    def evaluate_all_models(self, models_dict, val_dataset, class_names):
        """
        Evaluate all models and compare performance.

        Args:
            models_dict: Dictionary of trained models
            val_dataset: Validation dataset
            class_names: List of class names

        Returns:
            Dictionary with evaluation results for all models
        """
        evaluation_results = {}

        print("📊 Evaluating all models on validation dataset...")
        print("=" * 60)

        for model_name, model in models_dict.items():
            try:
                results = self.evaluate_model(model, model_name, val_dataset, class_names)
                evaluation_results[model_name] = results

            except Exception as e:
                print(f"❌ Error evaluating {model_name}: {e}")
                continue

        return evaluation_results

    def get_predictions_and_labels(self, model, dataset, class_names):
        """
        Get predictions and true labels from dataset.

        Args:
            model: Trained model
            dataset: TensorFlow dataset
            class_names: List of class names

        Returns:
            true_labels, predicted_labels, predicted_probs
        """
        true_labels = []
        predicted_labels = []
        predicted_probs = []

        for images, labels in dataset:
            # Get predictions
            predictions = model.predict(images, verbose=0)

            # Convert one-hot labels to class indices
            true_indices = np.argmax(labels.numpy(), axis=1)
            pred_indices = np.argmax(predictions, axis=1)

            true_labels.extend(true_indices)
            predicted_labels.extend(pred_indices)
            predicted_probs.extend(predictions)

        return np.array(true_labels), np.array(predicted_labels), np.array(predicted_probs)

    def plot_confusion_matrix(self, confusion_matrix, class_names, model_name, save_path=None):
        """
        Plot confusion matrix.

        Args:
            confusion_matrix: Confusion matrix array
            class_names: List of class names
            model_name: Name of the model
            save_path: Path to save plot (optional)
        """
        plt.figure(figsize=(12, 10))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)

        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Confusion matrix saved to {save_path}")

        plt.show()

    def plot_model_comparison(self, evaluation_results, save_path=None):
        """
        Plot accuracy comparison across all models.

        Args:
            evaluation_results: Dictionary with evaluation results
            save_path: Path to save plot (optional)
        """
        model_names = list(evaluation_results.keys())
        accuracies = [results['accuracy'] for results in evaluation_results.values()]

        plt.figure(figsize=(12, 6))
        bars = plt.bar(model_names, accuracies, color='skyblue')
        plt.title('Model Accuracy Comparison')
        plt.xlabel('Model')
        plt.ylabel('Validation Accuracy')
        plt.ylim(0, 1)
        plt.xticks(rotation=45, ha='right')

        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.4f}', ha='center', va='bottom')

        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Model comparison plot saved to {save_path}")

        plt.show()

    def find_best_model(self, evaluation_results):
        """
        Find the best performing model based on validation accuracy.

        Args:
            evaluation_results: Dictionary with evaluation results

        Returns:
            Best model name and its metrics
        """
        if not evaluation_results:
            return None, None

        # Sort by accuracy (descending)
        sorted_results = sorted(
            evaluation_results.items(),
            key=lambda x: x[1]['accuracy'],
            reverse=True
        )

        best_model_name = sorted_results[0][0]
        best_metrics = sorted_results[0][1]

        return best_model_name, best_metrics

    def print_evaluation_summary(self, evaluation_results):
        """Print comprehensive evaluation summary."""
        print(f"\n{'='*70}")
        print("📈 MODEL EVALUATION SUMMARY")
        print(f"{'='*70}")

        if not evaluation_results:
            print("❌ No evaluation results available")
            return

        # Sort by accuracy
        sorted_results = sorted(
            evaluation_results.items(),
            key=lambda x: x[1]['accuracy'],
            reverse=True
        )

        print("\n🏆 Model Rankings by Validation Accuracy:")
        print("-" * 70)

        for rank, (model_name, metrics) in enumerate(sorted_results, 1):
            accuracy = metrics['accuracy']
            loss = metrics['loss']
            indicator = "🏆 BEST" if rank == 1 else ""
            print(f"{rank}. {model_name:15s} | Accuracy: {accuracy:.4f} ({accuracy*100:6.2f}%) | Loss: {loss:.4f} {indicator}")

        # Best model details
        best_model, best_metrics = self.find_best_model(evaluation_results)
        if best_model:
            print(f"\n🎯 Best Model: {best_model}")
            print(f"🎯 Best Accuracy: {best_metrics['accuracy']:.4f} ({best_metrics['accuracy']*100:.2f}%)")

    def save_evaluation_results(self, evaluation_results, filename="evaluation_results.json"):
        """
        Save evaluation results to JSON file.

        Args:
            evaluation_results: Dictionary with evaluation results
            filename: Output filename
        """
        filepath = os.path.join(self.metrics_dir, filename)

        try:
            with open(filepath, 'w') as f:
                json.dump(evaluation_results, f, indent=4)
            print(f"✅ Evaluation results saved to {filepath}")
        except Exception as e:
            print(f"❌ Error saving results: {e}")

    def generate_detailed_report(self, evaluation_results, class_names, save_path=None):
        """
        Generate detailed evaluation report.

        Args:
            evaluation_results: Dictionary with evaluation results
            class_names: List of class names
            save_path: Path to save report (optional)
        """
        report = f"""
BONE FRACTURE CLASSIFICATION - EVALUATION REPORT
{'='*60}

Dataset: {len(class_names)} classes
Models Evaluated: {len(evaluation_results)}

"""

        # Sort by accuracy
        sorted_results = sorted(
            evaluation_results.items(),
            key=lambda x: x[1]['accuracy'],
            reverse=True
        )

        report += "\nMODEL PERFORMANCE RANKING:\n"
        report += "-" * 40 + "\n"

        for rank, (model_name, metrics) in enumerate(sorted_results, 1):
            report += f"{rank}. {model_name}\n"
            report += f"   Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n"
            report += f"   Loss: {metrics['loss']:.4f}\n\n"

        # Best model details
        best_model, best_metrics = self.find_best_model(evaluation_results)
        if best_model:
            report += f"BEST MODEL: {best_model}\n"
            report += f"BEST ACCURACY: {best_metrics['accuracy']:.4f} ({best_metrics['accuracy']*100:.2f}%)\n\n"

            # Classification report for best model
            report += f"CLASSIFICATION REPORT - {best_model}:\n"
            report += "-" * 40 + "\n"
            report += classification_report(
                best_metrics['true_labels'],
                best_metrics['predicted_labels'],
                target_names=class_names,
                digits=4
            )

        if save_path:
            try:
                with open(save_path, 'w') as f:
                    f.write(report)
                print(f"✅ Detailed report saved to {save_path}")
            except Exception as e:
                print(f"❌ Error saving report: {e}")

        return report


# Example usage
if __name__ == "__main__":
    from data_loader import DataLoader
    from model_builder import ModelBuilder
    from trainer import ModelTrainer

    # Initialize components
    loader = DataLoader()
    builder = ModelBuilder()
    trainer = ModelTrainer()
    evaluator = ModelEvaluator()

    # Load and preprocess data
    train_ds, val_ds, class_names, num_classes = loader.load_dataset()
    aug = loader.create_data_augmentation()
    train_ds = loader.preprocess_dataset(train_ds, aug)
    val_ds = loader.preprocess_dataset(val_ds)

    # Build and train models
    models = builder.build_all_models(num_classes)
    trained_models, histories = trainer.train_all_models(models, train_ds, val_ds)

    # Evaluate all models
    evaluation_results = evaluator.evaluate_all_models(trained_models, val_ds, class_names)

    # Print summary
    evaluator.print_evaluation_summary(evaluation_results)

    # Plot confusion matrix for best model
    best_model, best_metrics = evaluator.find_best_model(evaluation_results)
    if best_model:
        evaluator.plot_confusion_matrix(
            best_metrics['confusion_matrix'],
            class_names,
            best_model
        )

    # Save results
    evaluator.save_evaluation_results(evaluation_results)