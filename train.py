"""
Simple Training Script for Bone Fracture Classification

This script trains all 5 models and saves them for deployment.
Run this instead of the Jupyter notebook for quick training.
"""

import sys
import os
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.append(str(src_path))

# Import custom modules
from data_loader import DataLoader
from model_builder import ModelBuilder
from trainer import ModelTrainer
from evaluator import ModelEvaluator
from utils import Utils

def main():
    """Main training function."""
    print("🦴 Bone Fracture Classification - Training Script")
    print("=" * 60)

    try:
        # Initialize components
        print("🔄 Initializing components...")
        data_loader = DataLoader()
        model_builder = ModelBuilder()
        trainer = ModelTrainer()
        evaluator = ModelEvaluator()
        utils = Utils()

        # Load and preprocess data
        print("📊 Loading dataset...")
        data_loader = DataLoader(dataset_path="dataset")
        train_ds, val_ds, class_names, num_classes = data_loader.load_dataset()

        print(f"✅ Dataset loaded: {num_classes} classes")
        print(f"   Classes: {class_names}")

        # Create data augmentation
        print("🔄 Setting up data augmentation...")
        augmentation = data_loader.create_data_augmentation()
        train_ds = data_loader.preprocess_dataset(train_ds, augmentation)
        val_ds = data_loader.preprocess_dataset(val_ds)

        # Build models
        print("🏗️ Building models...")
        models = model_builder.build_all_models(num_classes)
        print(f"✅ Built {len(models)} models: {list(models.keys())}")

        # Train all models
        print("🚀 Starting training...")
        print("This may take 5-10 minutes with reduced epochs for quick testing.")
        trained_models, training_histories = trainer.train_all_models(models, train_ds, val_ds, epochs=3)

        # Evaluate models
        print("🔍 Evaluating models...")
        evaluation_results = evaluator.evaluate_all_models(trained_models, val_ds, class_names)

        # Print results
        evaluator.print_evaluation_summary(evaluation_results)

        # Save models and results
        print("💾 Saving models and results...")

        # Save trained models
        for model_name, model in trained_models.items():
            model_path = utils.save_model(model, model_name)
            print(f"   Saved {model_name}: {model_path}")

        # Save evaluation results
        evaluator.save_evaluation_results(evaluation_results)

        # Save training histories
        for model_name, history in training_histories.items():
            utils.save_history(history, model_name)

        # Find and save best model
        best_model_name, best_metrics = evaluator.find_best_model(evaluation_results)
        if best_model_name:
            print(f"🏆 Best Model: {best_model_name} (Accuracy: {best_metrics['accuracy']:.4f})")

            # Save best model with special name
            best_model_path = utils.save_model(
                trained_models[best_model_name],
                "best_model"
            )
            print(f"   Best model saved: {best_model_path}")

        print("\n🎉 Training completed successfully!")
        print("📁 Check the 'outputs' folder for all results.")
        print("🌐 Run 'streamlit run app/app.py' to start the web application.")

    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)