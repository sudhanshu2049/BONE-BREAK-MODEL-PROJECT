"""
Data Loader Module for Bone Fracture Classification

This module handles:
- Loading dataset from directory structure
- Data preprocessing and normalization
- Data augmentation for training
- Train/validation split
"""

import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers
import matplotlib.pyplot as plt


class DataLoader:
    """Handles data loading, preprocessing, and augmentation for bone fracture classification."""

    def __init__(self, dataset_path="dataset", img_size=(224, 224), batch_size=32):
        """
        Initialize DataLoader.

        Args:
            dataset_path: Path to dataset directory
            img_size: Image dimensions (height, width)
            batch_size: Batch size for data loading
        """
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.batch_size = batch_size

    def load_dataset(self, validation_split=0.2, seed=42):
        """
        Load dataset using image_dataset_from_directory.

        Args:
            validation_split: Fraction of data for validation
            seed: Random seed for reproducibility

        Returns:
            train_dataset, val_dataset, class_names, num_classes
        """
        print(f"Loading dataset from: {self.dataset_path}")

        # Load full dataset
        full_dataset = image_dataset_from_directory(
            self.dataset_path,
            seed=seed,
            image_size=self.img_size,
            batch_size=self.batch_size,
            label_mode='categorical',
            shuffle=True
        )

        # Get class information
        class_names = full_dataset.class_names
        num_classes = len(class_names)

        # Calculate split sizes
        dataset_size = tf.data.experimental.cardinality(full_dataset).numpy()
        train_size = int((1 - validation_split) * dataset_size)

        # Split into train and validation
        train_dataset = full_dataset.take(train_size)
        val_dataset = full_dataset.skip(train_size)

        print(f"Dataset loaded: {num_classes} classes, {len(class_names)} total")
        print(f"Training batches: {train_size}, Validation batches: {dataset_size - train_size}")

        return train_dataset, val_dataset, class_names, num_classes

    def create_data_augmentation(self):
        """
        Create data augmentation pipeline.

        Returns:
            Keras Sequential model with augmentation layers
        """
        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.2),
        ])

        print("Data augmentation created: RandomFlip, RandomRotation, RandomZoom")
        return data_augmentation

    def preprocess_dataset(self, dataset, data_augmentation=None, normalize=True):
        """
        Apply preprocessing to dataset.

        Args:
            dataset: TensorFlow dataset
            data_augmentation: Augmentation model (optional)
            normalize: Whether to normalize pixel values

        Returns:
            Preprocessed dataset
        """
        def preprocess(images, labels):
            if data_augmentation:
                images = data_augmentation(images, training=True)
            if normalize:
                images = images / 255.0
            return images, labels

        dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def visualize_samples(self, dataset, class_names, num_samples=3):
        """
        Visualize sample images from dataset.

        Args:
            dataset: TensorFlow dataset
            class_names: List of class names
            num_samples: Number of samples per class to show
        """
        plt.figure(figsize=(15, 10))

        samples_shown = {class_name: 0 for class_name in class_names}

        for images, labels in dataset.take(1):
            images, labels = images.numpy(), labels.numpy()

            for i in range(len(images)):
                class_idx = tf.argmax(labels[i]).numpy()
                class_name = class_names[class_idx]

                if samples_shown[class_name] < num_samples:
                    plt.subplot(len(class_names), num_samples,
                              class_idx * num_samples + samples_shown[class_name] + 1)
                    plt.imshow(images[i].astype('uint8'))
                    plt.title(f'{class_name}', fontsize=10)
                    plt.axis('off')
                    samples_shown[class_name] += 1

                if all(count >= num_samples for count in samples_shown.values()):
                    break

        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Initialize data loader
    loader = DataLoader()

    # Load dataset
    train_ds, val_ds, class_names, num_classes = loader.load_dataset()

    # Create augmentation
    aug = loader.create_data_augmentation()

    # Preprocess datasets
    train_ds = loader.preprocess_dataset(train_ds, aug, normalize=True)
    val_ds = loader.preprocess_dataset(val_ds, normalize=True)

    # Visualize samples
    loader.visualize_samples(train_ds, class_names)