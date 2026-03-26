"""
Model Builder Module for Bone Fracture Classification

This module defines and builds 5 different neural network architectures:
1. MobileNetV2 (Transfer Learning)
2. ResNet50 (Transfer Learning)
3. EfficientNetB0 (Transfer Learning)
4. DenseNet121 (Transfer Learning)
5. Custom CNN (Built from scratch)
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import (
    MobileNetV2, ResNet50, EfficientNetB0, DenseNet121
)


class ModelBuilder:
    """Builds different neural network architectures for classification."""

    def __init__(self, input_shape=(224, 224, 3)):
        """
        Initialize ModelBuilder.

        Args:
            input_shape: Input shape for models (height, width, channels)
        """
        self.input_shape = input_shape

    def create_mobilenetv2(self, num_classes):
        """
        Create MobileNetV2 model with transfer learning.

        Args:
            num_classes: Number of output classes

        Returns:
            Compiled Keras model
        """
        # Load pretrained MobileNetV2
        base_model = MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False

        # Custom classification head
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])

        return model

    def create_resnet50(self, num_classes):
        """
        Create ResNet50 model with transfer learning.

        Args:
            num_classes: Number of output classes

        Returns:
            Compiled Keras model
        """
        # Load pretrained ResNet50
        base_model = ResNet50(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False

        # Custom classification head
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])

        return model

    def create_efficientnetb0(self, num_classes):
        """
        Create EfficientNetB0 model with transfer learning.

        Args:
            num_classes: Number of output classes

        Returns:
            Compiled Keras model
        """
        # Load pretrained EfficientNetB0
        base_model = EfficientNetB0(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False

        # Custom classification head
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])

        return model

    def create_densenet121(self, num_classes):
        """
        Create DenseNet121 model with transfer learning.

        Args:
            num_classes: Number of output classes

        Returns:
            Compiled Keras model
        """
        # Load pretrained DenseNet121
        base_model = DenseNet121(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False

        # Custom classification head
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])

        return model

    def create_custom_cnn(self, num_classes):
        """
        Create custom CNN model built from scratch.

        Args:
            num_classes: Number of output classes

        Returns:
            Compiled Keras model
        """
        model = models.Sequential([
            # Block 1
            layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                         input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Block 2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Block 3
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Classification head
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])

        return model

    def build_all_models(self, num_classes):
        """
        Build all 5 model architectures.

        Args:
            num_classes: Number of output classes

        Returns:
            Dictionary of model names and models
        """
        models_dict = {}

        # Build each model
        try:
            models_dict['MobileNetV2'] = self.create_mobilenetv2(num_classes)
            print("✓ MobileNetV2 model created")
        except Exception as e:
            print(f"✗ Error creating MobileNetV2: {e}")

        try:
            models_dict['ResNet50'] = self.create_resnet50(num_classes)
            print("✓ ResNet50 model created")
        except Exception as e:
            print(f"✗ Error creating ResNet50: {e}")

        try:
            models_dict['EfficientNetB0'] = self.create_efficientnetb0(num_classes)
            print("✓ EfficientNetB0 model created")
        except Exception as e:
            print(f"✗ Error creating EfficientNetB0: {e}")

        try:
            models_dict['DenseNet121'] = self.create_densenet121(num_classes)
            print("✓ DenseNet121 model created")
        except Exception as e:
            print(f"✗ Error creating DenseNet121: {e}")

        try:
            models_dict['Custom CNN'] = self.create_custom_cnn(num_classes)
            print("✓ Custom CNN model created")
        except Exception as e:
            print(f"✗ Error creating Custom CNN: {e}")

        print(f"\nSuccessfully built {len(models_dict)}/{5} models")
        return models_dict

    def compile_model(self, model, model_name):
        """
        Compile a model with standard settings.

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


# Example usage
if __name__ == "__main__":
    # Initialize model builder
    builder = ModelBuilder()

    # Build all models (assuming 10 classes for bone fractures)
    num_classes = 10
    models = builder.build_all_models(num_classes)

    # Compile all models
    for name, model in models.items():
        models[name] = builder.compile_model(model, name)

    print(f"\nAll models ready for training: {list(models.keys())}")