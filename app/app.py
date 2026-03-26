"""
Streamlit Web Application for Bone Fracture Classification

This app provides:
- Image upload and classification
- Model comparison and visualization
- Real-time predictions
- Results interpretation
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import tensorflow as tf
import os
import json
from pathlib import Path
import sys

# Add src directory to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.append(str(src_path))

# Import custom modules
from utils import Utils

# Configure page
st.set_page_config(
    page_title="Bone Fracture Classification",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize utilities
utils = Utils()

# Load class names
CLASS_NAMES = [
    'Avulsion fracture', 'Comminuted fracture', 'Fracture Dislocation',
    'Greenstick fracture', 'Hairline Fracture', 'Impacted fracture',
    'Longitudinal fracture', 'Oblique fracture', 'Pathological fracture',
    'Spiral Fracture'
]

# Model paths
MODEL_DIR = project_root / "outputs" / "models"
MODEL_PATH = MODEL_DIR / "best_model.keras"

EVALUATION_FILE = project_root / "outputs" / "metrics" / "evaluation_results.json"


@st.cache_resource
def load_models():
    """Load all available trained models."""
    loaded_models = {}

    model_files = list(MODEL_DIR.glob("*.h5")) + list(MODEL_DIR.glob("*.keras"))

    if not model_files:
        st.sidebar.error("❌ No models found in models folder")
        return loaded_models

    for model_file in model_files:
        try:
            model_name = model_file.stem
            model = tf.keras.models.load_model(model_file)
            loaded_models[model_name] = model
            st.sidebar.success(f"✅ Loaded {model_name}")
        except Exception as e:
            st.sidebar.error(f"❌ Failed to load {model_file.name}: {e}")

    return loaded_models
    

@st.cache_data
def load_evaluation_results():
    """Load evaluation results if available."""
    if EVALUATION_FILE.exists():
        try:
            with open(EVALUATION_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading evaluation results: {e}")
    return None


def display_model_predictions(predictions, class_names):
    """Display prediction results for each selected model."""
    st.subheader("🎯 Prediction Results")

    result_rows = []
    for model_name, pred in predictions.items():
        if pred is None:
            continue
        top_idx = int(np.argmax(pred))
        top_class = class_names[top_idx]
        top_prob = float(pred[top_idx])
        result_rows.append((model_name, top_class, top_prob))

    if not result_rows:
        st.error("No valid predictions available.")
        return

    # Show each model result as metric cards
    cols = st.columns(len(result_rows))
    for idx, (model_name, top_class, top_prob) in enumerate(result_rows):
        with cols[idx]:
            st.metric(f"{model_name}", f"{top_class} ({top_prob:.1%})")

    # See all predictions for first model (as baseline)
    first_model = next(iter(predictions.keys()))
    first_preds = predictions[first_model]
    if first_preds is not None:
        st.markdown("---")
        st.write(f"### {first_model} probability breakdown")
        sorted_indices = np.argsort(first_preds)[::-1]
        sorted_classes = [class_names[i] for i in sorted_indices]
        sorted_probs = first_preds[sorted_indices]

        import pandas as pd
        df = pd.DataFrame({
            'Fracture Type': sorted_classes,
            'Probability': sorted_probs,
            'Percentage': [f"{p:.1%}" for p in sorted_probs]
        })
        st.dataframe(df, use_container_width=True)


def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess uploaded image for model prediction.

    Args:
        image: PIL Image object
        target_size: Target size for model input

    Returns:
        Preprocessed image array
    """
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Resize image
    image = image.resize(target_size)

    # Convert to array and normalize
    img_array = np.array(image) / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def predict_with_models(image, models):
    """
    Get predictions from all loaded models.

    Args:
        image: Preprocessed image array
        models: Dictionary of loaded models

    Returns:
        Dictionary of predictions
    """
    predictions = {}

    for model_name, model in models.items():
        try:
            pred = model.predict(image, verbose=0)[0]
            predictions[model_name] = pred
        except Exception as e:
            st.error(f"Error predicting with {model_name}: {e}")
            predictions[model_name] = None

    return predictions


def plot_prediction_bar_chart(predictions, class_names):
    """Plot prediction probabilities as a bar chart."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Choose the most confident model (max top probability)
    best_model = None
    best_score = -1
    best_preds = None
    for model_name, model_preds in predictions.items():
        if model_preds is None:
            continue
        score = float(np.max(model_preds))
        if score > best_score:
            best_score = score
            best_model = model_name
            best_preds = model_preds

    if best_preds is None:
        return None

    # Create bar chart
    bars = ax.barh(class_names, best_preds, color='skyblue')

    # Highlight top prediction
    top_idx = int(np.argmax(best_preds))
    bars[top_idx].set_color('darkblue')

    ax.set_xlabel('Probability')
    ax.set_title(f'Fracture Type Probabilities (Model: {best_model})')
    ax.grid(True, alpha=0.3)

    # Add probability values on bars (top 5 only for clarity)
    sorted_idx = np.argsort(best_preds)[::-1]
    for i in sorted_idx[:5]:
        prob = best_preds[i]
        bar = bars[i]
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{prob:.3f}', ha='left', va='center', color='black', fontsize=10)

    plt.tight_layout()
    return fig


def plot_model_comparison(predictions, class_names):
    """Plot prediction comparison across models."""
    fig, ax = plt.subplots(figsize=(14, 8))

    model_names = list(predictions.keys())
    num_classes = len(class_names)

    # Create data matrix
    pred_matrix = np.zeros((len(model_names), num_classes))
    for i, (model_name, preds) in enumerate(predictions.items()):
        if preds is not None:
            pred_matrix[i] = preds

    # Create heatmap
    sns.heatmap(pred_matrix.T, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=model_names, yticklabels=class_names, ax=ax)

    ax.set_title('Model Predictions Comparison')
    ax.set_xlabel('Model')
    ax.set_ylabel('Fracture Type')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    return fig


def display_prediction_results(predictions, class_names):
    """Display prediction results in a formatted way."""
    st.subheader("🎯 Prediction Results")

    # Get the first valid prediction for display
    first_model = next(iter(predictions.keys()))
    first_preds = predictions[first_model]

    if first_preds is None:
        st.error("No valid predictions available.")
        return

    # Find top prediction
    top_idx = np.argmax(first_preds)
    top_class = class_names[top_idx]
    top_prob = first_preds[top_idx]

    # Display top prediction prominently
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.success(f"**Top Prediction: {top_class}**")
        st.metric("Confidence", f"{top_prob:.1%}")

    with col2:
        st.info(f"Probability: {top_prob:.4f}")

    with col3:
        if top_prob > 0.8:
            st.success("High Confidence")
        elif top_prob > 0.5:
            st.warning("Medium Confidence")
        else:
            st.error("Low Confidence")

    # Display all predictions
    st.subheader("📊 All Predictions")

    # Sort predictions by probability
    sorted_indices = np.argsort(first_preds)[::-1]
    sorted_classes = [class_names[i] for i in sorted_indices]
    sorted_probs = first_preds[sorted_indices]

    # Create a dataframe for display
    import pandas as pd
    df = pd.DataFrame({
        'Fracture Type': sorted_classes,
        'Probability': sorted_probs,
        'Percentage': [f"{p:.1%}" for p in sorted_probs]
    })

    st.dataframe(df, use_container_width=True)


def main():
    """Main application function."""
    st.title("🦴 Bone Fracture Classification System")
    st.markdown("---")

    # Sidebar
    st.sidebar.title("🔧 Control Panel")

    models = load_models()
    if models:
        st.sidebar.success(f"✅ {len(models)} model(s) available")
        st.session_state.models = models
    else:
        st.sidebar.error("❌ No trained models found in outputs/models. Please train first.")

    with st.sidebar.expander("📥 Model Controls", expanded=True):
        st.write("Automatically loaded from outputs/models")
        if st.button("Reload Models"):
            st.session_state.models = load_models()
            st.rerun()

    model_names = sorted(models.keys()) if models else []
    selected_models = st.sidebar.multiselect("Select models for prediction", model_names, default=model_names)

    if not selected_models:
        st.sidebar.warning("Please select at least one model for prediction")

    # Load evaluation results
    eval_results = load_evaluation_results()
    if eval_results:
        st.sidebar.success("✅ Evaluation results loaded")

    # Main content
    st.markdown("""
    ### Welcome to the Bone Fracture Classification System!

    This application uses deep learning to classify X-ray images into 10 different types of bone fractures:

    - Avulsion fracture
    - Comminuted fracture
    - Fracture Dislocation
    - Greenstick fracture
    - Hairline Fracture
    - Impacted fracture
    - Longitudinal fracture
    - Oblique fracture
    - Pathological fracture
    - Spiral Fracture

    **Instructions:**
    1. Upload an X-ray image (JPG, PNG, JPEG)
    2. Click "Classify Image" to get predictions
    3. View results and model comparisons
    """)

    # File uploader
    st.subheader("📤 Upload X-ray Image")
    uploaded_file = st.file_uploader(
        "Choose an X-ray image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear X-ray image for fracture classification"
    )

    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("📷 Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded X-ray", use_column_width=True)

        with col2:
            st.subheader("🔍 Analysis")

            # Classification button
            if st.button("🔬 Classify Image", type="primary", use_container_width=True):
                if 'models' not in st.session_state:
                    st.error("❌ Please load models first using the sidebar.")
                    return

                with st.spinner("Analyzing image..."):
                    # Preprocess image
                    processed_image = preprocess_image(image)

                    # Get predictions from selected models only
                    models_for_prediction = {
                        name: st.session_state.models[name]
                        for name in selected_models
                        if name in st.session_state.models
                    }

                    if not models_for_prediction:
                        st.error("No selected models available. Please select models in the sidebar.")
                        return

                    predictions = predict_with_models(processed_image, models_for_prediction)

                    # Store predictions in session state
                    st.session_state.predictions = predictions
                    st.session_state.processed_image = processed_image

                st.success("✅ Analysis complete!")

    # Display results if available
    if 'predictions' in st.session_state:
        st.markdown("---")

        # Display prediction results (clear, high confidence, top class)
        display_model_predictions(st.session_state.predictions, CLASS_NAMES)

        # Visualization tabs
        tab1, tab2, tab3 = st.tabs(["📊 Probability Chart", "🔄 Model Comparison", "📋 Model Performance"])

        with tab1:
            st.subheader("Probability Distribution")
            fig = plot_prediction_bar_chart(st.session_state.predictions, CLASS_NAMES)
            if fig:
                st.pyplot(fig)

        with tab2:
            st.subheader("Model Predictions Comparison")
            if len(st.session_state.models) > 1:
                fig = plot_model_comparison(st.session_state.predictions, CLASS_NAMES)
                if fig:
                    st.pyplot(fig)
            else:
                st.info("Load multiple models to see comparison.")

        with tab3:
            st.subheader("Model Performance Summary")
            if eval_results:
                # Create performance summary
                model_performance = []
                for model_name, metrics in eval_results.items():
                    model_performance.append({
                        'Model': model_name,
                        'Accuracy': f"{metrics['accuracy']:.1%}",
                        'Loss': f"{metrics['loss']:.4f}"
                    })

                import pandas as pd
                perf_df = pd.DataFrame(model_performance)
                st.dataframe(perf_df, use_container_width=True)

                # Find best model
                best_model = max(eval_results.items(), key=lambda x: x[1]['accuracy'])
                st.success(f"🏆 Best Model: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.1%})")
            else:
                st.info("No evaluation results found. Run the training notebook to generate results.")

    # Footer
    st.markdown("---")
    st.markdown("""
    ### 📚 About This Project

    This application is part of a comprehensive deep learning project for bone fracture classification.
    The system uses transfer learning with pre-trained models including:

    - MobileNetV2
    - ResNet50
    - EfficientNetB0
    - DenseNet121
    - Custom CNN

    **For more information, check the project documentation and training notebook.**
    """)

    # System info
    with st.sidebar.expander("💻 System Info"):
        system_info = utils.get_system_info()
        st.write(f"**TensorFlow:** {system_info['tensorflow_version']}")
        st.write(f"**GPU Available:** {system_info['gpu_available']}")
        st.write(f"**Python:** {system_info['python_version']}")


if __name__ == "__main__":
    main()