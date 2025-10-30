import os
import json
import pickle
import numpy as np
import streamlit as st
import shap  # Add this at the top for SHAP explainability
import matplotlib.pyplot as plt  # Needed for plotting
from sklearn.svm import SVR  # <-- Import SVR for isinstance check

# Paths (default candidates)
DEFAULT_CANDIDATE_DIRS = [
    os.path.join('retail', 'models'),
    os.path.join('retail', 'retail', 'models'),
    'models',
]
METADATA_NAME = 'metadata.json'
SCALER_NAME = 'scaler.pkl'

# Default categorical options used when metadata lacks mappings
DEFAULT_CATEGORICAL_OPTIONS = {
    'Category': ['Clothing', 'Electronics', 'Furniture', 'Groceries', 'Toys'],
    'Region': ['East', 'North', 'South', 'West'],
    'Weather': ['Cloudy', 'Rainy', 'Snowy', 'Sunny'],
    'Seasonality': ['Autumn', 'Spring', 'Summer', 'Winter'],
}


def resolve_models_dir(override_dir: str | None = None) -> str:
    # If user provided override and it exists, use it
    if override_dir and os.path.isdir(override_dir):
        return override_dir
    # Try environment variable
    env_dir = os.environ.get('MODELS_DIR')
    if env_dir and os.path.isdir(env_dir):
        return env_dir
    # Probe default candidates
    for candidate in DEFAULT_CANDIDATE_DIRS:
        if os.path.isdir(candidate):
            return candidate
    # Fallback to first candidate (even if missing) for clear error
    return DEFAULT_CANDIDATE_DIRS[0]


@st.cache_resource
def load_artifacts(models_dir: str):
    metadata_path = os.path.join(models_dir, METADATA_NAME)
    scaler_path = os.path.join(models_dir, SCALER_NAME)

    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"{METADATA_NAME} not found in {models_dir}. Run the notebook save cell first.")

    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"{SCALER_NAME} not found in {models_dir}.")

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    available_models = {}
    for name in metadata.get('models', []):
        model_path = os.path.join(models_dir, f"{name}.pkl")
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                available_models[name] = pickle.load(f)

    return metadata, scaler, available_models


# --- UI helpers ---

def normalize(name: str) -> str:
    return ''.join(ch.lower() for ch in name if ch.isalnum())


# Known feature aliases -> canonical keys
ALIASES = {
    'inventory': {'inventory', 'inventorylevel'},
    'sales': {'sales', 'unitssold'},
    'orders': {'orders', 'unitsordered'},
    'price': {'price'},
    'discount': {'discount'},
    'competitorprice': {'competitorprice', 'competitorpricing'},
    'promotion': {'promotion', 'holidaypromotion', 'isholiday', 'holiday'},
    'category': {'category'},
    'region': {'region'},
    'weather': {'weather', 'weathercondition'},
    'seasonality': {'seasonality'},
}

# Reverse lookup from normalized feature name to canonical key
def canonical_key(feature_name: str) -> str | None:
    n = normalize(feature_name)
    for key, variants in ALIASES.items():
        if n in variants:
            return key
    return None


def _encode_from_defaults(feature_name: str, col):
    # Use default options; encode as index of alphabetically-sorted options
    options = DEFAULT_CATEGORICAL_OPTIONS.get(feature_name)
    if not options:
        # Try using canonical key lookup
        key = canonical_key(feature_name)
        key_to_title = {
            'category': 'Category',
            'region': 'Region',
            'weather': 'Weather',
            'seasonality': 'Seasonality',
        }
        options = DEFAULT_CATEGORICAL_OPTIONS.get(key_to_title.get(key, ''), [])
    options_sorted = sorted([str(o) for o in options]) if options else []
    if options_sorted:
        selected = col.selectbox(feature_name, options=options_sorted, index=0)
        return float(options_sorted.index(str(selected)))
    # No defaults available -> fall back numeric
    return col.number_input(feature_name, value=0.0, step=1.0, format="%f")


def render_input_for_feature(feature_name: str, col, categorical_mappings: dict = None):
    key = canonical_key(feature_name)
    
    # Categorical features first: try metadata mappings
    if categorical_mappings and feature_name in categorical_mappings:
        mapping = categorical_mappings[feature_name]
        options = mapping.get('values', [])
        if options:
            show_options = [str(v) for v in options]
            selected = col.selectbox(feature_name, options=show_options, index=0)
            encoded = mapping.get('encoded', [])
            values = [str(v) for v in mapping.get('values', [])]
            if selected in values:
                idx = values.index(selected)
                return float(encoded[idx]) if idx < len(encoded) else float(idx)
        # If mapping present but empty, continue to default behavior below

    # If not provided via metadata, use friendly dropdowns for known categorical keys
    if key in {'category', 'region', 'weather', 'seasonality'}:
        return _encode_from_defaults(feature_name, col)

    # Numeric defaults
    if key in {'inventory', 'sales', 'orders'}:
        return col.number_input(feature_name, value=0, step=1, format="%d")

    if key in {'price', 'competitorprice'}:
        return col.number_input(feature_name, value=0.0, step=0.01, format="%f")

    if key == 'discount':
        return col.slider(feature_name, min_value=0, max_value=100, value=0, step=1)

    if key == 'promotion':
        # Check if promotion has categorical mapping first
        if categorical_mappings and feature_name in categorical_mappings:
            mapping = categorical_mappings[feature_name]
            options = mapping.get('values', [])
            if options:
                selected = col.selectbox(feature_name, options=[str(v) for v in options], index=0)
                encoded = mapping.get('encoded', [])
                values = mapping.get('values', [])
                if str(selected) in [str(v) for v in values]:
                    idx = [str(v) for v in values].index(str(selected))
                    return float(encoded[idx]) if idx < len(encoded) else 0.0
        # Fallback to Yes/No
        label = col.selectbox(feature_name, options=["No", "Yes"], index=0)
        return 1.0 if label == "Yes" else 0.0

    # Fallback generic numeric
    return col.number_input(feature_name, value=0.0, step=1.0, format="%f")


def main():
    st.set_page_config(page_title="Retail Demand Prediction", page_icon="📦", layout="centered")
    st.title("📦 Retail Demand Prediction")
    st.caption("Load saved .pkl models and predict demand from features.")

    st.sidebar.header("Settings")
    suggested_dir = resolve_models_dir()
    models_dir_input = st.sidebar.text_input("Models directory", value=suggested_dir)

    try:
        models_dir = resolve_models_dir(models_dir_input)
        metadata, scaler, models = load_artifacts(models_dir)
        st.sidebar.success(f"Using: {models_dir}")
    except Exception as e:
        st.sidebar.error(str(e))
        st.stop()

    feature_names = metadata.get('feature_names', [])
    target_name = metadata.get('target_name', 'Demand')
    categorical_mappings = metadata.get('categorical_mappings', {})

    if not feature_names:
        st.error("No feature names found in metadata.json")
        st.stop()

    if not models:
        st.error("No models available. Ensure .pkl models were saved.")
        st.stop()

    st.sidebar.subheader("Model")
    model_name = st.sidebar.selectbox("Select model", list(models.keys()))
    model = models[model_name]

    st.subheader("Input Features")
    cols = st.columns(2)
    user_inputs = {}

    for idx, feature in enumerate(feature_names):
        with cols[idx % 2]:
            user_inputs[feature] = render_input_for_feature(feature, cols[idx % 2], categorical_mappings)

    if st.button("Predict"):
        try:
            x_raw = np.array([[user_inputs[f] for f in feature_names]], dtype=float)
            x_scaled = scaler.transform(x_raw)
            y_pred = model.predict(x_scaled)
            st.success(f"Predicted {target_name}: {float(y_pred[0]):.2f}")

            # ---- SHAP Explainability ----
            st.markdown("#### Model Explainability (SHAP)")
            explainer = None
            try:
                if isinstance(model, SVR):
                    st.info("Using SHAP KernelExplainer for SVR. This may be slow — only current input is explained.")
                    # Use a simple background: zeros or random samples
                    # If you have training data, use a small random sample from it instead of zeros
                    background = np.zeros((10, x_scaled.shape[1]))
                    explainer = shap.KernelExplainer(model.predict, background)
                    shap_values = explainer.shap_values(x_scaled)
                    fig, ax = plt.subplots()
                    shap.waterfall_plot(shap.Explanation(values=shap_values[0], base_values=explainer.expected_value,
                                                          data=x_scaled[0], feature_names=feature_names), max_display=10, show=False)
                    st.pyplot(fig)
                else:
                    # Other models (tree-based, linear)
                    explainer = shap.Explainer(model, x_scaled)
                    shap_values = explainer(x_scaled)
                    fig, ax = plt.subplots()
                    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
                    st.pyplot(fig)
            except Exception as shap_exc:
                st.warning(f"SHAP explanation failed: {shap_exc}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    with st.expander("About"):
        st.write("""
        Inputs are displayed in a user-friendly way:
        - Category, Region, Weather, Seasonality as dropdown selectors (from metadata; otherwise sensible defaults)
        - Inventory/Sales/Orders as integers
        - Price/Competitor Price as decimals
        - Discount as 0–100 slider
        - Promotion as Yes/No (or mapping from metadata)
        Other unknown features default to numeric inputs.
        """)


if __name__ == "__main__":
    main()
