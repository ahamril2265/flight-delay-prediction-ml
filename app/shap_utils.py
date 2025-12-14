import shap
import numpy as np

def get_shap_explanation(model, input_df):
    """
    Generate SHAP values safely for a single prediction
    from a sklearn Pipeline with preprocessing.
    """

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    # Transform input
    X_transformed = preprocessor.transform(input_df)

    # 🔴 CRITICAL FIX: force numeric dtype
    X_transformed = np.asarray(X_transformed, dtype=float)

    # SHAP explainer
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_transformed)

    return shap_values, explainer.expected_value
