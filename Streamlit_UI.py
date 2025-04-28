import streamlit as st
import joblib
import pandas as pd


MODELS = {
    "AdaBoost": "models/AdaBoost_model.pkl",
    "XGBoost": "models/XGBoost_model.pkl",
    "GBDT": "models/GBDT_model.pkl",
    "RF": "modelsmodels/RF_model.pkl",
    "MLP": "models/MLP_model.pkl",  
    "SVR": "models/SVR_model.pkl",
    "KNN": "models/KNN_model.pkl",
    "KRR": "models/KRR_model.pkl"
}


@st.cache_resource
def load_model(model_path):
    try:
        model = joblib.load(model_path)
        if not hasattr(model, 'estimators_'):  # AdaBoost 特有检查
            raise ValueError("Model not trained！")
        return model
    except Exception as e:
        st.error(f"Loading failed: {str(e)}")
        return None


def main():
    st.title("🔮 Multi-Model Prediction System")

    st.sidebar.header("Model Configuration")
    selected_model_name = st.sidebar.selectbox(
        "Select Prediction Model",
        list(MODELS.keys()),
        help="Choose model based on task type"
    )

    st.header("1. Upload Data")

    st.markdown("""
    **📌 Data Format Requirements:**  
    - Upload a CSV/Excel file with **7 input parameters** (7 rows × n columns)  
    """)

    uploaded_file = st.file_uploader(
        "Upload CSV/Excel File (7 rows × n columns)",
        type=["csv", "xlsx"]
    )

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Data loaded successfully！")
        st.write("Data Preview：", df.head(3))

        st.header("2. Model Information")
        model_path = MODELS[selected_model_name]
        st.code(f"Selected Model: {selected_model_name}\nPath: {model_path}")

        if st.button("🚀  Run Prediction", type="primary"):
            with st.spinner("Loading model..."):
                try:
                    model = load_model(model_path)
                    st.success("Pre-trained model loaded successfully!")

                    if "clf" in model_path.lower():  
                        predictions = model.predict(df)
                        st.write("Predicted Classes：", predictions)
                    else:  # 回归模型
                        predictions = model.predict(df)
                        st.write("Predicted Values：", predictions)

                    result_df = pd.DataFrame({"Predictions": predictions})
                    st.download_button("📥 Download Results",
                                       result_df.to_csv(),
                                       "predictions.csv")

                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    main()
