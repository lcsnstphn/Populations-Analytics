import streamlit as st
import pandas as pd
import data_processing as dp
import visualizations as vis
import model as md
import evaluation as ev
import prediction as pr

st.set_page_config(page_title="Population Analytics", layout="wide")

st.title("🌍 Population Descriptive & Predictive Analytics")

# Sidebar File Upload
st.sidebar.header("📁 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is None:
    # Default to actual system file if user has not uploaded a new one
    try:
        default_path = '/content/population.csv.xlsx.csv'
        df_raw = pd.read_csv(default_path)
        st.sidebar.success("Loaded default database successfully!")
    except Exception:
        df_raw = None
        st.warning("Please upload a population dataset to activate full features.")
else:
    df_raw = dp.load_data(uploaded_file)
    st.sidebar.success("File uploaded successfully!")

if df_raw is not None:
    df = dp.clean_and_prepare_data(df_raw)
    
    tab1, tab2, tab3 = st.tabs(["📊 Descriptive Analytics", "🤖 Predictive Analytics", "🔮 Single Live Prediction"])
    
    with tab1:
        st.header("Descriptive Analytics & Visualizations")
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(vis.plot_population_distribution(df), use_container_width=True)
            st.plotly_chart(vis.plot_youth_analysis(df), use_container_width=True)
            
        with col2:
            st.plotly_chart(vis.plot_gender_analysis(df), use_container_width=True)
            st.subheader("Dataset Summary Metrics")
            st.dataframe(df.describe(), use_container_width=True)
            
    with tab2:
        st.header("Classification Models & Evaluation")
        
        X_train, X_test, y_train, y_test, scaler = md.prepare_model_data(df)
        trained_models = md.train_classification_models(X_train, y_train)
        metrics_df, roc_curves, conf_matrices = ev.evaluate_all_models(trained_models, X_test, y_test)
        
        st.subheader("Model Performance Summary Table")
        st.dataframe(metrics_df.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']), use_container_width=True)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(ev.plot_roc_curves(roc_curves), use_container_width=True)
        with col2:
            selected_model = st.selectbox("Select model for Confusion Matrix:", list(trained_models.keys()))
            st.plotly_chart(ev.plot_confusion_matrix(conf_matrices[selected_model], selected_model), use_container_width=True)
            
    with tab3:
        st.header("Make Real-time Population Trend Prediction")
        st.markdown("Predict if a region/cluster is classified as a High Density/Growth segment (Class 1) based on current demographic parameters.")
        
        col1, col2 = st.columns(2)
        with col1:
            age_input = st.number_input("Median Age of Area:", min_value=1, max_value=100, value=30)
            pop_input = st.number_input("Total Sample Cohort Population:", min_value=10, max_value=1000000, value=5000)
            is_youth_input = st.selectbox("Is this area primarily defined as a Youth Cohort?", ["No", "Yes"])
            is_youth_numeric = 1 if is_youth_input == "Yes" else 0
            
            pred_model_name = st.selectbox("Classifier to Use:", list(trained_models.keys()))
            
        with col2:
            st.write("### Prediction Verdict")
            model_obj = trained_models[pred_model_name]
            prediction, probability = pr.make_single_prediction(model_obj, scaler, age_input, pop_input, is_youth_numeric)
            
            if prediction == 1:
                st.error(f"🚨 Classified as Class 1 (High Density/Growth Area) with {probability:.1%} probability.")
            else:
                st.success(f"✅ Classified as Class 0 (Normal Area) with {(1 - probability):.1%} confidence.")
