import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================
# Load dataset
# =====================================
@st.cache_data
def load_data():
    df = pd.read_csv("merged_dataset.csv")

    # ✅ If encoded columns exist, decode them
    status_map = {0: "Cancelled", 1: "Completed", 2: "No-show", 3: "Scheduled"}
    specialization_map = {0: "Cardiology", 1: "Surgery", 2: "Pediatrics", 3: "Orthopedics"}

    if "status" in df.columns:
        df["status"] = df["status"].map(status_map).fillna(df["status"])
    if "specialization" in df.columns:
        df["specialization"] = df["specialization"].map(specialization_map).fillna(df["specialization"])

    return df

df = load_data()

# =====================================
# Dashboard Title & Intro
# =====================================
st.title(" Hospital Data Management Dashboard")
st.markdown("""
This dashboard was created as part of **Technical Programming Assessment 4**.  
It provides an overview of the hospital dataset and compares machine learning models (Decision Tree & Random Forest) used to predict patient appointment outcomes.
""")

# Sidebar Navigation
tab = st.sidebar.radio("📌 Select Tab", ["📊 Data Overview", "🤖 Machine Learning Results", "📈 Insights & Visualizations"])

# =====================================
# TAB 1: Data Overview
# =====================================
if tab == "📊 Data Overview":
    st.header("🔎 Data Overview")

    st.subheader("📋 Sample Data")
    st.dataframe(df.head())

    st.subheader("📊 Summary Statistics")
    numeric_summary = df.describe().T
    st.markdown("### 🔢 Numeric Summary")
    st.dataframe(numeric_summary)

    # Categorical summary (only if object columns exist)
    cat_cols = df.select_dtypes(include=["object"]).columns
    if len(cat_cols) > 0:
        categorical_summary = df[cat_cols].describe().T
        st.markdown("### 🔠 Categorical Summary")
        st.dataframe(categorical_summary)
    else:
        st.info("No categorical (object) columns available after encoding.")

    # Example Chart: Doctors per Branch
    if "hospital_branch" in df.columns:
        branch_counts = df["hospital_branch"].value_counts()
        st.subheader("🏥 Doctors per Hospital Branch")
        st.bar_chart(branch_counts)

    # Example Chart: Age distribution
    if "age" in df.columns:
        st.subheader("👨‍⚕️ Age Distribution of Patients")
        fig, ax = plt.subplots()
        sns.histplot(df["age"].dropna(), bins=20, kde=True, ax=ax)
        st.pyplot(fig)

# =====================================
# TAB 2: Machine Learning Results
# =====================================
elif tab == "🤖 Machine Learning Results":
    st.header("🤖 Machine Learning Model Evaluation")

    # Model Accuracies (from your Colab)
    dt_accuracy = 0.69
    rf_accuracy = 0.62

    results = pd.DataFrame({
        "Model": ["Decision Tree", "Random Forest"],
        "Accuracy": [dt_accuracy, rf_accuracy]
    })

    st.subheader("📋 Model Comparison Table")
    st.dataframe(results)

    st.subheader("📊 Accuracy Comparison")
    st.bar_chart(results.set_index("Model"))

    st.markdown("""
    ✅ **Decision Tree Accuracy: 69%**  
    ✅ **Random Forest Accuracy: 62%**  

    The Decision Tree outperformed the Random Forest in this dataset.  
    This suggests that a single tree captured the patterns more effectively than an ensemble of multiple trees.
    """)

# =====================================
# TAB 3: Insights & Visualizations
# =====================================
elif tab == "📈 Insights & Visualizations":
    st.header("📈 Data Insights & Visualizations")

    # Status distribution
    if "status" in df.columns:
        st.subheader("📌 Appointment Status Distribution")
        status_counts = df["status"].value_counts()
        fig, ax = plt.subplots()
        sns.barplot(x=status_counts.index, y=status_counts.values, ax=ax)
        ax.set_ylabel("Count")
        ax.set_xlabel("Appointment Status")
        st.pyplot(fig)

    # Specialization distribution
    if "specialization" in df.columns:
        st.subheader("🩺 Specialization Distribution")
        spec_counts = df["specialization"].value_counts()
        fig, ax = plt.subplots()
        sns.barplot(x=spec_counts.index, y=spec_counts.values, ax=ax)
        ax.set_ylabel("Count")
        ax.set_xlabel("Doctor Specialization")
        st.pyplot(fig)

# =====================================
# Conclusion
# =====================================
st.markdown("---")
st.markdown("""
### ✅ Conclusion
- This dashboard provided an overview of hospital appointment data, key visualizations, and machine learning results.  
- **Decision Tree (69% accuracy)** performed better than **Random Forest (62% accuracy)**.  
- Label encoding was applied to categorical features for model training, but values were decoded back for better readability in this dashboard.  

""")


