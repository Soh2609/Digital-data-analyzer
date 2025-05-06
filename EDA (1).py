
# app.py

import streamlit as st
import pandas as pd
from io import BytesIO
from data_cleaner import basic_data_cleaning, standardize_numeric_columns
import plotly.express as px

st.title("🧹Digital Data Analyser & Smart Data Cleaning App")
with st.sidebar:
    st.markdown("## 👤 About the Creator")
    st.image("https://avatars.githubusercontent.com/u/soh2609", width=100)  # Optional avatar
    st.markdown("""
    **Name**: Soham Mahadev Masute  
    **Role**: Final Year B.Tech(AI & ML) 
    **College**: GHRCEM,Pune
    
    **LinkedIn**: [💼 LinkedIn](https://www.linkedin.com/in/soham-masute-2a34b3241)
    
    **Mail**:[📂Gmail](mailto:sohammasuteofficial@gmail.com)
    """)
    st.markdown("OPEN TO WORK! AND WANT TO CONTRIBUTE MY SKILLS TO BUILD EVEN BETTER TECHNOLOGY")

uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])
if "cleaned_df" not in st.session_state:
  st.session_state.cleaned_df = None
if "encoded_df" not in st.session_state:
    st.session_state.encoded_df = None


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file,encoding='latin')
    st.write("📊 Raw Dataset Preview:")
    st.dataframe(df.head())

    st.markdown("---")

    # Column drop selection
    st.subheader("❓ Columns to Drop(optional)")
    drop_cols = st.multiselect("Select columns you want to remove", options=df.columns.tolist())

    # Missing value handling
    st.subheader("🛠 Missing Value Strategy")
    strategy = st.radio(
        "How should missing values be handled?",
        options=["drop", "mean", "median"],
        horizontal=True
    )

    # Trigger cleaning
    if st.button("🚀 Clean My Data"):
        cleaned_df = basic_data_cleaning(df, missing_strategy=strategy, columns_to_drop=drop_cols)
        if cleaned_df.empty or cleaned_df.shape[1] == 0:
            st.error("❌ Cleaning resulted in an empty dataset. Please revise your column selections.")
            st.stop()
        st.session_state.cleaned_df = cleaned_df

        st.success("✅ Data cleaned successfully!")
        st.write("🔍 Cleaned Dataset Preview:")
        st.dataframe(cleaned_df.head())

    # Download link

        @st.cache_data
        def convert_df(df):
          return df.to_csv(index=False).encode('utf-8')

        csv = convert_df(cleaned_df)
        st.download_button(
            label="📥 Download Cleaned Data",
            data=csv,
            file_name='cleaned_data.csv',
            mime='text/csv',
            key="download_cleaned"
        )
# Auto EDA section
if st.session_state.cleaned_df is not None:
    st.markdown("---")
    st.header("🔍 Automated Exploratory Data Analysis")

    df = st.session_state.cleaned_df

    st.subheader("🧾 Dataset Overview")
    st.write(f"**Shape:** {df.shape}")
    st.write(f"**Columns:** {df.columns.tolist()}")
    st.write("**Missing Values (per column):**")
    st.write(df.isna().sum())

    st.subheader("📊 Descriptive Statistics")
    st.dataframe(df.describe(include='all').T)

    # Correlation Heatmap
    import plotly.express as px
    import plotly.graph_objects as go

    numeric_df = df.select_dtypes(include=['int64', 'float64'])

    if not numeric_df.empty:
        st.subheader("📈 Interactive Correlation Heatmap")
        corr = numeric_df.corr()
        fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r")
        st.plotly_chart(fig, use_container_width=True)


    st.subheader("📉 Interactive Feature Distribution")
    selected_column = st.selectbox("Choose column for histogram", numeric_df.columns, key="hist")

    if selected_column:
        fig = px.histogram(df, x=selected_column, marginal="box", nbins=30, title=f"Distribution of {selected_column}")
        st.plotly_chart(fig, use_container_width=True)


    st.subheader("🧪 Outlier Detection (Boxplot)")
    outlier_col = st.selectbox("Select column for boxplot", numeric_df.columns, key="box")

    if outlier_col:
        fig = px.box(df, y=outlier_col, points="all", title=f"Boxplot of {outlier_col}")
        st.plotly_chart(fig, use_container_width=True)
    if st.checkbox("📏 Standardize numeric columns"):
        st.session_state.cleaned_df = standardize_numeric_columns(st.session_state.cleaned_df)
        st.success("✅ Dataset standardized successfully!")
        st.subheader("🔍 Standardized Dataset Preview")
        st.dataframe(st.session_state.cleaned_df.head())

    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    csv = convert_df(st.session_state.cleaned_df)
    st.download_button(
        label="📥 Download Cleaned Data",
        data=csv,
        file_name='final_cleaned_data.csv',
        mime='text/csv',
        key="download_standardized"
    )
    from sklearn.preprocessing import OneHotEncoder, LabelEncoder
    st.markdown("---")

    st.header("🔢 Encode Categorical Variables")
    df = st.session_state.cleaned_df.copy()

    # Detect categorical columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if len(cat_cols) == 0:
        st.info("✅ No categorical columns detected to encode.")
        st.session_state.numeric_df = df
    else:
        st.write("🧠 Categorical Columns Detected:", cat_cols)

        encoding_method = st.radio(
            "Choose Encoding Method:",
            options=["One-Hot Encoding", "Label Encoding"],
            horizontal=True
        )

        if st.button("🚀 Encode Now"):
            encoded_df = df.copy()

            if encoding_method == "One-Hot Encoding":
                encoded_df = pd.get_dummies(encoded_df, columns=cat_cols, drop_first=True)
            else:
                label_encoder = LabelEncoder()
                for col in cat_cols:
                    encoded_df[col] = label_encoder.fit_transform(encoded_df[col].astype(str))

            st.session_state.encoded_df = encoded_df
            st.success("✅ Encoding completed!")
            st.dataframe(encoded_df.head())
            csv = encoded_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Encoded Data", data=csv, file_name="encoded_data.csv", mime="text/csv",key='encoded_download')
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC

    # Only proceed if encoded_df exists
    if "encoded_df" in st.session_state and st.session_state.encoded_df is not None:
        st.markdown("---")
        st.header("🤖 Machine Learning Model Trainer(Optional)")
        st.subheader("🎯 Results might vary greatly in comparison with custom trained model")

        df = st.session_state.encoded_df.copy()

        # Select target column
        target_column = st.selectbox("🎯 Select Target Column", df.columns)

        # Validate that target column is not one-hot encoded
        if df[target_column].nunique() <= 1:
            st.warning("⚠️ Target column appears to be one-hot encoded. Please choose an appropriate target.")
        else:
            # Train/Test Split
            split_ratio = st.slider("📊 Train/Test Split Ratio", 0.1, 0.9, 0.8, step=0.05)

            # Model selection
            classifier_name = st.selectbox("🧠 Choose Classifier", [
                "Logistic Regression", "Random Forest", "K-Nearest Neighbors", "Support Vector Machine"
            ])

            if st.button("🚀 Train Model"):
                X = df.drop(columns=[target_column])
                y = df[target_column]

                # Ensure target column is label encoded if it's categorical
                if y.dtype == "object" or y.nunique() < len(y):
                    le = LabelEncoder()
                    y = le.fit_transform(y)

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1 - split_ratio), random_state=42)

                # Select classifier
                if classifier_name == "Logistic Regression":
                    model = LogisticRegression(max_iter=1000)
                elif classifier_name == "Random Forest":
                    model = RandomForestClassifier()
                elif classifier_name == "K-Nearest Neighbors":
                    model = KNeighborsClassifier()
                elif classifier_name == "Support Vector Machine":
                    model = SVC()

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Show metrics
                st.subheader("📈 Evaluation Metrics(might vary greatly,work is in progress)")
                st.write(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.2f}")
                st.write(f"📊 Precision: {precision_score(y_test, y_pred, average='weighted'):.2f}")
                st.write(f"📊 Recall: {recall_score(y_test, y_pred, average='weighted'):.2f}")
                st.write(f"📊 F1 Score: {f1_score(y_test, y_pred, average='weighted'):.2f}")

