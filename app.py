import streamlit as st
import pandas as pd
import joblib

model = joblib.load("extra_tree_credit_model.pkl")
encoders = {col : joblib.load(f"{col}_encoder.pkl") for col in ["Sex","Housing","Saving accounts","Checking account"]}

st.set_page_config(page_title="Credit Risk Prediction", layout="wide")
st.title("💳Credit Risk Prediction App")
st.info("Fill in applicant details from the sidebar and click Predict.")
st.markdown("""This application predicts whether a customer is a **Good Risk** or **Bad Risk**
based on financial and personal information.""")
st.markdown("---")

st.sidebar.title("📋 Applicant Details")

st.sidebar.subheader("👤 Personal Information")
age = st.sidebar.number_input("Age",min_value = 18, max_value= 80, value= 30)
sex = st.sidebar.selectbox("Sex", ["male","female"])
job = st.sidebar.number_input("Job (0-3)",min_value = 0, max_value= 3, value= 1)
housing = st.sidebar.selectbox("Housing", ["own","rent","free"])

st.sidebar.subheader("💰 Financial Information")
saving_accounts = st.sidebar.selectbox("Saving Accounts", ["little","moderate","rich","quite rich"])
checking_account = st.sidebar.selectbox("Checking Accounts", ["little","moderate","rich"])
credit_amount = st.sidebar.number_input("Credit Amount",min_value = 0, value= 100)
duration = st.sidebar.number_input("Duration(months)",min_value = 1, value= 12)

input_credit_data = pd.DataFrame({
    "Age" : [age],
    "Sex" : [encoders["Sex"].transform([sex])[0]],
    "Job" : [job],
    "Housing" : [encoders["Housing"].transform([housing])[0]],
    "Saving accounts" : [encoders["Saving accounts"].transform([saving_accounts])[0]],
    "Checking account" : [encoders["Checking account"].transform([checking_account])[0]],
    "Credit amount" : [credit_amount],
    "Duration" : [duration]
})

st.write("")
with st.container():
    st.subheader("📊 Prediction Results")
    if st.button("🔍 Predict Credit Risk"):
        pred = model.predict(input_credit_data)[0]
        probability = model.predict_proba(input_credit_data)
        confidence = max(probability[0]) * 100

        tab1, tab2, tab3 = st.tabs([
                     "📊 Prediction",
                     "📋 Applicant Details",
                     "📈 Model Performance"
        ])
        
        with tab1:
            col1, col2 = st.columns(2)
        
            with col1:
                if pred == 1:
                    st.success("✅ GOOD Credit Risk")
                else:
                    st.error("⚠️ BAD Credit Risk")
        
            with col2:
                st.metric(label= "Prediction Confidence", value= f"{confidence:.2f}%")
                
            st.write("### Risk Confidence Level")
            st.progress(int(confidence))
            st.markdown("---")
 
            st.write("### 🧠 Risk Analysis")
            
            if credit_amount > 5000:
                st.warning("💰 High credit amount may increase repayment risk.")
            
            if duration > 24:
                st.warning("⏳ Longer loan duration may increase financial risk.")
            
            if saving_accounts == "little":
                st.warning("🏦 Low savings account balance may indicate financial instability.")
            
            if checking_account == "little":
                st.warning("📉 Low checking account balance may affect credit reliability.")
            
            if age < 25:
                st.info("👤 Younger applicants may have limited credit history.")
                
        with tab2:
            with st.expander("View Applicant Summary"):
        
                summary_col1, summary_col2 = st.columns(2)
            
                with summary_col1:
                    st.markdown(f"""
                    ### 👤 Personal Details
                    
                    - **Age:** {age}
                    - **Sex:** {sex}
                    - **Housing:** {housing}
                    """)
        
                with summary_col2:
                    st.markdown(f"""
                    ### 💰 Financial Details
                    
                    - **Credit Amount:** {credit_amount}
                    - **Duration:** {duration} months
                    """)
        with tab3:
        
            st.write("### 📈 Model Comparison")
        
            metric_col1, metric_col2 = st.columns(2)
        
            with metric_col1:
        
                st.metric(
                    "Decision Tree Accuracy",
                    "67.5%"
                )
        
                st.metric(
                    "Random Forest Accuracy",
                    "73.5%"
                )
        
            with metric_col2:
        
                st.metric(
                    "Extra Trees Accuracy",
                    "71.5%"
                )
        
                st.metric(
                    "XGBoost Accuracy",
                    "73.5%"
                )
        
            st.markdown("---")
        
            st.success(
                "Best Performing Models: Random Forest & XGBoost"
            )
            
st.markdown("---")
st.markdown("Developed using Streamlit • Machine Learning • Scikit-learn")