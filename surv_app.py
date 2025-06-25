import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Recurrence Risk Prediction", layout="centered")
st.title("Personalized Recurrence Risk Prediction Tool")
st.write("**Input patient features to predict 1/2/3-year recurrence probability (Random Survival Forest Model)**")

# 加载模型
@st.cache_resource
def load_model():
    return joblib.load("RandomForest.joblib")
model = load_model()

# 定义特征顺序，必须和训练时一致
feature_names = ['tsize', 'tgrade', 'pnodes', 'progrec']

# 默认值（你的示例）
default_data = {'tsize': 52, 'tgrade': 2, 'pnodes': 1, 'progrec': 0}

# 构建表单
with st.form("input_form"):
    tsize = st.number_input("Tumor size (tsize)", min_value=0.0, value=float(default_data['tsize']))
    tgrade = st.selectbox("Tumor grade (tgrade)", options=[1, 2, 3], index=default_data['tgrade']-1)
    pnodes = st.number_input("Positive lymph nodes (pnodes)", min_value=0.0, value=float(default_data['pnodes']))
    progrec = st.number_input("progrec", min_value=0.0, value=float(default_data['progrec']))
    submit = st.form_submit_button("Predict")

if submit:
    X_input = pd.DataFrame([[tsize, tgrade, pnodes, progrec]], columns=feature_names)
    years = [1, 2, 3]
    if hasattr(model, "predict_survival_function"):
        surv_fn = model.predict_survival_function(X_input)[0]
        surv_probs = [surv_fn(t) for t in years]
        risks = [1 - p for p in surv_probs]
        # Bar chart
        fig, ax = plt.subplots()
        ax.bar([f"{y} year" for y in years], risks, color='tomato')
        ax.set_ylabel("Recurrence Risk Probability")
        ax.set_ylim(0, 1)
        ax.set_title("1/2/3-Year Recurrence Risk")
        for i, r in enumerate(risks):
            ax.text(i, r + 0.03, f"{r:.1%}", ha="center", fontsize=13)
        st.pyplot(fig)
        # Info string as you requested
        info_str = ""
        for y, r in zip(years, risks):
            info_str += f"Recurrence risk probabilities {y} year: {r:.1%}\n"
        st.info(info_str)
    else:
        st.error("The model does not support survival probability prediction. Please check your model or training method.")



