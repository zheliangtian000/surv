import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="复发风险预测", layout="centered")
st.title("个体化复发风险预测工具")
st.write("**输入患者特征，预测1/2/3年复发概率（随机生存森林模型）**")

# 加载模型
@st.cache_resource
def load_model():
    return joblib.load("outputs/models/RandomForest.joblib")
model = load_model()

# 定义特征顺序，必须和训练时一致
feature_names = ['tsize', 'tgrade', 'pnodes', 'progrec']

# 默认值（你的示例）
default_data = {'tsize': 52, 'tgrade': 2, 'pnodes': 1, 'progrec': 0}

# 构建表单
with st.form("input_form"):
    tsize = st.number_input("肿瘤大小 tsize", min_value=0.0, value=float(default_data['tsize']))
    tgrade = st.selectbox("分级 tgrade", options=[1, 2, 3], index=default_data['tgrade']-1)
    pnodes = st.number_input("阳性淋巴结数 pnodes", min_value=0.0, value=float(default_data['pnodes']))
    progrec = st.number_input("progrec", min_value=0.0, value=float(default_data['progrec']))
    submit = st.form_submit_button("点击预测")

if submit:
    # 组装特征表
    X_input = pd.DataFrame([[tsize, tgrade, pnodes, progrec]], columns=feature_names)
    years = [1, 2, 3]
    if hasattr(model, "predict_survival_function"):
        surv_fn = model.predict_survival_function(X_input)[0]
        surv_probs = [surv_fn(t) for t in years]
        risks = [1 - p for p in surv_probs]
        # 柱状图
        fig, ax = plt.subplots()
        ax.bar([f"{y}年" for y in years], risks, color='tomato')
        ax.set_ylabel("复发风险概率")
        ax.set_ylim(0, 1)
        ax.set_title("1/2/3年复发风险")
        for i, r in enumerate(risks):
            ax.text(i, r + 0.03, f"{r:.1%}", ha="center", fontsize=13)
        st.pyplot(fig)
        # 展示文字
        st.info("复发风险概率：")
        st.write({f"{y}年": f"{r:.1%}" for y, r in zip(years, risks)})
    else:
        st.error("模型不支持生存概率预测，请检查模型或训练方法。")
