import streamlit as st
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor

# 1. 预定义模型信息（模型文件路径 + 显示名称）
MODELS = {
    "AdaBoost": "models/AdaBoost_model.pkl",
    "XGBoost": "models/XGBoost_model.pkl",
    "GBDT": "models/GBDT_model.pkl",
    "RF": "modelsmodels/RF_model.pkl",
    "MLP": "models/MLP_model.pkl",  # 如果是Keras模型需特殊处理
    "SVR": "models/SVR_model.pkl",
    "KNN": "models/KNN_model.pkl",
    "KRR": "models/KRR_model.pkl"
}

# 2. 加载模型的函数（带缓存）
@st.cache_resource
def load_model(model_path):
    try:
        model = joblib.load(model_path)
        if not hasattr(model, 'estimators_'):  # AdaBoost 特有检查
            raise ValueError("模型未训练！")
        return model
    except Exception as e:
        st.error(f"加载失败: {str(e)}")
        return None


# 3. 主界面
def main():
    st.title("🔮 多模型预测系统")

    # ---- 侧边栏：模型选择 ----
    st.sidebar.header("模型配置")
    selected_model_name = st.sidebar.selectbox(
        "选择预测模型",
        list(MODELS.keys()),
        help="根据任务类型选择模型"
    )

    # ---- 主区域：数据上传和预测 ----
    st.header("1. 上传数据")
    uploaded_file = st.file_uploader("上传CSV文件", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("✅ 数据加载成功！")
        st.write("数据预览：", df.head(3))

        # 显示当前选择的模型
        st.header("2. 模型信息")
        model_path = MODELS[selected_model_name]
        st.code(f"已选择模型: {selected_model_name}\n路径: {model_path}")

        # 加载模型并预测
        if st.button("🚀 运行预测", type="primary"):
            with st.spinner("模型加载中..."):
                try:
                    # 关键点：直接加载预训练模型，不执行fit！
                    model = load_model(model_path)
                    st.success("预训练模型加载完成！")

                    # 执行预测（根据模型类型适配）
                    if "clf" in model_path.lower():  # 分类模型
                        predictions = model.predict(df)
                        st.write("预测类别：", predictions)
                    else:  # 回归模型
                        predictions = model.predict(df)
                        st.write("预测数值：", predictions)

                    # 下载结果
                    result_df = pd.DataFrame({"预测结果": predictions})
                    st.download_button("📥 下载结果",
                                       result_df.to_csv(),
                                       "predictions.csv")

                except Exception as e:
                    st.error(f"预测失败: {str(e)}")


if __name__ == "__main__":
    main()