import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, roc_auc_score, roc_curve, confusion_matrix,
                             precision_score, recall_score, f1_score)
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
from datetime import datetime
import io
import base64

# 尝试导入shap，如果没有就跳过
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available")

# 尝试导入lifelines，如果没有就跳过
try:
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False
    print("Lifelines not available")

warnings.filterwarnings('ignore')

# 页面配置
st.set_page_config(
    page_title="MINIC3智能预测系统",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        background: linear-gradient(45deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
        margin-bottom: 1rem;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .info-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .risk-low { color: #27ae60; font-weight: bold; font-size: 1.2rem; }
    .risk-medium { color: #f39c12; font-weight: bold; font-size: 1.2rem; }
    .risk-high { color: #e74c3c; font-weight: bold; font-size: 1.2rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🧠 MINIC3抗CTLA-4抗体智能预测系统</div>', unsafe_allow_html=True)
st.markdown("### 基于多模态机器学习的疗效与安全性双任务预测平台")

# ==================== 生成增强版模拟数据 ====================
@st.cache_data
def generate_clinical_data():
    """生成模拟临床数据"""
    np.random.seed(42)
    n_patients = 500  # 减少样本量避免超时
    
    # 基础特征
    data = {
        '患者ID': [f'P{str(i).zfill(4)}' for i in range(1, n_patients + 1)],
        '剂量水平(mg/kg)': np.random.choice([0.3, 1.0, 3.0, 10.0], n_patients, p=[0.15, 0.25, 0.35, 0.25]),
        '年龄': np.random.normal(60, 12, n_patients).astype(int).clip(25, 85),
        '性别': np.random.choice(['男', '女'], n_patients, p=[0.52, 0.48]),
        '体重(kg)': np.random.normal(70, 15, n_patients).astype(int).clip(40, 120),
        'BMI': np.random.normal(24, 4, n_patients).round(1),
    }
    
    # 肿瘤相关特征
    tumor_data = {
        '基线肿瘤大小(mm)': np.random.exponential(30, n_patients).round(1).clip(5, 150),
        'ECOG评分': np.random.choice([0, 1, 2, 3], n_patients, p=[0.25, 0.45, 0.25, 0.05]),
        '既往治疗线数': np.random.choice([0, 1, 2, 3, 4], n_patients, p=[0.1, 0.3, 0.3, 0.2, 0.1]),
        '肿瘤类型': np.random.choice(['非小细胞肺癌', '黑色素瘤', '肾细胞癌', '尿路上皮癌', '头颈鳞癌'], n_patients),
        '转移部位数': np.random.poisson(2, n_patients).clip(0, 5),
        '肝转移': np.random.choice([0, 1], n_patients, p=[0.7, 0.3]),
        '脑转移': np.random.choice([0, 1], n_patients, p=[0.85, 0.15]),
    }
    
    # 生物标志物
    biomarker_data = {
        'PD-L1表达': np.random.choice(['阴性(<1%)', '低表达(1-49%)', '高表达(≥50%)'], n_patients, p=[0.3, 0.4, 0.3]),
        'TMB(mut/Mb)': np.random.exponential(8, n_patients).round(1).clip(0, 50),
        'MSI状态': np.random.choice(['MSS', 'MSI-L', 'MSI-H'], n_patients, p=[0.8, 0.15, 0.05]),
        '中性粒细胞计数': np.random.normal(4.5, 2, n_patients).round(2).clip(1, 15),
        '淋巴细胞计数': np.random.normal(2.0, 0.8, n_patients).round(2).clip(0.3, 5),
        '血小板计数': np.random.normal(250, 80, n_patients).round(0).clip(100, 500),
        '白蛋白(g/L)': np.random.normal(38, 5, n_patients).round(1).clip(25, 50),
        'LDH(U/L)': np.random.normal(200, 80, n_patients).round(0).clip(100, 600),
        'CRP(mg/L)': np.random.exponential(15, n_patients).round(1).clip(1, 150),
    }
    
    df = pd.DataFrame({**data, **tumor_data, **biomarker_data})
    
    # 计算衍生指标
    df['NLR'] = (df['中性粒细胞计数'] / df['淋巴细胞计数']).round(2)
    df['PLR'] = (df['血小板计数'] / df['淋巴细胞计数']).round(2)
    df['LIPI评分'] = np.where(
        (df['LDH(U/L)'] > 250) & (df['NLR'] > 3), '高风险',
        np.where((df['LDH(U/L)'] > 250) | (df['NLR'] > 3), '中风险', '低风险')
    )
    
    # 复杂的疗效生成逻辑
    def calculate_response_prob(row):
        base_prob = 0.25
        dose_effect = {0.3: -0.1, 1.0: 0, 3.0: 0.15, 10.0: 0.25}
        pdl1_effect = {'阴性(<1%)': -0.1, '低表达(1-49%)': 0.05, '高表达(≥50%)': 0.2}
        ecog_effect = -0.15 * row['ECOG评分']
        metastasis_effect = -0.05 * row['转移部位数']
        
        prob = (base_prob + dose_effect[row['剂量水平(mg/kg)']] + 
                pdl1_effect[row['PD-L1表达']] + ecog_effect + metastasis_effect)
        return np.clip(prob, 0.05, 0.85)
    
    def calculate_ae_prob(row):
        base_prob = 0.35
        dose_ae_effect = {0.3: -0.2, 1.0: -0.1, 3.0: 0.1, 10.0: 0.25}
        age_effect = 0.01 * (row['年龄'] - 60) if row['年龄'] > 60 else 0
        prob = base_prob + dose_ae_effect[row['剂量水平(mg/kg)']] + age_effect
        return np.clip(prob, 0.1, 0.9)
    
    # 生成结果
    response_probs = df.apply(calculate_response_prob, axis=1)
    ae_probs = df.apply(calculate_ae_prob, axis=1)
    
    df['疗效概率'] = response_probs.round(3)
    df['AE概率'] = ae_probs.round(3)
    df['是否缓解'] = np.random.binomial(1, response_probs)
    df['是否发生AE'] = np.random.binomial(1, ae_probs)
    
    # 生成PFS时间
    df['PFS_月'] = np.where(
        df['是否缓解'] == 1,
        np.random.normal(15, 5, len(df)),
        np.random.normal(5, 2, len(df))
    ).clip(1, 36).round(1)
    
    df['OS_月'] = df['PFS_月'] + np.random.exponential(8, len(df)).round(1)
    df['OS_月'] = df['OS_月'].clip(1, 48).round(1)
    df['事件'] = np.random.binomial(1, 0.8, len(df))
    
    # 风险分层
    df['风险评分'] = (df['ECOG评分'] * 2 + (df['LDH(U/L)'] > 250).astype(int) * 3 + 
                     (df['转移部位数'] > 2).astype(int) * 2 + (df['NLR'] > 4).astype(int) * 2)
    df['风险分层'] = pd.cut(df['风险评分'], bins=[0, 3, 6, 10], labels=['低风险', '中风险', '高风险'])
    
    # 肿瘤缓解状态文本
    df['肿瘤缓解状态'] = np.where(
        df['是否缓解'] == 1,
        np.random.choice(['完全缓解(CR)', '部分缓解(PR)'], len(df), p=[0.2, 0.8]),
        np.random.choice(['疾病稳定(SD)', '疾病进展(PD)'], len(df), p=[0.4, 0.6])
    )
    
    return df

# ==================== 机器学习模型 ====================
class AdvancedPredictiveModel:
    def __init__(self):
        self.model_ae = None
        self.model_response = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.importance_df = None
        self.roc_data = None
        self.cv_scores = None
        self.metrics = None
        
    def prepare_features(self, df, fit_scaler=False):
        """准备特征"""
        feature_df = df.copy()
        
        # 编码分类变量
        feature_df['性别编码'] = feature_df['性别'].map({'男': 0, '女': 1})
        feature_df['PD-L1编码'] = feature_df['PD-L1表达'].map({
            '阴性(<1%)': 0, '低表达(1-49%)': 1, '高表达(≥50%)': 2
        })
        feature_df['MSI编码'] = feature_df['MSI状态'].map({'MSS': 0, 'MSI-L': 1, 'MSI-H': 2})
        feature_df['LIPI编码'] = feature_df['LIPI评分'].map({'低风险': 0, '中风险': 1, '高风险': 2})
        
        # 选择特征
        self.feature_columns = [
            '剂量水平(mg/kg)', '年龄', '性别编码', 'BMI', 'ECOG评分',
            '既往治疗线数', '转移部位数', '肝转移', '脑转移',
            'PD-L1编码', 'TMB(mut/Mb)', 'MSI编码', 'NLR', 'PLR',
            '白蛋白(g/L)', 'LDH(U/L)', 'CRP(mg/L)', 'LIPI编码'
        ]
        
        X = feature_df[self.feature_columns].fillna(0)
        
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
            
        return pd.DataFrame(X_scaled, columns=self.feature_columns)
    
    def train(self, df):
        """训练模型"""
        with st.spinner('正在训练集成模型...'):
            X = self.prepare_features(df, fit_scaler=True)
            y_response = df['是否缓解']
            y_ae = df['是否发生AE']
            
            # 划分训练集和测试集
            X_train, X_test, y_response_train, y_response_test = train_test_split(
                X, y_response, test_size=0.2, random_state=42, stratify=y_response
            )
            _, _, y_ae_train, y_ae_test = train_test_split(
                X, y_ae, test_size=0.2, random_state=42, stratify=y_ae
            )
            
            # 训练模型
            self.model_response = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            self.model_response.fit(X_train, y_response_train)
            
            self.model_ae = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            self.model_ae.fit(X_train, y_ae_train)
            
            # 计算交叉验证得分
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            self.cv_scores = {
                'response': cross_val_score(self.model_response, X, y_response, cv=cv, scoring='roc_auc'),
                'ae': cross_val_score(self.model_ae, X, y_ae, cv=cv, scoring='roc_auc')
            }
            
            # 预测概率
            y_response_prob = self.model_response.predict_proba(X_test)[:, 1]
            y_ae_prob = self.model_ae.predict_proba(X_test)[:, 1]
            
            # ROC数据
            fpr_res, tpr_res, _ = roc_curve(y_response_test, y_response_prob)
            fpr_ae, tpr_ae, _ = roc_curve(y_ae_test, y_ae_prob)
            
            self.roc_data = {
                'response': {
                    'fpr': fpr_res, 'tpr': tpr_res,
                    'auc': roc_auc_score(y_response_test, y_response_prob)
                },
                'ae': {
                    'fpr': fpr_ae, 'tpr': tpr_ae,
                    'auc': roc_auc_score(y_ae_test, y_ae_prob)
                }
            }
            
            # 特征重要性
            self.importance_df = pd.DataFrame({
                '特征': self.feature_columns,
                '重要性': self.model_response.feature_importances_
            }).sort_values('重要性', ascending=False)
            
            # 计算各种指标
            y_response_pred = self.model_response.predict(X_test)
            y_ae_pred = self.model_ae.predict(X_test)
            
            self.metrics = {
                'response': {
                    'accuracy': accuracy_score(y_response_test, y_response_pred),
                    'precision': precision_score(y_response_test, y_response_pred),
                    'recall': recall_score(y_response_test, y_response_pred),
                    'f1': f1_score(y_response_test, y_response_pred),
                    'auc': self.roc_data['response']['auc']
                },
                'ae': {
                    'accuracy': accuracy_score(y_ae_test, y_ae_pred),
                    'precision': precision_score(y_ae_test, y_ae_pred),
                    'recall': recall_score(y_ae_test, y_ae_pred),
                    'f1': f1_score(y_ae_test, y_ae_pred),
                    'auc': self.roc_data['ae']['auc']
                }
            }
            
            return self.metrics
    
    def predict_patient(self, patient_features):
        """预测单个患者"""
        features_scaled = self.scaler.transform(patient_features)
        response_prob = self.model_response.predict_proba(features_scaled)[0][1]
        ae_prob = self.model_ae.predict_proba(features_scaled)[0][1]
        
        return {
            'response_prob': response_prob,
            'ae_prob': ae_prob
        }

# ==================== 初始化 ====================
if 'model' not in st.session_state:
    st.session_state.model = AdvancedPredictiveModel()
    with st.spinner('正在生成临床数据并训练模型...'):
        df = generate_clinical_data()
        st.session_state.df = df
        metrics = st.session_state.model.train(df)
        st.session_state.metrics = metrics

df = st.session_state.df

# ==================== 侧边栏 ====================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=80)
    st.title("📌 导航菜单")
    
    page = st.radio(
        "",
        ["🏥 临床数据总览", "🎯 智能预测系统", "📊 模型性能分析", "📈 生存分析"]
    )
    
    st.markdown("---")
    st.markdown("### 系统状态")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("患者总数", f"{len(df):,}")
    with col2:
        st.metric("特征维度", len(st.session_state.model.feature_columns))
    
    st.markdown("---")
    st.caption(f"© 2024 MINIC3预测系统 v3.0")
    st.caption(f"最后更新: {datetime.now().strftime('%Y-%m-%d')}")

# ==================== 页面1：临床数据总览 ====================
if page == "🏥 临床数据总览":
    st.markdown('<div class="sub-header">📊 临床数据总览</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("总体有效率", f"{df['是否缓解'].mean()*100:.1f}%")
    with col2:
        st.metric("不良事件率", f"{df['是否发生AE'].mean()*100:.1f}%")
    with col3:
        st.metric("中位PFS", f"{df['PFS_月'].median():.1f} 月")
    with col4:
        st.metric("中位OS", f"{df['OS_月'].median():.1f} 月")
    
    tab1, tab2 = st.tabs(["数据预览", "患者分布"])
    
    with tab1:
        st.dataframe(df.head(20), use_container_width=True)
        
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(df, names='肿瘤类型', title='肿瘤类型分布', hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.histogram(df, x='年龄', color='性别', nbins=30, title='年龄分布')
            st.plotly_chart(fig, use_container_width=True)

# ==================== 页面2：智能预测系统 ====================
elif page == "🎯 智能预测系统":
    st.markdown('<div class="sub-header">🎯 智能预测系统</div>', unsafe_allow_html=True)
    
    with st.expander("📝 输入患者信息", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            dose = st.selectbox("剂量水平 (mg/kg)", [0.3, 1.0, 3.0, 10.0])
            age = st.slider("年龄", 25, 85, 60)
            gender = st.selectbox("性别", ["男", "女"])
            ecog = st.selectbox("ECOG评分", [0, 1, 2, 3])
            
        with col2:
            pdl1 = st.selectbox("PD-L1表达", ["阴性(<1%)", "低表达(1-49%)", "高表达(≥50%)"])
            tmb = st.number_input("TMB (mut/Mb)", 0, 50, 8)
            nlr = st.number_input("NLR", 0.5, 15.0, 2.5, 0.1)
            ldh = st.number_input("LDH (U/L)", 100, 600, 200)
    
    if st.button("🔮 开始预测", type="primary", use_container_width=True):
        input_data = pd.DataFrame([{
            '剂量水平(mg/kg)': dose, '年龄': age, '性别': gender,
            'BMI': 24, 'ECOG评分': ecog, '既往治疗线数': 1,
            '转移部位数': 1, '肝转移': 0, '脑转移': 0,
            'PD-L1表达': pdl1, 'TMB(mut/Mb)': tmb, 'MSI状态': 'MSS',
            'NLR': nlr, 'PLR': 150, '白蛋白(g/L)': 38,
            'LDH(U/L)': ldh, 'CRP(mg/L)': 10, 'LIPI评分': '中风险'
        }])
        
        input_data['性别编码'] = input_data['性别'].map({'男': 0, '女': 1})
        input_data['PD-L1编码'] = input_data['PD-L1表达'].map({'阴性(<1%)':0, '低表达(1-49%)':1, '高表达(≥50%)':2})
        input_data['MSI编码'] = 0
        input_data['LIPI编码'] = 1
        
        features = st.session_state.model.prepare_features(input_data)
        predictions = st.session_state.model.predict_patient(features)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("治疗有效概率", f"{predictions['response_prob']*100:.1f}%")
            if predictions['response_prob'] > 0.5:
                st.success("✅ 高概率有效")
            elif predictions['response_prob'] > 0.3:
                st.warning("⚠️ 中等概率有效")
            else:
                st.error("❌ 低概率有效")
        
        with col2:
            st.metric("不良事件风险", f"{predictions['ae_prob']*100:.1f}%")
            if predictions['ae_prob'] < 0.3:
                st.success("✅ 低风险")
            elif predictions['ae_prob'] < 0.6:
                st.warning("⚠️ 中等风险")
            else:
                st.error("❌ 高风险")

# ==================== 页面3：模型性能分析 ====================
elif page == "📊 模型性能分析":
    st.markdown('<div class="sub-header">📊 模型性能分析</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["ROC曲线", "特征重要性", "性能指标"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=st.session_state.model.roc_data['response']['fpr'],
                y=st.session_state.model.roc_data['response']['tpr'],
                mode='lines',
                name=f"疗效预测 (AUC={st.session_state.model.roc_data['response']['auc']:.3f})",
                line=dict(color='blue', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='随机猜测',
                line=dict(color='gray', dash='dash')
            ))
            fig.update_layout(title="疗效预测ROC曲线")
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=st.session_state.model.roc_data['ae']['fpr'],
                y=st.session_state.model.roc_data['ae']['tpr'],
                mode='lines',
                name=f"AE预测 (AUC={st.session_state.model.roc_data['ae']['auc']:.3f})",
                line=dict(color='red', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='随机猜测',
                line=dict(color='gray', dash='dash')
            ))
            fig.update_layout(title="不良事件预测ROC曲线")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = px.bar(
            st.session_state.model.importance_df.head(10),
            x='重要性', y='特征',
            orientation='h',
            title='特征重要性排名',
            color='重要性'
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 疗效预测模型")
            metrics_df = pd.DataFrame([
                ['准确率', f"{st.session_state.metrics['response']['accuracy']:.3f}"],
                ['精确率', f"{st.session_state.metrics['response']['precision']:.3f}"],
                ['召回率', f"{st.session_state.metrics['response']['recall']:.3f}"],
                ['F1分数', f"{st.session_state.metrics['response']['f1']:.3f}"],
                ['AUC', f"{st.session_state.metrics['response']['auc']:.3f}"]
            ], columns=['指标', '数值'])
            st.dataframe(metrics_df, use_container_width=True)
            
        with col2:
            st.markdown("#### 不良事件预测模型")
            metrics_df = pd.DataFrame([
                ['准确率', f"{st.session_state.metrics['ae']['accuracy']:.3f}"],
                ['精确率', f"{st.session_state.metrics['ae']['precision']:.3f}"],
                ['召回率', f"{st.session_state.metrics['ae']['recall']:.3f}"],
                ['F1分数', f"{st.session_state.metrics['ae']['f1']:.3f}"],
                ['AUC', f"{st.session_state.metrics['ae']['auc']:.3f}"]
            ], columns=['指标', '数值'])
            st.dataframe(metrics_df, use_container_width=True)

# ==================== 页面4：生存分析 ====================
elif page == "📈 生存分析":
    st.markdown('<div class="sub-header">📈 生存分析</div>', unsafe_allow_html=True)
    
    if LIFELINES_AVAILABLE:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for dose in sorted(df['剂量水平(mg/kg)'].unique()):
            dose_data = df[df['剂量水平(mg/kg)'] == dose]
            kmf = KaplanMeierFitter()
            kmf.fit(dose_data['PFS_月'], event_observed=dose_data['事件'], 
                   label=f'{dose} mg/kg')
            kmf.plot_survival_function(ax=ax, ci_show=True)
        
        ax.set_xlabel('时间 (月)', fontsize=12)
        ax.set_ylabel('生存率', fontsize=12)
        ax.set_title('各剂量组Kaplan-Meier生存曲线', fontsize=14)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)
        
        st.subheader("中位生存时间")
        for dose in sorted(df['剂量水平(mg/kg)'].unique()):
            dose_data = df[df['剂量水平(mg/kg)'] == dose]
            median = dose_data['PFS_月'].median()
            st.metric(f"{dose} mg/kg", f"{median:.1f} 月")
    else:
        st.info("生存分析需要安装lifelines包")
