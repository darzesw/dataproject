import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm

# 1. 페이지 설정
st.set_page_config(page_title="기온 및 일교차 분석 대시보드", layout="wide")

st.title("🌡️ 기온 변화 및 일교차 상관관계 심층 분석 웹앱")
st.markdown("""
이 앱은 장기 기상 데이터를 바탕으로 **시간에 따른 일교차의 추이**와 
**평균기온과 일교차 간의 기상학적 상관관계**를 논리적·수식적으로 분석합니다.
""")

# 2. 데이터 로드 및 전처리 함수
@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='cp949')
    
    # 컬럼명 공백 제거
    df.columns = df.columns.str.strip()
    
    if '날짜' in df.columns:
        # 데이터 내부의 탭(\t), 쌍따옴표, 공백 제거 후 날짜 변환
        df['날짜'] = df['날짜'].astype(str).str.replace(r'[\t"\s]', '', regex=True)
        df['날짜'] = pd.to_datetime(df['날짜'], errors='coerce')
    
    required_cols = ['평균기온(℃)', '최저기온(℃)', '최고기온(℃)', '날짜']
    df = df.dropna(subset=[col for col in required_cols if col in df.columns])
    
    if not df.empty:
        # 파생 변수 생성
        df['일교차'] = df['최고기온(℃)'] - df['최저기온(℃)']
        df['연도'] = df['날짜'].dt.year
        df['월'] = df['날짜'].dt.month
        
        # 계절 분류 정의
        def get_season(month):
            if month in [3, 4, 5]: return '봄'
            elif month in [6, 7, 8]: return '여름'
            elif month in [9, 10, 11]: return '가을'
            else: return '겨울'
        df['계절'] = df['월'].apply(get_season)
        
    return df

# 데이터 파일 지정 (업로드된 파일명)
DATA_PATH = "ta_20260601093156.csv"

try:
    df = load_data(DATA_PATH)
    
    if df.empty:
        st.error("데이터를 불러왔으나 내용이 비어있습니다. 컬럼명을 확인해주세요.")
    else:
        st.sidebar.success("📊 데이터 로드 성공!")
        
        # 사이드바 필터
        st.sidebar.header("🔍 분석 조건 설정")
        min_year, max_year = int(df['연도'].min()), int(df['연도'].max())
        year_range = st.sidebar.slider("분석 연도 범위", min_year, max_year, (min_year, max_year))
        
        # 데이터 필터링
        f_df = df[(df['연도'] >= year_range[0]) & (df['연도'] <= year_range[1])]
        
        # 메인 탭 구성
        tab1, tab2, tab3 = st.tabs(["⏳ 시간에 따른 일교차 추이", "🔗 평균기온과의 상관관계 분석", "🧮 통계적 수식 요약"])
        
        # -------------------------------------------------------------
        # TAB 1: 시간에 따른 일교차 추이
        # -------------------------------------------------------------
        with tab1:
            st.header("⏳ 연도별 평균 일교차의 장기적 변화 경향")
            
            yearly_df = f_df.groupby('연도').agg({'일교차': 'mean', '평균기온(℃)': 'mean'}).reset_index()
            
            # 선형 회귀 (시간 vs 일교차)
            X1 = sm.add_constant(yearly_df['연도'])
            model1 = sm.OLS(yearly_df['일교차'], X1).fit()
            slope1 = model1.params['연도']
            
            st.metric(label="연간 일교차 변화율 (기울기 β₁)", value=f"{slope1:.4f} ℃/년")
            
            fig1 = px.scatter(yearly_df, x='연도', y='일교차', trendline="ols",
                              title="연도별 평균 일교차 장기 추세 (OLS 회귀선)",
                              labels={'일교차': '평균 일교차 (℃)', '연도': '연도'},
                              template="plotly_white")
            st.plotly_chart(fig1, use_container_width=True)
            
            st.markdown(f"""
            **📝 추이 논리 해석:**
            * 현재 선택된 기간 동안 일교차는 매년 평균 **{slope1:.4f}℃**씩 변화하고 있습니다.
            * 기상학적으로 기온 상승(온난화) 속도 환경에서 온실효과를 유발하는 수증기량이나 도시 열섬 현상이 심해지면 야간 최저기온이 크게 올라 일교차가 감소(음의 기울기)하는 특성을 보입니다.
            """)

        # -------------------------------------------------------------
        # TAB 2: 평균기온과의 상관관계 분석 (핵심 추가/보완)
        # -------------------------------------------------------------
        with tab2:
            st.header("🔗 평균기온과 일교차의 상관관계")
            
            # 전체 피어슨 상관계수
            overall_corr = f_df['평균기온(℃)'].corr(f_df['일교차'])
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.subheader("📊 통계적 지표")
                st.metric(label="전체 피어슨 상관계수 (r)", value=f"{overall_corr:.3f}")
                
                st.markdown("""
                * **r > 0**: 평균기온이 높을수록 일교차 증가 (양의 상관관계)
                * **r < 0**: 평균기온이 높을수록 일교차 감소 (음의 상관관계)
                
                **💡 계절별 상관계수 파헤치기:**
                전체 데이터를 묶으면 비선형적 특성 때문에 상관관계가 왜곡될 수 있습니다. 아래에서 계절별로 나누어 변수 간의 관계를 파악해 보세요.
                """)
                
                # 계절별 상관계수 표 출력
                seasons = ['봄', '여름', '가을', '겨울']
                sect_corr = {}
                for s in seasons:
                    s_data = f_df[f_df['계절'] == s]
                    if len(s_data) > 1:
                        sect_corr[s] = s_data['평균기온(℃)'].corr(s_data['일교차'])
                
                st.write("**[계절별 독립 상관계수 (r)]**")
                st.json({k: f"{v:.3f}" for k, v in sect_corr.items()})
                
            with col2:
                # 대량 데이터 시각화 안정성을 위한 샘플링
                sample_df = f_df.sample(n=min(5000, len(f_df)), random_state=42)
                
                # 계절별로 색상을 나누어 산점도 매핑
                fig2 = px.scatter(sample_df, x='평균기온(℃)', y='일교차', color='계절',
                                  color_discrete_map={'봄':'#2ecc71', '여름':'#e74c3c', '가을':'#f39c12', '겨울':'#3498db'},
                                  category_orders={"계절": ["봄", "여름", "가을", "겨울"]},
                                  opacity=0.5, trendline="ols",
                                  title="평균기온 vs 일교차 산점도 (계절별 회귀선 포함)",
                                  labels={'평균기온(℃)': '평균기온 (℃)', '일교차': '일교차 (℃)'},
                                  template="plotly_white")
                st.plotly_chart(fig2, use_container_width=True)
                
            st.markdown("""
            **📝 상관관계 분석 해석:**
            * 그래프를 보면 **여름철(빨간색)** 군집은 기온이 매우 높음에도 불구하고 일교차가 낮게 형성되어 있는 것을 볼 수 있습니다. 이는 여름철 높은 대기 습도와 잦은 구름이 야간 열방출을 막기 때문입니다.
            * 반면 **봄(초록색)과 가을(주황색)**은 기온이 적당하면서도 일교차가 위쪽으로 높게 분포합니다. 대기가 건조하여 낮과 밤의 기온 차이가 극명해지기 때문입니다.
            * 따라서 전체 상관계수가 음(-) 혹은 양(+)으로 한쪽으로 치우치더라도, 그것은 대기 수증기량과 계절적 주기가 개입된 결과로 해석하는 것이 논리적입니다.
            """)

        # -------------------------------------------------------------
        # TAB 3: 통계적 수식 요약
        # -------------------------------------------------------------
        with tab3:
            st.header("🧮 수학적 모델링 및 통계 요약")
            
            st.markdown("""
            ### 1. 피어슨 상관계수 공식 (Pearson Correlation Coefficient)
            두 변수 $X$(평균기온)와 $Y$(일교차)의 선형 연관성을 측정합니다.
            $$r = \\frac{\\sum (X_i - \\bar{X})(Y_i - \\bar{Y})}{\\sqrt{\\sum (X_i - \\bar{X})^2 \\sum (Y_i - \\bar{Y})^2}}$$
            
            ### 2. 일교차 장기 시계열 회귀 모델 분석 결과
            """)
            
            # OLS 요약 정보 서식 출력
            st.text(model1.summary().as_text())
            
            st.subheader("📋 필터링된 데이터의 기술 통계량")
            st.dataframe(f_df[['평균기온(℃)', '최저기온(℃)', '최고기온(℃)', '일교차']].describe())

except FileNotFoundError:
    st.error(f"데이터 파일(`{DATA_PATH}`)을 찾을 수 없습니다. 파일이 스크립트와 동일한 디렉토리에 있는지 확인하세요.")
