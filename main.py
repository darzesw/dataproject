import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose

# 1. 페이지 설정
st.set_page_config(page_title="기온 및 일교차 주기 분석", layout="wide")

st.title("🌡️ 기온 변화, 일교차 및 최고/최저 주기 심층 분석 웹앱")
st.markdown("""
이 애플리케이션은 장기 기상 데이터를 바탕으로 **일교차의 변화**, **평균기온과의 상관관계**, 
그리고 **최고/최저기온의 장기 주기(시계열 분해)**를 논리적·수식적으로 분석합니다.
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
        
        # 메인 탭 구성 (주기 분석 탭 추가)
        tab1, tab2, tab3, tab4 = st.tabs([
            "⏳ 시간에 따른 일교차 추이", 
            "🔗 평균기온과의 상관관계 분석", 
            "🕒 최고/최저기온 주기 분석", 
            "🧮 통계적 수식 요약"
        ])
        
        # -------------------------------------------------------------
        # TAB 1: 시간에 따른 일교차 추이
        # -------------------------------------------------------------
        with tab1:
            st.header("⏳ 연도별 평균 일교차의 장기적 변화 경향")
            
            yearly_df = f_df.groupby('연도').agg({'일교차': 'mean', '평균기온(℃)': 'mean'}).reset_index()
            
            X1 = sm.add_constant(yearly_df['연도'])
            model1 = sm.OLS(yearly_df['일교차'], X1).fit()
            slope1 = model1.params['연도']
            
            st.metric(label="연간 일교차 변화율 (기울기 β₁)", value=f"{slope1:.4f} ℃/년")
            
            fig1 = px.scatter(yearly_df, x='연도', y='일교차', trendline="ols",
                              title="연도별 평균 일교차 장기 추세 (OLS 회귀선)",
                              labels={'일교차': '평균 일교차 (℃)', '연도': '연도'},
                              template="plotly_white")
            st.plotly_chart(fig1, use_container_width=True)
            
            st.markdown(f"**📝 추이 논리 해석:** 현재 선택된 기간 동안 일교차는 매년 평균 **{slope1:.4f}℃**씩 변하고 있습니다.")

        # -------------------------------------------------------------
        # TAB 2: 평균기온과의 상관관계 분석
        # -------------------------------------------------------------
        with tab2:
            st.header("🔗 평균기온과 일교차의 상관관계")
            overall_corr = f_df['평균기온(℃)'].corr(f_df['일교차'])
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.subheader("📊 통계적 지표")
                st.metric(label="전체 피어슨 상관계수 (r)", value=f"{overall_corr:.3f}")
                
                seasons = ['봄', '여름', '가을', '겨울']
                sect_corr = {}
                for s in seasons:
                    s_data = f_df[f_df['계절'] == s]
                    if len(s_data) > 1:
                        sect_corr[s] = s_data['평균기온(℃)'].corr(s_data['일교차'])
                
                st.write("**[계절별 독립 상관계수 (r)]**")
                st.json({k: f"{v:.3f}" for k, v in sect_corr.items()})
                
            with col2:
                sample_df = f_df.sample(n=min(5000, len(f_df)), random_state=42)
                fig2 = px.scatter(sample_df, x='평균기온(℃)', y='일교차', color='계절',
                                  color_discrete_map={'봄':'#2ecc71', '여름':'#e74c3c', '가을':'#f39c12', '겨울':'#3498db'},
                                  category_orders={"계절": ["봄", "여름", "가을", "겨울"]},
                                  opacity=0.5, trendline="ols",
                                  title="평균기온 vs 일교차 산점도 (계절별 회귀선 포함)",
                                  labels={'평균기온(℃)': '평균기온 (℃)', '일교차': '일교차 (℃)'},
                                  template="plotly_white")
                st.plotly_chart(fig2, use_container_width=True)

        # -------------------------------------------------------------
        # TAB 3: 최고/최저기온 주기 분석 (★ 새로 추가된 핵심 기능)
        # -------------------------------------------------------------
        with tab3:
            st.header("🕒 최고기온과 최저기온의 주기 및 추세 분해 분석")
            st.markdown("""
            일교차($DTR = T_{max} - T_{min}$)의 장기적 성질을 이해하기 위해, 최고기온과 최저기온 데이터로부터 
            **1년 단위의 계절적 주기(Seasonal)**와 **장기적인 변동 추세(Trend)**를 수학적으로 분리(Decomposition)했습니다.
            """)
            
            # 시계열 분석을 위해 데이터를 '월별 평균'으로 정렬 및 연속성 보장
            ts_df = f_df.set_index('날짜').resample('M').agg({
                '최고기온(℃)': 'mean',
                '최저기온(℃)': 'mean'
            }).interpolate(method='linear')
            
            if len(ts_df) < 24:
                st.warning("주기 분석(시계열 분해)을 수행하려면 최소 2년(24개월) 이상의 데이터 범위가 필요합니다.")
            else:
                # Statsmodels를 이용한 가법 모형 시계열 분해 (주기=12개월)
                decomp_max = seasonal_decompose(ts_df['최고기온(℃)'], model='additive', period=12)
                decomp_min = seasonal_decompose(ts_df['최저기온(℃)'], model='additive', period=12)
                
                # 시각화를 위한 데이터프레임 구축
                analysis_df = pd.DataFrame({
                    '최고_장기추세': decomp_max.trend,
                    '최저_장기추세': decomp_min.trend,
                    '최고_연간주기': decomp_max.seasonal,
                    '최저_연간주기': decomp_min.seasonal
                }, index=ts_df.index).dropna()
                
                # 차트 1: 장기 추세 변동 비교
                st.subheader("📉 1. 최고기온 vs 최저기온 장기 추세(Trend) 비교")
                st.markdown("계절적 주기(1년 진폭)를 제거한 순수 기온의 장기 baseline 변화 패턴입니다.")
                
                fig_trend = go.Figure()
                fig_trend.add_trace(go.Scatter(x=analysis_df.index, y=analysis_df['최고_장기추세'], name='최고기온 추세', line=dict(color='red', width=2)))
                fig_trend.add_trace(go.Scatter(x=analysis_df.index, y=analysis_df['최저_장기추세'], name='최저기온 추세', line=dict(color='blue', width=2)))
                fig_trend.update_layout(title="최고 및 최저기온의 장기 추세 변동선", xaxis_title="연도", yaxis_title="기온 추세값 (℃)", template="plotly_white")
                st.plotly_chart(fig_trend, use_container_width=True)
                
                # 차트 2: 1년 단위 고정 주기 파형 비교
                st.subheader("🔁 2. 1년 주기성(Seasonal Pattern) 비교")
                st.markdown("모든 연도에 공통적으로 나타나는 12개월 주기 데이터 패턴의 형태와 진폭을 비교합니다.")
                
                # 1년치 주기성 데이터만 슬라이싱 (12개월 추출)
                one_year_cycle = analysis_df.iloc[:12].copy()
                one_year_cycle['월'] = one_year_cycle.index.month
                one_year_cycle = one_year_cycle.sort_values('월')
                
                fig_season = go.Figure()
                fig_season.add_trace(go.Scatter(x=one_year_cycle['월'], y=one_year_cycle['최고_연간주기'], name='최고기온 주기성', mode='lines+markers', line=dict(color='red')))
                fig_season.add_trace(go.Scatter(x=one_year_cycle['월'], y=one_year_cycle['최저_연간주기'], name='최저기온 주기성', mode='lines+markers', line=dict(color='blue')))
                fig_season.update_layout(title="1년 주기 내 기온 변동 진폭(달별 가중치)", xaxis=dict(tickmode='linear', tick0=1, dtick=1), xaxis_title="월", yaxis_title="주기 성분값 (℃)", template="plotly_white")
                st.plotly_chart(fig_season, use_container_width=True)
                
                # 논리적 결론 도출
                st.markdown("""
                **📝 주기 및 추세 해석 매트릭스:**
                1. **추세선(Trend)의 불균형:** 만약 파란색 선(최저기온 추세)이 빨간색 선(최고기온 추세)보다 가파르게 상승하고 있다면, 이는 기후 변화나 도시화 과정에서 **밤 기온이 낮 기온보다 빠르게 상승**하고 있다는 뜻입니다. 두 선의 간격이 좁아지는 현상이 곧 **장기적 일교차 감소**의 원인입니다.
                2. **연간 주기(Seasonal) 파형의 진폭:** 최고기온 주기성과 최저기온 주기성의 진폭 차이는 계절마다 다르게 나타납니다. 봄·가을철 구간에 두 주기의 차이가 크게 벌어지고, 여름철에 좁혀지는 규칙적인 파형 역학을 시각적으로 명백히 증명해 줍니다.
                """)

        # -------------------------------------------------------------
        # TAB 4: 통계적 수식 요약
        # -------------------------------------------------------------
        with tab4:
            st.header("🧮 수학적 모델링 및 통계 요약")
            st.markdown("""
            ### 1. 시계열 분해 공식 (Additive Time-Series Decomposition)
            최고/최저기온 데이터($Y_t$)를 세 가지 독립된 확률 변수로 분해합니다.
            $$Y_t = Trend_t + Seasonal_t + \\epsilon_t$$
            * 단, 주기($period$)는 지구의 공전 주기인 **12개월**을 기준으로 설정되었습니다.
            """)
            st.text(model1.summary().as_text())
            st.dataframe(f_df[['평균기온(℃)', '최저기온(℃)', '최고기온(℃)', '일교차']].describe())

except FileNotFoundError:
    st.error(f"데이터 파일(`{DATA_PATH}`)을 찾을 수 없습니다.")
