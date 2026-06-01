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
        
        # 메인 탭 구성
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
        # TAB 3: 최고/최저기온 주기 분석 (★ 에러 완벽 수정본)
        # -------------------------------------------------------------
        with tab3:
            st.header("🕒 최고기온과 최저기온의 주기 및 추세 분해 분석")
            st.markdown("""
            일교차의 성질을 깊이 있게 이해하기 위해, 최고기온과 최저기온 데이터로부터 
            **1년 단위의 계절적 주기(Seasonal)**와 **장기적인 변동 추세(Trend)**를 수학적으로 분리(Decomposition)했습니다.
            """)
            
            # 1. 시계열 분석을 위해 월별 평균 리샘플링
            ts_df = f_df.set_index('날짜').resample('M').agg({
                '최고기온(℃)': 'mean',
                '최저기온(℃)': 'mean'
            })
            
            # 2. ★ 오류 방지 핵심 전처리: 선형 보간 후 양 끝단 결측치 완전히 메우기
            ts_df = ts_df.interpolate(method='linear').ffill().bfill()
            
            # 3. ★ 오류 방지 핵심 설정: 데이터프레임의 시간 빈도(Frequency)를 월단위('M')로 강제 고정
            ts_df = ts_df.asfreq('M')
            ts_df = ts_df.ffill().bfill()  # 재설정 과정에서 생길 수 있는 미세 결측치 차단
            
            if len(ts_df) < 24:
                st.warning("주기 분석(시계열 분해)을 수행하려면 최소 2년(24개월) 이상의 데이터 범위가 필요합니다. 사이드바에서 연도 범위를 늘려주세요.")
            else:
                try:
                    # 4. Statsmodels 가법 모형 시계열 분해 실행 (주기 = 12개월)
                    decomp_max = seasonal_decompose(ts_df['최고기온(℃)'], model='additive', period=12)
                    decomp_min = seasonal_decompose(ts_df['최저기온(℃)'], model='additive', period=12)
                    
                    # 5. 데이터 가공 (장기 추세 데이터프레임 구성 - 양 끝단 Moving Average 공백은 드롭)
                    trend_df = pd.DataFrame({
                        '최고_장기추세': decomp_max.trend,
                        '최저_장기추세': decomp_min.trend
                    }, index=ts_df.index).dropna()
                    
                    # 차트 1: 장기 추세 변동 비교
                    st.subheader("📉 1. 최고기온 vs 최저기온 장기 추세(Trend) 비교")
                    st.markdown("1년 주기의 계절 변동 진폭을 수학적으로 걷어낸 순수 기온의 장기 baseline 변화 패턴입니다.")
                    
                    fig_trend = go.Figure()
                    fig_trend.add_trace(go.Scatter(x=trend_df.index, y=trend_df['최고_장기추세'], name='최고기온 장기추세', line=dict(color='#e74c3c', width=2.5)))
                    fig_trend.add_trace(go.Scatter(x=trend_df.index, y=trend_df['최저_장기추세'], name='최저기온 장기추세', line=dict(color='#3498db', width=2.5)))
                    fig_trend.update_layout(title="최고 및 최저기온의 장기 추세 변동 곡선", xaxis_title="연도", yaxis_title="기온 추세 성분 (℃)", template="plotly_white")
                    st.plotly_chart(fig_trend, use_container_width=True)
                    
                    # 차트 2: 1년 고정 주기성 파형 비교 (안전하게 Groupby로 12개월 추출)
                    st.subheader("🔁 2. 1년 주기성(Seasonal Pattern) 비교")
                    st.markdown("연도별 노이즈를 제거하고 모든 해에 공통적으로 반복되는 12개월 주기 파형의 형태와 진폭을 비교합니다.")
                    
                    seasonal_pattern_df = pd.DataFrame({
                        '최고_연간주기': decomp_max.seasonal,
                        '최저_연간주기': decomp_min.seasonal,
                        '월': ts_df.index.month
                    }, index=ts_df.index)
                    
                    # 월별로 평균내어 고정된 1~12월 주기 생성 (슬라이싱 오류 완벽 방지)
                    one_year_cycle = seasonal_pattern_df.groupby('월').mean().reset_index()
                    
                    fig_season = go.Figure()
                    fig_season.add_trace(go.Scatter(x=one_year_cycle['월'], y=one_year_cycle['최고_연간주기'], name='최고기온 주기성 (달별 가중치)', mode='lines+markers', line=dict(color='#e74c3c', width=2)))
                    fig_season.add_trace(go.Scatter(x=one_year_cycle['월'], y=one_year_cycle['최저_연간주기'], name='최저기온 주기성 (달별 가중치)', mode='lines+markers', line=dict(color='#3498db', width=2)))
                    fig_season.update_layout(title="1년 주기 내 최고/최저기온 변동 진폭", xaxis=dict(tickmode='linear', tick0=1, dtick=1), xaxis_title="월", yaxis_title="주기 성분값 (℃)", template="plotly_white")
                    st.plotly_chart(fig_season, use_container_width=True)
                    
                    st.markdown("""
                    **📝 주기 및 추세 논리 해석 가이드:**
                    * **최저기온 장기추세(파란선)의 상승 탄력:** 최고기온 추세선(빨간선)보다 최저기온 추세선이 더 가파르게 위로 오르고 있다면, 밤 기온이 더 빠르게 온난화되고 있다는 증거입니다. 이 두 선의 간격이 점점 좁아지는 현상이 일교차 감소의 직접적 원인이 됩니다.
                    * **주기성 파형 변동:** 최고기온과 최저기온의 월별 주기성 성분의 격차가 봄·가을에 커지고 여름에 좁아지는 현상을 시각적으로 증명해 줍니다.
                    """)
                except Exception as e:
                    st.error(f"시계열 분석 모델 계산 중 예상치 못한 에러가 발생했습니다: {e}")

        # -------------------------------------------------------------
        # TAB 4: 통계적 수식 요약
        # -------------------------------------------------------------
        with tab4:
            st.header("🧮 수학적 모델링 및 통계 요약")
            st.markdown("""
            ### 1. 시계열 분해 공식 (Additive Time-Series Decomposition)
            최고/최저기온 데이터($Y_t$)를 세 가지 독립된 확률 변수로 분해합니다.
            $$Y_t = Trend_t + Seasonal_t + \\epsilon_t$$
            """)
            st.text(model1.summary().as_text())
            st.dataframe(f_df[['평균기온(℃)', '최저기온(℃)', '최고기온(℃)', '일교차']].describe())

except FileNotFoundError:
    st.error(f"데이터 파일(`{DATA_PATH}`)을 찾을 수 없습니다. 파일이 스크립트와 동일한 폴더에 있는지 확인하세요.")
