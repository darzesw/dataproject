import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm

# 1. 페이지 설정
st.set_page_config(page_title="기온 데이터 분석 앱", layout="wide")

st.title("🌡️ 기온 변화 및 일교차 상관관계 분석 웹앱")
st.markdown("""
이 애플리케이션은 장기 기상 데이터를 바탕으로 **시간 경과에 따른 일교차의 변화 추이**와 
**평균기온과 일교차 간의 수식적 상관관계**를 논리적으로 분석합니다.
""")

# 2. 데이터 로드 및 전처리 함수
@st.cache_data
def load_data(file_path):
    # 인코딩 오류 방지 (utf-8 실패 시 cp949로 재시도)
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='cp949')
    
    # 컬럼명 공백 및 문자열 정리
    df.columns = df.columns.str.strip()
    
    if '날짜' in df.columns:
        # 데이터 안의 \t, 쌍따옴표("), 공백을 모두 제거하여 완전한 날짜 문자열로 변환
        df['날짜'] = df['날짜'].astype(str).str.replace(r'[\t"\s]', '', regex=True)
        df['날짜'] = pd.to_datetime(df['날짜'], errors='coerce')
    
    # 필요한 핵심 컬럼 리스트 확인 후 결측치 제거
    required_cols = ['평균기온(℃)', '최저기온(℃)', '최고기온(℃)', '날짜']
    df = df.dropna(subset=[col for col in required_cols if col in df.columns])
    
    # 데이터가 정상적으로 존재할 때 파생 변수 생성
    if not df.empty:
        df['일교차'] = df['최고기온(℃)'] - df['최저기온(℃)']
        df['연도'] = df['날짜'].dt.year
        df['월'] = df['날짜'].dt.month
    
    return df

# 데이터 불러오기 (사용자 파일명에 맞춤)
DATA_PATH = "ta_20260601093156.csv"

try:
    df = load_data(DATA_PATH)
    
    if df.empty:
        st.error("데이터 전처리 후 남은 데이터가 없습니다. CSV 파일의 컬럼명과 날짜 형식을 확인해주세요.")
    else:
        st.sidebar.success("📊 데이터 로드 완료!")
        
        # 사이드바 필터 설정 (연0 오타 수정 및 안전성 확보)
        st.sidebar.header("🔍 분석 필터 설정")
        min_year = int(df['연도'].min())
        max_year = int(df['연도'].max())
        
        year_range = st.sidebar.slider(
            "분석 연도 범위 선택",
            min_year, max_year, (min_year, max_year)
        )
        
        # 데이터 필터링
        filtered_df = df[(df['연도'] >= year_range[0]) & (df['연도'] <= year_range[1])]
        
        if filtered_df.empty:
            st.warning("선택한 연도 범위에 해당하는 데이터가 없습니다.")
        else:
            # --- 메인 탭 구성 ---
            tab1, tab2, tab3 = st.tabs(["📈 시간에 따른 일교차 추이", "🔍 평균기온과의 상관관계", "🧮 수식 및 데이터 요약"])
            
            # -------------------------------------------------------------
            # TAB 1: 시간에 따른 일교차 추이
            # -------------------------------------------------------------
            with tab1:
                st.header("⏳ 연도별 평균 일교차의 장기적 변화")
                
                # 연도별 데이터 그룹화
                yearly_df = filtered_df.groupby('연도').agg({
                    '일교차': 'mean',
                    '평균기온(℃)': 'mean'
                }).reset_index()
                
                # 선형 회귀 분석 (OLS)
                X = sm.add_constant(yearly_df['연도'])
                model = sm.OLS(yearly_df['일교차'], X).fit()
                slope = model.params['연도']
                r_squared = model.rsquared
                
                st.metric(label="연간 일교차 변화율 (기울기 β₁)", value=f"{slope:.4f} ℃/년")
                
                # 시각화
                fig1 = px.scatter(yearly_df, x='연度' if '연度' in yearly_df else '연도', y='일교차', trendline="ols",
                                  title="연도별 평균 일교차 추이 (회귀선 포함)",
                                  labels={'일교차': '평균 일교차 (℃)', '연도': '연도'},
                                  template="plotly_white")
                st.plotly_chart(fig1, use_container_width=True)
                
                st.markdown("### 📝 추이 해석")
                if slope < 0:
                    st.write(f"기울기가 **{slope:.4f}**로 음의 값을 가집니다. 이는 시간이 흐를수록 일교차가 장기적으로 **감소**하는 경향이 있음을 뜻합니다. 도시 열섬 현상 등으로 인한 야간 최저기온 상승이 주원인일 수 있습니다.")
                else:
                    st.write(f"기울기가 **{slope:.4f}**로 양의 값을 가집니다. 이는 시간이 흐를수록 일교차가 장기적으로 **증가**하는 경향이 있음을 의미합니다.")
                    
            # -------------------------------------------------------------
            # TAB 2: 평균기온과의 상관관계
            # -------------------------------------------------------------
            with tab2:
                st.header("🔗 평균기온과 일교차의 상관관계")
                
                # 피어슨 상관계수 계산
                corr_value = filtered_df['평균기온(℃)'].corr(filtered_df['일교차'])
                
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.subheader("상관계수 (Pearson r)")
                    st.metric(label="Correlation Coefficient", value=f"{corr_value:.3f}")
                    
                    if abs(corr_value) >= 0.7:
                        strength = "매우 강한"
                    elif abs(corr_value) >= 0.4:
                        strength = "다소 강한"
                    else:
                        strength = "약한"
                    direction = "양의 상관관계(비례)" if corr_value > 0 else "음의 상관관계(반비례)"
                    
                    st.write(f"두 변수는 현재 **{strength} {direction}**를 보이고 있습니다.")
                    
                with col2:
                    # 데이터가 너무 많을 경우 시각화 속도 향상을 위해 샘플링 적용
                    sample_size = min(5000, len(filtered_df))
                    sample_df = filtered_df.sample(n=sample_size, random_state=42)
                    fig2 = px.scatter(sample_df, x='평균기온(℃)', y='일교차', opacity=0.4,
                                      trendline="ols", trendline_color_override="red",
                                      title=f"평균기온 vs 일교차 산점도 ({sample_size:,}개 샘플링)",
                                      labels={'평균기온(℃)': '평균기온 (℃)', '일교차': '일교차 (℃)'},
                                      template="plotly_white")
                    st.plotly_chart(fig2, use_container_width=True)
                    
                # 월별 세부 분석
                st.subheader("🗓️ 월별 평균기온과 일교차 패턴")
                monthly_df = filtered_df.groupby('월').agg({'평균기온(℃)': 'mean', '일교차': 'mean'}).reset_index()
                fig3 = px.bar(monthly_df, x='월', y='일교차', color='평균기온(℃)',
                              title="월별 평균 일교차 크기 (색상: 평균기온)",
                              labels={'일교차': '평균 일교차 (℃)', '월': '월'},
                              color_continuous_scale="Reds")
                st.plotly_chart(fig3, use_container_width=True)

            # -------------------------------------------------------------
            # TAB 3: 수식 및 데이터 요약
            # -------------------------------------------------------------
            with tab3:
                st.header("🧮 수학적 배경 및 통계 요약")
                
                st.markdown("""
                ### 1. 일교차 정의
                $$DTR = T_{max} - T_{min}$$
                
                ### 2. 선형 회귀 분석 결과 (시간 추이)
                """)
                # 통계 요약표 출력
                st.text(model.summary().as_text())
                
                st.markdown(f"""
                * **결정계수 ($R^2$):** {r_squared:.4f} (이 값이 1에 가까울수록 연도별 일교차가 회귀선 패턴에 엄격하게 부합함을 의미합니다.)
                """)
                
                st.subheader("📋 분석에 사용된 데이터 요약 (통계량)")
                st.dataframe(filtered_df[['평균기온(℃)', '최저기온(℃)', '최고기온(℃)', '일교차']].describe())

except FileNotFoundError:
    st.error(f"지정된 위치에서 데이터를 찾을 수 없습니다. 파일명이 `{DATA_PATH}` 이며 `app.py`와 같은 폴더에 있는지 확인해주세요.")
