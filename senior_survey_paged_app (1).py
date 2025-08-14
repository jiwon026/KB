import streamlit as st
import pandas as pd

# 세션 상태 단축 변수
ss = st.session_state

# 기본 설정값들 (기존 코드에서 가져온 것으로 추정)
LOCK_INFERRED_FIELDS = False
SHOW_PROBA_CHART = False
SHOW_SUCCESS_TOAST = True
DEFAULT_DISPLAY_TYPE = "안정형"

def render_main_home():
    # 커스텀 CSS 스타일
    st.markdown("""
    <style>
      /* 전체 페이지 배경 */
      .stApp {
        max-width: 350px;
        margin: 0 auto;
        background-color: #f8f9fa;
        padding: 20px;
    }
      
      /* 메인 컨테이너 카드 */
      .main-container {
        max-width: 480px;
        margin: 2rem auto;
        padding: 2.5rem;
        background: white;
        border-radius: 20px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        text-align: center;
      }
      
      /* KB 로고 및 브랜드 */
      .brand-section {
        margin-bottom: 1.5rem;
      }
      
      .kb-logo {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, #FFD700, #FFA500);
        color: #8B4513;
        font-weight: 900;
        font-size: 28px;
        padding: 10px 16px;
        border-radius: 10px;
        margin-right: 15px;
        border: 2px solid #FF8C00;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
      }
      
      .elderly-icons {
        font-size: 50px;
        margin-left: 10px;
      }
      
      .app-title {
        font-size: 36px;
        font-weight: 800;
        color: #2c3e50;
        margin: 1.5rem 0;
        line-height: 1.2;
      }
      
      /* 메뉴 버튼들 */
      .stButton > button {
        width: 100% !important;
        height: 80px !important;
        border-radius: 20px !important;
        font-size: 18px !important;
        font-weight: bold !important;
        border: none !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
        transition: all 0.2s ease !important;
        white-space: pre-line !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
    }
      
      /* Streamlit 기본 버튼 스타일 오버라이드 */
      .stButton > button {
        width: 100% !important;
        padding: 22px 24px !important;
        margin: 12px 0 !important;
        border: none !important;
        border-radius: 16px !important;
        font-size: 22px !important;
        font-weight: 700 !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        min-height: 70px !important;
      }
      
      /* 각 버튼의 색상 적용 */
      .stButton:nth-child(1) > button {
        background: #FFE4B5 !important;
        color: #8B4513 !important;
      }
      
      .stButton:nth-child(2) > button {
        background: #E6F3FF !important;
        color: #0066CC !important;
      }
      
      .stButton:nth-child(3) > button {
        background: #E8F5E8 !important;
        color: #2E8B57 !important;
      }
      
      .stButton:nth-child(4) > button {
        background: #FFE4E1 !important;
        color: #DC143C !important;
      }
      
      .stButton:nth-child(5) > button {
        background: #F0E6FF !important;
        color: #6A0DAD !important;
      }
      
      .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(0,0,0,0.15) !important;
      }
      
      /* 하단 설명 텍스트 */
      .footer-text {
        margin-top: 1.5rem;
        font-size: 16px;
        color: #7f8c8d;
        font-style: italic;
      }
      
      /* 반응형 디자인 */
      @media (max-width: 480px) {
        .main-container {
          margin: 1rem;
          padding: 2rem;
        }
        
        .app-title {
          font-size: 30px;
        }
        
        .kb-logo {
          font-size: 24px;
          padding: 8px 14px;
        }
        
        .elderly-icons {
          font-size: 40px;
        }
        
        .menu-button, .stButton > button {
          font-size: 20px !important;
          padding: 20px 22px !important;
          min-height: 65px !important;
        }
      }
    </style>
    """, unsafe_allow_html=True)

    # 메인 컨테이너
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # 브랜드 섹션
    st.markdown("""
    <div class="brand-section">
        <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 1rem;">
            <div class="kb-logo">KB</div>
            <div class="elderly-icons">👨‍🦳👩‍🦳</div>
        </div>
        <div class="app-title">시니어 연금 계산기</div>
    </div>
    """, unsafe_allow_html=True)
    
    # 메뉴 버튼들
    st.markdown('<div class="menu-section">', unsafe_allow_html=True)
    
    # 1) 내 금융 유형 보기
    if st.button("내 금융 유형 보기", key="home_btn_type"):
        ss.flow = "survey"
        st.rerun()
    
    # 2) 연금 계산하기
    if st.button("연금 계산하기", key="home_btn_predict"):
        ss.flow = "predict"
        st.rerun()
    
    # 3) 노후 시뮬레이션
    if st.button("노후 시뮬레이션", key="home_btn_sim"):
        ss.flow = "sim"
        st.rerun()
    
    # 4) 맞춤 상품 추천
    if st.button("맞춤 상품 추천", key="home_btn_reco"):
        ss.flow = "recommend"
        st.rerun()
    
    # 5) 설문 다시하기
    if st.button("설문 다시하기", key="home_btn_reset"):
        ss.flow = "survey"
        if hasattr(ss, 'question_step'):
            ss.question_step = 1
        if hasattr(ss, 'answers'):
            ss.answers = {}
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)  # menu-section 닫기
    
    # 하단 설명
    st.markdown(
        '<div class="footer-text">버튼을 눌러 다음 단계로 이동하세요</div>',
        unsafe_allow_html=True
    )
    
    st.markdown('</div>', unsafe_allow_html=True)  # main-container 닫기


def render_survey_form(defaults=None, lock_inferred=False):
    """기존 render_survey_form 함수와 호환되는 새로운 설문 UI"""
    
    # 설문 전용 CSS 스타일
    st.markdown("""
    <style>
      /* 전체 페이지 배경 */
      .stApp {
        max-width: 350px;
        margin: 0 auto;
        background-color: #f8f9fa;
        padding: 20px;
      }
      
      /* 메인 컨테이너 카드 */
      .survey-container {
        max-width: 480px;
        margin: 2rem auto;
        padding: 2.5rem;
        background: white;
        border-radius: 20px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        text-align: center;
      }
      
      /* KB 로고 및 브랜드 */
      .brand-section {
        margin-bottom: 2rem;
      }
      
      .kb-logo {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, #FFD700, #FFA500);
        color: #8B4513;
        font-weight: 900;
        font-size: 28px;
        padding: 10px 16px;
        border-radius: 10px;
        margin-right: 15px;
        border: 2px solid #FF8C00;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
      }
      
      .elderly-icons {
        font-size: 50px;
        margin-left: 10px;
      }
      
      .survey-title {
        font-size: 24px;
        font-weight: 700;
        color: #2c3e50;
        margin: 1rem 0;
        line-height: 1.3;
      }
      
      /* 질문 섹션 */
      .question-section {
        margin: 2rem 0;
        text-align: left;
      }
      
      .question-text {
        font-size: 22px;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 1.5rem;
        line-height: 1.4;
        text-align: center;
      }
      
      /* 진행률 바 */
      .progress-container {
        margin: 1.5rem 0;
        background-color: #f0f0f0;
        border-radius: 10px;
        height: 8px;
        overflow: hidden;
      }
      
      .progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #FFD700, #FFA500);
        border-radius: 10px;
        transition: width 0.3s ease;
      }
      
      /* 입력 필드 스타일 */
      .stTextInput > div > div > input,
      .stNumberInput > div > div > input {
        height: 60px !important;
        font-size: 18px !important;
        border-radius: 15px !important;
        border: 2px solid #e0e0e0 !important;
        padding: 15px 20px !important;
        text-align: center !important;
      }
      
      .stTextInput > div > div > input:focus,
      .stNumberInput > div > div > input:focus {
        border-color: #FFA500 !important;
        box-shadow: 0 0 0 2px rgba(255, 165, 0, 0.2) !important;
      }
      
      .stSelectbox > div > div > div {
        height: 60px !important;
        font-size: 18px !important;
        border-radius: 15px !important;
        border: 2px solid #e0e0e0 !important;
      }
      
      /* 선택 버튼들 */
      .choice-buttons {
        display: flex;
        flex-direction: column;
        gap: 15px;
        margin: 1.5rem 0;
      }
      
      .stButton > button {
        width: 100% !important;
        height: 70px !important;
        border-radius: 15px !important;
        font-size: 18px !important;
        font-weight: 600 !important;
        border: 2px solid #e0e0e0 !important;
        background: white !important;
        color: #2c3e50 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
        transition: all 0.3s ease !important;
      }
      
      .stButton > button:hover {
        background: #FFF8DC !important;
        border-color: #FFA500 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.12) !important;
      }
      
      /* 폼 제출 버튼 */
      .stFormSubmitButton > button {
        background: linear-gradient(135deg, #28a745, #20c997) !important;
        color: white !important;
        border: none !important;
        font-weight: 700 !important;
        font-size: 20px !important;
        height: 80px !important;
        width: 100% !important;
        border-radius: 15px !important;
      }
      
      .stFormSubmitButton > button:hover {
        background: linear-gradient(135deg, #218838, #1abc9c) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(40, 167, 69, 0.3) !important;
      }
      
      /* 반응형 디자인 */
      @media (max-width: 480px) {
        .survey-container {
          margin: 1rem;
          padding: 2rem;
        }
        
        .survey-title {
          font-size: 20px;
        }
        
        .question-text {
          font-size: 20px;
        }
        
        .kb-logo {
          font-size: 24px;
          padding: 8px 14px;
        }
        
        .elderly-icons {
          font-size: 40px;
        }
      }
    </style>
    """, unsafe_allow_html=True)

    if defaults is None:
        defaults = {}
    
    answers = {}
    
    # 메인 컨테이너
    st.markdown('<div class="survey-container">', unsafe_allow_html=True)
    
    # 브랜드 섹션
    st.markdown("""
    <div class="brand-section">
        <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 1rem;">
            <div class="kb-logo">KB</div>
            <div class="elderly-icons">👨‍🦳👩‍🦳</div>
        </div>
        <div class="survey-title">금융 유형 설문</div>
    </div>
    """, unsafe_allow_html=True)
    
    # 설문 폼
    with st.form("survey_form", clear_on_submit=False):
        st.markdown('<div class="question-section">', unsafe_allow_html=True)
        
        # 1. 나이
        answers["age"] = st.number_input(
            "1. 나이를 입력해주세요",
            min_value=20, max_value=100, 
            value=defaults.get("age", 67),
            step=1,
            disabled=lock_inferred and "age" in defaults
        )
        
        # 2. 성별
        answers["gender"] = st.selectbox(
            "2. 성별을 선택해주세요",
            options=["남성", "여성"],
            index=0 if defaults.get("gender") == "남성" else 1,
            disabled=lock_inferred and "gender" in defaults
        )
        
        # 3. 가구원 수
        answers["family_size"] = st.number_input(
            "3. 가구원 수를 입력해주세요",
            min_value=1, max_value=10,
            value=defaults.get("family_size", 2),
            step=1
        )
        
        # 4. 피부양자
        answers["dependents"] = st.selectbox(
            "4. 피부양자가 있나요?",
            options=["아니오", "예"],
            index=1 if defaults.get("dependents") == "예" else 0
        )
        
        # 5. 현재 보유 금융자산
        answers["assets"] = st.number_input(
            "5. 현재 보유한 금융자산을 입력해주세요 (만원)",
            min_value=0, step=100,
            value=defaults.get("assets", 9000)
        )
        
        # 6. 월 수령 연금
        answers["pension"] = st.number_input(
            "6. 월 수령하는 연금 급여를 입력해주세요 (만원)",
            min_value=0, step=10,
            value=defaults.get("pension", 0)
        )
        
        # 7. 월 평균 지출
        answers["living_cost"] = st.number_input(
            "7. 월 평균 지출비를 입력해주세요 (만원)",
            min_value=0, step=10,
            value=defaults.get("living_cost", 130)
        )
        
        # 8. 평균 월소득
        answers["income"] = st.number_input(
            "8. 평균 월소득을 입력해주세요 (만원)",
            min_value=0, step=10,
            value=defaults.get("income", 0)
        )
        
        # 9. 투자 성향 (risk)
        answers["risk"] = st.selectbox(
            "9. 투자 성향을 선택해주세요",
            options=["안정형", "안정추구형", "위험중립형", "적극투자형"],
            index=0 if not defaults.get("risk") else 
                  ["안정형", "안정추구형", "위험중립형", "적극투자형"].index(defaults.get("risk", "안정형"))
        )
        
        st.markdown('</div>', unsafe_allow_html=True)  # question-section 닫기
        
        # 제출 버튼
        submitted = st.form_submit_button("✓ 설문 완료하기")
    
    st.markdown('</div>', unsafe_allow_html=True)  # survey-container 닫기
    
    return answers, submitted


# 더미 함수들 (실제 함수들이 없을 때 에러 방지용)
def map_survey_to_model_input(answers):
    """설문 답변을 모델 입력으로 변환하는 더미 함수"""
    return [[1, 2, 3, 4, 5]]  # 더미 데이터

def render_type_result():
    """유형 결과 화면 더미 함수"""
    st.markdown("""
    <div style="max-width: 480px; margin: 2rem auto; padding: 2.5rem; background: white; 
         border-radius: 20px; box-shadow: 0 20px 40px rgba(0,0,0,0.1); text-align: center;">
        <h2>🎯 당신의 금융 유형</h2>
        <div style="font-size: 24px; font-weight: 700; color: #2c3e50; margin: 2rem 0;">
            {}</div>
        <p>설문 결과를 바탕으로 분석한 결과입니다.</p>
    </div>
    """.format(ss.get("pred_label", "안정형")), unsafe_allow_html=True)
    
    if st.button("홈으로 돌아가기"):
        ss.flow = "main"
        st.rerun()

def render_final_screen(display_type, rec_df):
    """추천 결과 화면 더미 함수"""
    st.write(f"### 🎯 {display_type} 유형 맞춤 추천")
    st.dataframe(rec_df.head(3))

def recommend_fallback_split(user_pref):
    """상품 추천 더미 함수"""
    return pd.DataFrame({
        "상품명": ["KB 안정형 펀드", "KB 성장형 펀드", "KB 적극형 펀드"],
        "예상수익률": [0.03, 0.05, 0.07],
        "월예상수익금(만원)": [15, 25, 35],
        "투자기간(개월)": [12, 24, 36],
        "최소투자금액": ["100만원", "300만원", "500만원"]
    })

def retirement_simulation(current_age, end_age, current_assets, monthly_income, monthly_expense, inflation_rate=0.03, investment_return=0.02):
    """노후 시뮬레이션 더미 함수"""
    log = []
    assets = current_assets
    for age in range(current_age, end_age + 1):
        monthly_real_expense = monthly_expense * ((1 + inflation_rate) ** (age - current_age))
        net_monthly = monthly_income - monthly_real_expense
        assets = assets * (1 + investment_return) + (net_monthly * 12)
        log.append({"나이": age, "잔액": round(assets)})
        if assets <= 0:
            return log, age
    return log, None

def simulate_with_financial_product(current_age, end_age, current_assets, monthly_income, monthly_expense, invest_return=0.05):
    """금융상품 적용 시뮬레이션 더미 함수"""
    return retirement_simulation(current_age, end_age, current_assets, monthly_income, monthly_expense, investment_return=invest_return)

def get_invest_return_from_risk(risk_choice):
    """리스크에 따른 수익률 반환"""
    risk_map = {"안정형": 0.03, "안정추구형": 0.04, "위험중립형": 0.05, "적극투자형": 0.07}
    return risk_map.get(risk_choice, 0.05)

def recommend_reason_from_simulation(depletion_age, current_age, current_assets, monthly_income, monthly_expense, risk_choice):
    """추천 근거 텍스트 생성"""
    if depletion_age:
        return f"현재 자산으로는 {depletion_age}세에 고갈될 것으로 예상되어 {risk_choice} 상품을 추천드립니다."
    else:
        return f"안정적인 노후 자금 확보를 위해 {risk_choice} 상품을 추천드립니다."


def main():
    # 페이지 설정
    st.set_page_config(
        page_title="KB 시니어 연금 계산기",
        page_icon="👨‍🦳",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    
    # 세션 상태 초기화
    if 'flow' not in ss:
        ss.flow = "main"
    
    # 더미 모델 변수들
    survey_model = None
    survey_encoder = None
    reg_model = None
    
    # 메인 플로우 로직 (기존 코드와 동일)
    if ss.flow == "main":
        render_main_home()
        
    elif ss.flow == "survey":
        answers, submitted = render_survey_form(
            defaults=ss.get("prefill_survey", {}),
            lock_inferred=LOCK_INFERRED_FIELDS
        )

        # 제출 처리
        if submitted:
            if (survey_model is None) or (survey_encoder is None):
                # 모델이 없어도 설문 결과 저장 후 곧바로 결과 화면으로 이동
                ss.pred_label = answers.get("risk") or "안정형"
                ss.answers = answers
                ss.flow = "result"
                st.rerun()  # ← 즉시 결과 화면으로 전환
            else:
                try:
                    arr = map_survey_to_model_input(answers)
                    pred = survey_model.predict(arr)
                    tabnet_label = survey_encoder.inverse_transform(pred)[0].strip()
                    ss.tabnet_label = tabnet_label
                    ss.pred_label = tabnet_label
                    ss.answers = answers
        
                    # (선택) 예측 확률 막대차트
                    if SHOW_PROBA_CHART:
                        proba_method = getattr(survey_model, "predict_proba", None)
                        if callable(proba_method):
                            proba = proba_method(arr)
                            proba_df = pd.DataFrame(proba, columns=survey_encoder.classes_)
                            st.bar_chart(proba_df.T)
        
                    # (선택) 성공 메시지
                    if SHOW_SUCCESS_TOAST:
                        st.success(f"🧾 예측된 금융 유형: **{tabnet_label}**")
        
                    # 곧바로 유형 결과 화면으로 이동
                    ss.flow = "result"
                    st.rerun()  # ← 여기 추가가 핵심
                except Exception as e:
                    st.error(f"오류 발생: {e}")

        # 🔽 폼 '밖'에 보조 네비게이션 버튼들 (제출과 독립적)
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("메인으로", key="survey_nav_main"):
                ss.flow = "main"
                st.rerun()
        with col2:
            if st.button("추천으로", key="survey_nav_reco"):
                # 설문 미제출이어도 이동 허용 (필요 시 tabnet_label 체크해서 survey로 돌려보내도 됨)
                ss.flow = "recommend"
                st.rerun()
        with col3:
            if st.button("시뮬레이션으로", key="survey_nav_sim"):
                ss.flow = "recommend"  # 추천 화면 하단의 시뮬레이션 섹션에서 보이도록
                st.rerun()
                
    elif ss.flow == "result":
        render_type_result()
        
    elif ss.flow == "recommend":
        st.markdown("---")
        st.subheader("🧲 금융상품 추천")

        # 1) 입력 폼
        invest_amount  = st.number_input("투자금액(만원)", min_value=10, step=10, value=500, key="reco_amount")
        invest_period  = st.selectbox("투자기간(개월)", [6, 12, 24, 36], index=1, key="reco_period")
        risk_choice    = st.selectbox("리스크 허용도", ["안정형", "위험중립형", "공격형"], index=1, key="reco_risk")
        target_monthly = st.number_input("목표 월이자(만원)", min_value=1, step=1, value=10, key="reco_target")

        # 2) 추천 실행
        if st.button("추천 보기", key="reco_btn"):
            user_pref = {
                '투자금액':   int(invest_amount),
                '투자기간':   int(invest_period),
                '투자성향':   str(risk_choice),
                '목표월이자': float(target_monthly),
            }
            rec_df = recommend_fallback_split(user_pref)
            if "메시지" in rec_df.columns:
                st.warning(rec_df.iloc[0, 0])
            else:
                ss.rec_df = rec_df
                ss.display_type = ss.get("tabnet_label") or DEFAULT_DISPLAY_TYPE
                ss.risk_choice = risk_choice
                ss.show_reco = True
                ss.pop("selected_product", None)   # ★ 상세 선택 초기화
                st.rerun()

        # 3) 추천 결과 (카드 + 근거만)
        if ss.get("show_reco") and ("rec_df" in ss):
            rec_df = ss.rec_df
            display_type = ss.get("display_type", DEFAULT_DISPLAY_TYPE)
            risk_choice = ss.get("risk_choice", "위험중립형")

            render_final_screen(display_type, rec_df)
            # === 카드 아래 '자세히 보기' 버튼들 ===
            rec_records = rec_df.head(3).to_dict(orient="records")
            cols = st.columns(len(rec_records) if rec_records else 1)
            
            for i, (col, r) in enumerate(zip(cols, rec_records)):
                with col:
                    pname = str(r.get("상품명", "-"))
                    if st.button(f"🔍 {pname} 자세히 보기", key=f"prod_detail_{i}"):
                        ss.selected_product = r
                        st.rerun()
            
            # === 선택된 상품 상세 영역 ===
            sel = ss.get("selected_product")
            if sel:
                st.markdown("---")
                st.subheader("📋 상품 상세")
                # 예상수익률 표시는 (예상수익률(연) 있으면 그걸, 없으면 숫자를 %로 변환)
                rate_txt = sel.get("예상수익률(연)")
                if not rate_txt:
                    try:
                        rate_txt = f"{float(sel.get('예상수익률', 0.0))*100:.2f}%"
                    except Exception:
                        rate_txt = "-"
            
                rows = [
                    ("상품명", sel.get("상품명", "-")),
                    ("월예상수익금(만원)", sel.get("월예상수익금(만원)", "-")),
                    ("예상수익률", rate_txt),
                    ("투자기간", f"{sel.get('투자기간(개월)', sel.get('권장투자기간','-'))}개월"),
                    ("최소투자금액", sel.get("최소투자금액", "-")),
                ]
                st.table(pd.DataFrame(rows, columns=["항목", "값"]))
            
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("선택 해제", key="clear_selected_product"):
                        ss.pop("selected_product", None)
                        st.rerun()
                with c2:
                    if st.button("시뮬레이션으로 이동", key="go_sim_from_detail"):
                        ss.flow = "sim"
                        st.rerun()

            # 추천 근거(고갈 여부는 내부 계산해서 문장만)
            ans = ss.get("answers", {})
            current_age     = int(ans.get("age", 67))
            end_age         = 100
            current_assets  = float(ans.get("assets", 9000))
            pension_month   = float(ans.get("pension", 0))
            income_month    = float(ans.get("income", 0))
            monthly_income  = pension_month + income_month
            monthly_expense = float(ans.get("living_cost", 130))

            base_return = 0.02
            log_base, depletion_base = retirement_simulation(
                current_age, end_age, current_assets, monthly_income, monthly_expense,
                inflation_rate=0.03, investment_return=base_return
            )
            reason_text = recommend_reason_from_simulation(
                depletion_base, current_age, current_assets, monthly_income, monthly_expense, risk_choice
            )
            st.info(f"🔎 추천 근거: {reason_text}")

            # 다운로드
            csv_bytes = rec_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button("추천 결과 CSV 다운로드", csv_bytes, "recommendations.csv", "text/csv")
            col_go1, col_go2 = st.columns(2)
            with col_go1:
                if st.button("📈 노후 시뮬레이션으로", key="go_to_sim"):
                    ss.flow = "sim"          # 상태(rec_df 등) 그대로 유지한 채 이동
                    st.rerun()
            with col_go2:
                if st.button("🏠 메인으로", key="go_to_main_from_reco"):
                    ss.flow = "main"         # 상태는 유지(원하면 유지), '처음으로'와 역할 분리
                    st.rerun()

    elif ss.flow == "predict":
        st.subheader("📈 연금 계산기")

        # 폼으로 묶어 중복 버튼/리렌더 방지
        with st.form("predict_form"):
            income = st.number_input("평균 월소득(만원)", min_value=0, step=1, key="pred_income")
            years  = st.number_input("국민연금 가입기간(년)", min_value=0, max_value=50, step=1, key="pred_years")
            pred_submit = st.form_submit_button("연금 예측하기")

        if pred_submit:
            if reg_model is None:
                # 모델 없어도 설문으로 이동 가능하게 프리필 0원 세팅
                ss.prefill_survey = {"income": income, "pension": 0}
                st.info("연금 예측 모델이 없어 계산을 건너뜁니다.")
                ss.predicted = True
                ss.pred_amount = 0.0
                st.rerun()
            else:
                try:
                    X = pd.DataFrame([{"평균월소득(만원)": income, "가입기간(년)": years}])
                    amount = round(float(reg_model.predict(X)[0]), 1)

                    # 결과/프리필 저장
                    ss.prefill_survey = {"income": income, "pension": amount}
                    ss.pred_amount = amount
                    ss.predicted = True
                    ss.pred_amount = amount
                    st.rerun()
                except Exception as e:
                    st.exception(e)

        # 예측이 끝났으면 결과 + 네비게이션 버튼 노출
        if ss.get("predicted"):
            amt = ss.get("pred_amount", 0.0)

            # 보조설명(선택)
            def classify_pension_type(a):
                if a >= 90: return "완전노령연금"
                if a >= 60: return "조기노령연금"
                if a >= 30: return "감액노령연금"
                return "특례노령연금"
            ptype = classify_pension_type(amt)
            explains = {
                "조기노령연금": "※ 만 60세부터 수령 가능하나 최대 30% 감액될 수 있어요.",
                "완전노령연금": "※ 만 65세부터 감액 없이 정액 수령이 가능해요.",
                "감액노령연금": "※ 일정 조건을 만족하지 못할 경우 감액되어 수령됩니다.",
                "특례노령연금": "※ 가입기간이 짧더라도 일정 기준 충족 시 수령 가능."
            }

            st.success(f"💰 예측 연금 수령액: **{amt}만원/월**")
            st.caption(f"예측 연금 유형: **{ptype}**")
            st.info(explains[ptype])

            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("👉 설문으로 진행", key="pred_go_survey"):
                    ss.flow = "survey"
                    st.rerun()
            with c2:
                if st.button("🧲 바로 추천 보기", key="pred_go_reco"):
                    # 설문을 건너뛰는 경우도 있으니, 최소 기본값 보장
                    ss.answers = ss.get("answers", {})
                    ss.flow = "recommend"
                    st.rerun()
            with c3:
                if st.button("🏠 메인으로", key="pred_go_main"):
                    ss.flow = "main"
                    st.rerun()

        # 예측 전이라도 이동하고 싶다면(옵션)
        st.markdown("---")
        if st.button("건너뛰고 설문으로", key="pred_skip_to_survey"):
            ss.flow = "survey"
            st.rerun()

    elif ss.flow == "sim":
        st.subheader("📈 노후 시뮬레이션")

        has_reco = "rec_df" in ss and not ss.rec_df.empty
        rec_df = ss.rec_df if has_reco else pd.DataFrame()
        risk_choice = ss.get("risk_choice", "위험중립형")

        if not has_reco:
            st.info("추천 결과 없이도 기본 시뮬레이션을 먼저 볼 수 있어요. "
                    "'맞춤 상품 추천'에서 추천을 실행하면 상품별 탭이 추가됩니다.")

        # 설문값(없으면 기본값)
        ans = ss.get("answers", {})
        current_age     = int(ans.get("age", 67))
        end_age         = 100
        current_assets  = float(ans.get("assets", 9000))
        pension_month   = float(ans.get("pension", 0))
        income_month    = float(ans.get("income", 0))
        monthly_income  = pension_month + income_month
        monthly_expense = float(ans.get("living_cost", 130))

        base_return   = 0.02
        invest_return = get_invest_return_from_risk(risk_choice)

        log_base, depletion_base = retirement_simulation(
            current_age, end_age, current_assets, monthly_income, monthly_expense,
            inflation_rate=0.03, investment_return=base_return
        )
        log_invest, depletion_invest = simulate_with_financial_product(
            current_age, end_age, current_assets, monthly_income, monthly_expense,
            invest_return=invest_return
        )

        col1, col2 = st.columns(2)
        with col1:
            st.metric(f"기본 시나리오(연 {int(base_return*100)}%) 고갈 나이",
                      value=f"{depletion_base}세" if depletion_base else "고갈 없음")
        with col2:
            st.metric(f"금융상품 적용(연 {int(invest_return*100)}%) 고갈 나이",
                      value=f"{depletion_invest}세" if depletion_invest else "고갈 없음")

        st.markdown("### ⚙️ 시뮬레이션 가정값")
        with st.form("sim_form_only"):
            colA, colB = st.columns(2)
            with colA:
                inflation_pct = st.slider("물가상승률(연, %)", 0.0, 8.0, 3.0, 0.1, key="sim_inflation_only")
            with colB:
                base_return_pct = st.slider("기본 시나리오 수익률(연, %)", 0.0, 6.0, 2.0, 0.1, key="sim_base_return_only")
            submitted = st.form_submit_button("시뮬레이션 실행")

        if submitted:
            inflation = inflation_pct / 100.0
            base_r    = base_return_pct / 100.0

            log_base2, _ = retirement_simulation(
                current_age, end_age, current_assets, monthly_income, monthly_expense,
                inflation_rate=inflation, investment_return=base_r
            )
            df_b = (pd.DataFrame(log_base2)[['나이','잔액']]
                    .rename(columns={'잔액':'기본 시나리오'}) if log_base2 else pd.DataFrame())

            # 추천 결과가 있을 때만 상품 탭 렌더
            if has_reco:
                st.markdown("### 📈 추천 상품별 적용 시나리오")
                rec_records = rec_df.to_dict(orient="records")
                tabs = st.tabs([f"{i+1}. {r.get('상품명','-')}" for i, r in enumerate(rec_records)])

                for tab, r in zip(tabs, rec_records):
                    with tab:
                        if '예상수익률' in r and pd.notnull(r['예상수익률']):
                            prod_return_pct = float(r['예상수익률']) * 100.0
                        else:
                            txt = str(r.get('예상수익률(연)','0')).replace('%','')
                            try: prod_return_pct = float(txt)
                            except: prod_return_pct = 5.0
                        prod_r = prod_return_pct / 100.0

                        log_prod2, _ = retirement_simulation(
                            current_age, end_age, current_assets, monthly_income, monthly_expense,
                            inflation_rate=inflation, investment_return=prod_r
                        )
                        df_p = pd.DataFrame(log_prod2)[['나이','잔액']].rename(
                            columns={'잔액': f"{r.get('상품명','-')} 적용"}
                        )
                        st.caption(
                            f"가정 수익률: 기본 **{base_return_pct:.1f}%**, "
                            f"해당 상품 **{prod_return_pct:.1f}%** · 물가상승률 **{inflation_pct:.1f}%**"
                        )
                        chart_df = (pd.merge(df_b, df_p, on='나이', how='outer').set_index('나이')
                                    if not df_b.empty else df_p.set_index('나이'))
                        st.line_chart(chart_df)
            else:
                st.info("상품별 그래프는 추천 실행 후 표시됩니다. '맞춤 상품 추천'에서 추천을 실행해 주세요.")

        st.markdown("---")
        colX, colY = st.columns(2)
        with colX:
            if st.button("맞춤 상품 추천으로", key="sim_to_recommend"):
                ss.flow = "recommend"
                st.rerun()
        with colY:
            if st.button("메인으로", key="sim_to_main"):
                ss.flow = "main"
                st.rerun()


if __name__ == "__main__":
    main()
