import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import time

# FAISS 설정 (기존 코드 유지)
USE_FAISS = True
try:
    import faiss
except Exception as e:
    USE_FAISS = False
    from sklearn.neighbors import NearestNeighbors

# 페이지 설정
st.set_page_config(
    page_title="KB 시니어 연금 계산기",
    page_icon="🏦",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# CSS 스타일링
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 20px 0;
        margin-bottom: 30px;
        background-color: #f8f9fa;
        border-radius: 15px;
    }
    
    .kb-logo {
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 36px;
        font-weight: bold;
        margin-bottom: 15px;
    }
    
    .kb-star {
        color: #FFB800;
        margin-right: 8px;
    }
    
    .kb-text {
        color: #666;
        margin-right: 15px;
    }
    
    .elderly-emoji {
        font-size: 48px;
        margin-left: 10px;
    }
    
    .title {
        font-size: 24px;
        font-weight: bold;
        color: #333;
        margin-top: 15px;
    }
    
    .stApp {
        max-width: 400px;
        margin: 0 auto;
        background-color: #f8f9fa;
        padding: 20px;
    }
    
    /* Streamlit 버튼 스타일링 */
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
    
    /* 메인 화면 버튼들 */
    div[data-testid="stVerticalBlock"] > div:nth-child(1) .stButton > button {
        background: #FFE4B5 !important;
        color: #8B4513 !important;
    }
    
    div[data-testid="stVerticalBlock"] > div:nth-child(3) .stButton > button {
        background: #B8D4F0 !important;
        color: #2C5282 !important;
    }
    
    div[data-testid="stVerticalBlock"] > div:nth-child(5) div:nth-child(1) .stButton > button {
        background: #C6F6D5 !important;
        color: #22543D !important;
        height: 60px !important;
        font-size: 16px !important;
    }
    
    div[data-testid="stVerticalBlock"] > div:nth-child(5) div:nth-child(2) .stButton > button {
        background: #FECACA !important;
        color: #7F1D1D !important;
        height: 60px !important;
        font-size: 16px !important;
    }
    
    div[data-testid="stVerticalBlock"] > div:nth-child(7) div:nth-child(1) .stButton > button {
        background: #DDD6FE !important;
        color: #5B21B6 !important;
        height: 60px !important;
        font-size: 16px !important;
    }
    
    div[data-testid="stVerticalBlock"] > div:nth-child(7) div:nth-child(2) .stButton > button {
        background: #FDE68A !important;
        color: #92400E !important;
        height: 60px !important;
        font-size: 16px !important;
    }
    
    /* 선택 버튼 스타일링 */
    .choice-button .stButton > button {
        background: #E8F4FD !important;
        color: #1E40AF !important;
        border: 2px solid #60A5FA !important;
        border-radius: 15px !important;
        font-size: 18px !important;
        font-weight: bold !important;
        padding: 20px !important;
        margin: 10px 0 !important;
        transition: all 0.3s ease !important;
    }
    
    .choice-button .stButton > button:hover {
        background: #DBEAFE !important;
        border-color: #3B82F6 !important;
        transform: translateY(-2px) !important;
    }
    
    /* 텍스트 입력 스타일링 */
    .stTextInput > div > div > input {
        border-radius: 15px !important;
        border: 2px solid #E5E7EB !important;
        padding: 15px 20px !important;
        font-size: 16px !important;
        text-align: center !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #3B82F6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
    }
    
    /* 결과 카드 스타일 */
    .result-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 15px 0;
    }
    
    .product-card {
        border: 2px solid #E5E7EB;
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        background: white;
    }
    
    .product-card:hover {
        border-color: #3B82F6;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.1);
    }
    
    .consultation-info {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin: 20px 0;
    }
    
    .consultation-card {
        background: white;
        border: 2px solid #4F46E5;
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(79, 70, 229, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# =================================
# 기본 설정 및 유틸 함수들 (기존 코드 유지)
# =================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
MODELS_DIR = BASE_DIR
DEPOSIT_CSV = "금융상품_3개_통합본.csv"
FUND_CSV = "펀드_병합본.csv"

# 모델/데이터 로딩 함수들 (기존 코드 유지)
@st.cache_resource
def load_models():
    def safe_load(name):
        path = os.path.join(MODELS_DIR, name)
        if not os.path.exists(path):
            return None
        try:
            return joblib.load(path)
        except Exception as e:
            st.warning(f"{name} 로드 실패: {e}")
            return None
    
    survey_model = safe_load("tabnet_model.pkl")
    survey_encoder = safe_load("label_encoder.pkl")
    reg_model = safe_load("reg_model.pkl")
    type_model = safe_load("type_model.pkl")
    return survey_model, survey_encoder, reg_model, type_model

@st.cache_data
def load_deposit_csv():
    path = os.path.join(BASE_DIR, DEPOSIT_CSV)
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding='utf-8-sig')
    except:
        try:
            return pd.read_csv(path, encoding='cp949')
        except:
            return pd.DataFrame()

@st.cache_data
def load_fund_csv():
    path = os.path.join(BASE_DIR, FUND_CSV)
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding='utf-8-sig')
    except:
        try:
            return pd.read_csv(path, encoding='cp949')
        except:
            return pd.DataFrame()

# 모델 로딩
survey_model, survey_encoder, reg_model, type_model = load_models()

# 추천 시스템 함수들 (기존 코드 간소화)
def simple_recommend(user_data):
    """간단한 규칙 기반 추천"""
    age = user_data.get('age', 65)
    assets = user_data.get('assets', 5000)
    risk = user_data.get('risk', '안정형')
    
    recommendations = []
    
    if risk == '안정형':
        recommendations = [
            {'상품명': 'KB 안심적금', '예상수익률': '3.2%', '리스크': '낮음', '최소투자금액': '100만원'},
            {'상품명': 'KB 시니어예금', '예상수익률': '2.8%', '리스크': '낮음', '최소투자금액': '500만원'},
            {'상품명': 'KB 연금저축', '예상수익률': '4.1%', '리스크': '낮음', '최소투자금액': '1000만원'}
        ]
    elif risk == '위험중립형':
        recommendations = [
            {'상품명': 'KB 혼합형펀드', '예상수익률': '5.5%', '리스크': '중간', '최소투자금액': '500만원'},
            {'상품명': 'KB 균형투자', '예상수익률': '4.8%', '리스크': '중간', '최소투자금액': '300만원'},
            {'상품명': 'KB 안정성장', '예상수익률': '6.2%', '리스크': '중간', '최소투자금액': '1000만원'}
        ]
    else:  # 적극투자형
        recommendations = [
            {'상품명': 'KB 성장주펀드', '예상수익률': '8.1%', '리스크': '높음', '최소투자금액': '1000만원'},
            {'상품명': 'KB 글로벌투자', '예상수익률': '7.5%', '리스크': '높음', '최소투자금액': '500만원'},
            {'상품명': 'KB 테크펀드', '예상수익률': '9.3%', '리스크': '높음', '최소투자금액': '2000만원'}
        ]
    
    return recommendations

def calculate_pension_estimate(monthly_income, years):
    """간단한 연금 계산"""
    base_amount = monthly_income * 0.015 * years
    if base_amount > 150:
        return min(base_amount, 200)  # 최대 200만원
    return max(base_amount, 30)  # 최소 30만원

# =================================
# 세션 상태 초기화
# =================================
if 'page' not in st.session_state:
    st.session_state.page = 'main'
if 'question_step' not in st.session_state:
    st.session_state.question_step = 1
if 'answers' not in st.session_state:
    st.session_state.answers = {}
if 'user_type' not in st.session_state:
    st.session_state.user_type = None

# =================================
# 헤더 컴포넌트
# =================================
def render_header(title="시니어 연금 계산기"):
    st.markdown(f"""
    <div class="main-header">
        <div class="kb-logo">
            <span class="kb-star">★</span>
            <span class="kb-text">KB</span>
            <span class="elderly-emoji">👴👵</span>
        </div>
        <div class="title">{title}</div>
    </div>
    """, unsafe_allow_html=True)

# =================================
# 메인 페이지
# =================================
def render_main_page():
    render_header()
    
    # 내 금융유형 보기 버튼
    if st.button("내 금융유형\n보기", key="financial_type", use_container_width=True):
        st.session_state.page = 'survey'
        st.session_state.question_step = 1
        st.session_state.answers = {}
        st.rerun()
    
    st.markdown('<div style="margin: 15px 0;"></div>', unsafe_allow_html=True)
    
    # 연금 계산하기 버튼
    if st.button("연금\n계산하기", key="pension_calc", use_container_width=True):
        st.session_state.page = 'pension_input'
        st.rerun()
    
    st.markdown('<div style="margin: 20px 0;"></div>', unsafe_allow_html=True)
    
    # 하단 버튼들
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("노후\n시뮬레이션", key="simulation", use_container_width=True):
            st.session_state.page = 'simulation'
            st.rerun()
    
    with col2:
        if st.button("맞춤 상품\n추천", key="recommendation", use_container_width=True):
            if st.session_state.answers:
                st.session_state.page = 'recommendation'
            else:
                st.session_state.page = 'survey'
                st.session_state.question_step = 1
                st.session_state.answers = {}
            st.rerun()
    
    st.markdown('<div style="margin: 15px 0;"></div>', unsafe_allow_html=True)
    
    # 설문 다시하기와 전화 상담 버튼
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("설문\n다시하기", key="survey_reset", use_container_width=True):
            st.session_state.page = 'survey'
            st.session_state.question_step = 1
            st.session_state.answers = {}
            st.rerun()
    
    with col2:
        if st.button("📞 전화\n상담", key="phone_consultation", use_container_width=True):
            st.session_state.page = 'phone_consultation'
            st.rerun()

# =================================
# 전화 상담 페이지
# =================================
def render_phone_consultation_page():
    render_header("전화 상담")
    
    st.markdown("""
    <div class="consultation-info">
        <div style="text-align: center; margin-bottom: 20px;">
            <h2 style="margin: 0; color: white;">📞 전문 상담사와 1:1 상담</h2>
            <p style="margin: 10px 0 0 0; font-size: 16px; opacity: 0.9;">복잡한 연금 제도, 전문가가 쉽게 설명해드립니다</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="consultation-card">
        <h3 style="color: #4F46E5; margin-bottom: 15px;">📞 KB 시니어 연금 상담센터</h3>
        
        <div style="margin: 15px 0;">
            <strong style="color: #1F2937;">상담 전화번호:</strong>
            <span style="font-size: 24px; font-weight: bold; color: #4F46E5; margin-left: 10px;">1588-9999</span>
        </div>
        
        <div style="margin: 15px 0;">
            <strong style="color: #1F2937;">상담 시간:</strong>
            <ul style="margin: 10px 0; padding-left: 20px;">
                <li>평일: 오전 9시 ~ 오후 6시</li>
                <li>토요일: 오전 9시 ~ 오후 1시</li>
                <li>일요일 및 공휴일 휴무</li>
            </ul>
        </div>
        
        <div style="margin: 15px 0;">
            <strong style="color: #1F2937;">상담 가능 내용:</strong>
            <ul style="margin: 10px 0; padding-left: 20px;">
                <li>🏦 연금 상품 상세 안내</li>
                <li>📝 가입 절차 및 필요 서류</li>
                <li>💰 수령 방법 및 시기</li>
                <li>💸 세제 혜택 안내</li>
                <li>📊 개인 맞춤 포트폴리오 구성</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📋 상담 예약 신청")
    st.markdown("아래 정보를 입력하시면 전문 상담사가 먼저 연락드립니다.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        name = st.text_input("성함 *", placeholder="홍길동")
        consultation_type = st.selectbox(
            "상담 유형 *",
            ["선택해주세요", "연금 상품 문의", "가입 절차 문의", "수령 방법 상담", "세제 혜택 문의", "기타"]
        )
    
    with col2:
        phone = st.text_input("연락처 *", placeholder="010-1234-5678")
        preferred_time = st.selectbox(
            "희망 상담 시간",
            ["상관없음", "오전 (9시-12시)", "오후 (1시-3시)", "늦은 오후 (3시-6시)"]
        )
    
    inquiry = st.text_area("문의 내용", placeholder="궁금한 점이나 상담받고 싶은 내용을 자유롭게 적어주세요.", height=100)
    
    st.markdown('<div style="margin: 20px 0;"></div>', unsafe_allow_html=True)
    
    if st.button("📞 상담 신청하기", use_container_width=True):
        if name and phone and consultation_type != "선택해주세요":
            # 상담 신청 처리 로직
            st.balloons()
            st.success(f"""
            ✅ **상담 신청이 완료되었습니다!**
            
            **신청자:** {name}님  
            **연락처:** {phone}  
            **상담 유형:** {consultation_type}  
            **희망 시간:** {preferred_time}
            
            📞 영업일 기준 24시간 내에 전문 상담사가 연락드리겠습니다.
            """)
            
            # 세션에 상담 신청 정보 저장
            st.session_state.consultation_requested = {
                'name': name,
                'phone': phone,
                'type': consultation_type,
                'time': preferred_time,
                'inquiry': inquiry
            }
            
        else:
            st.error("⚠️ 필수 항목(*)을 모두 입력해주세요.")
    
    st.markdown('<div style="margin: 30px 0;"></div>', unsafe_allow_html=True)
    
    # 추가 정보
    st.info("""
    💡 **상담 전 준비사항**
    - 신분증 및 소득 관련 서류
    - 기존 가입 연금 정보
    - 투자 목표 및 위험 성향 파악
    """)
    
    if st.button("← 메인으로 돌아가기", use_container_width=True):
        st.session_state.page = 'main'
        st.rerun()

# =================================
# 설문 페이지
# =================================
def render_survey_page():
    # 질문 데이터
    questions = [
        {
            "title": "설문조사 1",
            "question": "1. 나이를\n입력해주세요.",
            "type": "input",
            "placeholder": "나이를 입력하세요",
            "key": "age"
        },
        {
            "title": "설문조사 2",
            "question": "2. 성별을\n선택해주세요.",
            "type": "choice",
            "options": ["남성", "여성"],
            "key": "gender"
        },
        {
            "title": "설문조사 3",
            "question": "3. 가구원 수를\n입력해주세요.",
            "type": "input",
            "placeholder": "가구원 수를 입력하세요",
            "key": "family_size"
        },
        {
            "title": "설문조사 4",
            "question": "4. 피부양자가\n있나요?",
            "type": "choice",
            "options": ["예", "아니오"],
            "key": "dependents"
        },
        {
            "title": "설문조사 5",
            "question": "5. 현재 보유한\n금융자산을\n입력해주세요.",
            "type": "input",
            "placeholder": "현재 보유 금융자산 (만원)",
            "key": "assets"
        },
        {
            "title": "설문조사 6",
            "question": "6. 월 수령하는\n연금 급여를\n입력해주세요.",
            "type": "input",
            "placeholder": "월 수령 연금 급여 (만원)",
            "key": "pension"
        },
        {
            "title": "설문조사 7",
            "question": "7. 월 평균\n지출비를\n입력해주세요.",
            "type": "input",
            "placeholder": "월 평균 지출비 (만원)",
            "key": "living_cost"
        },
        {
            "title": "설문조사 8",
            "question": "8. 월 평균 소득은\n얼마인가요?",
            "type": "input",
            "placeholder": "월 평균 소득 (만원)",
            "key": "income"
        },
        {
            "title": "설문조사 9",
            "question": "9. 투자 성향을\n선택해주세요.",
            "type": "choice",
            "options": ["안정형", "안정추구형", "위험중립형", "적극투자형", "공격투자형"],
            "key": "risk"
        }
    ]
    
    if st.session_state.question_step <= len(questions):
        current_q = questions[st.session_state.question_step - 1]
        
        render_header(current_q['title'])
        
        # 질문 표시
        st.markdown(f"""
        <div style="text-align: center; font-size: 20px; font-weight: bold; margin: 50px 0; line-height: 1.5; color: #333;">
            {current_q['question']}
        </div>
        """, unsafe_allow_html=True)
        
        # 답변 입력/선택
        if current_q['type'] == 'input':
            answer = st.text_input("", placeholder=current_q['placeholder'], key=f"survey_q{st.session_state.question_step}")
            
            if answer and answer.strip():
                with st.spinner('다음 단계로 이동 중...'):
                    time.sleep(1)
                
                st.session_state.answers[current_q['key']] = answer
                if st.session_state.question_step < len(questions):
                    st.session_state.question_step += 1
                    st.rerun()
                else:
                    # 설문 완료 - 유형 분석
                    analyze_user_type()
                    st.session_state.page = 'survey_result'
                    st.rerun()
        
        elif current_q['type'] == 'choice':
            st.markdown('<div style="margin: 30px 0;"></div>', unsafe_allow_html=True)
            
            for option in current_q['options']:
                if st.button(option, key=f"choice_{option}_{st.session_state.question_step}", use_container_width=True):
                    st.session_state.answers[current_q['key']] = option
                    with st.spinner('다음 단계로 이동 중...'):
                        time.sleep(0.5)
                    
                    if st.session_state.question_step < len(questions):
                        st.session_state.question_step += 1
                        st.rerun()
                    else:
                        # 설문 완료 - 유형 분석
                        analyze_user_type()
                        st.session_state.page = 'survey_result'
                        st.rerun()
        
        # 진행 상황 표시
        progress = st.session_state.question_step / len(questions)
        st.progress(progress)
        st.markdown(f"""
        <div style='text-align: center; margin-top: 15px; font-size: 16px; color: #666;'>
            {st.session_state.question_step}/{len(questions)} 단계
        </div>
        """, unsafe_allow_html=True)
        
        # 메인으로 돌아가기 버튼
        st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)
        if st.button("← 메인으로", key="back_to_main_from_survey"):
            st.session_state.page = 'main'
            st.rerun()

def analyze_user_type():
    """사용자 유형 분석"""
    answers = st.session_state.answers
    
    try:
        age = int(answers.get('age', 65))
        assets = float(answers.get('assets', 5000))
        pension = float(answers.get('pension', 100))
        income = float(answers.get('income', 200))
        living_cost = float(answers.get('living_cost', 150))
        risk = answers.get('risk', '안정형')
        
        # 간단한 유형 분류 로직
        if assets > 10000 and income > 300:
            user_type = "자산운용형"
        elif living_cost > income + pension:
            user_type = "위험취약형"
        elif risk in ['적극투자형', '공격투자형']:
            user_type = "적극투자형"
        elif assets < 3000 and pension < 80:
            user_type = "위험취약형"
        else:
            user_type = "균형형"
        
        st.session_state.user_type = user_type
        
    except:
        st.session_state.user_type = "균형형"

# =================================
# 설문 결과 페이지
# =================================
def render_survey_result_page():
    render_header("금융 유형 결과")
    
    user_type = st.session_state.user_type or "균형형"
    
    # 유형별 설명
    type_descriptions = {
        "자산운용형": {
            "icon": "💼",
            "description": "투자 여력이 충분한 유형으로, 운용 전략 중심의 포트폴리오가 적합합니다.",
            "color": "#4F46E5"
        },
        "위험취약형": {
            "icon": "⚠️",
            "description": "재무 위험이 높은 유형입니다. 지출 관리와 복지 연계가 필요합니다.",
            "color": "#EF4444"
        },
        "균형형": {
            "icon": "⚖️",
            "description": "자산과 연금이 안정적인 편으로, 보수적인 전략이 적합합니다.",
            "color": "#10B981"
        },
        "적극투자형": {
            "icon": "🚀",
            "description": "수익을 위해 적극적인 투자를 선호하는 유형입니다.",
            "color": "#F59E0B"
        }
    }
    
    type_info = type_descriptions.get(user_type, type_descriptions["균형형"])
    
    st.markdown(f"""
    <div class="result-card" style="text-align: center; border-left: 5px solid {type_info['color']};">
        <div style="font-size: 48px; margin-bottom: 20px;">{type_info['icon']}</div>
        <h2 style="color: {type_info['color']}; margin-bottom: 15px;">{user_type}</h2>
        <p style="font-size: 18px; line-height: 1.6; color: #666;">{type_info['description']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 추천 액션
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("맞춤 상품 추천 보기", use_container_width=True):
            st.session_state.page = 'recommendation'
            st.rerun()
    
    with col2:
        if st.button("노후 시뮬레이션 보기", use_container_width=True):
            st.session_state.page = 'simulation'
            st.rerun()
    
    st.markdown('<div style="margin: 20px 0;"></div>', unsafe_allow_html=True)
    
    if st.button("← 메인으로 돌아가기", use_container_width=True):
        st.session_state.page = 'main'
        st.rerun()

# =================================
# 연금 계산 페이지
# =================================
def render_pension_input_page():
    render_header("연금 계산기")
    
    st.markdown("""
    <div style="text-align: center; font-size: 18px; margin: 30px 0; color: #666;">
        평균 월소득과 가입기간을 입력하시면<br>예상 연금액을 계산해드립니다.
    </div>
    """, unsafe_allow_html=True)
    
    monthly_income = st.number_input("평균 월소득 (만원)", min_value=0, value=300, step=10)
    pension_years = st.number_input("국민연금 가입기간 (년)", min_value=0, value=25, step=1)
    
    if st.button("연금 계산하기", use_container_width=True):
        estimated_pension = calculate_pension_estimate(monthly_income, pension_years)
        
        st.session_state.pension_result = {
            'monthly_income': monthly_income,
            'pension_years': pension_years,
            'estimated_pension': estimated_pension
        }
        
        st.session_state.page = 'pension_result'
        st.rerun()
    
    if st.button("← 메인으로", key="pension_back"):
        st.session_state.page = 'main'
        st.rerun()

def render_pension_result_page():
    render_header("연금 계산 결과")
    
    result = st.session_state.get('pension_result', {})
    estimated_pension = result.get('estimated_pension', 0)
    monthly_income = result.get('monthly_income', 0)
    pension_years = result.get('pension_years', 0)
    
    st.markdown(f"""
    <div class="result-card" style="text-align: center;">
        <h3 style="color: #4F46E5; margin-bottom: 20px;">💰 예상 월 연금액</h3>
        <div style="font-size: 36px; font-weight: bold; color: #1F2937; margin: 20px 0;">
            {estimated_pension:,.0f}만원
        </div>
        <div style="font-size: 16px; color: #666; margin-top: 15px;">
            월소득 {monthly_income:,.0f}만원 × 가입기간 {pension_years}년 기준
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 연금 유형 분류
    if estimated_pension >= 90:
        pension_type = "완전노령연금"
        description = "만 65세부터 감액 없이 정액 수령이 가능합니다."
    elif estimated_pension >= 60:
        pension_type = "조기노령연금"
        description = "만 60세부터 수령 가능하나 최대 30% 감액될 수 있습니다."
    else:
        pension_type = "감액노령연금"
        description = "일정 조건을 만족하지 못할 경우 감액되어 수령됩니다."
    
    st.info(f"**{pension_type}**: {description}")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("상품 추천 받기"):
            st.session_state.page = 'recommendation'
            st.rerun()
    
    with col2:
        if st.button("← 메인으로"):
            st.session_state.page = 'main'
            st.rerun()

# =================================
# 상품 추천 페이지
# =================================
def render_recommendation_page():
    render_header("맞춤 상품 추천")
    
    if not st.session_state.answers:
        st.warning("먼저 설문조사를 완료해주세요.")
        if st.button("설문조사 하기"):
            st.session_state.page = 'survey'
            st.rerun()
        return
    
    user_type = st.session_state.user_type or "균형형"
    
    st.markdown(f"""
    <div style="text-align: center; margin: 20px 0;">
        <h3>🎯 {user_type} 맞춤 추천</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # 추천 상품 생성
    recommendations = simple_recommend(st.session_state.answers)
    
    for i, product in enumerate(recommendations, 1):
        st.markdown(f"""
        <div class="product-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <h4 style="margin: 0; color: #1F2937;">🏆 {i}. {product['상품명']}</h4>
                <span style="background: #10B981; color: white; padding: 4px 8px; border-radius: 8px; font-size: 14px; font-weight: bold;">{product['예상수익률']}</span>
            </div>
            <div style="color: #666; margin-bottom: 8px;">
                <strong>리스크:</strong> {product['리스크']} | 
                <strong>최소투자금액:</strong> {product['최소투자금액']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # 추가 서비스 버튼
    col1, col2 = st.columns(2)
    with col1:
        if st.button("노후 시뮬레이션 보기"):
            st.session_state.page = 'simulation'
            st.rerun()
    
    with col2:
        if st.button("← 메인으로"):
            st.session_state.page = 'main'
            st.rerun()

# =================================
# 노후 시뮬레이션 페이지
# =================================
def render_simulation_page():
    render_header("노후 시뮬레이션")
    
    if not st.session_state.answers:
        st.warning("먼저 설문조사를 완료하시면 더 정확한 시뮬레이션이 가능합니다.")
        # 기본값으로 시뮬레이션
        current_age = 65
        current_assets = 5000
        monthly_income = 200
        monthly_expense = 150
    else:
        answers = st.session_state.answers
        current_age = int(answers.get('age', 65))
        current_assets = float(answers.get('assets', 5000))
        pension = float(answers.get('pension', 100))
        income = float(answers.get('income', 100))
        monthly_income = pension + income
        monthly_expense = float(answers.get('living_cost', 150))
    
    st.markdown("### 📊 현재 상황 분석")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("현재 나이", f"{current_age}세")
    with col2:
        st.metric("보유 자산", f"{current_assets:,.0f}만원")
    with col3:
        st.metric("월 순수익", f"{monthly_income - monthly_expense:,.0f}만원")
    
    # 간단한 시뮬레이션
    years_left = 100 - current_age
    total_needed = monthly_expense * 12 * years_left
    total_income = monthly_income * 12 * years_left
    total_available = current_assets + total_income
    
    st.markdown("### 📈 100세까지 생활비 시뮬레이션")
    
    if total_available >= total_needed:
        st.success(f"✅ 현재 자산과 소득으로 100세까지 안정적인 생활이 가능합니다!")
        surplus = total_available - total_needed
        st.info(f"💰 예상 잉여자금: {surplus:,.0f}만원")
    else:
        shortage = total_needed - total_available
        st.warning(f"⚠️ 100세까지 생활하려면 {shortage:,.0f}만원이 부족할 수 있습니다.")
        st.info("💡 추가 투자나 부업을 고려해보시기 바랍니다.")
    
    # 투자 시나리오
    st.markdown("### 💹 투자 수익률별 시나리오")
    
    scenarios = [
        {"name": "안전투자 (연 3%)", "rate": 0.03},
        {"name": "균형투자 (연 5%)", "rate": 0.05},
        {"name": "적극투자 (연 7%)", "rate": 0.07}
    ]
    
    for scenario in scenarios:
        # 복리 계산
        investment_return = current_assets * (1 + scenario["rate"]) ** years_left
        final_total = investment_return + total_income
        
        if final_total >= total_needed:
            st.success(f"✅ {scenario['name']}: {final_total:,.0f}만원 (충분)")
        else:
            st.error(f"❌ {scenario['name']}: {final_total:,.0f}만원 (부족)")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("상품 추천 받기"):
            st.session_state.page = 'recommendation'
            st.rerun()
    
    with col2:
        if st.button("← 메인으로"):
            st.session_state.page = 'main'
            st.rerun()

# =================================
# 메인 앱 실행
# =================================
def main():
    if st.session_state.page == 'main':
        render_main_page()
    elif st.session_state.page == 'survey':
        render_survey_page()
    elif st.session_state.page == 'survey_result':
        render_survey_result_page()
    elif st.session_state.page == 'pension_input':
        render_pension_input_page()
    elif st.session_state.page == 'pension_result':
        render_pension_result_page()
    elif st.session_state.page == 'recommendation':
        render_recommendation_page()
    elif st.session_state.page == 'simulation':
        render_simulation_page()
    elif st.session_state.page == 'phone_consultation':
        render_phone_consultation_page()

if __name__ == "__main__":
    main()
