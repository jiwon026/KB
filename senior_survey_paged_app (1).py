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
# FAISS 인덱싱 함수들
def build_index(X: np.ndarray):
    """FAISS 또는 sklearn으로 인덱스 구축"""
    if USE_FAISS and X.shape[0] > 0:
        try:
            dim = X.shape[1]
            index = faiss.IndexFlatL2(dim)
            index.add(X)
            return ('faiss', index)
        except:
            pass
    
    # sklearn fallback
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=min(10, X.shape[0]), metric='euclidean')
    nn.fit(X)
    return ('sklearn', nn)

def index_search(index_info, query: np.ndarray, k: int):
    """인덱스에서 k개 최근접 검색"""
    index_type, index_obj = index_info
    
    if index_type == 'faiss':
        distances, indices = index_obj.search(query, k)
        return distances[0], indices[0]
    else:  # sklearn
        distances, indices = index_obj.kneighbors(query, n_neighbors=k)
        return distances[0], indices[0]
        
# 페이지 설정
st.set_page_config(
    page_title="노후愛",
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

def preprocess_products(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """
    필요한 컬럼 표준화. 없으면 기본값 채우기.
    기대 컬럼:
      - 상품명, 구분, 예상수익률(연), 리스크, 최소투자금액, 투자기간(개월)
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "상품명","구분","예상수익률(연)","리스크","최소투자금액","투자기간(개월)"
        ])

    out = df.copy()
    out["구분"] = kind

    # 컬럼 없으면 기본값 생성
    if "상품명" not in out.columns:
        out["상품명"] = out.get("펀드명", out.index.astype(str)).astype(str)

    # 예상수익률(연) → 숫자(%기호 제거)
    if "예상수익률(연)" not in out.columns:
        out["예상수익률(연)"] = 3.0
    out["예상수익률(연)"] = (
        out["예상수익률(연)"]
        .astype(str).str.replace("%","", regex=False)
        .astype(float)
        .fillna(0.0)
    )

    if "리스크" not in out.columns:
        out["리스크"] = "중간"

    if "최소투자금액" not in out.columns:
        out["최소투자금액"] = 0
    out["최소투자금액"] = pd.to_numeric(out["최소투자금액"], errors="coerce").fillna(0).astype(float)

    if "투자기간(개월)" not in out.columns:
        out["투자기간(개월)"] = 12
    out["투자기간(개월)"] = pd.to_numeric(out["투자기간(개월)"], errors="coerce").fillna(12).astype(int)

    return out[["상품명","구분","예상수익률(연)","리스크","최소투자금액","투자기간(개월)"]]

def rule_based_filter(df: pd.DataFrame, cond: dict) -> pd.DataFrame:
    """
    - 최소투자금액 <= 투자금액
    - 투자기간(개월)이 사용자가 선택한 기간과 크게 어긋나지 않는 상품 우선(±12개월)
    - 리스크 매칭(대략적)
    """
    if df.empty:
        return df

    invest = float(cond.get("투자금액", 0) or 0)
    period = int(cond.get("투자기간", 12) or 12)
    risk  = cond.get("투자성향", "위험중립형")

    # 리스크 레벨 매칭
    def risk_ok(x):
        x = str(x)
        if risk == "안정형":
            return ("낮" in x) or ("보수" in x) or (x in ["낮음","안정형"])
        if risk == "공격형":
            return ("높" in x) or ("공격" in x) or (x in ["높음","공격형"])
        return True  # 위험중립형은 모두 허용

    df2 = df.copy()
    df2 = df2[df2["최소투자금액"] <= invest]

    # 기간 차이 계산 후 가중치 컬럼
    df2["기간차"] = (df2["투자기간(개월)"] - period).abs()
    df2 = df2[df2["기간차"] <= 12] if not df2.empty else df2

    if not df2.empty:
        df2 = df2[df2["리스크"].apply(risk_ok)]

    return df2.drop(columns=["기간차"], errors="ignore") if not df2.empty else df2



def get_custom_recommendations_from_csv(investment_amount, period, risk_level, target_monthly):
    """실제 CSV 데이터에서 조건에 맞는 상품 추천"""
    try:
        dep_raw = load_deposit_csv()
        fun_raw = load_fund_csv()

        dep = preprocess_products(dep_raw, "예·적금")
        fun = preprocess_products(fun_raw, "펀드")

        all_products = pd.concat([dep, fun], ignore_index=True)
        if all_products.empty:
            return []

        user_conditions = {
            '투자금액': float(investment_amount),
            '투자기간': int(period),
            '투자성향': risk_level,
            '목표월이자': float(target_monthly)
        }

        filtered = rule_based_filter(all_products, user_conditions)
        if filtered.empty:
            return []

        # 월 예상수익 계산(투자금액 × 연수익률 / 12)
        filtered = filtered.copy()
        filtered["월예상수익금(만원)"] = (
            user_conditions["투자금액"] * (filtered["예상수익률(연)"] / 100.0) / 12.0
        )

        # 점수: 목표월이자에 얼마나 근접한가
        filtered["추천점수"] = (100 - (filtered["월예상수익금(만원)"] - user_conditions["목표월이자"]).abs() * 2).clip(lower=0)

        # 정렬: 점수↓, 연수익률↓
        filtered = filtered.sort_values(["추천점수","예상수익률(연)"], ascending=False)

        result = []
        for _, row in filtered.head(5).iterrows():
            result.append({
                '상품명': row.get('상품명', '상품명 없음'),
                '구분': row.get('구분', '기타'),
                '월수령액': f"{row.get('월예상수익금(만원)', 0):.1f}만원",
                '연수익률': f"{row.get('예상수익률(연)', 0):.1f}%",
                '리스크': row.get('리스크', '중간'),
                '최소투자금액': f"{int(row.get('최소투자금액', 0))}만원",
                '투자기간': f"{int(row.get('투자기간(개월)', period))}개월",
                '추천점수': float(row.get('추천점수', 0))
            })
        return result

    except Exception as e:
        st.error(f"추천 시스템 오류: {e}")
        return get_fallback_recommendations(
            investment_amount=int(investment_amount),
            period=int(period),
            risk_level=risk_level,
            target_monthly=float(target_monthly)
        )
        
def get_fallback_recommendations(investment_amount, period, risk_level, target_monthly):
    """CSV 로딩 실패시 폴백 추천"""
    base_products = {
        '안정형': [
            {'상품명': 'KB 안심정기예금', '기본수익률': 3.2, '최소투자': 100},
            {'상품명': 'KB 시니어적금', '기본수익률': 3.5, '최소투자': 50},
            {'상품명': 'KB 연금저축예금', '기본수익률': 4.1, '최소투자': 300},
        ],
        '위험중립형': [
            {'상품명': 'KB 균형형펀드', '기본수익률': 5.5, '최소투자': 100},
            {'상품명': 'KB 혼합자산펀드', '기본수익률': 6.2, '최소투자': 200},
            {'상품명': 'KB 안정성장펀드', '기본수익률': 5.8, '최소투자': 300},
        ],
        '공격형': [
            {'상품명': 'KB 성장주펀드', '기본수익률': 8.1, '최소투자': 200},
            {'상품명': 'KB 테크성장펀드', '기본수익률': 9.3, '최소투자': 500},
            {'상품명': 'KB 글로벌성장펀드', '기본수익률': 7.8, '최소투자': 300},
        ]
    }
    
    products = base_products.get(risk_level, base_products['위험중립형'])
    result = []
    
    for product in products:
        if investment_amount >= product['최소투자']:
            annual_return = investment_amount * (product['기본수익률'] / 100)
            monthly_return = annual_return / 12
            
            result.append({
                '상품명': product['상품명'],
                '구분': '예·적금' if '예금' in product['상품명'] or '적금' in product['상품명'] else '펀드',
                '월수령액': f"{monthly_return:.1f}만원",
                '연수익률': f"{product['기본수익률']:.1f}%",
                '리스크': risk_level,
                '최소투자금액': f"{product['최소투자']}만원",
                '투자기간': f"{period}개월",
                '추천점수': max(0, 100 - abs(monthly_return - target_monthly) * 2)
            })
    
    return sorted(result, key=lambda x: x['추천점수'], reverse=True)[:3]

def _defaults_from_survey(answers: dict):
    """설문 답변에서 기본 추천 입력치(투자금액/기간/리스크/목표월이자)를 뽑아 UI에 프리필"""
    age    = int(float(answers.get('age', 65) or 65))
    assets = float(answers.get('assets', 5000) or 5000)
    income = float(answers.get('income', 200) or 200)
    risk   = str(answers.get('risk', '위험중립형') or '위험중립형')

    # 나이/자산으로 기본 투자금액/기간 가정
    if age >= 70:
        invest_amount = min(assets * 0.3, 3000)
        period = 12
    elif age >= 60:
        invest_amount = min(assets * 0.4, 5000)
        period = 24
    else:
        invest_amount = min(assets * 0.5, 8000)
        period = 36

    target_monthly = income * 0.1  # 소득의 10%를 목표 월이자(만원)로

    # 리스크 5단계 → 3단계 매핑
    risk_map = {
        '안정형':'안정형', '안정추구형':'안정형',
        '위험중립형':'위험중립형',
        '적극투자형':'공격형', '공격투자형':'공격형'
    }
    risk3 = risk_map.get(risk, '위험중립형')

    return {
        "investment_amount": int(round(invest_amount)),
        "period": int(period),
        "risk_level": risk3,
        "target_monthly": float(round(target_monthly, 1)),
    }


# 설문 기반 추천도 개선
def get_survey_based_recommendations(user_answers):
    """설문 결과를 바탕으로 한 추천 (CSV 데이터 활용)"""
    try:
        # 설문 답변을 추천 조건으로 변환
        age = int(user_answers.get('age', 65))
        assets = float(user_answers.get('assets', 5000))
        risk = user_answers.get('risk', '안정형')
        income = float(user_answers.get('income', 200))
        
        # 리스크 성향 매핑
        risk_mapping = {
            '안정형': '안정형',
            '안정추구형': '안정형', 
            '위험중립형': '위험중립형',
            '적극투자형': '공격형',
            '공격투자형': '공격형'
        }
        
        mapped_risk = risk_mapping.get(risk, '위험중립형')
        
        # 나이와 자산에 따른 추천 투자금액/기간 결정
        if age >= 70:
            invest_amount = min(assets * 0.3, 3000)  # 보수적
            invest_period = 12
        elif age >= 60:
            invest_amount = min(assets * 0.4, 5000)
            invest_period = 24
        else:
            invest_amount = min(assets * 0.5, 8000)
            invest_period = 36
            
        target_monthly = income * 0.1  # 소득의 10%를 목표 월수익
        
        # CSV 기반 추천 실행
        recommendations = get_custom_recommendations_from_csv(
            invest_amount, invest_period, mapped_risk, target_monthly
        )
        
        return recommendations
        
    except Exception as e:
        st.error(f"설문 기반 추천 오류: {e}")
        # 기본 추천으로 폴백
        return [
            {'상품명': 'KB 시니어 안심예금', '월수령액': '25만원', '연수익률': '3.2%', 
             '리스크': '낮음', '최소투자금액': '500만원', '구분': '예·적금'},
            {'상품명': 'KB 균형투자펀드', '월수령액': '42만원', '연수익률': '5.5%', 
             '리스크': '중간', '최소투자금액': '1000만원', '구분': '펀드'},
        ]

def render_custom_recommendation_result():
    render_header("맞춤 추천 결과")

    recs = st.session_state.get("custom_recommendations", [])
    cond = st.session_state.get("search_conditions", {})

    if not recs:
        st.warning("결과가 없습니다. 조건을 조정해 다시 시도해 주세요.")
        if st.button("← 조건 다시 입력"):
            st.session_state.page = 'custom_recommendation'
            st.rerun()
        return

    # 검색 조건 요약
    st.caption(
        f"검색 조건 · 투자금액 **{cond.get('investment_amount',0)}만원**, "
        f"기간 **{cond.get('period',0)}개월**, 리스크 **{cond.get('risk_level','-')}**, "
        f"목표 월이자 **{cond.get('target_monthly',0)}만원**"
    )
    st.caption("추천 소스: **CSV 기반**")

    # 카드 렌더
    for i, product in enumerate(recs, 1):
        badge = "최적" if product.get('추천점수',0) >= 90 else ("추천" if product.get('추천점수',0) >= 70 else "적합")
        st.markdown(f"""
        <div class="product-card" style="position: relative;">
            <div style="position: absolute; top: 15px; right: 15px;">
                <span style="background: #3B82F6; color: white; padding: 4px 8px; border-radius: 12px; font-size: 12px; font-weight: bold;">{badge}</span>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; margin-right:60px;">
                <h4 style="margin:0; color:#1F2937;">🏆 {i}. {product.get('상품명','-')}</h4>
                <span style="background:#10B981; color:white; padding:8px 12px; border-radius:8px; font-size:16px; font-weight:bold;">
                    {product.get('월수령액','-')}
                </span>
            </div>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; color:#666; font-size:14px;">
                <div><strong>구분:</strong> {product.get('구분','-')}</div>
                <div><strong>연수익률:</strong> {product.get('연수익률','-')}</div>
                <div><strong>리스크:</strong> {product.get('리스크','-')}</div>
                <div><strong>최소투자:</strong> {product.get('최소투자금액','-')}</div>
                <div><strong>투자기간:</strong> {product.get('투자기간','-')}</div>
                <div><strong>추천점수:</strong> {product.get('추천점수',0):.1f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("조건 바꿔 다시 추천"):
            st.session_state.page = 'custom_recommendation'
            st.rerun()
    with col2:
        if st.button("설문 기반으로 보기"):
            st.session_state.recommendation_mode = 'survey_based'
            st.session_state.page = 'recommendation_hub'
            st.rerun()
    with col3:
        if st.button("← 메인으로"):
            st.session_state.page = 'main'
            st.rerun()


# 맞춤 추천 입력 페이지 (업데이트)
def render_custom_recommendation_page():
    render_header("맞춤 투자 조건 입력")
    
    st.markdown("""
    <div style="text-align: center; margin: 20px 0; color: #666;">
        원하시는 투자 조건을 입력하시면<br>실제 금융상품 데이터에서 가장 적합한 상품을 추천해드립니다.
    </div>
    """, unsafe_allow_html=True)
    
    # CSV 데이터 로딩 상태 확인
    try:
        dep_raw = load_deposit_csv()
        fun_raw = load_fund_csv()
        data_status = f"✅ 상품 데이터 로딩 완료 (예·적금: {len(dep_raw)}개, 펀드: {len(fun_raw)}개)"
        st.success(data_status)
    except Exception as e:
        st.warning(f"⚠️ 상품 데이터 로딩 문제: {e} (기본 상품으로 추천)")
    
    # 투자 조건 입력
    col1, col2 = st.columns(2)
    
    with col1:
        investment_amount = st.number_input(
            "투자금액 (만원)", 
            min_value=50, 
            value=1000, 
            step=50,
            help="투자하실 금액을 입력해주세요"
        )
        
        risk_level = st.selectbox(
            "리스크 허용도",
            ["안정형", "위험중립형", "공격형"],
            help="투자 위험에 대한 성향을 선택해주세요"
        )
    
    with col2:
        period = st.selectbox(
            "투자 기간 (개월)",
            [6, 12, 24, 36],
            index=1,
            help="투자 유지 기간을 선택해주세요"
        )
        
        target_monthly = st.number_input(
            "목표 월이자 (만원)",
            min_value=0,
            value=30,
            step=5,
            help="매월 받고 싶은 이자 금액을 입력해주세요"
        )
    
    st.markdown('<div style="margin: 30px 0;"></div>', unsafe_allow_html=True)
    
    if st.button("🔍 맞춤 상품 찾기", use_container_width=True):
        with st.spinner('실제 금융상품 데이터에서 최적 상품을 찾는 중...'):
            # CSV 데이터 기반 추천
            recommendations = get_custom_recommendations_from_csv(
                investment_amount, period, risk_level, target_monthly
            )
        
        if recommendations:
            st.session_state.custom_recommendations = recommendations
            st.session_state.search_conditions = {
                'investment_amount': investment_amount,
                'period': period, 
                'risk_level': risk_level,
                'target_monthly': target_monthly
            }
            st.session_state.page = 'custom_recommendation_result'
            st.rerun()
        else:
            st.error("조건에 맞는 상품을 찾을 수 없습니다. 조건을 다시 설정해보세요.")
    
    if st.button("← 메인으로", key="custom_rec_back"):
        st.session_state.page = 'main'
        st.rerun()


# render_recommendation_page도 업데이트 (설문 기반 추천 개선)
# render_recommendation_page 함수 수정 (1100라인 근처)
def render_recommendation_hub():
    render_header("맞춤 상품 추천")
    
    st.markdown("""
    <div style="text-align: center; margin: 20px 0; color: #666;">
        원하시는 추천 방식을 선택해주세요
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    # 방식 1: 설문 기반 추천
    with col1:
        if st.button("📋 설문 기반 추천\n(간편 방식)", use_container_width=True):
            if not st.session_state.answers:
                st.warning("먼저 설문조사를 완료해주세요.")
                if st.button("설문조사 하러 가기"):
                    st.session_state.page = 'survey'
                    st.session_state.question_step = 1
                    st.session_state.answers = {}
                    st.rerun()
                return
            
            st.session_state.recommendation_mode = 'survey_based'
            st.rerun()
    
    # 방식 2: 맞춤 조건 입력
    with col2:
        if st.button("🎯 맞춤 조건 입력\n(상세 방식)", use_container_width=True):
            st.session_state.page = 'custom_recommendation'
            st.rerun()
    
    # 설문 기반 추천 결과 표시
    if st.session_state.get('recommendation_mode') == 'survey_based':
        user_type = st.session_state.user_type or "균형형"
        
        st.markdown(f"""
        <div style="text-align: center; margin: 20px 0;">
            <h3>🎯 {user_type} 맞춤 추천 (설문 기반)</h3>
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner('실제 상품 데이터에서 최적 상품을 분석 중...'):
            # CSV 기반 설문 추천 - simple_recommend 대신 get_survey_based_recommendations 사용
            recommendations = get_survey_based_recommendations(st.session_state.answers)
        
        if not recommendations:
            st.error("추천 가능한 상품을 찾을 수 없습니다.")
            return
            
        for i, product in enumerate(recommendations, 1):
            # 추천점수에 따른 배지
            score = product.get('추천점수', 0)
            if score >= 90:
                badge_color, badge_text = "#10B981", "최적"
            elif score >= 70:
                badge_color, badge_text = "#3B82F6", "추천"  
            else:
                badge_color, badge_text = "#F59E0B", "적합"
            
            st.markdown(f"""
            <div class="product-card" style="position: relative;">
                <div style="position: absolute; top: 15px; right: 15px;">
                    <span style="background: {badge_color}; color: white; padding: 4px 8px; border-radius: 12px; font-size: 12px; font-weight: bold;">{badge_text}</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; margin-right: 60px;">
                    <h4 style="margin: 0; color: #1F2937;">🏆 {i}. {product['상품명']}</h4>
                    <span style="background: #10B981; color: white; padding: 8px 12px; border-radius: 8px; font-size: 16px; font-weight: bold;">{product['월수령액']}</span>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; color: #666; font-size: 14px;">
                    <div><strong>구분:</strong> {product['구분']}</div>
                    <div><strong>연수익률:</strong> {product['연수익률']}</div>
                    <div><strong>리스크:</strong> {product['리스크']}</div>
                    <div><strong>최소투자:</strong> {product['최소투자금액']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # 하단 버튼들
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎯 조건 입력해서 다시 추천받기"):
                st.session_state.page = 'custom_recommendation'
                st.session_state.recommendation_mode = None
                st.rerun()
        
        with col2:
            if st.button("📞 전문가 상담받기"):
                st.session_state.page = 'phone_consultation'
                st.rerun()
        
        if st.button("← 추천 방식 다시 선택"):
            st.session_state.recommendation_mode = None
            st.rerun()
    
    # 하단 공통 서비스 버튼들
    if not st.session_state.get('recommendation_mode'):
        st.markdown('<div style="margin: 40px 0;"></div>', unsafe_allow_html=True)
        st.markdown("### 🔗 다른 서비스도 이용해보세요")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 노후 시뮬레이션 보기"):
                st.session_state.page = 'simulation'
                st.rerun()
        
        with col2:
            if st.button("💰 연금 계산하기"):
                st.session_state.page = 'pension_input'
                st.rerun()
        
        if st.button("← 메인으로 돌아가기"):
            st.session_state.page = 'main'
            st.session_state.recommendation_mode = None
            st.rerun()
            
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
def render_header(title="노후愛"):
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
        if st.session_state.get('user_type'):           # 설문 완료 → 결과 페이지로
            st.session_state.page = 'survey_result'
        else:                                           # 미완료 → 설문 시작
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
            if st.session_state.get('answers'):         # 설문 값 있으면 합친 화면으로
                st.session_state.page = 'survey_plus_custom'
            else:                                       # 없으면 설문부터
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
# 전화 상담 페이지 함수 수정
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
    
    # 상담센터 정보를 일반 마크다운으로 변경
    st.markdown("### 📞 KB 시니어 연금 상담센터")
    
    # 전화번호 표시
    st.markdown("""
    **상담 전화번호:** 
    ## 1588-9999
    """)
    
    # 상담 시간
    st.markdown("""
    **상담 시간:**
    - 평일: 오전 9시 ~ 오후 6시
    - 토요일: 오전 9시 ~ 오후 1시  
    - 일요일 및 공휴일 휴무
    """)
    
    # 상담 가능 내용
    st.markdown("""
    **상담 가능 내용:**
    - 🏦 연금 상품 상세 안내
    - 📝 가입 절차 및 필요 서류
    - 💰 수령 방법 및 시기
    - 💸 세제 혜택 안내
    - 📊 개인 맞춤 포트폴리오 구성
    """)
    
    st.markdown("---")
    
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

def _to_int(x, default):
    try:
        if x in (None, ""): return default
        return int(float(str(x).replace(",", "").strip()))
    except Exception:
        return default

def _to_float(x, default):
    try:
        if x in (None, ""): return default
        return float(str(x).replace(",", "").strip())
    except Exception:
        return default

def analyze_user_type():
    """문자/빈값/콤마 포함 입력도 안전하게 파싱해서 유형 분류"""
    a = st.session_state.get('answers', {})

    age         = _to_int(a.get('age'), 65)
    assets      = _to_float(a.get('assets'), 5000)
    pension     = _to_float(a.get('pension'), 100)
    income      = _to_float(a.get('income'), 200)
    living_cost = _to_float(a.get('living_cost'), 150)
    risk        = (a.get('risk') or '안정형').strip()

    # 간단 분류 로직(원래 쓰던 기준 그대로)
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
            st.session_state.page = 'survey_plus_custom'   # ← 여기!
            st.rerun()
    
    with col2:
        if st.button("노후 시뮬레이션 보기", use_container_width=True):
            st.session_state.page = 'simulation'
            st.rerun()
    
    st.markdown('<div style="margin: 20px 0;"></div>', unsafe_allow_html=True)
    
    if st.button("← 메인으로 돌아가기", use_container_width=True):
        st.session_state.page = 'main'
        st.rerun()
def render_survey_plus_custom_page():
    render_header("설문 + 맞춤 조건으로 추천")

    if not st.session_state.answers:
        st.warning("먼저 설문을 완료해주세요.")
        if st.button("설문 하러 가기"):
            st.session_state.page = 'survey'
            st.rerun()
        return

    # 설문에서 기본값 추출
    defaults = _defaults_from_survey(st.session_state.answers)

    col1, col2 = st.columns(2)
    with col1:
        investment_amount = st.number_input(
            "투자금액 (만원)",
            min_value=10, step=10,
            value=int(defaults["investment_amount"])
        )
        risk_level = st.selectbox(
            "리스크 허용도",
            ["안정형","위험중립형","공격형"],
            index=["안정형","위험중립형","공격형"].index(defaults["risk_level"])
        )
    with col2:
        period = st.selectbox(
            "투자 기간 (개월)",
            [6,12,24,36],
            index=[6,12,24,36].index(int(defaults["period"]))
        )
        target_monthly = st.number_input(
            "목표 월이자 (만원)",
            min_value=0.0, step=1.0,
            value=float(defaults["target_monthly"])
        )

    st.markdown('<div style="margin: 8px 0 16px 0;"></div>', unsafe_allow_html=True)

    if st.button("🔍 추천 받기", use_container_width=True):
        with st.spinner("CSV에서 조건에 맞는 상품을 찾는 중..."):
            recs = get_custom_recommendations_from_csv(
                investment_amount, period, risk_level, target_monthly
            )
        if not recs:
            # 비었으면 폴백 사용
            recs = get_fallback_recommendations(investment_amount, period, risk_level, target_monthly)

        st.session_state.spc_last_input = {
            "investment_amount": investment_amount,
            "period": period,
            "risk_level": risk_level,
            "target_monthly": target_monthly,
        }
        st.session_state.spc_recs = recs
        st.rerun()

    # 결과 표시
    if "spc_recs" in st.session_state:
        cond = st.session_state.get("spc_last_input", {})
        st.caption(
            f"검색 조건 · 투자금액 **{cond.get('investment_amount',0)}만원**, "
            f"기간 **{cond.get('period',0)}개월**, 리스크 **{cond.get('risk_level','-')}**, "
            f"목표 월이자 **{cond.get('target_monthly',0)}만원** · 소스: **CSV 기반**"
        )
        for i, p in enumerate(st.session_state.spc_recs, 1):
            st.markdown(f"""
            <div class="product-card">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
                <h4 style="margin:0;color:#1F2937;">🏆 {i}. {p.get('상품명','-')}</h4>
                <span style="background:#10B981;color:#fff;padding:8px 12px;border-radius:8px;font-weight:700;">
                  {p.get('월수령액','-')}
                </span>
              </div>
              <div style="color:#666;font-size:14px;display:grid;grid-template-columns:1fr 1fr;gap:8px;">
                <div><strong>구분:</strong> {p.get('구분','-')}</div>
                <div><strong>연수익률:</strong> {p.get('연수익률','-')}</div>
                <div><strong>리스크:</strong> {p.get('리스크','-')}</div>
                <div><strong>최소투자:</strong> {p.get('최소투자금액','-')}</div>
                <div><strong>투자기간:</strong> {p.get('투자기간','-')}</div>
                <div><strong>추천점수:</strong> {p.get('추천점수',0):.1f}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("조건 바꿔 다시 추천"):
                st.session_state.pop("spc_recs", None)
                st.rerun()
        with c2:
            if st.button("노후 시뮬레이션으로"):
                st.session_state.page = 'simulation'
                st.rerun()
        with c3:
            if st.button("메인으로"):
                st.session_state.page = 'main'
                st.rerun()

# =================================
# 연금 계산 페이지
# =================================
def calculate_pension_estimate(monthly_income: float, pension_years: int) -> float:
    """
    매우 단순한 추정식: 과거 평균소득의 일부 × 가입연수 보정
    """
    accrual = min(max(pension_years, 0), 40) / 40.0   # 0~1
    base_ratio = 0.45                                  # 임의 계수(조정 가능)
    est = monthly_income * base_ratio * accrual
    return round(est, 1)


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
def simple_recommend(answers: dict):
    """
    설문 답변을 이용해 CSV 기반 추천을 호출하고,
    비어있으면 폴백 추천을 돌려주는 래퍼.
    """
    # 기본값/파싱
    age = int(answers.get('age', 65) or 65)
    assets = float(answers.get('assets', 5000) or 5000)
    risk = answers.get('risk', '위험중립형') or '위험중립형'
    income = float(answers.get('income', 200) or 200)

    # 추천 입력값 추정(설문 기반 함수 로직과 일치)
    if age >= 70:
        invest_amount = min(assets * 0.3, 3000)
        period = 12
    elif age >= 60:
        invest_amount = min(assets * 0.4, 5000)
        period = 24
    else:
        invest_amount = min(assets * 0.5, 8000)
        period = 36
    target_monthly = income * 0.1

    # CSV 기반 추천 시도
    recs = get_custom_recommendations_from_csv(
        investment_amount=invest_amount,
        period=period,
        risk_level=risk if risk in ["안정형","위험중립형","공격형"] else (
            "안정형" if "안정" in risk else "공격형" if "공격" in risk or "적극" in risk else "위험중립형"
        ),
        target_monthly=target_monthly
    )
    if recs:
        return recs

    # 폴백
    return get_fallback_recommendations(
        investment_amount=int(invest_amount),
        period=int(period),
        risk_level=risk if risk in ["안정형","위험중립형","공격형"] else "위험중립형",
        target_monthly=float(target_monthly)
    )


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
                <span style="background: #10B981; color: white; padding: 8px 12px; border-radius: 8px; font-size: 16px; font-weight: bold;">{product['월수령액']}</span>
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
    elif st.session_state.page == 'survey_plus_custom':   # ← 추가
        render_survey_plus_custom_page()
    elif st.session_state.page == 'pension_input':
        render_pension_input_page()
    elif st.session_state.page == 'pension_result':
        render_pension_result_page()
    elif st.session_state.page == 'recommendation':
        render_recommendation_page()  # (원하면 유지/삭제 자유)
    elif st.session_state.page == 'simulation':
        render_simulation_page()
    elif st.session_state.page == 'phone_consultation':
        render_phone_consultation_page()



if __name__ == "__main__":
    main()
