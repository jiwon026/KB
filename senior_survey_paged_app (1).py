# app.py
import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# =========================
# FAISS (옵션)
# =========================
USE_FAISS = True
try:
    import faiss  # pip install faiss-cpu
except Exception:
    USE_FAISS = False
    from sklearn.neighbors import NearestNeighbors  # noqa: F401


# =================================
# 공통 유틸 (인덱스 빌드/검색)
# =================================
def build_index(X: np.ndarray):
    """FAISS 우선, 실패 시 sklearn NN"""
    if X.size == 0:
        return ('none', None)
    if USE_FAISS:
        try:
            idx = faiss.IndexFlatL2(X.shape[1])
            idx.add(X.astype('float32'))
            return ('faiss', idx)
        except Exception:
            pass
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=min(10, len(X)), metric='euclidean')
    nn.fit(X)
    return ('sklearn', nn)

def index_search(index_info, query: np.ndarray, k: int):
    typ, obj = index_info
    if typ == 'faiss':
        D, I = obj.search(query.astype('float32'), k)
        return D[0], I[0]
    elif typ == 'sklearn':
        D, I = obj.kneighbors(query, n_neighbors=k)
        return D[0], I[0]
    return [], []


# =========================
# 페이지 & 스타일
# =========================
st.set_page_config(
    page_title="노후愛",
    page_icon="🏦",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .main-header { text-align:center; padding:20px 0; margin-bottom:30px; background:#f8f9fa; border-radius:15px; }
    .kb-logo { display:flex; align-items:center; justify-content:center; font-size:36px; font-weight:bold; margin-bottom:15px; }
    .kb-star { color:#FFB800; margin-right:8px; }
    .kb-text { color:#666; margin-right:15px; }
    .elderly-emoji { font-size:48px; margin-left:10px; }
    .title { font-size:24px; font-weight:bold; color:#333; margin-top:15px; }
    .stApp { max-width: 420px; margin:0 auto; background:#f8f9fa; padding:20px; }

    .stButton > button {
        width:100% !important; height:72px !important; border-radius:20px !important;
        font-size:18px !important; font-weight:bold !important; border:none !important;
        box-shadow:0 2px 8px rgba(0,0,0,0.1) !important; transition:all .2s ease !important; white-space:pre-line !important;
    }
    .stButton > button:hover { transform:translateY(-1px) !important; box-shadow:0 4px 12px rgba(0,0,0,0.15) !important; }

    .result-card { background:#fff; padding:20px; border-radius:15px; box-shadow:0 4px 15px rgba(0,0,0,0.1); margin:15px 0; }
    .product-card { border:2px solid #E5E7EB; border-radius:15px; padding:15px; margin:10px 0; background:#fff; }
    .product-card:hover { border-color:#3B82F6; box-shadow:0 4px 15px rgba(59,130,246,0.1); }
    .consultation-info { background:linear-gradient(135deg,#667eea 0%,#764ba2 100%); color:#fff; padding:20px; border-radius:15px; margin:20px 0; }
</style>
""", unsafe_allow_html=True)


# =========================
# 경로/파일
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
MODELS_DIR = BASE_DIR
DEPOSIT_CSV = "금융상품_3개_통합본.csv"
FUND_CSV    = "펀드_병합본.csv"


# =========================
# 로딩 (캐시)
# =========================
@st.cache_resource
def load_models():
    def safe_load(name):
        p = os.path.join(MODELS_DIR, name)
        if not os.path.exists(p): return None
        try:
            return joblib.load(p)
        except Exception as e:
            st.warning(f"{name} 로드 실패: {e}")
            return None
    return (
        safe_load("tabnet_model.pkl"),
        safe_load("label_encoder.pkl"),
        safe_load("reg_model.pkl"),
        safe_load("type_model.pkl"),
    )

@st.cache_data
def load_deposit_csv():
    p = os.path.join(BASE_DIR, DEPOSIT_CSV)
    if not os.path.exists(p): return pd.DataFrame()
    for enc in ("utf-8-sig", "cp949"):
        try: return pd.read_csv(p, encoding=enc)
        except Exception: pass
    return pd.read_csv(p)

@st.cache_data
def load_fund_csv():
    p = os.path.join(BASE_DIR, FUND_CSV)
    if not os.path.exists(p): return pd.DataFrame()
    for enc in ("utf-8-sig", "cp949"):
        try: return pd.read_csv(p, encoding=enc)
        except Exception: pass
    return pd.read_csv(p)

survey_model, survey_encoder, reg_model, type_model = load_models()


# =========================
# 유틸: 안전 파싱
# =========================
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


# =========================
# (참고코드 이식) CSV 전처리 & 규칙 필터 & 랭킹 추천
# =========================
def preprocess_products(df: pd.DataFrame, group_name: str = "") -> pd.DataFrame:
    """두 CSV를 공통 스키마로 통일."""
    if df is None or df.empty:
        return pd.DataFrame()
    np.random.seed(42)
    out = df.copy()
    out.columns = out.columns.str.strip()

    # 상품명 추출
    if '상품명' in out.columns:
        names = out['상품명'].fillna('무명상품').astype(str)
    elif '펀드명' in out.columns:
        names = out['펀드명'].fillna('무명상품').astype(str)
    elif '출처파일명' in out.columns:
        names = out['출처파일명'].apply(lambda x: str(x).split('.')[0] if pd.notnull(x) else '무명상품')
    else:
        names = [f"무명상품_{i}" for i in range(len(out))]

    # 최소 투자금액
    if '최소가입금액' in out.columns:
        min_invest = pd.to_numeric(out['최소가입금액'], errors='coerce')
        miss = min_invest.isna()
        if miss.any():
            min_invest.loc[miss] = np.random.randint(100, 1000, miss.sum())
    elif '최고한도' in out.columns:
        min_invest = pd.to_numeric(out['최고한도'], errors='coerce').fillna(0)
        zero_mask = (min_invest == 0)
        if zero_mask.any():
            min_invest.loc[zero_mask] = np.random.randint(100, 1000, zero_mask.sum())
    else:
        min_invest = pd.Series(np.random.randint(100, 1000, len(out)), index=out.index)

    # 예상수익률 (소수, 예: 0.052)
    cand_cols = [c for c in out.columns if any(k in c for k in ["기본금리","이자율","세전","%","수익률","수익"])]
    rate_col = cand_cols[0] if cand_cols else None
    if rate_col:
        raw = (out[rate_col].astype(str).str.replace(",", "", regex=False)
               .str.extract(r"([\d\.]+)")[0])
        est = pd.to_numeric(raw, errors="coerce")
        rand = pd.Series(np.random.uniform(1.0, 8.0, len(out)), index=out.index)
        est_return = (est.fillna(rand) / 100.0).astype(float).round(4)
    else:
        low, high = (0.01, 0.08) if group_name != "펀드" else (0.03, 0.15)
        est_return = pd.Series(np.round(np.random.uniform(low, high, len(out)), 4), index=out.index)

    # 리스크
    if '위험등급' in out.columns:
        rr = out['위험등급'].astype(str)
        risk = rr.apply(lambda x: '높음' if ('5' in x or '4' in x) else ('중간' if '3' in x else '낮음'))
    else:
        if group_name == "펀드":
            risk = pd.Series(np.random.choice(['낮음','중간','높음'], len(out), p=[0.2,0.4,0.4]), index=out.index)
        else:
            risk = pd.Series(np.random.choice(['낮음','중간','높음'], len(out), p=[0.6,0.3,0.1]), index=out.index)

    duration = pd.Series(np.random.choice([6,12,24,36], len(out)), index=out.index)
    profile  = pd.Series(np.random.choice(['안정형','위험중립형','공격형'], len(out)), index=out.index)

    ret = pd.DataFrame({
        '구분': group_name if group_name else '기타',
        '상품명': names,
        '최소투자금액': min_invest.astype(int),
        '예상수익률': est_return,          # 0.05
        '리스크': risk,
        '권장투자기간': duration,
        '투자성향': profile
    })
    ret = ret[ret['상품명'] != '무명상품'].drop_duplicates(subset=['상품명']).reset_index(drop=True)
    return ret

def rule_based_filter(df: pd.DataFrame, user: dict) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
    risk_choice = (user.get('투자성향') or '위험중립형')
    invest_amt  = int(user.get('투자금액', 0) or 0)
    invest_per  = int(user.get('투자기간', 0) or 0)
    risk_pref_map = {
        '안정형': ['낮음','중간'],
        '위험중립형': ['중간','낮음','높음'],
        '공격형': ['높음','중간']
    }
    allowed = risk_pref_map.get(risk_choice, ['낮음','중간','높음'])
    f = df[
        (pd.to_numeric(df['최소투자금액'], errors='coerce').fillna(10**9) <= invest_amt) &
        (pd.to_numeric(df['권장투자기간'], errors='coerce').fillna(10**9) <= invest_per) &
        (df['리스크'].isin(allowed))
    ]
    return f.sort_values('예상수익률', ascending=False).head(500).reset_index(drop=True)

def _get_feature_vector(df: pd.DataFrame) -> np.ndarray:
    return np.vstack([
        df['최소투자금액'].astype(float) / 1000.0,
        df['예상수익률'].astype(float) * 100.0,
        df['권장투자기간'].astype(float) / 12.0
    ]).T.astype('float32')

def _get_user_vector(user: dict) -> np.ndarray:
    return np.array([
        user['투자금액'] / 1000.0,
        user['목표월이자'],
        user['투자기간'] / 12.0
    ], dtype='float32').reshape(1, -1)

def _add_explain(df: pd.DataFrame, user: dict) -> pd.DataFrame:
    out = df.copy()
    out['월예상수익금(만원)'] = (out['예상수익률'].astype(float) * user['투자금액'] / 12.0).round(1)
    out['투자기간(개월)'] = out['권장투자기간'].astype(int)
    out['예상수익률(연)'] = (out['예상수익률'].astype(float) * 100).round(2).astype(str) + '%'
    return out[['구분','상품명','월예상수익금(만원)','예상수익률','예상수익률(연)','리스크','투자기간(개월)','최소투자금액','투자성향']]

def recommend_fallback_split(user: dict) -> pd.DataFrame:
    dep_raw = load_deposit_csv()
    fun_raw = load_fund_csv()
    dep = preprocess_products(dep_raw, "예·적금")
    fun = preprocess_products(fun_raw, "펀드")
    dep_f = rule_based_filter(dep, user)
    fun_f = rule_based_filter(fun, user)

    if dep_f.empty and fun_f.empty:
        return pd.DataFrame({'메시지': ['조건에 맞는 상품이 없어요 😢']})

    # 예·적금 2개
    if not dep_f.empty:
        Xd = _get_feature_vector(dep_f)
        idxd = build_index(Xd)
        _, idd = index_search(idxd, _get_user_vector(user), min(2, len(dep_f)))
        rec_dep = dep_f.iloc[idd].copy().head(2) if hasattr(idd, '__len__') else dep_f.head(2)
    else:
        rec_dep = pd.DataFrame(columns=dep_f.columns)

    # 펀드 1개
    if not fun_f.empty:
        Xf = _get_feature_vector(fun_f)
        idxf = build_index(Xf)
        _, idf = index_search(idxf, _get_user_vector(user), min(1, len(fun_f)))
        rec_fun = fun_f.iloc[idf].copy().head(1) if hasattr(idf, '__len__') else fun_f.head(1)
    else:
        rec_fun = pd.DataFrame(columns=fun_f.columns)

    out = pd.concat([rec_dep, rec_fun], ignore_index=True)
    out = out.drop_duplicates(subset=['상품명']).reset_index(drop=True)
    return _add_explain(out, user)


# =========================
# [NEW] 노후 시뮬레이션 유틸 (참고코드 이식)
# =========================
def retirement_simulation(current_age, end_age, current_assets, monthly_income, monthly_expense,
                          inflation_rate=0.03, investment_return=0.02):
    asset = float(current_assets)
    yearly_log = []
    expense = float(monthly_expense)
    depletion_age = None

    for age in range(int(current_age), int(end_age) + 1):
        annual_income = float(monthly_income) * 12
        annual_expense = float(expense) * 12
        delta = annual_income - annual_expense
        asset += delta
        if asset > 0:
            asset *= (1 + float(investment_return))

        yearly_log.append({
            "나이": age,
            "수입": round(annual_income),
            "지출": round(annual_expense),
            "증감": round(delta),
            "잔액": round(asset)
        })

        if asset <= 0 and depletion_age is None:
            depletion_age = age
            break

        expense *= (1 + float(inflation_rate))

    return yearly_log, depletion_age

def simulate_with_financial_product(current_age, end_age, current_assets, monthly_income, monthly_expense,
                                    invest_return=0.05):
    return retirement_simulation(current_age, end_age, current_assets, monthly_income, monthly_expense,
                                 inflation_rate=0.03, investment_return=invest_return)

def get_invest_return_from_risk(risk_level: str) -> float:
    if risk_level in ["안정형", "안정추구형"]:
        return 0.03
    if risk_level in ["위험중립형"]:
        return 0.05
    if risk_level in ["적극투자형", "공격투자형", "공격형"]:
        return 0.07
    return 0.05

def recommend_reason_from_simulation(depletion_age, current_age, current_assets,
                                     monthly_income, monthly_expense, risk_level: str):
    surplus = monthly_income - monthly_expense
    if depletion_age:
        if surplus <= 0:
            return f"{depletion_age}세에 자산 고갈 예상 · 현금흐름 보강이 시급합니다."
        if current_assets < 10000:
            return f"{depletion_age}세 자산 고갈 위험 · 절세형/분산형 상품으로 수익률 제고가 필요합니다."
        return f"{depletion_age}세 자산 고갈 위험 · 위험도('{risk_level}')에 맞는 수익원 다변화가 필요합니다."
    if current_assets >= 20000 and surplus > 0:
        return f"자산/현금흐름이 양호합니다 · '{risk_level}'에 맞춘 분산투자로 실질가치(물가 3%) 방어를 권장합니다."
    return "지출 구조를 점검하고 비과세/저비용 상품으로 실질 수익률을 높이세요."


# =========================
# 타입 설명(결과 카드용)
# =========================
RISK_STYLE_DESCRIPTIONS = {
    "안정형": "자산/연금 비율이 안정적이고 원금 보전을 선호해요. 예·적금과 초저위험 상품 위주가 좋아요.",
    "안정추구형": "수익과 안정의 균형을 중시해요. 예·적금 + 초저위험 펀드를 소폭 섞는 구성이 적합해요.",
    "위험중립형": "위험/수익을 균형 있게 받아들여요. 채권형·혼합형과 적금을 혼합하면 좋아요.",
    "적극투자형": "수익을 위해 변동성을 일정 수준 허용해요. 혼합형/주식형 비중을 조금 더 높여요.",
    "공격투자형": "높은 수익을 위해 변동성 감내도가 높아요. 주식형·테마형 등 성장지향 상품을 고려해요.",
    "위험취약형": "손실 회피 성향이 매우 큽니다. 원금 보전이 최우선이며 예·적금, MMF, 초저위험 채권형 위주가 적합합니다."
}
TABNET_TYPE_DESCRIPTIONS = {
    "자산운용형": "💼 투자 여력이 충분한 유형으로, 운용 전략 중심의 포트폴리오가 적합합니다.",
    "위험취약형": "⚠️ 재무 위험이 높은 유형입니다. 지출 관리와 복지 연계가 필요합니다.",
    "균형형": "⚖️ 자산과 연금이 안정적인 편으로, 보수적인 전략이 적합합니다.",
    "고소비형": "💳 소비가 많은 유형으로 절세 전략 및 예산 재조정이 필요합니다.",
    "자산의존형": "🏦 연금보다는 자산에 의존도가 높으며, 자산 관리 전략이 중요합니다.",
    "연금의존형": "📥 자산보다 연금에 의존하는 경향이 강한 유형입니다.",
}
DEFAULT_DISPLAY_TYPE = "균형형"


# =========================
# 세션 초기화
# =========================
ss = st.session_state
ss.setdefault('page', 'main')
ss.setdefault('question_step', 1)
ss.setdefault('answers', {})
ss.setdefault('user_type', None)
# 추천/시뮬 공유 상태
ss.setdefault('rec_df', pd.DataFrame())
ss.setdefault('display_type', DEFAULT_DISPLAY_TYPE)
ss.setdefault('risk_choice', '위험중립형')


# =========================
# 공통 헤더
# =========================
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


# =========================
# 메인
# =========================
def render_main_page():
    render_header()

    if st.button("내 금융유형\n보기", key="financial_type", use_container_width=True):
        if ss.get('user_type'):
            ss.page = 'survey_result'
        else:
            ss.page = 'survey'; ss.question_step = 1; ss.answers = {}
        st.rerun()

    st.markdown('<div style="margin: 15px 0;"></div>', unsafe_allow_html=True)

    if st.button("연금\n계산하기", key="pension_calc", use_container_width=True):
        ss.page = 'pension_input'; st.rerun()

    st.markdown('<div style="margin: 20px 0;"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("노후\n시뮬레이션", key="simulation", use_container_width=True):
            ss.page = 'simulation'; st.rerun()
    with col2:
        if st.button("맞춤 상품\n추천", key="recommendation", use_container_width=True):
            if ss.get('answers'):
                ss.page = 'survey_plus_custom'
            else:
                ss.page = 'survey'; ss.question_step = 1; ss.answers = {}
            st.rerun()

    st.markdown('<div style="margin: 15px 0;"></div>', unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        if st.button("설문\n다시하기", key="survey_reset", use_container_width=True):
            ss.page = 'survey'; ss.question_step = 1; ss.answers = {}; ss.user_type = None; st.rerun()
    with col4:
        if st.button("📞 전화\n상담", key="phone_consultation", use_container_width=True):
            ss.page = 'phone_consultation'; st.rerun()


# =========================
# 전화 상담
# =========================
def render_phone_consultation_page():
    render_header("전화 상담")
    st.markdown("""
    <div class="consultation-info">
      <div style="text-align:center; margin-bottom:20px;">
        <h2 style="margin:0; color:#fff;">📞 전문 상담사와 1:1 상담</h2>
        <p style="margin:10px 0 0 0; font-size:16px; opacity:.9;">복잡한 연금 제도, 전문가가 쉽게 설명해드립니다</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📞 KB 시니어 연금 상담센터")
    st.markdown("**상담 전화번호:**  \n## 1588-9999")
    st.markdown("**상담 시간:**\n- 평일: 9:00~18:00\n- 토요일: 9:00~13:00\n- 일/공휴일 휴무")

    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("성함 *", placeholder="홍길동")
        ctype = st.selectbox("상담 유형 *", ["선택해주세요","연금 상품 문의","가입 절차 문의","수령 방법 상담","세제 혜택 문의","기타"])
    with col2:
        phone = st.text_input("연락처 *", placeholder="010-1234-5678")
        pref  = st.selectbox("희망 상담 시간", ["상관없음","오전 (9-12시)","오후 (1-3시)","늦은 오후 (3-6시)"])
    inquiry = st.text_area("문의 내용", placeholder="궁금한 점을 자유롭게 적어주세요.", height=100)

    if st.button("📞 상담 신청하기", use_container_width=True):
        if name and phone and ctype != "선택해주세요":
            st.balloons()
            st.success(f"✅ 상담 신청 완료!\n\n**신청자:** {name}\n\n**연락처:** {phone}\n\n**상담 유형:** {ctype}\n\n**희망 시간:** {pref}")
            ss.consultation_requested = {'name':name,'phone':phone,'type':ctype,'time':pref,'inquiry':inquiry}
        else:
            st.error("⚠️ 필수 항목(*)을 모두 입력해주세요.")

    if st.button("← 메인으로 돌아가기", use_container_width=True):
        ss.page = 'main'; st.rerun()


# =========================
# 설문
# =========================
def render_survey_page():
    questions = [
        {"title":"설문조사 1","question":"1. 나이를\n입력해주세요.","type":"input","placeholder":"나이를 입력하세요","key":"age"},
        {"title":"설문조사 2","question":"2. 성별을\n선택해주세요.","type":"choice","options":["남성","여성"],"key":"gender"},
        {"title":"설문조사 3","question":"3. 가구원 수를\n입력해주세요.","type":"input","placeholder":"가구원 수","key":"family_size"},
        {"title":"설문조사 4","question":"4. 피부양자가\n있나요?","type":"choice","options":["예","아니오"],"key":"dependents"},
        {"title":"설문조사 5","question":"5. 현재 보유한\n금융자산(만원)","type":"input","placeholder":"예: 5,000","key":"assets"},
        {"title":"설문조사 6","question":"6. 월 수령하는\n연금 급여(만원)","type":"input","placeholder":"예: 120","key":"pension"},
        {"title":"설문조사 7","question
