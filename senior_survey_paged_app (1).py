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

<style>
/* 공통 버튼 스타일 유지 */
.stButton > button {
    width:100% !important; height:72px !important; border-radius:20px !important;
    font-size:18px !important; font-weight:bold !important; border:none !important;
    box-shadow:0 2px 8px rgba(0,0,0,0.1) !important; transition:all .2s ease !important; white-space:pre-line !important;
}

/* ====== 메인 버튼별 색상 (help 속성으로 타겟팅) ====== */
/* 내 금융유형 보기: 노란색 */
button[title="btn-fin-type"]{
  background:#FFD700 !important; color:#1f2937 !important;
}
button[title="btn-fin-type"]:hover{ filter:brightness(0.98); }

/* 연금 계산하기: 하늘색 */
button[title="btn-pension-calc"]{
  background:#87CEFA !important; color:#1f2937 !important;
}
button[title="btn-pension-calc"]:hover{ filter:brightness(0.98); }

/* 노후시뮬레이션: 핑크 */
button[title="btn-simulation"]{
  background:#FFB6C1 !important; color:#1f2937 !important;
}
button[title="btn-simulation"]:hover{ filter:brightness(0.98); }

/* 맞춤 상품 추천: 연두색 */
button[title="btn-recommend"]{
  background:#90EE90 !important; color:#1f2937 !important;
}
button[title="btn-recommend"]:hover{ filter:brightness(0.98); }

/* 다시 설문하기 & 전화상담: 연주황 */
button[title="btn-survey-reset"],
button[title="btn-phone"]{
  background:#FFA07A !important; color:#1f2937 !important;
}
button[title="btn-survey-reset"]:hover,
button[title="btn-phone"]:hover{ filter:brightness(0.98); }
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


# =========================
# 메인
# =========================
def render_main_page():
    render_header()

    # 내 금융유형 보기 (노란색)
    if st.button("내 금융유형\n보기", key="financial_type", help="btn-fin-type", use_container_width=True):
        if ss.get('user_type'):
            ss.page = 'survey_result'
        else:
            ss.page = 'survey'; ss.question_step = 1; ss.answers = {}
        st.rerun()

    st.markdown('<div style="margin: 15px 0;"></div>', unsafe_allow_html=True)

    # 연금 계산하기 (하늘색)
    if st.button("연금\n계산하기", key="pension_calc", help="btn-pension-calc", use_container_width=True):
        ss.page = 'pension_input'; st.rerun()

    st.markdown('<div style="margin: 20px 0;"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        # 노후시뮬레이션 (핑크)
        if st.button("노후\n시뮬레이션", key="simulation", help="btn-simulation", use_container_width=True):
            ss.page = 'simulation'; st.rerun()
    with col2:
        # 맞춤 상품 추천 (연두색)
        if st.button("맞춤 상품\n추천", key="recommendation", help="btn-recommend", use_container_width=True):
            if ss.get('answers'):
                ss.page = 'survey_plus_custom'
            else:
                ss.page = 'survey'; ss.question_step = 1; ss.answers = {}
            st.rerun()

    st.markdown('<div style="margin: 15px 0;"></div>', unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        # 다시 설문하기 (연주황)
        if st.button("설문\n다시하기", key="survey_reset", help="btn-survey-reset", use_container_width=True):
            ss.page = 'survey'; ss.question_step = 1; ss.answers = {}; ss.user_type = None; st.rerun()
    with col4:
        # 전화상담 (연주황)
        if st.button("📞 전화\n상담", key="phone_consultation", help="btn-phone", use_container_width=True):
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
        {"title":"설문조사 7","question":"7. 월 평균\n지출비(만원)","type":"input","placeholder":"예: 180","key":"living_cost"},
        {"title":"설문조사 8","question":"8. 월 평균\n소득(만원)","type":"input","placeholder":"예: 220","key":"income"},
        {"title":"설문조사 9","question":"9. 투자 성향을\n선택해주세요.","type":"choice","options":["안정형","안정추구형","위험중립형","적극투자형","공격투자형"],"key":"risk"},
    ]

    if ss.question_step <= len(questions):
        q = questions[ss.question_step - 1]
        render_header(q['title'])
        st.markdown(f"""
        <div style="text-align:center; font-size:20px; font-weight:bold; margin:50px 0; line-height:1.5; color:#333;">
            {q['question']}
        </div>
        """, unsafe_allow_html=True)

        if q['type'] == 'input':
            num_keys = {"age","family_size","assets","pension","living_cost","income"}
            if q['key'] in num_keys:
                answer = st.number_input("", min_value=0, step=1, key=f"survey_q{ss.question_step}")
            else:
                answer = st.text_input("", placeholder=q['placeholder'], key=f"survey_q{ss.question_step}")
        
            # ✅ 버튼을 눌러야만 진행
            if st.button("다음", key=f"next_{ss.question_step}"):
                # 숫자 검증(필요시): 나이/가구원은 1 이상 등
                if q['key'] in {"age","family_size"} and answer < 1:
                    st.error("1 이상의 값을 입력해주세요.")
                else:
                    ss.answers[q['key']] = answer
                    if ss.question_step < len(questions):
                        ss.question_step += 1; st.rerun()
                    else:
                        analyze_user_type(); ss.page='survey_result'; st.rerun()


        elif q['type'] == 'choice':
            st.markdown('<div style="margin: 30px 0;"></div>', unsafe_allow_html=True)
            for opt in q['options']:
                if st.button(opt, key=f"choice_{opt}_{ss.question_step}", use_container_width=True):
                    ss.answers[q['key']] = opt
                    with st.spinner('다음 단계로 이동 중...'): time.sleep(0.2)
                    if ss.question_step < len(questions):
                        ss.question_step += 1; st.rerun()
                    else:
                        analyze_user_type(); ss.page='survey_result'; st.rerun()

        progress = ss.question_step / len(questions)
        st.progress(progress)
        st.markdown(f"<div style='text-align:center; margin-top:15px; color:#666;'>{ss.question_step}/{len(questions)} 단계</div>", unsafe_allow_html=True)

        if st.button("← 메인으로", key="back_to_main_from_survey"):
            ss.page = 'main'; st.rerun()

def analyze_user_type():
    a = ss.answers
    age         = _to_int(a.get('age'), 65)
    assets      = _to_float(a.get('assets'), 5000)
    pension     = _to_float(a.get('pension'), 100)
    income      = _to_float(a.get('income'), 200)
    living_cost = _to_float(a.get('living_cost'), 150)
    risk        = (a.get('risk') or '안정형').strip()

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

    ss.user_type = user_type


# =========================
# 설문 결과
# =========================
def render_survey_result_page():
    render_header("금융 유형 결과")
    user_type = ss.user_type or "균형형"
    info = {
        "자산운용형": {"icon":"💼","desc":"투자 여력이 충분하며 운용 전략 중심 포트폴리오가 적합합니다.","color":"#4F46E5"},
        "위험취약형": {"icon":"⚠️","desc":"재무 위험이 높아 지출 관리와 복지 연계가 필요합니다.","color":"#EF4444"},
        "균형형":     {"icon":"⚖️","desc":"자산과 연금이 안정적이며 보수적 전략이 적합합니다.","color":"#10B981"},
        "적극투자형": {"icon":"🚀","desc":"수익을 위해 변동성을 감내하며 적극적 투자를 선호합니다.","color":"#F59E0B"},
    }[user_type]

    st.markdown(f"""
    <div class="result-card" style="text-align:center; border-left:5px solid {info['color']};">
        <div style="font-size:48px; margin-bottom:20px;">{info['icon']}</div>
        <h2 style="color:{info['color']}; margin-bottom:15px;">{user_type}</h2>
        <p style="font-size:18px; line-height:1.6; color:#666;">{info['desc']}</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("설문 + 맞춤 조건으로 추천 보기", use_container_width=True):
            ss.page = 'survey_plus_custom'; st.rerun()
    with c2:
        if st.button("노후 시뮬레이션 보기", use_container_width=True):
            ss.page = 'simulation'; st.rerun()

    if st.button("← 메인으로 돌아가기", use_container_width=True):
        ss.page = 'main'; st.rerun()


# =========================
# 설문 + 맞춤 조건으로 추천 (참고코드 로직 이식)
# =========================
def _defaults_from_survey(answers: dict):
    age    = _to_int(answers.get('age'), 65)
    assets = _to_float(answers.get('assets'), 5000)
    income = _to_float(answers.get('income'), 200)
    risk   = str(answers.get('risk', '위험중립형') or '위험중립형')

    if age >= 70:
        invest_amount = min(assets * 0.3, 3000); period = 12
    elif age >= 60:
        invest_amount = min(assets * 0.4, 5000); period = 24
    else:
        invest_amount = min(assets * 0.5, 8000); period = 36

    target_monthly = income * 0.1
    risk_map = {'안정형':'안정형','안정추구형':'안정형','위험중립형':'위험중립형','적극투자형':'공격형','공격투자형':'공격형'}
    risk3 = risk_map.get(risk, '위험중립형')
    return {
        "investment_amount": int(round(invest_amount)),
        "period": int(period),
        "risk_level": risk3,
        "target_monthly": float(round(target_monthly, 1)),
    }
def _goto(page_name: str):
    st.session_state.page = page_name
    st.rerun()
    st.stop()
    
def render_survey_plus_custom_page():
    render_header("설문 + 맞춤 조건으로 추천")

    # 0) 설문 체크
    if not ss.answers:
        st.warning("먼저 설문을 완료해주세요.")
        if st.button("설문 하러 가기", key="spc_go_survey"):
            _goto('survey')
        return

    # 1) 기본값 생성
    defaults = _defaults_from_survey(ss.answers)

    # 2) 조건 입력 폼
    col1, col2 = st.columns(2)
    with col1:
        investment_amount = st.number_input(
            "투자금액 (만원)", min_value=10, step=10,
            value=int(defaults["investment_amount"]),
            key="spc_invest_amount"
        )
        risk_level = st.selectbox(
            "리스크 허용도", ["안정형","위험중립형","공격형"],
            index=["안정형","위험중립형","공격형"].index(defaults["risk_level"]),
            key="spc_risk_level"
        )
    with col2:
        period = st.selectbox(
            "투자 기간 (개월)", [6,12,24,36],
            index=[6,12,24,36].index(int(defaults["period"])),
            key="spc_period"
        )
        target_monthly = st.number_input(
            "목표 월이자 (만원)", min_value=0.0, step=1.0,
            value=float(defaults["target_monthly"]),
            key="spc_target_monthly"
        )

    st.markdown('<div style="margin: 8px 0 16px 0;"></div>', unsafe_allow_html=True)

    # 3) 추천 실행
    if st.button("🔍 추천 받기", use_container_width=True, key="spc_do_reco"):
        user_pref = {
            '투자금액':   int(investment_amount),
            '투자기간':   int(period),
            '투자성향':   str(risk_level),
            '목표월이자': float(target_monthly),
        }
        with st.spinner("CSV에서 조건에 맞는 상품을 찾는 중..."):
            rec_df = recommend_fallback_split(user_pref)

        # 상태 저장 (시뮬 화면과 공유)
        ss.spc_last_input = user_pref
        if "메시지" in rec_df.columns or rec_df.empty:
            ss.spc_recs = []      # 결과 없음
        else:
            ss.spc_recs = rec_df.to_dict(orient="records")
            ss.rec_df   = rec_df  # 시뮬 탭 그래프에서 사용
            ss.display_type = ss.get('user_type') or DEFAULT_DISPLAY_TYPE
            ss.risk_choice  = risk_level

        st.rerun()
        return  # rerun 후 즉시 종료

    # 4) 추천 결과 렌더
    if "spc_recs" in ss and ss.spc_recs:
        cond = ss.get("spc_last_input", {})
        st.caption(
            f"검색 조건 · 투자금액 **{cond.get('투자금액',0)}만원**, "
            f"기간 **{cond.get('투자기간',0)}개월**, 리스크 **{cond.get('투자성향','-')}**, "
            f"목표 월이자 **{cond.get('목표월이자',0)}만원** · 소스: **CSV 기반**"
        )

        # 카드들
        for i, p in enumerate(ss.spc_recs[:3], 1):
            st.markdown(f"""
            <div class="product-card">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
                <h4 style="margin:0;color:#1F2937;">🏆 {i}. {p.get('상품명','-')}</h4>
                <span style="background:#10B981;color:#fff;padding:8px 12px;border-radius:8px;font-weight:700;">
                  {p.get('월예상수익금(만원)','-')}
                </span>
              </div>
              <div style="color:#666;font-size:14px;display:grid;grid-template-columns:1fr 1fr;gap:8px;">
                <div><strong>구분:</strong> {p.get('구분','-')}</div>
                <div><strong>연수익률:</strong> {p.get('예상수익률(연)','-')}</div>
                <div><strong>리스크:</strong> {p.get('리스크','-')}</div>
                <div><strong>최소투자:</strong> {p.get('최소투자금액','-')}</div>
                <div><strong>투자기간:</strong> {p.get('투자기간(개월)','-')}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        # 추천 근거 메시지
        a = ss.answers
        current_age     = _to_int(a.get('age'), 65)
        current_assets  = _to_float(a.get('assets'), 5000)
        pension         = _to_float(a.get('pension'), 100)
        income          = _to_float(a.get('income'), 100)
        monthly_income  = pension + income
        monthly_expense = _to_float(a.get('living_cost'), 150)

        _, depletion_base = retirement_simulation(
            current_age, 100, current_assets, monthly_income, monthly_expense,
            inflation_rate=0.03, investment_return=0.02
        )
        st.info("🔎 추천 근거: " + recommend_reason_from_simulation(
            depletion_base, current_age, current_assets, monthly_income, monthly_expense, ss.get("risk_choice","위험중립형")
        ))

        # 다운로드 버튼
        try:
            rec_df = pd.DataFrame(ss.spc_recs)
            csv_bytes = rec_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button("추천 결과 CSV 다운로드", csv_bytes, "recommendations.csv", "text/csv", key="spc_dl_csv")
        except Exception:
            pass

        st.markdown("---")
        c1, c2, c3 = st.columns(3)

        with c1:
            if st.button("조건 바꿔 다시 추천", key="spc_reset"):
                # 결과만 지우고 입력값/리스크는 유지 → 조건 화면으로
                ss.pop("spc_recs", None)
                st.rerun()
                return

        with c2:
            if st.button("노후 시뮬레이션으로", key="spc_to_sim"):
                _goto('simulation')  # rerun & stop

        with c3:
            if st.button("메인으로", key="spc_to_main"):
                _goto('main')        # rerun & stop

    else:
        # 아직 추천 실행 전이거나 결과 없음
        st.info("조건을 설정한 후 **‘🔍 추천 받기’**를 눌러주세요.")



# =========================
# 연금 계산기
# =========================
def calculate_pension_estimate(monthly_income: float, pension_years: int) -> float:
    accrual = min(max(pension_years, 0), 40) / 40.0
    base_ratio = 0.45
    return round(monthly_income * base_ratio * accrual, 1)

def render_pension_input_page():
    render_header("연금 계산기")
    st.markdown('<div style="text-align:center; color:#666;">평균 월소득과 가입기간을 입력하시면 예상 연금액을 계산해드립니다.</div>', unsafe_allow_html=True)
    monthly_income = st.number_input("평균 월소득 (만원)", min_value=0, value=300, step=10)
    pension_years = st.number_input("국민연금 가입기간 (년)", min_value=0, value=25, step=1)
    if st.button("연금 계산하기", use_container_width=True):
        ss.pension_result = {'monthly_income':monthly_income,'pension_years':pension_years,'estimated_pension':calculate_pension_estimate(monthly_income,pension_years)}
        ss.page='pension_result'; st.rerun()
    if st.button("← 메인으로", key="pension_back"):
        ss.page='main'; st.rerun()

def render_pension_result_page():
    render_header("연금 계산 결과")
    result = ss.get('pension_result', {})
    estimated = result.get('estimated_pension', 0)
    income    = result.get('monthly_income', 0)
    years     = result.get('pension_years', 0)
    st.markdown(f"""
    <div class="result-card" style="text-align:center;">
        <h3 style="color:#4F46E5; margin-bottom:20px;">💰 예상 월 연금액</h3>
        <div style="font-size:36px; font-weight:bold; color:#1F2937; margin:20px 0;">{estimated:,.0f}만원</div>
        <div style="font-size:16px; color:#666; margin-top:15px;">월소득 {income:,.0f}만원 × 가입기간 {years}년 기준</div>
    </div>
    """, unsafe_allow_html=True)

    if estimated >= 90:
        ptype, desc = "완전노령연금", "만 65세부터 감액 없이 정액 수령이 가능합니다."
    elif estimated >= 60:
        ptype, desc = "조기노령연금", "만 60세부터 수령 가능하나 최대 30% 감액될 수 있습니다."
    else:
        ptype, desc = "감액노령연금", "일정 조건을 만족하지 못할 경우 감액되어 수령됩니다."
    st.info(f"**{ptype}**: {desc}")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("설문 + 맞춤 추천 받기"):
            ss.page='survey_plus_custom'; st.rerun()
    with c2:
        if st.button("← 메인으로"):
            ss.page='main'; st.rerun()


# =========================
# 노후 시뮬레이션 (참고코드 기능 이식)
# =========================
def render_simulation_page():
    render_header("노후 시뮬레이션")

    # 설문값(없으면 기본값)
    if not ss.answers:
        st.warning("먼저 설문을 완료하시면 더 정확한 시뮬레이션이 가능합니다.")
        current_age, current_assets, pension, income, monthly_expense = 65, 5000, 100, 100, 150
    else:
        a = ss.answers
        current_age     = _to_int(a.get('age'), 65)
        current_assets  = _to_float(a.get('assets'), 5000)
        pension         = _to_float(a.get('pension'), 0)
        income          = _to_float(a.get('income'), 0)
        monthly_expense = _to_float(a.get('living_cost'), 150)

    monthly_income = pension + income
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("현재 나이", f"{current_age}세")
    with col2: st.metric("보유 자산", f"{current_assets:,.0f}만원")
    with col3: st.metric("월 순수익", f"{monthly_income - monthly_expense:,.0f}만원")

    # 기본/상품 적용 시뮬
    base_return   = 0.02
    invest_return = get_invest_return_from_risk(ss.get('risk_choice', '위험중립형'))

    log_base, depletion_base = retirement_simulation(
        current_age, 100, current_assets, monthly_income, monthly_expense,
        inflation_rate=0.03, investment_return=base_return
    )
    log_invest, depletion_invest = simulate_with_financial_product(
        current_age, 100, current_assets, monthly_income, monthly_expense,
        invest_return=invest_return
    )

    c1, c2 = st.columns(2)
    with c1:
        st.metric(f"기본 시나리오(연 {int(base_return*100)}%) 고갈 나이",
                  value=f"{depletion_base}세" if depletion_base else "고갈 없음")
    with c2:
        st.metric(f"금융상품 적용(연 {int(invest_return*100)}%) 고갈 나이",
                  value=f"{depletion_invest}세" if depletion_invest else "고갈 없음")

    # 가정값 조정 + 그래프
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
            current_age, 100, current_assets, monthly_income, monthly_expense,
            inflation_rate=inflation, investment_return=base_r
        )
        df_b = (pd.DataFrame(log_base2)[['나이','잔액']]
                .rename(columns={'잔액':'기본 시나리오'}) if log_base2 else pd.DataFrame())

        # 추천 결과가 있으면 상품별 탭 시나리오
        has_reco = isinstance(ss.get('rec_df'), pd.DataFrame) and not ss.rec_df.empty
        if has_reco:
            st.markdown("### 📈 추천 상품별 적용 시나리오")
            rec_records = ss.rec_df.to_dict(orient="records")
            tabs = st.tabs([f"{i+1}. {r.get('상품명','-')}" for i, r in enumerate(rec_records)])

            for tab, r in zip(tabs, rec_records):
                with tab:
                    # 예상수익률(연) → % 숫자
                    if '예상수익률' in r and pd.notnull(r['예상수익률']):
                        prod_return_pct = float(r['예상수익률']) * 100.0
                    else:
                        txt = str(r.get('예상수익률(연)','0')).replace('%','')
                        try: prod_return_pct = float(txt)
                        except: prod_return_pct = 5.0
                    prod_r = prod_return_pct / 100.0

                    log_prod2, _ = retirement_simulation(
                        current_age, 100, current_assets, monthly_income, monthly_expense,
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
            st.info("상품별 그래프는 ‘맞춤 상품 추천’에서 추천을 실행하면 표시됩니다.")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("설문 + 맞춤 추천으로"):
            ss.page = 'survey_plus_custom'; st.rerun()
    with c2:
        if st.button("← 메인으로"):
            ss.page = 'main'; st.rerun()


# =========================
# 라우터
# =========================
def main():
    if ss.page == 'main':
        render_main_page()
    elif ss.page == 'survey':
        render_survey_page()
    elif ss.page == 'survey_result':
        render_survey_result_page()
    elif ss.page == 'survey_plus_custom':
        render_survey_plus_custom_page()
    elif ss.page == 'pension_input':
        render_pension_input_page()
    elif ss.page == 'pension_result':
        render_pension_result_page()
    elif ss.page == 'simulation':
        render_simulation_page()
    elif ss.page == 'phone_consultation':
        render_phone_consultation_page()


if __name__ == "__main__":
    main()
