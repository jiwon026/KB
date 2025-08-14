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
    import faiss
except Exception:
    USE_FAISS = False
    from sklearn.neighbors import NearestNeighbors  # noqa: F401


def build_index(X: np.ndarray):
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
# CSV 전처리 & 필터 & 추천
# =========================
def preprocess_products(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["상품명","구분","예상수익률(연)","리스크","최소투자금액","투자기간(개월)"])
    out = df.copy()
    out["구분"] = kind

    if "상품명" not in out.columns:
        out["상품명"] = out.get("펀드명", out.index.astype(str)).astype(str)

    if "예상수익률(연)" not in out.columns:
        out["예상수익률(연)"] = 3.0
    out["예상수익률(연)"] = (
        out["예상수익률(연)"].astype(str).str.replace("%","", regex=False)
        .astype(float).fillna(0.0)
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
    if df.empty: return df
    invest = float(cond.get("투자금액", 0) or 0)
    period = int(cond.get("투자기간", 12) or 12)
    risk   = str(cond.get("투자성향", "위험중립형") or "위험중립형")

    def risk_ok(x):
        x = str(x)
        if risk == "안정형":
            return ("낮" in x) or ("보수" in x) or (x in ["낮음","안정형"])
        if risk == "공격형":
            return ("높" in x) or ("공격" in x) or (x in ["높음","공격형"])
        return True

    df2 = df[df["최소투자금액"] <= invest].copy()
    df2["기간차"] = (df2["투자기간(개월)"] - period).abs()
    df2 = df2[df2["기간차"] <= 12] if not df2.empty else df2
    if not df2.empty:
        df2 = df2[df2["리스크"].apply(risk_ok)]
    return df2.drop(columns=["기간차"], errors="ignore") if not df2.empty else df2

def get_custom_recommendations_from_csv(investment_amount, period, risk_level, target_monthly):
    try:
        dep = preprocess_products(load_deposit_csv(), "예·적금")
        fun = preprocess_products(load_fund_csv(), "펀드")
        all_products = pd.concat([dep, fun], ignore_index=True)
        if all_products.empty: return []

        user = {
            '투자금액': float(investment_amount),
            '투자기간': int(period),
            '투자성향': risk_level,
            '목표월이자': float(target_monthly)
        }

        filtered = rule_based_filter(all_products, user)
        if filtered.empty: return []

        filtered = filtered.copy()
        filtered["월예상수익금(만원)"] = user["투자금액"] * (filtered["예상수익률(연)"]/100.0) / 12.0
        filtered["추천점수"] = (100 - (filtered["월예상수익금(만원)"] - user["목표월이자"]).abs() * 2).clip(lower=0)
        filtered = filtered.sort_values(["추천점수","예상수익률(연)"], ascending=False)

        out = []
        for _, r in filtered.head(5).iterrows():
            out.append({
                '상품명': r.get('상품명', '상품명 없음'),
                '구분': r.get('구분', '기타'),
                '월수령액': f"{r.get('월예상수익금(만원)', 0):.1f}만원",
                '연수익률': f"{r.get('예상수익률(연)', 0):.1f}%",
                '리스크': r.get('리스크', '중간'),
                '최소투자금액': f"{int(r.get('최소투자금액', 0))}만원",
                '투자기간': f"{int(r.get('투자기간(개월)', period))}개월",
                '추천점수': float(r.get('추천점수', 0))
            })
        return out
    except Exception as e:
        st.error(f"추천 시스템 오류: {e}")
        return []

def get_fallback_recommendations(investment_amount, period, risk_level, target_monthly):
    base = {
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
    products = base.get(risk_level, base['위험중립형'])
    out = []
    for p in products:
        if investment_amount >= p['최소투자']:
            annual = investment_amount * (p['기본수익률']/100)
            monthly = annual / 12
            out.append({
                '상품명': p['상품명'],
                '구분': '예·적금' if ('예금' in p['상품명'] or '적금' in p['상품명']) else '펀드',
                '월수령액': f"{monthly:.1f}만원",
                '연수익률': f"{p['기본수익률']:.1f}%",
                '리스크': risk_level,
                '최소투자금액': f"{p['최소투자']}만원",
                '투자기간': f"{period}개월",
                '추천점수': max(0, 100 - abs(monthly - target_monthly) * 2)
            })
    return sorted(out, key=lambda x: x['추천점수'], reverse=True)[:3]


# =========================
# 세션 초기화
# =========================
ss = st.session_state
ss.setdefault('page', 'home_v2')   # ← 새 홈을 기본으로
ss.setdefault('question_step', 1)
ss.setdefault('answers', {})
ss.setdefault('user_type', None)


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


# =================================
# UI 흐름: 새 메인 홈 화면
# =================================
def reset_app_state(go: str | None = None):
    """앱 상태 초기화 후 go 페이지로 이동(선택)."""
    for k in [
        "pred_amount", "answers", "prefill_survey", "pred_label",
        "tabnet_label", "rec_df", "display_type", "risk_choice",
        "show_reco", "show_sim", "sim_ready", "sim_inputs",
        *[kk for kk in st.session_state.keys() if str(kk).startswith("survey_")],
    ]:
        st.session_state.pop(k, None)
    if go:
        st.session_state["page"] = go
    st.rerun()

def render_main_home():
    st.title("💬 시니어 금융 설문 & 추천 시스템")

    # 커스텀 CSS (모바일 닫힘 괄호 보완)
    st.markdown("""
    <style>
      .stApp { max-width: 350px; margin: 0 auto; background-color: #f8f9fa; padding: 20px; }
      .main-container { max-width: 480px; margin: 2rem auto; padding: 2.5rem; background: white; border-radius: 20px;
                        box-shadow: 0 20px 40px rgba(0,0,0,0.1); text-align: center; }
      .brand-section { margin-bottom: 1.5rem; }
      .kb-logo { display: inline-flex; align-items: center; justify-content: center;
                 background: linear-gradient(135deg, #FFD700, #FFA500); color: #8B4513; font-weight: 900; font-size: 28px;
                 padding: 10px 16px; border-radius: 10px; margin-right: 15px; border: 2px solid #FF8C00; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
      .elderly-icons { font-size: 50px; margin-left: 10px; }
      .app-title { font-size: 36px; font-weight: 800; color: #2c3e50; margin: 1.5rem 0; line-height: 1.2; }

      .stButton > button {
        width: 100% !important; padding: 22px 24px !important; margin: 12px 0 !important;
        border: none !important; border-radius: 16px !important; font-size: 22px !important; font-weight: 700 !important;
        cursor: pointer !important; transition: all 0.3s ease !important; min-height: 70px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important; white-space: pre-line !important;
      }
      .stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 20px rgba(0,0,0,0.15) !important; }

      .footer-text { margin-top: 1.5rem; font-size: 16px; color: #7f8c8d; font-style: italic; }

      @media (max-width: 480px) {
        .main-container { margin: 1rem; padding: 2rem; }
        .app-title { font-size: 30px; }
        .kb-logo { font-size: 24px; padding: 8px 14px; }
        .elderly-icons { font-size: 40px; }
        .stButton > button { font-size: 20px !important; padding: 20px 22px !important; min-height: 65px !important; }
      }
      @media (max-width: 400px) {
        .stButton > button { font-size: 18px !important; padding: 20px 15px !important; }
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
        if st.session_state.get('user_type'):
            st.session_state.page = "survey_result"
        else:
            st.session_state.page = "survey"
            st.session_state.question_step = 1
            st.session_state.answers = {}
        st.rerun()

    # 2) 연금 계산하기
    if st.button("연금 계산하기", key="home_btn_predict"):
        st.session_state.page = "pension_input"
        st.rerun()

    # 3) 노후 시뮬레이션
    if st.button("노후 시뮬레이션", key="home_btn_sim"):
        st.session_state.page = "simulation"
        st.rerun()

    # 4) 맞춤 상품 추천
    if st.button("맞춤 상품 추천", key="home_btn_reco"):
        if st.session_state.get('answers'):
            st.session_state.page = "survey_plus_custom"   # 설문 + 조건 입력 화면
        else:
            st.session_state.page = "survey"
            st.session_state.question_step = 1
            st.session_state.answers = {}
        st.rerun()

    # 5) 설문 다시하기
    if st.button("설문 다시하기", key="home_btn_reset"):
        reset_app_state(go="survey")

    st.markdown('</div>', unsafe_allow_html=True)  # menu-section 닫기

    # 하단 설명
    st.markdown(
        '<div class="footer-text">버튼을 눌러 다음 단계로 이동하세요</div>',
        unsafe_allow_html=True
    )

    st.markdown('</div>', unsafe_allow_html=True)  # main-container 닫기


# =========================
# 기존 메인 (보존용, 원하면 사용)
# =========================
def render_main_page():
    render_header()

    if st.button("내 금융유형\n보기", key="financial_type", use_container_width=True):
        if ss.get('user_type'):
            ss.page = 'survey_result'
        else:
            ss.page = 'survey'
            ss.question_step = 1
            ss.answers = {}
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
        ss.page = 'home_v2'; st.rerun()


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
                step = 1 if q['key'] in {"age","family_size"} else 1
                answer = st.number_input("", min_value=0, step=step, key=f"survey_q{ss.question_step}")
            else:
                answer = st.text_input("", placeholder=q['placeholder'], key=f"survey_q{ss.question_step}")

            if (answer or (isinstance(answer,(int,float)) and answer==0)):
                with st.spinner('다음 단계로 이동 중...'): time.sleep(0.3)
                ss.answers[q['key']] = answer
                if ss.question_step < len(questions):
                    ss.question_step += 1; st.rerun()
                else:
                    analyze_user_type()
                    ss.page = 'survey_result'; st.rerun()

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
            ss.page = 'home_v2'; st.rerun()

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
        ss.page = 'home_v2'; st.rerun()


# =========================
# 설문 + 맞춤 조건으로 추천
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

def render_survey_plus_custom_page():
    render_header("설문 + 맞춤 조건으로 추천")
    if not ss.answers:
        st.warning("먼저 설문을 완료해주세요.")
        if st.button("설문 하러 가기"): ss.page='survey'; st.rerun()
        return

    defaults = _defaults_from_survey(ss.answers)

    col1, col2 = st.columns(2)
    with col1:
        investment_amount = st.number_input("투자금액 (만원)", min_value=10, step=10, value=int(defaults["investment_amount"]))
        risk_level = st.selectbox("리스크 허용도", ["안정형","위험중립형","공격형"],
                                  index=["안정형","위험중립형","공격형"].index(defaults["risk_level"]))
    with col2:
        period = st.selectbox("투자 기간 (개월)", [6,12,24,36], index=[6,12,24,36].index(int(defaults["period"])))
        target_monthly = st.number_input("목표 월이자 (만원)", min_value=0.0, step=1.0, value=float(defaults["target_monthly"]))

    st.markdown('<div style="margin: 8px 0 16px 0;"></div>', unsafe_allow_html=True)

    if st.button("🔍 추천 받기", use_container_width=True):
        with st.spinner("CSV에서 조건에 맞는 상품을 찾는 중..."):
            recs = get_custom_recommendations_from_csv(investment_amount, period, risk_level, target_monthly)
        if not recs:
            recs = get_fallback_recommendations(investment_amount, period, risk_level, target_monthly)
        ss.spc_last_input = {"investment_amount":investment_amount,"period":period,"risk_level":risk_level,"target_monthly":target_monthly}
        ss.spc_recs = recs
        st.rerun()

    if "spc_recs" in ss:
        cond = ss.get("spc_last_input", {})
        st.caption(
            f"검색 조건 · 투자금액 **{cond.get('investment_amount',0)}만원**, "
            f"기간 **{cond.get('period',0)}개월**, 리스크 **{cond.get('risk_level','-')}**, "
            f"목표 월이자 **{cond.get('target_monthly',0)}만원** · 소스: **CSV 기반**"
        )
        for i, p in enumerate(ss.spc_recs, 1):
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
                ss.pop("spc_recs", None); st.rerun()
        with c2:
            if st.button("노후 시뮬레이션으로"):
                ss.page = 'simulation'; st.rerun()
        with c3:
            if st.button("메인으로"):
                ss.page = 'home_v2'; st.rerun()


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
        ss.page='home_v2'; st.rerun()

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
            ss.page='home_v2'; st.rerun()


# =========================
# 간단 시뮬레이션
# =========================
def render_simulation_page():
    render_header("노후 시뮬레이션")
    if not ss.answers:
        st.warning("먼저 설문을 완료하시면 더 정확한 시뮬레이션이 가능합니다.")
        current_age, current_assets, monthly_income, monthly_expense = 65, 5000, 200, 150
    else:
        a = ss.answers
        current_age     = _to_int(a.get('age'), 65)
        current_assets  = _to_float(a.get('assets'), 5000)
        pension         = _to_float(a.get('pension'), 100)
        income          = _to_float(a.get('income'), 100)
        monthly_income  = pension + income
        monthly_expense = _to_float(a.get('living_cost'), 150)

    col1, col2, col3 = st.columns(3)
    with col1: st.metric("현재 나이", f"{current_age}세")
    with col2: st.metric("보유 자산", f"{current_assets:,.0f}만원")
    with col3: st.metric("월 순수익", f"{monthly_income - monthly_expense:,.0f}만원")

    years_left   = 100 - current_age
    total_needed = monthly_expense * 12 * years_left
    total_income = monthly_income * 12 * years_left
    total_avail  = current_assets + total_income

    st.markdown("### 📈 100세까지 생활비 시뮬레이션")
    if total_avail >= total_needed:
        st.success("✅ 현재 자산과 소득으로 100세까지 안정적인 생활이 가능합니다!")
        st.info(f"💰 예상 잉여자금: {(total_avail-total_needed):,.0f}만원")
    else:
        st.warning(f"⚠️ 100세까지 {total_needed-total_avail:,.0f}만원이 부족할 수 있습니다.")
        st.info("💡 추가 투자나 부업을 고려해보세요.")

    scenarios = [{"name":"안전투자 (연 3%)","rate":0.03},{"name":"균형투자 (연 5%)","rate":0.05},{"name":"적극투자 (연 7%)","rate":0.07}]
    for s in scenarios:
        inv_ret = current_assets * (1 + s["rate"]) ** years_left
        final_total = inv_ret + total_income
        if final_total >= total_needed:
            st.success(f"✅ {s['name']}: {final_total:,.0f}만원 (충분)")
        else:
            st.error(f"❌ {s['name']}: {final_total:,.0f}만원 (부족)")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("설문 + 맞춤 추천으로"):
            ss.page='survey_plus_custom'; st.rerun()
    with c2:
        if st.button("← 메인으로"):
            ss.page='home_v2'; st.rerun()


# =========================
# 라우터
# =========================
def main():
    if ss.page == 'home_v2':                     # 새 홈
        render_main_home()
    elif ss.page == 'main':                      # 기존 홈(보존용)
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
