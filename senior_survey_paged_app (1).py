# app.py
import os
import pandas as pd
import numpy as np
import joblib
import faiss
import streamlit as st

# =================================
# 📂 경로 설정 (Git 리포 상대경로 안전하게)
# =================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "금융상품_3개_통합본.csv")

products_df = pd.read_csv(csv_path)

# =================================
# 🔹 모델 로딩 (캐시)
# =================================

def load_models():
    survey_model   = joblib.load(os.path.join(MODELS_DIR, "tabnet_model.pkl"))
    survey_encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
    reg_model      = joblib.load(os.path.join(MODELS_DIR, "reg_model.pkl"))
    type_model     = joblib.load(os.path.join(MODELS_DIR, "type_model.pkl"))
    return survey_model, survey_encoder, reg_model, type_model

survey_model, survey_encoder, reg_model, type_model = load_models()

# =================================
# 🔧 상품 전처리 & 추천 유틸
# =================================
def preprocess_products(df: pd.DataFrame) -> pd.DataFrame:
    np.random.seed(42)
    df.columns = df.columns.str.strip()

    # 상품명
    if '상품명' in df.columns:
        names = df['상품명'].fillna('무명상품').astype(str)
    elif '펀드명' in df.columns:
        names = df['펀드명'].fillna('무명상품').astype(str)
    elif '출처파일명' in df.columns:
        names = df['출처파일명'].apply(lambda x: str(x).split('.')[0] if pd.notnull(x) else '무명상품')
    else:
        names = [f"무명상품_{i}" for i in range(len(df))]

    # 최소 투자금액(없으면 샘플값)
    if '최고한도' in df.columns:
        min_invest = pd.to_numeric(df['최고한도'], errors='coerce').fillna(0)
        z = (min_invest == 0)
        if z.any():
            min_invest[z] = np.random.randint(100, 1000, z.sum())
    else:
        min_invest = np.random.randint(100, 1000, len(df))

    # 수익률(%) → 소수
    rate_col = '기본금리( %)' if '기본금리( %)' in df.columns else ('세전\n이자율' if '세전\n이자율' in df.columns else None)
    if rate_col and rate_col in df.columns:
        est_return = pd.to_numeric(df[rate_col].astype(str).str.extract(r"([\d.]+)")[0], errors='coerce')
        est_return = est_return.fillna(np.random.uniform(1.0, 8.0, len(df))) / 100.0
    else:
        est_return = np.round(np.random.uniform(0.01, 0.08, len(df)), 4)

    # 리스크
    if '위험등급' in df.columns:
        raw_risk = df['위험등급'].astype(str)
        risk = raw_risk.apply(lambda x: '높음' if ('5' in x or '4' in x) else ('중간' if '3' in x else '낮음'))
    else:
        risk = np.random.choice(['낮음', '중간', '높음'], len(df))

    # 권장기간/투자성향(필터용)
    duration = np.random.choice([6, 12, 24, 36], len(df))
    profile = np.random.choice(['안정형', '위험중립형', '공격형'], len(df))

    out = pd.DataFrame({
        '상품명': names,
        '최소투자금액': min_invest.astype(int),
        '예상수익률': np.round(est_return, 4),
        '리스크': risk,
        '권장투자기간': duration,
        '투자성향': profile
    })
    return out[out['상품명'] != '무명상품'].drop_duplicates(subset=['상품명'])

def rule_based_filter(df: pd.DataFrame, user: dict) -> pd.DataFrame:
    risk_pref_map = {
        '안정형': ['낮음','중간'],
        '위험중립형': ['중간','낮음','높음'],
        '공격형': ['높음','중간']
    }
    allowed_risks = risk_pref_map.get(user['투자성향'], ['낮음','중간','높음'])

    filtered = df[
        (df['최소투자금액'] <= user['투자금액']) &
        (df['권장투자기간'] <= user['투자기간']) &
        (df['리스크'].isin(allowed_risks)) &
        (df['투자성향'] == user['투자성향'])
    ]
    if filtered.empty:
        # 너무 빡세면 성향만 완화
        filtered = df[
            (df['최소투자금액'] <= user['투자금액']) &
            (df['권장투자기간'] <= user['투자기간']) &
            (df['리스크'].isin(allowed_risks))
        ]
    return filtered.sort_values('예상수익률', ascending=False).head(200).reset_index(drop=True)

def _get_feature_vector(df: pd.DataFrame) -> np.ndarray:
    return np.vstack([
        df['최소투자금액'] / 1000.0,
        df['예상수익률'] * 100.0,
        df['권장투자기간'] / 12.0
    ]).T.astype('float32')

def _get_user_vector(user: dict) -> np.ndarray:
    return np.array([
        user['투자금액'] / 1000.0,
        user['목표월이자'],
        user['투자기간'] / 12.0
    ], dtype='float32').reshape(1, -1)

def _explain_product(row: pd.Series, user: dict) -> dict:
    expected_monthly = round((user['투자금액'] * float(row['예상수익률'])) / 12.0, 1)
    return {
        '상품명': row['상품명'],
        '월예상수익금(만원)': expected_monthly,
        '리스크': row['리스크'],
        '투자기간(개월)': int(row['권장투자기간']),
        '예상수익률(연)': f"{round(float(row['예상수익률'])*100,2)}%"
    }

def recommend_products(processed_df: pd.DataFrame, user: dict, topk: int = 10):
    filtered = rule_based_filter(processed_df, user)
    if filtered.empty:
        return pd.DataFrame({'메시지': ['조건에 맞는 상품이 없어요 😢']}), None

    filtered = filtered.drop_duplicates(subset=['상품명'])
    X = _get_feature_vector(filtered)
    index = faiss.IndexFlatL2(X.shape[1])
    index.add(X)

    user_vec = _get_user_vector(user)
    _, idx = index.search(user_vec, k=min(topk, len(filtered)))
    rec = filtered.iloc[idx[0]].drop_duplicates(subset=['상품명']).head(topk).reset_index(drop=True)

    results = pd.DataFrame([_explain_product(row, user) for _, row in rec.iterrows()])
    return results, index

@st.cache_data
def load_default_products():
    path = os.path.join(DATA_DIR, "sample_products.csv")
    if os.path.exists(path):
        return preprocess_products(pd.read_csv(path))
    return None

# =================================
# 🖥 Streamlit UI
# =================================
st.set_page_config(page_title="시니어 금융 설문", page_icon="💸", layout="centered")
st.title("💬 시니어 금융 설문 & 추천 시스템")

# 상태 관리
ss = st.session_state
ss.setdefault("flow", "choose")          # choose → predict → survey → result → recommend
ss.setdefault("pred_amount", None)       # 미수령자의 예측 연금액
ss.setdefault("survey_answers", {})      # 설문 결과 캐시
ss.setdefault("products", load_default_products())

# 사이드바: 상품 CSV 업로드(옵션)
st.sidebar.header("📦 상품 데이터")
up = st.sidebar.file_uploader("CSV 업로드(없으면 data/sample_products.csv 사용)", type=["csv"])
if up is not None:
    ss.products = preprocess_products(pd.read_csv(up))
    st.sidebar.success(f"상품 {len(ss.products):,}건 로드됨")
elif ss.products is not None:
    st.sidebar.info(f"기본 상품 {len(ss.products):,}건 사용 중")
else:
    st.sidebar.warning("상품 데이터가 없습니다. 업로드하거나 data/sample_products.csv를 넣어주세요.")

# -------------------------------
# 공통: 설문 함수
# -------------------------------
QUESTIONS = [
    ("나이를 입력해주세요.", "number", "age"),
    ("성별을 선택해주세요.", "select", "gender", ["남성", "여성"]),
    ("가구원 수를 입력해주세요.", "number", "family_size"),
    ("피부양자가 있나요?", "select", "dependents", ["예", "아니오"]),
    ("현재 보유한 금융자산(만원)을 입력해주세요.", "number", "assets"),
    ("월 수령하는 연금 금액(만원)을 입력해주세요.", "number", "pension"),
    ("월 평균 지출비(만원)은 얼마인가요?", "number", "living_cost"),
    ("월 평균 소득은 얼마인가요?", "number", "income"),
    ("투자 성향을 선택해주세요.", "select", "risk", ["안정형", "안정추구형", "위험중립형", "적극투자형", "공격투자형"]),
]

def render_survey():
    st.subheader("📝 설문")
    answers = {}
    for q in QUESTIONS:
        title, kind, key = q[0], q[1], q[2]
        if kind == "number":
            answers[key] = st.number_input(title, min_value=0, step=1, key=f"q_{key}")
        elif kind == "select":
            answers[key] = st.selectbox(title, q[3], key=f"q_{key}")
    return answers

def map_survey_to_model_input(r):
    gender = 0 if r["gender"] == "남성" else 1
    dependents = 1 if r["dependents"] == "예" else 0
    risk_map = {"안정형": 0, "안정추구형": 1, "위험중립형": 2, "적극투자형": 3, "공격투자형": 4}
    risk = risk_map[r["risk"]]
    arr = np.array([[
        float(r["age"]), gender, float(r["family_size"]), dependents,
        float(r["assets"]), float(r["pension"]), float(r["living_cost"]),
        float(r["income"]), risk
    ]])
    return arr

# -------------------------------
# 1) 연금 수령 여부 선택
# -------------------------------
if ss.flow == "choose":
    st.markdown("### 1️⃣ 현재 연금을 받고 계신가요?")
    choice = st.radio("연금 수령 여부를 선택해주세요.", ["선택하세요", "예(수령 중)", "아니오(미수령)"], index=0)
    if choice == "예(수령 중)":
        ss.flow = "survey"
    elif choice == "아니오(미수령)":
        ss.flow = "predict"

# -------------------------------
# 2-1) 미수령자 → 연금 계산기
# -------------------------------
if ss.flow == "predict":
    st.subheader("📈 연금 계산기")
    income = st.number_input("평균 월소득(만원)", min_value=0, step=1, key="pred_income")
    years  = st.number_input("국민연금 가입기간(년)", min_value=0, max_value=50, step=1, key="pred_years")

    if st.button("연금 예측하기"):
        X = pd.DataFrame([{"평균월소득(만원)": income, "가입기간(년)": years}])
        amount = round(float(reg_model.predict(X)[0]), 1)
        ss.pred_amount = amount

        # 안내
        def classify_pension_type(a):
            if a >= 90: return "완전노령연금"
            if a >= 60: return "조기노령연금"
            if a >= 30: return "감액노령연금"
            return "특례노령연금"

        ptype = classify_pension_type(amount)
        explains = {
            "조기노령연금": "※ 만 60세부터 수령 가능하나 최대 30% 감액될 수 있어요.",
            "완전노령연금": "※ 만 65세부터 감액 없이 정액 수령이 가능해요.",
            "감액노령연금": "※ 일정 조건을 만족하지 못할 경우 감액되어 수령됩니다.",
            "특례노령연금": "※ 가입기간이 짧더라도 일정 기준 충족 시 수령 가능."
        }
        st.success(f"💰 예측 연금 수령액: **{amount}만원/월**")
        st.markdown(f"📂 예측 연금 유형: **{ptype}**")
        st.info(explains[ptype])

        # 다음 단계: 설문
        ss.flow = "survey"

# -------------------------------
# 2) 수령자/미수령자 공통 → 설문
# -------------------------------
if ss.flow == "survey":
    answers = render_survey()
    if st.button("유형 분류하기"):
        # 설문 → TabNet 분류
        arr = map_survey_to_model_input(answers)
        pred = survey_model.predict(arr)
        label = survey_encoder.inverse_transform(pred)[0]

        proba = survey_model.predict_proba(arr)
        proba_df = pd.DataFrame(proba, columns=survey_encoder.classes_)
        predicted_proba = float(proba_df[label].values[0])

        st.success(f"🧾 예측된 금융 유형: **{label}** (확률 {predicted_proba*100:.1f}%)")
        st.bar_chart(proba_df.T)

        # 다음 단계(추천)로 진행
        ss.survey_answers = answers
        ss.flow = "recommend"

# -------------------------------
# 3) 추천: 설문 + 투자조건 입력 → 추천
# -------------------------------
if ss.flow == "recommend":
    st.markdown("---")
    st.subheader("🧲 금융상품 추천")

    if ss.products is None:
        st.info("상품 CSV를 업로드하거나 data/sample_products.csv 를 넣어주세요.")
    else:
        # 추천 조건 입력
        invest_amount = st.number_input("투자금액(만원)", min_value=10, step=10, value=500)
        invest_period = st.selectbox("투자기간(개월)", [6, 12, 24, 36], index=1)
        risk_choice   = st.selectbox("리스크 허용도", ["안정형", "위험중립형", "공격형"], index=1)
        target_monthly = st.number_input("목표 월이자(만원)", min_value=1, step=1, value=10)

        if st.button("추천 보기"):
            user_pref = {
                '투자금액': invest_amount,
                '투자기간': invest_period,
                '투자성향': risk_choice,
                '목표월이자': target_monthly
            }
            rec_df, faiss_index = recommend_products(ss.products, user_pref)

            if "메시지" in rec_df.columns:
                st.warning(rec_df.iloc[0, 0])
            else:
                st.dataframe(rec_df, use_container_width=True)
                csv_bytes = rec_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button("추천 결과 CSV 다운로드", csv_bytes, "recommendations.csv", "text/csv")

                # 인덱스 저장(옵션)
                idx_path = os.path.join(MODELS_DIR, "faiss_index.idx")
                faiss.write_index(faiss_index, idx_path)
                st.caption(f"FAISS 인덱스 저장됨: models/faiss_index.idx")

    # 흐름 리셋 버튼
    if st.button("처음으로 돌아가기"):
        for k in ["flow","pred_amount","survey_answers"]:
            if k in ss: del ss[k]
        st.rerun()
