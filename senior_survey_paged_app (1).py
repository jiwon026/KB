import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 모델 및 인코더 불러오기
@st.cache_resource
def load_model():
    model = joblib.load("tabnet_model.pkl")
    encoder = joblib.load("label_encoder.pkl")
    return model, encoder

model, encoder = load_model()

# 페이지 설정
st.set_page_config(page_title="시니어 금융 설문", page_icon="💸", layout="centered")
st.title("💬 시니어 금융 유형 설문")
st.markdown("**아래 질문에 순차적으로 응답해주세요.**")

# 상태 초기화
if "page" not in st.session_state:
    st.session_state.page = 0
if "responses" not in st.session_state:
    st.session_state.responses = {}

# 설문 문항
questions = [
    ("나이를 입력해주세요.", "number", "age"),
    ("성별을 선택해주세요.", "selectbox", "gender", ["남성", "여성"]),
    ("가구원 수를 입력해주세요.", "number", "family_size"),
    ("피부양자가 있나요?", "selectbox", "dependents", ["예", "아니오"]),
    ("현재 보유한 금융자산(만원)을 입력해주세요.", "number", "assets"),
    ("월 수령하는 연금 금액(만원)을 입력해주세요.", "number", "pension"),
    ("월 평균 지출비(만원)은 얼마인가요?", "number", "living_cost"),
    ("월 평균 소득은 얼마인가요?", "number", "income"),
    ("투자 성향을 선택해주세요.", "selectbox", "risk", ["안정형", "안정추구형", "위험중립형", "적극투자형", "공격투자형"]),
]

# 다음 페이지 이동
def next_page():
    if st.session_state.get("input_value") is not None:
        current_q = questions[st.session_state.page]
        st.session_state.responses[current_q[2]] = st.session_state.input_value
        st.session_state.page += 1
        st.session_state.input_value = None

# 질문 출력
if st.session_state.page < len(questions):
    q = questions[st.session_state.page]
    st.markdown(f"**Q{st.session_state.page + 1}. {q[0]}**")
    if q[1] == "number":
        st.number_input(" ", key="input_value", step=1, format="%d", on_change=next_page, label_visibility="collapsed")
    elif q[1] == "selectbox":
        st.selectbox(" ", options=q[3], key="input_value", on_change=next_page, label_visibility="collapsed")

# 모든 질문 완료 시
else:
    st.success("✅ 모든 질문에 응답하셨습니다!")
    r = st.session_state.responses

    # 입력값 가공
    gender = 0 if r["gender"] == "남성" else 1
    dependents = 1 if r["dependents"] == "예" else 0
    risk_map = {"안정형": 0, "안정추구형": 1, "위험중립형": 2, "적극투자형": 3, "공격투자형": 4}
    risk = risk_map[r["risk"]]

    input_array = np.array([[
        float(r["age"]),
        gender,
        float(r["family_size"]),
        dependents,
        float(r["assets"]),
        float(r["pension"]),
        float(r["living_cost"]),
        float(r["income"]),
        risk
    ]])
    # 예측
    prediction = model.predict(input_array)
    label = encoder.inverse_transform(prediction)[0]

    # 예측 확률
    proba = model.predict_proba(input_array)
    proba_df = pd.DataFrame(proba, columns=encoder.classes_)

    # 해당 예측 유형의 확률
    predicted_proba = proba_df[label].values[0]

    # 결과 출력
    st.markdown(f"## 🧾 예측된 당신의 금융 유형: **{label}**")
    st.markdown(f"**확률: {predicted_proba * 100:.1f}%**")
    st.info("이 결과는 TabNet 모델이 입력값을 기반으로 예측한 결과입니다.")

    # 확률 바 차트
    st.markdown("### 📊 각 금융유형에 대한 예측 확률")
    st.bar_chart(proba_df.T)

    # 유형 설명
    descriptions = {
        "자산운용형": "💼 투자 여력이 충분한 유형으로, 운용 전략 중심의 포트폴리오가 적합합니다.",
        "위험취약형": "⚠️ 재무 위험이 높은 유형입니다. 지출 관리와 복지 연계가 필요합니다.",
        "균형형": "⚖️ 자산과 연금이 안정적인 편으로, 보수적인 전략이 적합합니다.",
        "고소비형": "💳 소비가 많은 유형으로 절세 전략 및 예산 재조정이 필요합니다.",
        "자산의존형": "🏦 연금보다는 자산에 의존도가 높으며, 자산 관리 전략이 중요합니다.",
        "연금의존형": "📥 자산보다 연금에 의존하는 경향이 강한 유형입니다.",
        "소득취약형": "📉 낮은 소득과 자산 구조로, 기초 재정 안정이 중요합니다.",
        "복합형": "🔀 복합적인 특성을 지니며, 맞춤형 분석과 전략 수립이 요구됩니다."
    }

    st.markdown("### 📝 유형 설명")
    st.markdown(descriptions.get(label, ""))


