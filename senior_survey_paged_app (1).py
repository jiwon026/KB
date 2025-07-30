import streamlit as st

st.set_page_config(page_title="시니어 금융 설문", page_icon="💸", layout="centered")

st.markdown("### 💬 시니어 금융 유형 설문")
st.markdown("**아래 질문에 순차적으로 응답해주세요.**")

if "page" not in st.session_state:
    st.session_state.page = 0
if "responses" not in st.session_state:
    st.session_state.responses = {}

questions = [
    ("나이를 입력해주세요.", "number", "age"),
    ("성별을 선택해주세요.", "selectbox", "gender", ["남성", "여성"]),
    ("가구원 수를 입력해주세요.", "number", "family_size"),
    ("현재 보유한 금융자산(만원)을 입력해주세요.", "number", "assets"),
    ("월 수령하는 연금 금액(만원)을 입력해주세요.", "number", "pension"),
    ("월 평균 생활비(만원)은 얼마인가요?", "number", "living_cost"),
    ("월 평균 취미/여가비(만원)는 얼마인가요?", "number", "hobby_cost"),
    ("투자 성향을 선택해주세요.", "selectbox", "risk", ["안정형", "중립형", "공격형"])
]

def next_page():
    if st.session_state.get("input_value") is not None:
        current_q = questions[st.session_state.page]
        st.session_state.responses[current_q[2]] = st.session_state.input_value
        st.session_state.page += 1
        st.session_state.input_value = None

if st.session_state.page < len(questions):
    q = questions[st.session_state.page]
    st.markdown(f"**Q{st.session_state.page + 1}. {q[0]}**")

    if q[1] == "number":
        st.number_input(
            label=" ",
            key="input_value",
            step=1,
            format="%d",
            on_change=next_page,
            label_visibility="collapsed"
        )
    elif q[1] == "selectbox":
        st.selectbox(
            label=" ",
            options=q[3],
            key="input_value",
            on_change=next_page,
            label_visibility="collapsed"
        )
else:
    st.success("✅ 모든 질문에 응답하셨습니다!")
    r = st.session_state.responses

    # 점수화 예시
    score = 0
    score += (r["assets"] or 0) * 0.003
    score += (r["pension"] or 0) * 0.05
    score -= (r["living_cost"] or 0) * 0.02
    score -= (r["hobby_cost"] or 0) * 0.01
    score += 1.0 if r["risk"] == "공격형" else (-0.5 if r["risk"] == "안정형" else 0)

    if score >= 7:
        category = "자산운용형"
    elif score >= 4:
        category = "균형형"
    else:
        category = "안정추구형"

    st.markdown(f"### 🧾 결과: **{category}**")
    st.markdown("👉 당신에게 맞는 금융 상품을 추천해드릴게요.")
