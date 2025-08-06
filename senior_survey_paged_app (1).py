import streamlit as st

# 세션 상태 초기화
if "started" not in st.session_state:
    st.session_state.started = False
if "page" not in st.session_state:
    st.session_state.page = 0
if "responses" not in st.session_state:
    st.session_state.responses = {}

# 💬 0단계: 연금 수령 여부 확인
pension_receive = st.radio("연금 수령 여부", ("예", "아니오"))

# 🎯 설문 시작 버튼
if pension_receive == "예" and not st.session_state.started:
    if st.button("설문 시작하기"):
        st.session_state.started = True
        st.success("설문을 시작합니다.")

# 🚫 연금 수령 안함 선택 시
elif pension_receive == "아니오":
    st.warning("이 설문은 연금 수령자를 대상으로 하고 있습니다.")

# ✅ 1단계: 설문 진행
if pension_receive == "예" and st.session_state.started:

    # 문항 정의
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

    total_questions = len(questions)
    current_page = st.session_state.page
    q = questions[current_page]

    # 현재 문항 출력
    st.markdown(f"**Q{current_page+1}. {q[0]}**")

    # 입력 위젯 처리
    if q[1] == "number":
        answer = st.number_input("", min_value=0, key=q[2])
    elif q[1] == "selectbox":
        answer = st.selectbox("", q[3], key=q[2])

    # 다음 버튼
    if st.button("다음"):
        st.session_state.responses[q[2]] = answer
        if current_page + 1 < total_questions:
            st.session_state.page += 1
        else:
            st.success("설문이 완료되었습니다.")

    # 설문 완료 후 결과 출력
    if current_page + 1 == total_questions:
        st.subheader("📊 입력한 설문 결과 요약:")
        st.write(st.session_state.responses)
