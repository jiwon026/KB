import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# 📌 대표값과 수령액 추정 함수는 기존 코드에서 그대로 사용 가능

# 👉 사용자 입력
st.title("💰 노후 시나리오 시뮬레이터")

gender = st.selectbox("성별을 선택하세요", ["남자", "여자"])
period = st.selectbox("연금 가입 기간", ["가입기간 10~19년", "가입기간 20년 이상", "조기"])
risk_tolerance = st.selectbox("위험 성향", ["낮음", "중간", "높음"])

current_age = st.slider("현재 나이", 55, 80, 67)
end_age = st.slider("예상 생존 나이", 85, 110, 100)
current_assets = st.number_input("현재 자산 (만원)", value=9000)
monthly_expense = st.number_input("월 지출 예상 (만원)", value=130)
other_income = st.number_input("기타 월 수입 (만원)", value=10)

# 📁 CSV 파일 불러오기
uploaded_file = st.file_uploader("(202503공시)2-6-1 노령연금 수급자 수-노령연금 종류별성별_월 수급금액별", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding='cp949')
    
    estimated_pension = estimate_average_pension(df, gender, period)
    monthly_income = estimated_pension + other_income

    # 💡 시뮬레이션 실행
    base_log, base_depletion = retirement_simulation(
        current_age, end_age, current_assets, monthly_income, monthly_expense
    )
    invest_log, invest_depletion = simulate_with_financial_product(
        current_age, end_age, current_assets, monthly_income, monthly_expense
    )
    recommendation = recommend_financial_product(
        base_depletion, current_age, current_assets,
        monthly_income, monthly_expense, risk_tolerance
    )

    # 💬 결과 출력
    st.markdown(f"### ▶️ 예상 국민연금 수령액: **{estimated_pension}만원/월**")
    if base_depletion:
        st.warning(f"⚠️ 자산이 **{base_depletion}세**에 고갈될 수 있습니다.")
    else:
        st.success("🎉 자산이 고갈되지 않고 유지될 수 있어요!")

    st.markdown(f"### ✅ 추천 상품: **{recommendation['추천']}**")
    st.markdown(f"📌 추천 이유: {recommendation['이유']}")

    # 📊 시각화
    df_base = pd.DataFrame(base_log)
    df_invest = pd.DataFrame(invest_log)

    st.line_chart({
        "기본 시나리오 (2%)": df_base.set_index("나이")["잔액"],
        "금융상품 시나리오 (5%)": df_invest.set_index("나이")["잔액"]
    }) 
