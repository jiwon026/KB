import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# 📌 데이터 로딩 (같은 폴더에 있어야 함)
@st.cache_data
def load_data():
    file_path = "노령연금_수급자통계.csv"
    return pd.read_csv(file_path, encoding='cp949')

df = load_data()

# 📌 월 수급 구간 대표값 설정
representative_values = {
    "20만원 미만": 10,
    "20만원∼40만원 미만": 30,
    "40만원∼60만원 미만": 50,
    "60만원∼80만원 미만": 70,
    "80만원∼100만원 미만": 90,
    "100만원∼130만원 미만": 115,
    "130만원∼160만원 미만": 145,
    "160만원∼200만원 미만": 180,
    "200만원 이상": 210
}

# 📌 국민연금 수령액 추정
def estimate_average_pension(df, gender='여자', period='가입기간 10~19년'):
    column_map = {
        '가입기간 10~19년': {'남자': '남자(가입기간 10~19년)', '여자': '여자(가입기간 10~19년)'},
        '가입기간 20년이상': {'남자': '남자(가입기간 20년이상)', '여자': '여자(가입기간 20년이상)'},
        '조기': {'남자': '남자(조기)', '여자': '여자(조기)'}
    }
    try:
        target_col = column_map[period][gender]
    except KeyError:
        return None

    total_people = 0
    total_amount = 0

    for _, row in df.iterrows():
        bracket = row['구분'].strip()
        if bracket in representative_values:
            people = row[target_col]
            avg_value = representative_values[bracket]
            total_people += people
            total_amount += people * avg_value

    if total_people == 0:
        return None
    else:
        return round(total_amount / total_people, 1)

# 📌 시뮬레이션 함수
def retirement_simulation(current_age, end_age, current_assets, monthly_income, monthly_expense,
                          inflation_rate=0.03, investment_return=0.02):
    asset = current_assets
    yearly_log = []
    expense = monthly_expense
    depletion_age = None

    for age in range(current_age, end_age + 1):
        annual_income = monthly_income * 12
        annual_expense = expense * 12
        delta = annual_income - annual_expense
        asset += delta
        if asset > 0:
            asset *= (1 + investment_return)

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

        expense *= (1 + inflation_rate)

    return yearly_log, depletion_age

# 📌 투자상품 적용 시
def simulate_with_investment(current_age, end_age, current_assets, monthly_income, monthly_expense):
    return retirement_simulation(current_age, end_age, current_assets, monthly_income, monthly_expense,
                                 inflation_rate=0.03, investment_return=0.05)

# 📌 투자상품 적용 시
def simulate_with_investment(current_age, end_age, current_assets, monthly_income, monthly_expense):
    return retirement_simulation(current_age, end_age, current_assets, monthly_income, monthly_expense,
                                 inflation_rate=0.03, investment_return=0.05)


    surplus = monthly_income - monthly_expense
    if surplus > 0:
        if risk_level == '공격형':
            return {
                "추천": "📊 고위험 자산 (해외 ETF, AI펀드 등)",
                "이유": "위험 감수 성향이 높고, 수입이 지출보다 많기 때문에 적극적 투자 가능"
            }
        elif risk_level == '중립형':
            return {
                "추천": "📈 중위험 자산 (국내 ETF, 채권혼합형 등)",
                "이유": "안정성과 수익성을 균형 있게 고려한 포트폴리오 추천"
            }
        else:
            return {
                "추천": "🏦 정기예금 또는 원금보장형 상품",
                "이유": "안전 위주의 투자 성향으로, 예적금 또는 원금보장 상품이 적합"
            }
    else:
        return {
            "추천": "📉 소비조절 컨설팅 또는 소액 적립식 저축",
            "이유": "현재 지출이 소득보다 많아 자산이 줄고 있어 소비 구조 조정이 우선입니다."
        }

# 📌 Streamlit 인터페이스
st.title("🧓 국민연금 기반 노후 시뮬레이션")
st.write("📊 국민연금 통계 기반 자동 계산")

col1, col2 = st.columns(2)
gender = col1.selectbox("성별", ['남자', '여자'])
period = col2.selectbox("수급 유형", ['가입기간 10~19년', '가입기간 20년이상', '조기'])
current_age = st.slider("현재 나이", 60, 80, 67)
end_age = st.slider("예상 수명", 85, 100, 95)
current_assets = st.number_input("현재 자산 (만원)", 0, 100000, 9000)
monthly_expense = st.number_input("월 지출 (만원)", 50, 300, 130)
other_income = st.number_input("기타 월 수입 (만원)", 0, 200, 10)
risk_level = st.radio("투자 성향", ['안정형', '중립형', '공격형'])

# 📌 실행
estimated_pension = estimate_average_pension(df, gender, period)
if estimated_pension is None:
    st.error("해당 조건의 국민연금 데이터를 찾을 수 없습니다.")
    st.stop()

monthly_income = estimated_pension + other_income
log_base, depletion_base = retirement_simulation(current_age, end_age, current_assets,
                                                 monthly_income, monthly_expense)
log_invest, depletion_invest = simulate_with_investment(current_age, end_age, current_assets,
                                                        monthly_income, monthly_expense)
recommendation = recommend_financial_product(depletion_base, current_age, current_assets,
                                             monthly_income, monthly_expense, risk_level)

# 📌 결과 출력
st.success(f"▶️ 자동 추정된 국민연금 수령액: {estimated_pension}만원/월")
if depletion_base:
    st.warning(f"⚠️ 현재 지출 기준으로는 약 **{depletion_base}세**에 자산이 고갈될 수 있어요.")
else:
    st.info("🎉 자산이 고갈되지 않고 유지될 수 있어요.")

st.markdown(f"""
✅ **[맞춤형 금융상품 추천]**  
- 추천 상품: {recommendation['추천']}  
- 추천 이유: {recommendation['이유']}
""")

# 📌 그래프 시각화
df_base = pd.DataFrame(log_base)
df_invest = pd.DataFrame(log_invest)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df_base['나이'], df_base['잔액'], label='기본 시나리오 (2%)')
ax.plot(df_invest['나이'], df_invest['잔액'], label='투자 시나리오 (5%)', linestyle='--')
ax.axhline(0, color='gray', linestyle=':')
ax.set_title("💰 자산 시나리오 비교")
ax.set_xlabel("나이")
ax.set_ylabel("잔액 (만원)")
ax.legend()
ax.grid(True)
st.pyplot(fig)
