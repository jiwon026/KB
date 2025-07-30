import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# 📌 데이터 로딩 
@st.cache_data
def load_data():
    file_path = "(202503공시)2-6-1 노령연금 수급자 수-노령연금 종류별성별_월 수급금액별.csv"
    return pd.read_csv(file_path, encoding='cp949')

df = load_data()


# 📌 대표값 설정
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

# 📌 연금 수령액 추정 함수
def estimate_average_pension(df, gender='여자', period='가입기간 10~19년'):
    column_map = {
        '가입기간 10~19년': {
            '남자': '남자(가입기간 10~19년)',
            '여자': '여자(가입기간 10~19년)'
        },
        '가입기간 20년 이상': {
            '남자': '남자(가입기간 20년이상)',
            '여자': '여자(가입기간 20년이상)'
        },
        '조기': {
            '남자': '남자(조기)',
            '여자': '여자(조기)'
        }
    }
    target_col = column_map[period][gender]
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

# 📌 금융상품 적용 시 시뮬레이션
def simulate_with_financial_product(current_age, end_age, current_assets, monthly_income, monthly_expense):
    return retirement_simulation(current_age, end_age, current_assets, monthly_income, monthly_expense,
                                 inflation_rate=0.03, investment_return=0.05)

# 📌 맞춤형 추천 로직
def recommend_financial_product(depletion_age, current_age, current_assets, monthly_income, monthly_expense, risk):
    if depletion_age:
        return {
            "추천": "💡 연금형 금융상품",
            "이유": f"{depletion_age}세에 자산 고갈이 예상되므로, 매달 일정 수입을 주는 상품이 적합합니다."
        }

    surplus = monthly_income - monthly_expense
    if surplus > 0:
        if risk == "높음":
            return {
                "추천": "📈 주식형 펀드 / ETF",
                "이유": "높은 수익률을 기대하는 성향이므로 주식형 상품이 적합합니다."
            }
        elif risk == "중간":
            return {
                "추천": "📊 혼합형 펀드 / 채권 ETF",
                "이유": "중간 수준의 위험 감수 성향에는 안정성과 수익성이 균형 잡힌 상품이 적절합니다."
            }
        else:  # 위험 성향 낮음
            if current_assets > 20000:
                return {
                    "추천": "🏦 정기예금 / 채권형 펀드",
                    "이유": "자산이 넉넉한 편이므로 안정적 상품으로 보존이 유리합니다."
                }
            else:
                return {
                    "추천": "🔐 원금보장형 상품 (ELB 등)",
                    "이유": "자산이 많지 않으므로 손실 없는 안정적 상품이 우선입니다."
                }
    return {
        "추천": "📉 소비 구조 점검 및 지출 조정 컨설팅",
        "이유": "지출이 수입보다 많아 저축이 어려우므로, 소비 개선이 필요합니다."
    }

# 📌 Streamlit 시작
st.set_page_config(page_title="노후 시나리오 시뮬레이터", page_icon="💸")
st.title("💸 노후 시나리오 시뮬레이터")

# 사용자 입력
with st.sidebar:
    st.header("📝 사용자 입력")
    gender = st.selectbox("성별", ["남자", "여자"])
    period = st.selectbox("국민연금 가입기간", ["가입기간 10~19년", "가입기간 20년 이상", "조기"])
    risk = st.radio("📊 위험 성향", ["낮음", "중간", "높음"])

    current_age = st.slider("현재 나이", 55, 80, 67)
    end_age = st.slider("예상 수명", 85, 110, 100)
    current_assets = st.number_input("현재 자산 (만원)", value=9000)
    monthly_expense = st.number_input("월 지출 예상 (만원)", value=130)
    other_income = st.number_input("기타 월 수입 (만원)", value=10)
    uploaded_file = st.file_uploader("📁 국민연금 수급자 통계 파일 업로드 (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding='cp949')
    estimated_pension = estimate_average_pension(df, gender, period)
    if estimated_pension is None:
        st.error("❌ 유효한 연금 수급액을 계산할 수 없습니다. 파일을 확인해주세요.")
    else:
        monthly_income = estimated_pension + other_income

        log_base, depletion_base = retirement_simulation(
            current_age, end_age, current_assets, monthly_income, monthly_expense
        )
        log_invest, depletion_invest = simulate_with_financial_product(
            current_age, end_age, current_assets, monthly_income, monthly_expense
        )

        recommendation = recommend_financial_product(
            depletion_base, current_age, current_assets,
            monthly_income, monthly_expense, risk
        )

        st.subheader("📌 결과 요약")
        st.markdown(f"▶️ 자동 추정된 국민연금 수령액: **{estimated_pension}만원/월**")
        if depletion_base:
            st.warning(f"⚠️ 자산은 **{depletion_base}세**에 고갈될 수 있어요.")
        else:
            st.success("🎉 자산이 고갈되지 않고 유지될 수 있어요!")

        st.markdown("### ✅ [맞춤형 금융상품 추천]")
        st.markdown(f"- **추천 상품**: {recommendation['추천']}")
        st.markdown(f"- **추천 이유**: {recommendation['이유']}")

        # 📊 자산 추이 시각화
        df_base = pd.DataFrame(log_base)
        df_invest = pd.DataFrame(log_invest)

        st.subheader("📈 자산 시나리오 비교")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df_base['나이'], df_base['잔액'], label='기본 시나리오 (2%)')
        ax.plot(df_invest['나이'], df_invest['잔액'], label='금융상품 적용 (5%)', linestyle='--')
        ax.axhline(0, color='gray', linestyle=':')
        ax.set_xlabel("나이")
        ax.set_ylabel("잔액 (만원)")
        ax.set_title("자산 변화 비교 그래프")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
else:
    st.info("📁 국민연금 수급자 파일을 업로드하면 분석이 시작됩니다.")
