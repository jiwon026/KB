import streamlit as st
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager as fm
from streamlit_folium import folium_static

# ✅ 한글 폰트 적용
font_path = "./NanumGothic-Regular.ttf"
fontprop = fm.FontProperties(fname=font_path)
mpl.rcParams['axes.unicode_minus'] = False
plt.rc('font', family=fontprop.get_name())

# ✅ 페이지 설정
st.set_page_config(page_title="시니어 금융 설문", layout="centered")

# ✅ CSS 스타일링
st.markdown("""
<style>
.big-title {
    font-size: 40px;
    font-weight: 900;
    color: black;
    margin-bottom: 0;
}
.sub-title {
    font-size: 24px;
    font-weight: 600;
    color: black;
    margin-top: 0;
}
</style>
""", unsafe_allow_html=True)

# ✅ 제목
st.markdown('<p class="big-title">💰 시니어 금융 설문 진단</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">설문을 통해 나의 금융 유형과 자산 고갈 시점을 예측하고, 맞춤형 상품을 추천받아보세요.</p>', unsafe_allow_html=True)

# ✅ 설문 입력
with st.form("survey_form"):
    age = st.number_input("현재 나이", min_value=55, max_value=100, value=65)
    assets = st.number_input("현재 총 자산 (만원)", min_value=0, step=100, value=50000)
    income = st.number_input("월 소득 (만원)", min_value=0, step=10, value=150)
    pension = st.number_input("월 연금 수령액 (만원)", min_value=0, step=10, value=80)
    living_cost = st.number_input("월 생활비 (만원)", min_value=0, step=10, value=200)
    risk = st.radio("투자 성향", ["낮은 위험", "중간 위험", "높은 위험"])
    submitted = st.form_submit_button("진단하기")

# ✅ 계산 함수
def estimate_depletion_age(start_asset, income, pension, cost, current_age):
    balance = start_asset
    age = current_age
    while balance > 0 and age <= 100:
        monthly_net = income + pension - cost
        balance += (monthly_net * 12)
        age += 1
    return age if balance <= 0 else None

def simulate_assets(start_asset, income, pension, cost, start_age=65, end_age=100):
    years = list(range(start_age, end_age + 1))
    assets = []
    balance = start_asset
    for age in years:
        monthly_net = income + pension - cost
        balance += (monthly_net * 12)
        if balance < 0:
            balance = 0
        assets.append(balance)
    return pd.DataFrame({"나이": years, "예상 자산": assets})

# ✅ 결과 출력
if submitted:
    st.divider()
    st.header("📊 진단 결과")

    depletion_age = estimate_depletion_age(assets, income, pension, living_cost, age)

    # ⚠️ 자산 고갈 메시지
    if depletion_age:
        st.warning(f"⚠️ 현재 자산은 약 **{depletion_age}세**에 고갈될 수 있어요.")
    else:
        st.success("✅ 현재 자산과 수입 구조로는 특별한 고갈 위험이 없습니다.")

    # ✅ 상품 추천
    st.markdown("### ✅ [맞춤형 상품 추천]")
    recommended_product = "📘 연금저축펀드"
    if depletion_age:
        reason = f"{depletion_age}세 자산 고갈 위험 + {risk} 성향 → 절세 + 수익 추구형이 적합합니다."
    else:
        reason = f"{risk} 성향에 적합한 절세형 상품입니다."
    st.markdown(f"- 추천 상품: {recommended_product}")
    st.markdown(f"- 추천 이유: {reason}")

    # 📈 자산 변화 그래프
    sim_df = simulate_assets(assets, income, pension, living_cost, age)
    st.altair_chart(
        alt.Chart(sim_df).mark_line().encode(
            x="나이",
            y=alt.Y("예상 자산", scale=alt.Scale(zero=True)),
            tooltip=["나이", "예상 자산"]
        ).properties(
            title="📉 자산 변화 시뮬레이션"
        ), use_container_width=True
    )
