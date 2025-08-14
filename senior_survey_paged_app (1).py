import streamlit as st

def render_main_home():
    # 커스텀 CSS 스타일
    st.markdown("""
    <style>
      /* 전체 페이지 배경 */
      .stApp {
        max-width: 350px;
        margin: 0 auto;
        background-color: #f8f9fa;
        padding: 20px;
    }
      
      /* 메인 컨테이너 카드 */
      .main-container {
        max-width: 480px;
        margin: 2rem auto;
        padding: 2.5rem;
        background: white;
        border-radius: 20px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        text-align: center;
      }
      
      /* KB 로고 및 브랜드 */
      .brand-section {
        margin-bottom: 1.5rem;
      }
      
      .kb-logo {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, #FFD700, #FFA500);
        color: #8B4513;
        font-weight: 900;
        font-size: 28px;
        padding: 10px 16px;
        border-radius: 10px;
        margin-right: 15px;
        border: 2px solid #FF8C00;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
      }
      
      .elderly-icons {
        font-size: 50px;
        margin-left: 10px;
      }
      
      .app-title {
        font-size: 36px;
        font-weight: 800;
        color: #2c3e50;
        margin: 1.5rem 0;
        line-height: 1.2;
      }
      
      /* 메뉴 버튼들 */
      .stButton > button {
        width: 100% !important;
        height: 80px !important;
        border-radius: 20px !important;
        font-size: 18px !important;
        font-weight: bold !important;
        border: none !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
        transition: all 0.2s ease !important;
        white-space: pre-line !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
    }
      
      /* 각 버튼별 색상 */
      .btn-type {
        background: #FFE4B5;
        color: #8B4513;
      }
      
      .btn-calc {
        background: #E6F3FF;
        color: #0066CC;
      }
      
      .btn-sim {
        background: #E8F5E8;
        color: #2E8B57;
      }
      
      .btn-reco {
        background: #FFE4E1;
        color: #DC143C;
      }
      
      .btn-reset {
        background: #F0E6FF;
        color: #6A0DAD;
      }
      
      .menu-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
      }
      
      /* Streamlit 기본 버튼 스타일 오버라이드 */
      .stButton > button {
        width: 100% !important;
        padding: 22px 24px !important;
        margin: 12px 0 !important;
        border: none !important;
        border-radius: 16px !important;
        font-size: 22px !important;
        font-weight: 700 !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        min-height: 70px !important;
      }
      
      /* 각 버튼의 색상 적용 */
      .stButton:nth-child(1) > button {
        background: #FFE4B5 !important;
        color: #8B4513 !important;
      }
      
      .stButton:nth-child(2) > button {
        background: #E6F3FF !important;
        color: #0066CC !important;
      }
      
      .stButton:nth-child(3) > button {
        background: #E8F5E8 !important;
        color: #2E8B57 !important;
      }
      
      .stButton:nth-child(4) > button {
        background: #FFE4E1 !important;
        color: #DC143C !important;
      }
      
      .stButton:nth-child(5) > button {
        background: #F0E6FF !important;
        color: #6A0DAD !important;
      }
      
      .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(0,0,0,0.15) !important;
      }
      
      /* 하단 설명 텍스트 */
      .footer-text {
        margin-top: 1.5rem;
        font-size: 16px;
        color: #7f8c8d;
        font-style: italic;
      }
      
      /* 반응형 디자인 */
      @media (max-width: 480px) {
        .main-container {
          margin: 1rem;
          padding: 2rem;
        }
        
        .app-title {
          font-size: 30px;
        }
        
        .kb-logo {
          font-size: 24px;
          padding: 8px 14px;
        }
        
        .elderly-icons {
          font-size: 40px;
        }
        
        .menu-button, .stButton > button {
          font-size: 20px !important;
          padding: 20px 22px !important;
          min-height: 65px !important;
        }

        /* 모바일 최적화 */
        @media (max-width: 400px) {
            .custom-button {
                font-size: 18px;
                padding: 20px 15px;
            }
            
            .green-button, .pink-button {
                font-size: 14px;
                padding: 18px 8px;
            }
            
            .kb-logo {
                font-size: 32px;
            }
            
            .title {
                font-size: 20px;
            }
        }
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
        st.session_state.flow = "survey"
        st.rerun()
    
    # 2) 연금 계산하기
    if st.button("연금 계산하기", key="home_btn_predict"):
        st.session_state.flow = "predict"
        st.rerun()
    
    # 3) 노후 시뮬레이션
    if st.button("노후 시뮬레이션", key="home_btn_sim"):
        st.session_state.flow = "sim"
        st.rerun()
    
    # 4) 맞춤 상품 추천
    if st.button("맞춤 상품 추천", key="home_btn_reco"):
        st.session_state.flow = "recommend"
        st.rerun()
    
    # 5) 설문 다시하기
    if st.button("설문 다시하기", key="home_btn_reset"):
        st.session_state.flow = "survey"
        st.session_state.question_step = 1
        st.session_state.answers = {}
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)  # menu-section 닫기
    
    # 하단 설명
    st.markdown(
        '<div class="footer-text">버튼을 눌러 다음 단계로 이동하세요</div>',
        unsafe_allow_html=True
    )
    
    st.markdown('</div>', unsafe_allow_html=True)  # main-container 닫기


def render_survey():
    # 설문 전용 CSS 스타일
    st.markdown("""
    <style>
      /* 전체 페이지 배경 */
      .stApp {
        max-width: 350px;
        margin: 0 auto;
        background-color: #f8f9fa;
        padding: 20px;
      }
      
      /* 메인 컨테이너 카드 */
      .survey-container {
        max-width: 480px;
        margin: 2rem auto;
        padding: 2.5rem;
        background: white;
        border-radius: 20px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        text-align: center;
      }
      
      /* KB 로고 및 브랜드 */
      .brand-section {
        margin-bottom: 2rem;
      }
      
      .kb-logo {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, #FFD700, #FFA500);
        color: #8B4513;
        font-weight: 900;
        font-size: 28px;
        padding: 10px 16px;
        border-radius: 10px;
        margin-right: 15px;
        border: 2px solid #FF8C00;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
      }
      
      .elderly-icons {
        font-size: 50px;
        margin-left: 10px;
      }
      
      .survey-title {
        font-size: 24px;
        font-weight: 700;
        color: #2c3e50;
        margin: 1rem 0;
        line-height: 1.3;
      }
      
      /* 질문 섹션 */
      .question-section {
        margin: 2rem 0;
        text-align: left;
      }
      
      .question-text {
        font-size: 22px;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 1.5rem;
        line-height: 1.4;
        text-align: center;
      }
      
      /* 진행률 바 */
      .progress-container {
        margin: 1.5rem 0;
        background-color: #f0f0f0;
        border-radius: 10px;
        height: 8px;
        overflow: hidden;
      }
      
      .progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #FFD700, #FFA500);
        border-radius: 10px;
        transition: width 0.3s ease;
      }
      
      /* 입력 필드 스타일 */
      .stTextInput > div > div > input {
        height: 60px !important;
        font-size: 18px !important;
        border-radius: 15px !important;
        border: 2px solid #e0e0e0 !important;
        padding: 15px 20px !important;
        text-align: center !important;
      }
      
      .stTextInput > div > div > input:focus {
        border-color: #FFA500 !important;
        box-shadow: 0 0 0 2px rgba(255, 165, 0, 0.2) !important;
      }
      
      /* 선택 버튼들 */
      .choice-buttons {
        display: flex;
        flex-direction: column;
        gap: 15px;
        margin: 1.5rem 0;
      }
      
      .stButton > button {
        width: 100% !important;
        height: 70px !important;
        border-radius: 15px !important;
        font-size: 18px !important;
        font-weight: 600 !important;
        border: 2px solid #e0e0e0 !important;
        background: white !important;
        color: #2c3e50 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
        transition: all 0.3s ease !important;
      }
      
      .stButton > button:hover {
        background: #FFF8DC !important;
        border-color: #FFA500 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.12) !important;
      }
      
      /* 네비게이션 버튼들 */
      .nav-buttons {
        display: flex;
        justify-content: space-between;
        margin-top: 2rem;
        gap: 15px;
      }
      
      .nav-button-prev {
        background: #f8f9fa !important;
        color: #6c757d !important;
        border: 2px solid #dee2e6 !important;
      }
      
      .nav-button-next {
        background: linear-gradient(135deg, #FFD700, #FFA500) !important;
        color: #8B4513 !important;
        border: 2px solid #FF8C00 !important;
        font-weight: 700 !important;
      }
      
      .nav-button-prev:hover {
        background: #e9ecef !important;
        transform: translateY(-1px) !important;
      }
      
      .nav-button-next:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(255, 165, 0, 0.3) !important;
      }
      
      /* 완료 버튼 */
      .complete-button {
        background: linear-gradient(135deg, #28a745, #20c997) !important;
        color: white !important;
        border: none !important;
        font-weight: 700 !important;
        font-size: 20px !important;
        height: 80px !important;
      }
      
      .complete-button:hover {
        background: linear-gradient(135deg, #218838, #1abc9c) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(40, 167, 69, 0.3) !important;
      }
      
      /* 반응형 디자인 */
      @media (max-width: 480px) {
        .survey-container {
          margin: 1rem;
          padding: 2rem;
        }
        
        .survey-title {
          font-size: 20px;
        }
        
        .question-text {
          font-size: 20px;
        }
        
        .kb-logo {
          font-size: 24px;
          padding: 8px 14px;
        }
        
        .elderly-icons {
          font-size: 40px;
        }
        
        .stButton > button {
          font-size: 16px !important;
          height: 60px !important;
        }
        
        .nav-buttons {
          flex-direction: column;
        }
      }
    </style>
    """, unsafe_allow_html=True)

    # 설문 질문 데이터
    receiving_questions = {
        1: {
            "title": "연금 계산기",
            "question": "1. 나이를\n입력해주세요.",
            "type": "input",
            "placeholder": "나이를 입력하세요"
        },
        2: {
            "title": "연금 계산기", 
            "question": "2. 성별을\n선택해주세요.",
            "type": "choice",
            "options": ["남성", "여성"]
        },
        3: {
            "title": "연금 계산기",
            "question": "3. 가구원 수를\n입력해주세요.",
            "type": "input",
            "placeholder": "가구원 수를 입력하세요"
        },
        4: {
            "title": "연금 계산기",
            "question": "4. 피부양자가\n있나요?",
            "type": "choice",
            "options": ["예", "아니오"]
        },
        5: {
            "title": "연금 계산기",
            "question": "5. 현재 보유한\n금융자산을\n입력해주세요.",
            "type": "input",
            "placeholder": "현재 보유 금융자산을 입력하세요 (만원)"
        },
        6: {
            "title": "연금 계산기",
            "question": "6. 월 수령하는\n연금 급여를\n입력해주세요.",
            "type": "input",
            "placeholder": "월 수령하는 연금 급여를 입력하세요 (만원)"
        },
        7: {
            "title": "연금 계산기",
            "question": "7. 월 평균\n지출비를\n입력해주세요.",
            "type": "input",
            "placeholder": "월 평균 지출비를 입력하세요 (만원)"
        },
        8: {
            "title": "연금 계산기",
            "question": "8. 평균 월소득을\n입력해주세요.",
            "type": "input",
            "placeholder": "평균 월소득을 입력하세요 (만원)"
        },
        9: {
            "title": "연금 계산기",
            "question": "9. 투자 성향을\n선택해주세요.",
            "type": "choice",
            "options": ["안정형", "안정추구형", "위험중립형", "적극투자형"]
        }
    }
    
    # 세션 상태 초기화
    if 'question_step' not in st.session_state:
        st.session_state.question_step = 1
    if 'answers' not in st.session_state:
        st.session_state.answers = {}
    
    current_q = receiving_questions[st.session_state.question_step]
    total_questions = len(receiving_questions)
    progress_percentage = (st.session_state.question_step / total_questions) * 100
    
    # 메인 컨테이너
    st.markdown('<div class="survey-container">', unsafe_allow_html=True)
    
    # 브랜드 섹션
    st.markdown("""
    <div class="brand-section">
        <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 1rem;">
            <div class="kb-logo">KB</div>
            <div class="elderly-icons">👨‍🦳👩‍🦳</div>
        </div>
        <div class="survey-title">{}</div>
    </div>
    """.format(current_q["title"]), unsafe_allow_html=True)
    
    # 진행률 표시
    st.markdown(f"""
    <div class="progress-container">
        <div class="progress-bar" style="width: {progress_percentage}%"></div>
    </div>
    <div style="text-align: center; margin-bottom: 1rem; color: #666; font-size: 14px;">
        {st.session_state.question_step} / {total_questions}
    </div>
    """, unsafe_allow_html=True)
    
    # 질문 섹션
    st.markdown('<div class="question-section">', unsafe_allow_html=True)
    st.markdown(f'<div class="question-text">{current_q["question"]}</div>', unsafe_allow_html=True)
    
    # 답변 입력/선택 영역
    if current_q["type"] == "input":
        answer = st.text_input(
            "", 
            placeholder=current_q["placeholder"],
            key=f"input_{st.session_state.question_step}",
            label_visibility="collapsed"
        )
        if answer:
            st.session_state.answers[st.session_state.question_step] = answer
            
    elif current_q["type"] == "choice":
        st.markdown('<div class="choice-buttons">', unsafe_allow_html=True)
        for i, option in enumerate(current_q["options"]):
            if st.button(option, key=f"choice_{st.session_state.question_step}_{i}"):
                st.session_state.answers[st.session_state.question_step] = option
                # 자동으로 다음 질문으로 이동
                if st.session_state.question_step < total_questions:
                    st.session_state.question_step += 1
                    st.rerun()
                else:
                    # 설문 완료
                    st.session_state.flow = "results"
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)  # question-section 닫기
    
    # 네비게이션 버튼들
    st.markdown('<div class="nav-buttons">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.question_step > 1:
            if st.button("← 이전", key="prev_btn", help="이전 질문으로"):
                st.session_state.question_step -= 1
                st.rerun()
        else:
            if st.button("← 홈으로", key="home_btn", help="홈으로 돌아가기"):
                st.session_state.flow = "home"
                st.rerun()
    
    with col2:
        # input 타입의 경우 다음 버튼 표시
        if current_q["type"] == "input":
            current_answer = st.session_state.answers.get(st.session_state.question_step)
            if current_answer:
                if st.session_state.question_step < total_questions:
                    if st.button("다음 →", key="next_btn", help="다음 질문으로"):
                        st.session_state.question_step += 1
                        st.rerun()
                else:
                    if st.button("완료 ✓", key="complete_btn", help="설문 완료"):
                        st.session_state.flow = "results"
                        st.rerun()
            else:
                st.button("다음 →", key="next_btn_disabled", disabled=True, help="답변을 입력해주세요")
    
    st.markdown('</div>', unsafe_allow_html=True)  # nav-buttons 닫기
    st.markdown('</div>', unsafe_allow_html=True)  # survey-container 닫기


def render_results():
    """설문 결과 페이지"""
    st.markdown("""
    <div style="max-width: 480px; margin: 2rem auto; padding: 2.5rem; background: white; 
         border-radius: 20px; box-shadow: 0 20px 40px rgba(0,0,0,0.1); text-align: center;">
        <div style="font-size: 24px; font-weight: 700; color: #2c3e50; margin-bottom: 2rem;">
            설문이 완료되었습니다! 🎉
        </div>
        <div style="font-size: 16px; color: #666; margin-bottom: 2rem;">
            입력하신 정보를 바탕으로 분석을 진행하겠습니다.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 답변 내용 표시
    if st.session_state.answers:
        st.write("### 입력된 답변:")
        for step, answer in st.session_state.answers.items():
            st.write(f"질문 {step}: {answer}")
    
    # 홈으로 돌아가기 버튼
    if st.button("홈으로 돌아가기"):
        st.session_state.flow = "home"
        st.rerun()


def render_predict():
    """연금 계산 페이지 (임시)"""
    st.markdown("""
    <div style="max-width: 480px; margin: 2rem auto; padding: 2.5rem; background: white; 
         border-radius: 20px; box-shadow: 0 20px 40px rgba(0,0,0,0.1); text-align: center;">
        <h2>연금 계산하기</h2>
        <p>연금 계산 기능을 구현할 페이지입니다.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("홈으로 돌아가기"):
        st.session_state.flow = "home"
        st.rerun()


def render_sim():
    """노후 시뮬레이션 페이지 (임시)"""
    st.markdown("""
    <div style="max-width: 480px; margin: 2rem auto; padding: 2.5rem; background: white; 
         border-radius: 20px; box-shadow: 0 20px 40px rgba(0,0,0,0.1); text-align: center;">
        <h2>노후 시뮬레이션</h2>
        <p>노후 시뮬레이션 기능을 구현할 페이지입니다.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("홈으로 돌아가기"):
        st.session_state.flow = "home"
        st.rerun()


def render_recommend():
    """맞춤 상품 추천 페이지 (임시)"""
    st.markdown("""
    <div style="max-width: 480px; margin: 2rem auto; padding: 2.5rem; background: white; 
         border-radius: 20px; box-shadow: 0 20px 40px rgba(0,0,0,0.1); text-align: center;">
        <h2>맞춤 상품 추천</h2>
        <p>맞춤 상품 추천 기능을 구현할 페이지입니다.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("홈으로 돌아가기"):
        st.session_state.flow = "home"
        st.rerun()


def main():
    # 페이지 설정
    st.set_page_config(
        page_title="KB 시니어 연금 계산기",
        page_icon="👨‍🦳",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    
    # 세션 상태 초기화
    if 'flow' not in st.session_state:
        st.session_state.flow = "home"
    if 'question_step' not in st.session_state:
        st.session_state.question_step = 1
    if 'answers' not in st.session_state:
        st.session_state.answers = {}
    
    # 페이지 라우팅
    if st.session_state.flow == "home":
        render_main_home()
    elif st.session_state.flow == "survey":
        render_survey()
    elif st.session_state.flow == "results":
        render_results()
    elif st.session_state.flow == "predict":
        render_predict()
    elif st.session_state.flow == "sim":
        render_sim()
    elif st.session_state.flow == "recommend":
        render_recommend()


if __name__ == "__main__":
    main()
