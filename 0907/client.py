import streamlit as st
import requests

# FastAPI 서버 URL
SERVER_URL = "http://localhost:8000"

st.title("한국 축제 정보 챗봇")

# 사용자 입력
user_question = st.text_input("축제에 대해 궁금한 점을 물어보세요:")

if st.button("질문하기"):
    if user_question:
        # 서버에 요청 보내기
        response = requests.post(f"{SERVER_URL}/ask", json={"text": user_question})
        
        if response.status_code == 200:
            answer = response.json()["answer"]
            st.write("답변:")
            st.write(answer)
        else:
            st.error("서버에서 응답을 받지 못했습니다. 다시 시도해 주세요.")
    else:
        st.warning("질문을 입력해 주세요.")

st.markdown("---")
st.write("이 챗봇은 2024년 한국 지역 축제 정보를 제공합니다.")
