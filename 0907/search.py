import requests
import streamlit as st
import pandas as pd
import re
import os

# SERVER_URL 정의 추가
SERVER_URL = "http://localhost:8000" 
# 페이지 설정을 스크립트의 가장 처음으로 이동
st.set_page_config(page_title="축제 정보 검색", page_icon="🎉")

import pandas as pd
import re

# 데이터 로드
@st.cache_data
def load_data():
    # 현재 파일의 디렉토리 경로를 가져옴
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, '2024년 지역축제 개최계획(수정).csv')
    
    # CSV 파일 읽기
    df = pd.read_csv(csv_path, encoding='utf-8')
    df.columns = df.iloc[0]
    df = df[1:]
    return df

df = load_data()
print(df.head(5))

# 검색 함수 수정
def search_data(query):
    mask = df.apply(lambda row: row.astype(str).str.contains(query, case=False).any(), axis=1)
    return df[mask]

# 굵게 표시 및 밑줄 함수 수정
def highlight_match(text, query):
    pattern = re.compile(f'({re.escape(query)})', re.IGNORECASE)
    return pattern.sub(r'<u>**\1**</u>', str(text))

# 탭별 검색 함수
def general_search():
    query = st.session_state.general_query
    # if query:
    #     # add_recent_search(query)
    #     results = search_data(query)
    #     display_results(results, query, "일반")
    if query:
        response = requests.post(f"{SERVER_URL}/search", json={"text": query})
        
        if response.status_code == 200:
            print(response)
            answer = response.json()["answer"]
            # 쿼리 변수 앞뒤 공백 제거
            display_results(answer, query, "일반")
        elif response.status_code == 429:
            st.error("요청 한도를 초과했습니다. 1분 후에 다시 시도해 주세요.")
        else:
            st.error("서버에서 응답을 받지 못했습니다. 다시 시도해 주세요.")
    else:
        st.warning("질문을 입력해 주세요.")

def embedding_search():
    query = st.session_state.embedding_query
    if query:
        # add_recent_search(query)
        # AI 임베딩 검색 로직 구현 필요
        st.warning("AI 임베딩 검색 기능은 아직 구현되지 않았습니다.")

def augmented_search():
    query = st.session_state.augmented_query
    if query:
        try:
            response = requests.post(f"{SERVER_URL}/ask", json={"text": query})
            print(f"Response status: {response.status_code}")
            print(f"Response content: {response.content}")
            answer = response.json()["answer"]
            answer = highlight_match(answer, query)
            # 쿼리 변수 앞뒤 공백 제거
            query2 = query.strip()
            st.session_state.search_history.append(f"**AI - RAG 검색어: {query2}**\n\n{answer}\n\n---\n\n")
        except Exception as e:
            st.error(f"Error during request: {str(e)}")
    else:
        st.warning("질문을 입력해 주세요.")

def display_results(results, query, search_type):
    response = f"{search_type} 검색 결과: {len(results)}개의 축제를 찾았습니다.\n\n"
    for row in results:
        festival_info = []
        if '축제명' in row:
            festival_info.append(highlight_match(row['축제명'], query))
        if '시군구명' in row:
            festival_info.append(highlight_match(row['시군구명'], query))
        if '개최기간' in row:
            festival_info.append(highlight_match(row['개최기간'], query))
        response += f"- {', '.join(festival_info)}\n"

    # 새로운 검색 결과를 세션 상태에 추가
    st.session_state.search_history.append(f"**{search_type} 검색어: {query}**\n\n{response}\n\n---\n\n")

# 세션 상태 초기화
if 'search_history' not in st.session_state:
    st.session_state.search_history = []
if 'general_query' not in st.session_state:
    st.session_state.general_query = ""
if 'embedding_query' not in st.session_state:
    st.session_state.embedding_query = ""
if 'augmented_query' not in st.session_state:
    st.session_state.augmented_query = ""
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "일반 검색"

# 최근 검색어 관련 함수 및 세션 상태 제거

# 사이드바에 세로로 나열된 탭 추가
with st.sidebar:
    st.title("축제 검색")
    
    st.session_state.current_tab = st.radio(
        "검색 유형 선택",
        #["일반 검색", "AI **임베딩** 검색", "AI **증강생성(RAG)** 검색"]
        ["일반 검색", "AI **증강생성(RAG)** 검색"]
    
    )
    
    if st.session_state.current_tab == "일반 검색":
        st.text_input('일반 검색어를 입력하세요', key='general_query', on_change=general_search)
        st.button('일반 검색', on_click=general_search)
    
    elif st.session_state.current_tab == "AI **임베딩** 검색":
        st.text_input('AI 임베딩 검색어를 입력하세요', key='embedding_query', on_change=embedding_search)
        st.button('AI 임베딩 검색', on_click=embedding_search)
    
    elif st.session_state.current_tab == "AI **증강생성(RAG)** 검색":
        st.text_input('AI 증강생성 검색어를 입력하세요', key='augmented_query', on_change=augmented_search)
        st.button('AI 증강생성 검색', on_click=augmented_search)

# 메인 화면
st.title("축제 정보")

# 검색 결과 표시
st.subheader("검색 기록")
for result in reversed(st.session_state.search_history):
    st.markdown(result, unsafe_allow_html=True)

# 자동 스크롤
st.markdown("<script>window.scrollTo(0,0);</script>", unsafe_allow_html=True)

# 결과 상세 보기
# if st.button('전체 결과 보기'):
#     st.dataframe(df)

# Streamlit 앱 실행 설정
if __name__ == '__main__':
    st.write("축제 정보 검색 앱이 실행되었습니다.")
    # 여기에 메인 앱 로직을 추가할 수 있습니다.