import pandas as pd
from typing import List
from openai import OpenAI
import chromadb
from chromadb.config import Settings
import os
from fastapi import FastAPI
from pydantic import BaseModel
import urllib.parse
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import logging
import time

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# .env 파일에서 키 불러오기
load_dotenv()
client = OpenAI(
    api_key="up_oLxU2aBXrVgGjZY3ejAbH021gtH0e",
    base_url="https://api.upstage.ai/v1/solar"
)

# ChromaDB 클라이언트 설정
chroma_client = chromadb.Client(Settings(persist_directory="./chroma_db"))

# 컬렉션 생성 또는 기존 컬렉션 가져오기
collection = chroma_client.get_or_create_collection(name="festival_collection3")

def prepare_documents(df: pd.DataFrame):
    logger.info("문서 준비 시작")
    documents = []
    for _, row in df.iterrows():
        doc = f"축제명: {row['축제명']}, 지역: {row['광역자치단체명']} {row['기초자치단체명']}, "
        doc += f"유형: {row['축제유형']}, 기간: {row['개최기간']}, 장소: {row['개최장소']}, "
        doc += f"방식: {row['개최방식']}, 예산: {row['합계 예산(백만원)']}, "
        doc += f"방문객 수: {row['방문객수(2023년) 합계']}"
        documents.append(doc)
    logger.info(f"총 {len(documents)}개의 문서 준비 완료")
    return documents

def add_documents(texts: List[str]):
    logger.info(f"{len(texts)}개의 문서에 대한 임베딩 생성 시작")
    embeddings = []
    for i, text in enumerate(texts):
        # 유니코드 문자를 안전하게 처리
        encoded_text = text.encode('utf-8', errors='ignore').decode('utf-8')
        try:
            response = client.embeddings.create(
                input=encoded_text,
                model="solar-embedding-1-large-query"
            )
            embeddings.append(response.data[0].embedding)
            if i % 100 == 0:
                logger.info(f"{i}개의 임베딩 생성 완료")
        except Exception as e:
            logger.error(f"텍스트 임베딩 생성 중 오류 발생: {e}")
            continue  # 오류가 발생한 텍스트는 건너뛰고 계속 진행

    # 임베딩이 생성된 텍스트만 ChromaDB에 추가
    valid_texts = [text for text, emb in zip(texts, embeddings) if emb]
    valid_embeddings = [emb for emb in embeddings if emb]

    if valid_texts and valid_embeddings:
        logger.info(f"{len(valid_texts)}개의 유효한 문서를 ChromaDB에 추가")
        for i in range(0, len(valid_texts), 10):
            batch_texts = valid_texts[i:i+10]
            batch_embeddings = valid_embeddings[i:i+10]
            batch_ids = [f"doc_{j}" for j in range(i, i+len(batch_texts))]
            
            collection.add(
                embeddings=batch_embeddings,
                documents=batch_texts,
                ids=batch_ids
            )
            
            logger.info(f"{i+len(batch_texts)}개의 문서 추가 완료")
            time.sleep(0.5)
    else:
        logger.warning("유효한 임베딩이 생성되지 않았습니다. 입력 텍스트와 API 키를 확인하세요.")

def query(question: str, k: int = 3):
    logger.info(f"질문에 대한 임베딩 생성 시작: {question}")
    # Solar 모델을 사용하여 질문 임베딩
    response = client.embeddings.create(
        input=question,
        model="solar-embedding-1-large-query"
    )
    query_embedding = response.data[0].embedding
    
    # 유사한 문서 검색
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )
    
    logger.info(f"{k}개의 유사한 문서 검색 완료")
    return results['documents'][0]

system_prompt = """
당신의 역할은 Context를 바탕으로 질문에 대한 답변을 제공하는 것입니다.
Context에 없는 내용은 절대 추가하지 말아주세요.
Question은 사용자의 질문입니다.
Answer은 사용자에게 제공할 답변입니다.
"""


def generate_answer(question: str, context: List[str]):
    logger.info("답변 생성 시작")
    # Solar API를 사용하여 답변 생성
    # print(context)
    for c in context:
        print(c)
        print("-"*100)
    
    prompt = f"Context:\n{''.join(context)}\n\nQuestion: {question}\nAnswer:"
    response = client.chat.completions.create(
        model="solar-1-mini-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )
    logger.info("답변 생성 완료")
    return response.choices[0].message.content.strip()

class Question(BaseModel):
    text: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("애플리케이션 시작")
    global df, documents
    # CSV 파일 읽기
    df = pd.read_csv('/Users/bokyung/Desktop/LLM-RAG-LANGCHAIN/0907/2024년 지역축제 개최계획(수정).csv', encoding='utf-8')
    df.columns = df.iloc[0]
    df = df[1:]
    
    # 열 이름에서 개행 문자 제거 및 공백 제거
    df.columns = df.columns.str.replace('\n', '').str.strip()
    
    logger.info("CSV 파일 로드 완료")
    
    # 문서 준비 및 추가
    documents = prepare_documents(df)
    logger.info("문서 준비 완료")
    add_documents(documents)
    logger.info("문서 추가 완료")
    
    yield
    
    logger.info("애플리케이션 종료")

app = FastAPI(lifespan=lifespan)

@app.post("/ask")
async def ask_question(question: Question):
    logger.info(f"질문 수신: {question.text}")
    try:
        relevant_docs = query(question.text)
        logger.info("관련 문서 검색 완료")
        answer = generate_answer(question.text, relevant_docs)
        logger.info("답변 생성 완료")
        return {"question": question.text, "answer": answer}
    except Exception as e:
        logger.error(f"질문 처리 중 오류 발생: {e}")
        return {"error": str(e)}, 500

if __name__ == "__main__":
    import uvicorn
    logger.info("서버 시작")
    uvicorn.run(app, host="0.0.0.0", port=8000)
