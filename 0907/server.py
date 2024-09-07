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

# .env 파일에서 키 불러오기
load_dotenv()
client = OpenAI(
    api_key="up_oLxU2aBXrVgGjZY3ejAbH021gtH0e",
    base_url="https://api.upstage.ai/v1/solar"
)

# ChromaDB 클라이언트 설정
chroma_client = chromadb.Client(Settings(persist_directory="./chroma_db"))

# 컬렉션 생성 또는 기존 컬렉션 가져오기
collection = chroma_client.get_or_create_collection(name="festival_collection")

def prepare_documents(df: pd.DataFrame):
    documents = []
    for _, row in df.iterrows():
        doc = f"축제명: {row['축제명']}, 지역: {row['광역자치단체명']} {row['기초자치단체명']}, "
        doc += f"유형: {row['축제유형']}, 기간: {row['개최기간']}, 장소: {row['개최장소']}, "
        doc += f"방식: {row['개최방식']}, 예산: {row['합계 예산(백만원)']}, "
        doc += f"방문객 수: {row['방문객수(2023년) 합계']}"
        documents.append(doc)
    return documents

def add_documents(texts: List[str]):
    # Solar 모델을 사용하여 문서 임베딩
    embeddings = []
    for text in texts:
        # 유니코드 문자를 안전하게 처리
        encoded_text = text.encode('utf-8', errors='ignore').decode('utf-8')
        try:
            response = client.embeddings.create(
                input=encoded_text,
                model="solar-embedding-1-large-query"
            )
            embeddings.append(response.data[0].embedding)
        except Exception as e:
            print(f"Error creating embedding for text: {encoded_text}")
            print(f"Error: {e}")
            continue  # 오류가 발생한 텍스트는 건너뛰고 계속 진행

    # 임베딩이 생성된 텍스트만 ChromaDB에 추가
    valid_texts = [text for text, emb in zip(texts, embeddings) if emb]
    valid_embeddings = [emb for emb in embeddings if emb]

    if valid_texts and valid_embeddings:
        collection.add(
            embeddings=valid_embeddings,
            documents=valid_texts,
            ids=[f"doc_{i}" for i in range(len(valid_texts))]
        )
    else:
        print("No valid embeddings were created. Check your input texts and API key.")

def query(question: str, k: int = 3):
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
    
    return results['documents'][0]

def generate_answer(question: str, context: List[str]):
    # Solar API를 사용하여 답변 생성
    prompt = f"Context:\n{''.join(context)}\n\nQuestion: {question}\nAnswer:"
    response = client.chat.completions.create(
        model="solar-1-mini-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant knowledgeable about Korean festivals."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

class Question(BaseModel):
    text: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 시작 시 실행될 코드
    global df, documents
    # CSV 파일 읽기
    df = pd.read_csv('/Users/bokyung/Desktop/LLM-RAG-LANGCHAIN/0907/2024년 지역축제 개최계획(수정).csv', encoding='utf-8')
    df.columns = df.iloc[0]
    df = df[1:]
    
    # 열 이름에서 개행 문자 제거 및 공백 제거
    df.columns = df.columns.str.replace('\n', '').str.strip()
    
    # 문서 준비 및 추가
    documents = prepare_documents(df)
    add_documents(documents)
    
    yield
    
    # 종료 시 실행될 코드

app = FastAPI()

@app.post("/ask")
async def ask_question(question: Question):
    try:
        relevant_docs = query(question.text)
        answer = generate_answer(question.text, relevant_docs)
        return {"question": question.text, "answer": answer}
    except Exception as e:
        print(f"Error processing question: {e}")
        return {"error": str(e)}, 500

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
