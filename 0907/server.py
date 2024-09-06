import pandas as pd
from typing import List
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import os
from fastapi import FastAPI
from pydantic import BaseModel

# OpenAI 클라이언트 설정
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# SentenceTransformer 모델 로드
model = SentenceTransformer('all-MiniLM-L6-v2')

# ChromaDB 클라이언트 설정
chroma_client = chromadb.Client(Settings(persist_directory="./chroma_db"))

# 컬렉션 생성 또는 기존 컬렉션 가져오기
collection = chroma_client.get_or_create_collection(name="festival_collection")

# CSV 파일 읽기
df = pd.read_csv('/Users/bokyung/Desktop/LLM-RAG-LANGCHAIN/0907/2024년 지역축제 개최계획(수정).csv', encoding='utf-8')
df.columns = df.iloc[0]
df = df[1:]

# 열 이름에서 개행 문자 제거 및 공백 제거
df.columns = df.columns.str.replace('\n', '').str.strip()

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
    # 문서 임베딩
    embeddings = model.encode(texts).tolist()
    
    # ChromaDB에 문서 추가
    collection.add(
        embeddings=embeddings,
        documents=texts,
        ids=[f"doc_{i}" for i in range(len(texts))]
    )

def query(question: str, k: int = 3):
    # 질문 임베딩
    query_embedding = model.encode(question).tolist()
    
    # 유사한 문서 검색
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )
    
    return results['documents'][0]

def generate_answer(question: str, context: List[str]):
    # OpenAI API를 사용하여 답변 생성
    prompt = f"Context:\n{''.join(context)}\n\nQuestion: {question}\nAnswer:"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant knowledgeable about Korean festivals."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

app = FastAPI()

class Question(BaseModel):
    text: str

@app.on_event("startup")
async def startup_event():
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

@app.post("/ask")
async def ask_question(question: Question):
    relevant_docs = query(question.text)
    answer = generate_answer(question.text, relevant_docs)
    return {"question": question.text, "answer": answer}