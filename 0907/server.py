import pandas as pd
from typing import List
from openai import OpenAI
import chromadb
from chromadb.config import Settings
import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import logging
import time
import uuid
import json

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# .env 파일에서 키 불러오기
load_dotenv()
client = OpenAI(
    api_key="up_oLxU2aBXrVgGjZY3ejAbH021gtH0e",
    base_url="https://api.upstage.ai/v1/solar",
    max_tokens=1024
)

# ChromaDB 클라이언트 설정
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# 컬렉션 생성 또는 기존 컬렉션 가져오기
collection = chroma_client.get_or_create_collection(name="festival_collection4")

def prepare_documents(df: pd.DataFrame):
    logger.info("문서 준비 시작")
    documents = []
    for _, row in df.iterrows():
        doc = f"축제명: {row['축제명']}, 광역자치단체명/기초자치단체명: {row['광역자치단체명']} {row['기초자치단체명']}, "
        doc += f"유형: {row['축제유형']}, 개최 기간: 2024. {row['개최기간']}, 장소: {row['개최장소']}, "
        doc += f"예산(단위:백만원): {row['합계 예산(백만원)']}, "
        doc += f"2023년 방문객 수: {row['방문객수(2023년) 합계']}명"
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
            #batch_ids = [f"doc_{j}" for j in range(i, i+len(batch_texts))]
            #batch_ids를 uuid로 변경
            batch_ids = [str(uuid.uuid4()) for j in range(i, i+len(batch_texts))]
            
            collection.add(
                embeddings=batch_embeddings,
                documents=batch_texts,
                ids=batch_ids
            )
            
            logger.info(f"{i+len(batch_texts)}개의 문서 추가 완료")
            time.sleep(0.5)
    else:
        logger.warning("유효한 임베딩이 생성되지 않았습니다. 입력 텍스트와 API 키를 확인하세요.")

def query(question: str, k: int = 10):
    # logger.info(f"질문에 대한 임베딩 생성 시작: {question}")
    # Solar 모델을 사용하여 질문 임베딩
    response = client.embeddings.create(
        input=question,
        model="solar-embedding-1-large-query"
    )
    query_embedding = response.data[0].embedding
    # print(len(collection.get()))
    # 유사한 문서 검색
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )
    # print(results)

    # logger.info(f"{k}개의 유사한 문서 검색 완료")
    return results['documents'][0]

system_prompt = """
당신의 역할은 Context를 바탕으로 질문에 대한 답변을 제공하는 것입니다.
만약, 사용자가 이해할 수 없는 질문을 하거나, 답변을 찾을 수 없는 경우에는 '죄송합니다. 정보를 찾을 수 없습니다.'라고 답변해주세요.
만약, 사용자가 행사, 축제, 지역명 등의 연관 정보를 요청하는 경우에는 관련 정보를 제공해주세요.
만약 Question과, Context가 아무 연관 없다면, '죄송합니다. 정보를 찾을 수 없습니다.'라고 답변해주세요.
Context에 없는 내용은 절대 추가하지 말아주세요.
Question은 사용자의 질문입니다.
Answer은 사용자에게 제공할 답변입니다. 최대한 정확하고 자세히 답변해주세요.
친근하고 사교적인 태도의 어조를 유지해주세요.

"""


def generate_answer(question: str, context: List[str]):
    # logger.info("답변 생성 시작")
    # Solar API를 사용하여 답변 생성
    # print(context)
    # for c in context:
    #    print(c)
    #    print("-"*100)

    contexts = '\n'.join(context)
    prompt = f"""Context:
    {contexts}

    Question: {question}

    Answer:"""

    response = client.chat.completions.create(
        model="solar-1-mini-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )
    #logger.info("답변 생성 완료")
    logger.info(question)
    logger.info(response.choices[0].message.content)
    return response.choices[0].message.content.strip()

class Question(BaseModel):
    text: str

class SearchQuery(BaseModel):
    text: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("애플리케이션 시작")
    global documents
    # CSV 파일 읽기

    logger.info("CSV 파일 로드 완료")

    # 문서 준비 및 추가
    documents = prepare_documents(df)
    logger.info("문서 준비 완료")
    add_documents(documents)
    logger.info("문서 추가 완료")

    yield

    logger.info("애플리케이션 종료")

limiter = Limiter(key_func=get_remote_address)
# app = FastAPI(lifespan=lifespan)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CSV 파일 읽기
df = pd.read_csv('/Users/bokyung/Desktop/LLM-RAG-LANGCHAIN/0907/2024년 지역축제 개최계획(수정).csv', encoding='utf-8')
df.columns = df.iloc[0]
df = df[1:]

# 열 이름에서 개행 문자 제거 및 공백 제거
df.columns = df.columns.str.replace('\n', '').str.strip()

def search_data(query):
    mask = df.apply(lambda row: row.astype(str).str.contains(query, case=False).any(), axis=1)
    return df[mask]

@app.post("/ask")
@limiter.limit("10/minute")
async def ask_question(request: Request, question: Question):
    try:
        relevant_docs = query(question.text)
        answer = generate_answer(question.text, relevant_docs)
        return {"question": question.text, "answer": answer}
    except RateLimitExceeded:
        raise HTTPException(status_code=429, detail="요청 한도를 초과했습니다. 1분 후에 다시 시도해 주세요.")
    except Exception as e:
        logger.error(f"질문 처리 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
@limiter.limit("10/minute")
async def search(request: Request, query: SearchQuery):
    try:
        results = search_data(query.text)
        return {"answer": json.loads(results.to_json(orient='records'))}
    except RateLimitExceeded:
        return JSONResponse(
            status_code=429,
            content={"message": "요청 한도를 초과했습니다. 1분 후에 다시 시도해 주세요."}
        )
    except Exception as e:
        logger.error(f"검색 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    logger.info("서버 시작")
    # reload=True로 설정하여 코드 변경 시 자동으로 서버 재시작
    uvicorn.run(app, host="0.0.0.0", port=8000)
