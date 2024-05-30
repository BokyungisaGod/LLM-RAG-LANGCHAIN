import json

# LLM RAG LANGCHAIN.json 파일 읽기
with open('/Users/bokyung/Desktop/LLM-RAG-LANGCHAIN/0531/'
          'LLM RAG LANGCHAIN.json', 'r', encoding='utf-8') as f:
    langchain = json.load(f)

print(langchain)


# read = r
# write = w
