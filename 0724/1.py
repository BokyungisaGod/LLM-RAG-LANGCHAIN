import os
import json
import base64
import io
from openai import OpenAI
from PIL import Image
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

prompt3 = """The content of the image is created in PowerPoint.
The most important thing is to accurately extract the sentences from the image.
If there is a table, do not create a 2D table.
Write out each row as a sentence.
If there is a diagram, write its meaning.
Always write in Korean.

Output format: Structured text (use only heading 2 (##) for Subject and heading 3 (###) for Section) 

"""

prompt = "get me text from image"

results = []
png_files = sorted([f for f in os.listdir('/Users/bokyung/Desktop/LLM-RAG-LANGCHAIN/0724/data') if f.endswith('.png')])

for filename in png_files:
    file_path = os.path.join('/Users/bokyung/Desktop/LLM-RAG-LANGCHAIN/0724/data', filename)

    # 이미지 파일 열기 및 base64 인코딩
    with Image.open(file_path) as img:
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}}
                ]
            }
        ],
        max_tokens=4096
    )

    # 결과 저장
    results.append({
        'content': response.choices[0].message.content,
        'page_no': filename
    })

    with open('image_to_copus__gpt4o.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
