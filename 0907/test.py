from openai import OpenAI # openai==1.2.0

client = OpenAI(
    api_key="up_oLxU2aBXrVgGjZY3ejAbH021gtH0e",
    base_url="https://api.upstage.ai/v1/solar"
)

stream = client.chat.completions.create(
    model="solar-1-mini-chat",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "Hello!"
        }
    ],
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
