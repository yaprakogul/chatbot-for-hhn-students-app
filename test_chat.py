import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant for new students at Heilbronn University. You always answer in English."},
        {"role": "user", "content": "Hi! Just say 'ready' if the API works."}
    ]
)

print(resp.choices[0].message.content.strip())
