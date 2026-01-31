import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"), base_url=os.environ.get("BASE_URL")
)

response = client.chat.completions.create(
    model=os.environ.get("MODEL_NAME"),
    messages=[{"role": "user", "content": "你好，你是谁"}],
)

print(response.choices[0].message)

response = client.chat.completions.create(
    model=os.environ.get("MODEL_NAME"),
    messages=[{"role": "system", "content": "刚才用户问了你你是谁"}]
    + [{"role": "user", "content": "我知道了"}],
)

print(response.choices[0].message)
