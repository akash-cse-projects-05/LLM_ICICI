import requests

GROQ_API_KEY = "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"

url = "https://api.groq.com/openai/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

# Ask any basic question
user_question = input("Ask a question: ")

data = {
    "model": "llama3-8b-8192",
    "messages": [
        {"role": "system", "content": "You are a smart and friendly assistant."},
        {"role": "user", "content": user_question}
    ],
    "temperature": 0.7
}

response = requests.post(url, headers=headers, json=data)

if response.status_code == 200:
    answer = response.json()["choices"][0]["message"]["content"]
    print("\n🤖 Answer:", answer)
else:
    print("❌ Error:", response.status_code)
    print(response.text)
