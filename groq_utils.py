import openai
GROQ_API_KEY = "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"
openai.api_key = "YOUR_GROQ_API_KEY"
openai.api_base = "https://api.groq.com/openai/v1"

def ask_llm_groq(query, clauses):
    context = "\n\n".join(clauses)
    prompt = f"""
You are an expert claim processor.

User Query:
"{query}"

Relevant Clauses:
{context}

Based on the clauses, return a JSON:
- decision: approved / rejected
- amount: if applicable
- justification: short explanation
- source_clauses: short excerpts that justify your answer
"""
    res = openai.ChatCompletion.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return res['choices'][0]['message']['content']
