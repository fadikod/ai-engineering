from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

def change_tone(text:str, tone:str) ->str:
    system_prompt = (
        "You are an expert editor. Rewrite the provided input text in the requested tone "
        "(formal, friendly, or direct). Maintain the exact original meaning and intent, "
        "only changing the tone of voice. Return only the rewritten text without extra commentary."
    )
    
    user_prompt = f"Target Tone: {tone}\nOriginal Text: {text}"

    response = client.chat.completions.create(
        model="qwen2.5-coder:7b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3
    )
    
    return response.choices[0].message.content.strip()

texts = [
    "Hey, can you send me the report by tomorrow? Thanks!",
    "I'm sorry but we are unable to refund your order at this time.",
    "The meeting has been moved to 3 PM, update your calendar.",
]

for text in texts:
    for tone in ["formal", "friendly", "direct"]:
        print(f"[{tone.upper()}] {change_tone(text, tone)}")
    print("=" * 60)