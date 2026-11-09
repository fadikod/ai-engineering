from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field



class Review(BaseModel):
    sentiment: str
    score: int = Field(ge=1, le=5)
    improvement : str




llm = ChatOllama(
    model = "qwen2.5-coder:7b",
    reasoning = False,
    temperature = 0
    )

prompt = ChatPromptTemplate([
    ("system", "you are a professional reviews evaluator, return sentiment of the review, score from 1 to 5 and suggestions for improvements"),
    ("human", "Evaluate this review: {review}")
])

chain = prompt | llm.with_structured_output(Review)

reviews = [
    {"review": "The product is great, but the delivery was slow."},
    {"review": "I love this product! It exceeded my expectations."},
    {"review": "The product is okay, but the packaging was damaged."},
    {"review": "I am very disappointed with this product. It broke after one use."},
    {"review": "The product is decent, but the customer service was unhelpful."},
    {"review": "The product is amazing! I would definitely recommend it to others."},
    {"review": "The product is not worth the price. I expected better quality."},
    {"review": "The product is good, but the instructions were confusing."}
]

the_results = chain.batch(reviews)

for results in the_results:
    print(results.sentiment)
    print(results.score)
    print(results.improvement)

improvements = []
for result in the_results:
    improvements.append(result.improvement)
print("improvements: ", improvements)

total = 0
for results in the_results:
    total +=results.score
avg = total/len(the_results)
print("average score: ", avg)