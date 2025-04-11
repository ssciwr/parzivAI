from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from llama_cpp import Llama
import asyncio

# select the model you want to use
# small model for testing
model = "llama3.2:1b"
# or provide the path to a model
model_path = (
    "/usr/share/ollama/.ollama/models/manifests/registry.ollama.ai/library/llama3.2/1b"
)
# temperature, this sets the creativity of the model
temperature = 1.1
# max_tokens, this sets the maximum number of tokens to generate
max_tokens = 350
# probability of generating nonsense, between 1 and 100 (higher value = more diverse answers)
top_k = 48
# same as top_k, but between 0 and 1 (higher value = more diverse answers)
top_p = 0.91

# Define the system prompt
SYSTEM_PROMPT = """
You are a helpful translator. Translate the user sentence exactly to French and repeat it before your answer.
You are an AI designed to support students in a safe and educational environment.
When you detect sensitive topics such as self-harm, suicide, violence, abuse, or any
inappropriate content, you must not respond directly. Instead, you should look for
a file named "emergency.txt" in the local system and provide the contents of this
file as your response.

The sensitive topics include but are not limited to:
- Self-harm
- Suicide
- Violence
- Abuse
- Inappropriate language or content

Additionally, if you receive greetings, inquiries about your well-being,
development, purpose, or jokes, respond directly without performing a web search.

Your goal is to ensure the safety and well-being of the students by redirecting
them to appropriate resources provided in the "emergency.txt" file and providing
direct responses to simple inquiries.
"""

# Initialize LLM
# using path to model
# llm = Llama(
# 	model_path=model_path,
# )
# or using ChatOllama
# optionen sind so ok
llm = ChatOllama(
    model=model,
    temperature=temperature,
    keep_alive=10,
    num_predict=max_tokens,
    top_p=top_p,
    top_k=top_k,
)
messages = [
    ("system", SYSTEM_PROMPT),
    ("human", "I love programming."),
]

# ai = AIMessage(content="")
# human = HumanMessage(content="Hallo ich winke")
# messages = [human, ai]
generation = llm.invoke(messages)
# generation = llm.invoke(("human", "Hallo"))
print(generation)
print("end")

messages = [
    ("system", SYSTEM_PROMPT),
    ("human", "I love programming."),
    ("human", "I love cats."),
]


async def batch(messages):
    generation = await llm.abatch(messages)
    return generation


generation = asyncio.run(batch(messages))
print([i.content for i in generation])
