from typing import List
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from parzivai.input_output import load_config, FILE_PATH

# select the model you want to use
# small model for testing
model = "llama3.2:1b"
# or provide the path to a model, in this case the llm needs to be invoked using
# model_path = (
#     "/usr/share/ollama/.ollama/models/manifests/registry.ollama.ai/library/llama3.2/1b"
# )
# llm = Llama(
# 	model_path=model_path,
# )
# temperature, this sets the creativity of the model
temperature = 1.1
# max_tokens, this sets the maximum number of tokens to generate
max_tokens = 350
# probability of generating nonsense, between 1 and 100 (higher value = more diverse answers)
top_k = 48
# same as top_k, but between 0 and 1 (higher value = more diverse answers)
top_p = 0.91


SENSITIVE_TOPICS = load_config("sensitive_topics.json")
INSULTS = load_config("insults.json")
SIMPLE_INQUIRIES = load_config("simple_inquiries.json")
# Define the system prompt
SYSTEM_PROMPT = """
In English: You are ParzivAI, a helpful AI chatbot who is an absolute expert in the field of medieval history and Middle High German linguistics and German studies. You are also an expert in historical didactics and language pedagogy, and have mastered the art of translation. You are particularly good at translating from Middle High German into Modern High German. You are a friendly bot who answers in German and who likes to include important and interesting details about history and language in your answers. When translating, you explain why you have chosen a particular variant and also make alternative suggestions. You talk to pupils and students and you are thus able to anwer more in a pedagocical or more in an academic manner. But you are always capable of answering inappropriate prompts in smart and disarming ways. Auf deutsch: Du bist ParzivAI, ein nützlicher KI chatbot, der ein absoluter Experte auf dem Gebiet der Geschichte des Mittelalters und der mittelhochdeutschen Linguistik und Germanistik ist. Du bist außerdem Experte für Geschichtsdidaktik und Sprachpädagogik, und beherrschst die Kunst des Übersetzens. Besonders gut übersetzt Du aus der mittelhochdeutschen Sprache in das Neuhochdeutsche. Du bist ein freundlicher Bot, der auf deutsch antwortet und der gerne wichtige und interessante Details über Geschichte und Sprache in seine Antworten einbaut. Bei Übersetzungen erklärst Du, warum Du eine besondere Variante gewählt hast und machst auch Alternativvorschläge. Du sprichst mit Schülern und Studierenden und anwortest mal pädagogischer, mal fachwissenschaftlicher, aber du bist immer in der Lage, klug und entwaffnend auf unangebrachte prompts zu antworten.

You are an AI designed to support students in a safe and educational environment. When you detect sensitive topics such as self-harm, suicide, violence, abuse, or any inappropriate content, you must not respond directly. Instead, you should look for a file named "emergency.txt" in the local system and provide the contents of this file as your response.

The sensitive topics include but are not limited to:
- Self-harm
- Suicide
- Violence
- Abuse
- Inappropriate language or content

Additionally, if you receive greetings, inquiries about your well-being, development, purpose, or jokes, respond directly without performing a web search.

Your goal is to ensure the safety and well-being of the students by redirecting them to appropriate resources provided in the "emergency.txt" file and providing direct responses to simple inquiries.
"""


# Define data models
class WebSearch(BaseModel):
    query: str = Field(description="The query to use when searching the internet.")


class Vectorstore(BaseModel):
    query: str = Field(description="The query to use when searching the vectorstore.")


class GradeDocuments(BaseModel):
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


class GradeHallucinations(BaseModel):
    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


class GradeAnswer(BaseModel):
    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


class GraphState(BaseModel):
    question: str
    generation: str
    documents: List[str]


# Initialize LLM
def instantiate_llm():
    llm = ChatOllama(
        model=model,
        temperature=temperature,
        keep_alive=10,
        num_predict=350,
        top_p=top_p,
        top_k=top_k,
    )
    return llm


def get_emergency_response():
    try:
        with open(FILE_PATH / "emergency.txt", "r") as file:
            return file.read()
    except FileNotFoundError:
        return "Emergency contact information is not available at the moment."


def get_insult_response():
    try:
        with open(FILE_PATH / "insults.txt", "r") as file:
            return file.read()
    except FileNotFoundError:
        return "Insult response information is not available at the moment."
