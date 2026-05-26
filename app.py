from fastapi import FastAPI
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain_ollama import ChatOllama

app = FastAPI()


class ChatRequest(BaseModel):
    message: str


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"The weather in {city} is sunny and 75°F."


def calculator(expression: str) -> str:
    """Calculate a simple math expression."""
    try:
        return str(eval(expression))
    except Exception:
        return "Invalid calculation."


def check_transaction_status(transaction_id: str) -> str:
    """Check demo transaction status using transaction ID."""
    return f"Transaction {transaction_id} was declined due to fraud verification."


model = ChatOllama(model="llama3.1")

agent = create_agent(
    model=model,
    tools=[get_weather, calculator, check_transaction_status],
    system_prompt="You are a helpful AI assistant. Use tools when needed.",
)


@app.post("/chat")
def chat(request: ChatRequest):
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": request.message,
                }
            ]
        }
    )
    return {"response": result["messages"][-1].content}
