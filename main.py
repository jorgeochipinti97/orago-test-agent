from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Orago Test Agent")

class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None

class ChatResponse(BaseModel):
    response: str
    conversation_id: str | None = None

@app.post("/chat")
async def chat(req: ChatRequest) -> ChatResponse:
    # Simple echo agent for testing
    return ChatResponse(
        response=f"Test Agent received: '{req.message}'. I'm an AI agent deployed on Orago!",
        conversation_id=req.conversation_id or "test-conv-1",
    )

@app.get("/health")
async def health():
    return {"status": "healthy", "agent": "orago-test-agent"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
