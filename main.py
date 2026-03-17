from __future__ import annotations

import json
import os
import uuid
from typing import Any

import boto3
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# App & Bedrock client
# ---------------------------------------------------------------------------
app = FastAPI(title="Orago Test Agent")

_bedrock: Any = None

MODEL_ID = os.getenv(
    "BEDROCK_MODEL_ID", "mistral.ministral-3-8b-instruct"
)
MAX_TOKENS = int(os.getenv("BEDROCK_MAX_TOKENS", "1024"))


def _get_bedrock():
    """Lazy-init the Bedrock Runtime client so env vars can be set after import."""
    global _bedrock
    if _bedrock is None:
        _bedrock = boto3.client(
            "bedrock-runtime",
            region_name=os.getenv("AWS_REGION", "us-east-1"),
        )
    return _bedrock


# ---------------------------------------------------------------------------
# In-memory conversation store  {conversation_id: [messages]}
# ---------------------------------------------------------------------------
conversations: dict[str, list[dict]] = {}

# ---------------------------------------------------------------------------
# Request / Response models  (Orago protocol)
# ---------------------------------------------------------------------------

class ToolCall(BaseModel):
    id: str
    name: str
    input: dict


class ToolResultItem(BaseModel):
    tool_use_id: str
    content: str


class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None
    context: str | None = None          # RAG context -> system prompt
    tools: list[dict] | None = None     # tool definitions
    tool_results: list[ToolResultItem] | None = None


class ChatResponse(BaseModel):
    response: str | None = None
    conversation_id: str
    tool_calls: list[ToolCall] | None = None
    handoff: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_system_prompt(context: str | None) -> str:
    base = "You are a helpful AI assistant deployed on Orago. Respond in the same language as the user."
    if context:
        return f"{base}\n\nRelevant context:\n{context}"
    return base


def _invoke_bedrock(
    messages: list[dict],
    system: str,
) -> str:
    """Call Bedrock with Mistral-compatible format and return text response."""
    client = _get_bedrock()

    # Prepend system message
    all_messages = [{"role": "system", "content": system}] + messages

    body: dict[str, Any] = {
        "messages": all_messages,
        "max_tokens": MAX_TOKENS,
    }

    response = client.invoke_model(
        modelId=MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body),
    )

    result = json.loads(response["body"].read())

    # Mistral format: {"choices": [{"message": {"content": "..."}}]}
    choices = result.get("choices", [])
    if choices:
        return choices[0].get("message", {}).get("content", "")
    return ""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/chat")
async def chat(req: ChatRequest) -> ChatResponse:
    conv_id = req.conversation_id or str(uuid.uuid4())
    history = conversations.setdefault(conv_id, [])

    # ---- Append the new user message -----------------------------------------
    if req.message.strip():
        history.append({
            "role": "user",
            "content": req.message,
        })

    # ---- If we got tool results, append them as assistant context ------------
    if req.tool_results:
        tool_info = "\n".join(
            f"Tool result ({tr.tool_use_id}): {tr.content}"
            for tr in req.tool_results
        )
        history.append({
            "role": "user",
            "content": f"Here are the tool results:\n{tool_info}",
        })

    # ---- Call Bedrock -------------------------------------------------------
    system = _build_system_prompt(req.context)

    try:
        text = _invoke_bedrock(
            messages=history,
            system=system,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Bedrock error: {exc}")

    # Save assistant turn in history
    history.append({"role": "assistant", "content": text})

    return ChatResponse(
        response=text or "I'm not sure how to help with that.",
        conversation_id=conv_id,
        handoff=False,
    )


@app.get("/health")
async def health():
    return {"status": "healthy", "agent": "orago-test-agent", "model": MODEL_ID}


# ---------------------------------------------------------------------------
# Dev entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
