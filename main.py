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
    "BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0"
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
    base = "You are a helpful AI assistant deployed on Orago."
    if context:
        return f"{base}\n\nRelevant context:\n{context}"
    return base


def _invoke_bedrock(
    messages: list[dict],
    system: str,
    tools: list[dict] | None = None,
) -> dict:
    """Call Bedrock converse API and return the raw response body."""
    client = _get_bedrock()

    body: dict[str, Any] = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": MAX_TOKENS,
        "system": system,
        "messages": messages,
    }

    if tools:
        body["tools"] = tools

    response = client.invoke_model(
        modelId=MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body),
    )

    return json.loads(response["body"].read())


def _extract_tool_calls(content: list[dict]) -> list[ToolCall]:
    """Pull tool_use blocks out of a Claude response."""
    return [
        ToolCall(id=block["id"], name=block["name"], input=block["input"])
        for block in content
        if block.get("type") == "tool_use"
    ]


def _extract_text(content: list[dict]) -> str:
    """Concatenate text blocks from a Claude response."""
    parts = [block["text"] for block in content if block.get("type") == "text"]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/chat")
async def chat(req: ChatRequest) -> ChatResponse:
    conv_id = req.conversation_id or str(uuid.uuid4())
    history = conversations.setdefault(conv_id, [])

    # ---- If the orchestrator sent tool_results, append them first ----------
    if req.tool_results:
        tool_result_content = [
            {
                "type": "tool_result",
                "tool_use_id": tr.tool_use_id,
                "content": tr.content,
            }
            for tr in req.tool_results
        ]
        history.append({"role": "user", "content": tool_result_content})

    # ---- Append the new user message (skip empty ones on tool-result turns) -
    if req.message.strip():
        history.append({
            "role": "user",
            "content": [{"type": "text", "text": req.message}],
        })

    # ---- Call Bedrock -------------------------------------------------------
    system = _build_system_prompt(req.context)

    try:
        result = _invoke_bedrock(
            messages=history,
            system=system,
            tools=req.tools,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Bedrock error: {exc}")

    assistant_content: list[dict] = result.get("content", [])

    # Save assistant turn in history
    history.append({"role": "assistant", "content": assistant_content})

    # ---- Tool use? Return tool_calls to the orchestrator --------------------
    tool_calls = _extract_tool_calls(assistant_content)
    if tool_calls:
        return ChatResponse(
            conversation_id=conv_id,
            tool_calls=tool_calls,
        )

    # ---- Normal text response -----------------------------------------------
    text = _extract_text(assistant_content)

    # If stop_reason indicates the model gave up, signal handoff
    stop_reason = result.get("stop_reason", "")
    handoff = stop_reason == "end_turn" and not text.strip()

    return ChatResponse(
        response=text or "I'm not sure how to help with that.",
        conversation_id=conv_id,
        handoff=handoff,
    )


@app.get("/health")
async def health():
    return {"status": "healthy", "agent": "orago-test-agent", "model": MODEL_ID}


# ---------------------------------------------------------------------------
# Dev entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
