import httpx
import json
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse

app = FastAPI()

OLLAMA_API_URL = "http://localhost:11434/api/chat"

async def ollama_stream_generator(messages: list):
    """
    An asynchronous generator that streams responses from Ollama,
    using the full conversation history.
    """
    payload = {
        "model": "phi3:mini",
        "messages": messages, # Pass the entire conversation
        "stream": True
    }

    async with httpx.AsyncClient(timeout=None) as client:
        try:
            async with client.stream("POST", OLLAMA_API_URL, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                            # The role 'assistant' is used by Ollama for model replies
                            content = chunk.get("message", {}).get("content", "")
                            yield content
                        except (json.JSONDecodeError, KeyError):
                            continue
        except httpx.HTTPStatusError as e:
            print(f"API error: {e}")
            yield f"Error: Failed to get response from model. Status: {e.response.status_code}"
        except httpx.RequestError as e:
            print(f"Network error: {e}")
            yield f"Error: Could not connect to Ollama. Is it running?"

@app.post("/api/chat")
async def chat_handler(request_data: dict):
    """
    Handles the chat request, which now contains the full message history.
    """
    messages = request_data.get("messages")
    if not messages:
        return {"error": "Messages list is missing"}, 400

    return StreamingResponse(ollama_stream_generator(messages), media_type="text/plain")

# Serves the static frontend files
app.mount("/", StaticFiles(directory="static", html=True), name="static")