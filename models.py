import asyncio

from pydantic import BaseModel


class GenerationRequest(BaseModel):
    prompt: str
    max_tokens: int


class Request:
    def __init__(self, request_id: str, prompt: str, max_tokens: int):
        self.request_id = request_id
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.generated_tokens: list[str] = []
        self.kv_cache = None
        self.finished: bool = False
        self.next_token_id = None
        self.token_queue: asyncio.Queue[str] = asyncio.Queue()
