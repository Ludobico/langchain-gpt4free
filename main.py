import asyncio
from typing import AsyncIterable, Awaitable
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
# from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from pydantic import BaseModel
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.manager import AsyncCallbackManagerForLLMRun,CallbackManagerForLLMRun

from testLLM.CustomLLM import TESTLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    content: str
async def send_message(content: str) -> AsyncIterable[str]:
    callback = AsyncIteratorCallbackHandler()
    llm = TESTLLM()

    template = """Question: {question}

    Answer: Let's think step by step."""

    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True, callbacks=[callback])
    question = "What NFL team won the Super Bowl in the year Justin Bieber was born?"
    tasks = [asyncio.create_task(llm_chain.arun(question))]

    task = asyncio.gather(*tasks)
    await task
    try:
        async for token in callback.aiter():
            yield token
    except Exception as e:
        print(f"Caught exception: {e}")
    finally:
        callback.done.set()

@app.post("/stream_chat/")
async def stream_chat(message: Message):
    generator = send_message(message.content)
    # return StreamingResponse(generator, media_type="text/event-stream")
    # sync_message()
    # return StreamingResponse(sync_message(), media_type="text/event-stream")
    return generator