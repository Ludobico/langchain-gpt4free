# from g4f import Provider, models
from langchain.llms.base import LLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
import sys
sys.path.append(r'C:\Users\aqs45\OneDrive\바탕 화면\repo\langchain-gpt4free')
from testLLM.CustomLLM import TESTLLM
import asyncio
from typing import AsyncIterable

from flask import Flask, Response
import time

# 동기
def main():
  llm = TESTLLM()
  callbacks = [StreamingStdOutCallbackHandler()]
  # res = llm('can you talk to korean?')
  # print(res)
  template = """Question: {question}

  Answer: Let's think step by step."""

  prompt = PromptTemplate(template=template, input_variables=["question"])
  llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)
  question = "What NFL team won the Super Bowl in the year Justin Bieber was born?"

  # print(llm_chain.run(question))
  llm_chain.run(question, callbacks=callbacks)



# 비동기
async def async_generate(chain, question):
  resp = await chain.arun(question)
  for token in resp:
    print(token)
    time.sleep(0.1)

async def generate_concurrently():
  llm = TESTLLM()
  callback = AsyncIteratorCallbackHandler()
  template = """Question: {question}

  Answer: Let's think step by step."""

  prompt = PromptTemplate(template=template, input_variables=["question"])
  llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True, callbacks=[callback])
  question = "What NFL team won the Super Bowl in the year Justin Bieber was born?"

  tasks = [async_generate(llm_chain, question)]
  await asyncio.gather(*tasks)


if __name__ == "__main__":
  main()