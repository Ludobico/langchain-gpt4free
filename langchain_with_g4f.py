from typing import Optional, List, Any
from langchain.llms.base import LLM

import g4f

from functools import partial
from langchain.callbacks.manager import AsyncCallbackManagerForLLMRun

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class TESTLLM(LLM):
  @property
  def _llm_type(self) -> str:
    return "custom"
  
  def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
    model = g4f.models.gpt_35_turbo
    provider = g4f.Provider.GptGo
    out = g4f.ChatCompletion.create(
      model=model,
      provider=provider,
      messages=[{"role": "user", "content": prompt}],
    )

    if stop:
      stop_indexes = (out.find(s) for s in stop if s in out)
      min_stop = min(stop_indexes, default=-1)
      if min_stop > -1:
        out = out[:min_stop]
    return out
  
  async def _acall(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[AsyncCallbackManagerForLLMRun] = None, **kwargs: Any) -> str:
    text_callback = None
    model = g4f.models.gpt_35_turbo
    provider = g4f.Provider.GptGo
    if run_manager:
      text_callback = partial(run_manager.on_llm_new_token)
    
    text = ""
    for token in g4f.ChatCompletion.create(model=model,provider=provider,messages=[{"role": "user", "content": prompt}], stream=True):
      if text_callback:
        await text_callback(token)
      text += token
    return text

llm = TESTLLM()

prompt = PromptTemplate(
  input_types=['product'],
  template="What is a good name for a company that makes {product}?",
)

chain = LLMChain(llm=llm, prompt=prompt)
