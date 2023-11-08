from typing import Optional, List, Any
from langchain.llms.base import LLM

import g4f

from functools import partial
from langchain.callbacks.manager import AsyncCallbackManagerForLLMRun,CallbackManagerForLLMRun, AsyncCallbackManagerForChainRun
import tracemalloc
import asyncio
class TESTLLM(LLM):
  @property
  def _llm_type(self) -> str:
    tracemalloc.start()
    return "custom"
  
  def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None,) -> str:
    model = g4f.models.gpt_35_turbo
    provider = g4f.Provider.GptGo
    out = g4f.ChatCompletion.create(
      model=model,
      provider=provider,
      messages=[{"role": "user", "content": prompt}],
      stream=True
    )
    text = []
    for chunk in out:
       text.append(chunk)
       if run_manager:
          run_manager.on_llm_new_token(chunk)
    return "".join(text)

    # if stop:
    #   stop_indexes = (out.find(s) for s in stop if s in out)
    #   min_stop = min(stop_indexes, default=-1)
    #   if min_stop > -1:
    #     out = out[:min_stop]
    # return out
  
  async def _acall(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[AsyncCallbackManagerForLLMRun] = None, **kwargs: Any) -> str:
    text_callback = None
    model = g4f.models.gpt_35_turbo
    provider = g4f.Provider.GptGo
    if run_manager:
      text_callback = partial(run_manager.on_llm_new_token)
    
    text = ""
    completion = g4f.ChatCompletion.create(model=model, provider=provider, messages=[{"role": "user", "content": prompt}])
    for token in completion:
        if text_callback:
            await text_callback(token)
        text += token
    return text