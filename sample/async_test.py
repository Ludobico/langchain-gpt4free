import g4f, asyncio

async def run_provider():
    try:
        response = await g4f.ChatCompletion.create_async(
            model=g4f.models.gpt_35_turbo,
            messages=[{"role": "user", "content": "Hello"}],
            provider=g4f.Provider.GptGo,
            stream=True
        )
        print(response)
    except Exception as e:
        print(e)
        
async def run_all():
    calls = [
        run_provider()
    ]
    await asyncio.gather(*calls)

if __name__ == "__main__":
  asyncio.run(run_all())