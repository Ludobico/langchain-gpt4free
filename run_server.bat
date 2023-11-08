@echo off
call conda activate lcg4f

cd C:\Users\aqs45\OneDrive\바탕 화면\repo\langchain-gpt4free

uvicorn main:app --reload