import asyncio
import time

async def embedded_question(question):
    print(f"Starting embedding for question: {question}")
    await asyncio.sleep(1)
    print(f"Embedding question: {question}")
    return "Embedding Result"

async def search_database(question):
    print(f"Starting database search for question: {question}")
    await asyncio.sleep(2)
    print(f"Database search completed for question: {question}")
    return "Database Result"

async def check_cache(question):
    print(f"Starting cache check for question: {question}")
    await asyncio.sleep(1)
    print(f"Cache check completed for question: {question}")
    return "Cache Result"

async def main():
    start = time.time()
    question = "What is the capital of France?"
    results = await asyncio.gather(
        embedded_question(question),
        search_database(question),
        check_cache(question)   
    )
    elapsed_time = time.time() - start
    print(f"All tasks completed for question: {question} in {elapsed_time:.2f} seconds")

asyncio.run(main())