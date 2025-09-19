from savingsAgent.savings_agent import SavingsRAGAgent

import asyncio

async def main():
    agent = SavingsRAGAgent()
    rag_query = "what is monthly service fee?"
    context = await agent.run_query(rag_query)
    agent.get_final_answer(rag_query, context)

if __name__ == "__main__":
    asyncio.run(main())
