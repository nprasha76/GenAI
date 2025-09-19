import os
import asyncio
from dotenv import load_dotenv
from litellm import completion
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import FunctionTool
from .adk_functions import call_agent_async
from .adk_tools import vectorSearch
from chase_embeddings.src.faissembeddings import load_faiss_index, query_embedding
from chase_embeddings.src.embeddings import load_embeddings_from_meta
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner

class CreditRAGAgent:
    def __init__(self):
        load_dotenv()
        self.model = LiteLlm(
            model="openai/TechxGenus/Meta-Llama-3-8B-Instruct-GPTQ",
            api_base="http://localhost:8000/v1",
            api_key="EMPTY",
            provider="openai",
            config={"systemRoleSupported": False}
        )
        self.rag_tool = FunctionTool(func=vectorSearch)
        self.ragagent = LlmAgent(
            model=self.model,
            name="local_rag_agent",
            description="Call vectorSearch(<query>) tool to retrieve top documents from vector DB and use them to summarise and answer user query",
            instruction="""
                You are an assistant that must call vectorSearch(<query>) tool to retrieve top documents from vector DB and use them to summarize and answer user query.
                When the user asks a question, call vectorSearch(<query>) tool to retrieve top documents, 
                then use them to summarize and answer.
            """,
            tools=[vectorSearch]
        )
        self.session_service = InMemorySessionService()
        self.runner = Runner(
            agent=self.ragagent,
            app_name="RAG_GPT",
            session_service=self.session_service
        )
        self.user_id = "user_1_gpt"
        self.session_id = "rag_session_001_gpt"
        self.app_name = "RAG_GPT"

    async def create_session(self):
        await self.session_service.create_session(
            app_name=self.app_name,
            user_id=self.user_id,
            session_id=self.session_id
        )

    async def run_query(self, query: str):
        await self.create_session()
        print(f"Session created: App='{self.app_name}', User='{self.user_id}', Session='{self.session_id}'")
        print(f"Running RAG query: {query}")
        context = await call_agent_async(
            query=query,
            runner=self.runner,
            user_id=self.user_id,
            session_id=self.session_id
        )
        return context

    def get_final_answer(self, rag_query: str, context: str):
        rag_prompt = (
            f"from the following retrieved context {rag_query}. "
            f"Give answer in concise(2 lines max) ***Context: {context}"
        )
        print("\nFinal RAG Prompt to LLM:", rag_prompt)
        response = completion(
            model="openai/TechxGenus/Meta-Llama-3-8B-Instruct-GPTQ",
            messages=[{"role": "user", "content": rag_prompt}],
            api_base="http://localhost:8000/v1",
            api_key="NONE"
        )
        answer = response["choices"][0].message["content"]
        print("Response From LLM:", answer)
        return answer

# Usage example
async def main():
    agent = SavingsRAGAgent()
    rag_query = "what is monthly service fee?"
    context = await agent.run_query(rag_query)
    agent.get_final_answer(rag_query, context)

if __name__ == "__main__":
    asyncio.run(main())