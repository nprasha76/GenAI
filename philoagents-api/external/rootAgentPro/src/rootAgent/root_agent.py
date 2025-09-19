from google.adk.agents.llm_agent import Agent
from google.genai import types
import os
import asyncio
from dotenv import load_dotenv
from litellm import completion
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import FunctionTool

from chase_embeddings.src.faissembeddings import load_faiss_index, query_embedding
from chase_embeddings.src.embeddings import load_embeddings_from_meta
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner

class RootAgent:
    
    async def creditAgent(query: str) -> str:

            from creditAgent.credit_agent import CreditRAGAgent
            agent = CreditRAGAgent()
            context = await agent.run_query(query)
            result = agent.get_final_answer(query, context)
            return result
        
    async def mortgageAgent(query: str) -> str:

            from mortgageAgent.mortgage_agent import MortgageRAGAgent
            agent = MortgageRAGAgent()
            context = await agent.run_query(query)
            result = agent.get_final_answer(query, context)
            return result
    async def autoAgent(query: str) -> str:

            from autoAgent.auto_agent import AutoRAGAgent
            agent = AutoRAGAgent()
            context = await agent.run_query(query)
            result = agent.get_final_answer(query, context)
            return result
    async def checkingAgent(self,query: str) -> str:

            from savingsAgent.savings_agent import SavingsRAGAgent
            agent = SavingsRAGAgent()
            context = await agent.run_query(query)
            result = agent.get_final_answer(query, context)
            return result   
    
    async def travelAgent(query: str) -> str:
            from travelAgent.travel_agent import TravelRAGAgent
            agent = TravelRAGAgent()
            context = await agent.run_query(query)
            result = agent.get_final_answer(query, context)
            return result
            
        
    
    def __init__(self):
        load_dotenv()
        self.model = LiteLlm(
            model="openai/TechxGenus/Meta-Llama-3-8B-Instruct-GPTQ",
            api_base="http://localhost:8000/v1",
            api_key="EMPTY",
            provider="openai",
            config={"systemRoleSupported": False}
        )

      
        
        self.rootagent = LlmAgent(
            model=self.model,
            name="root_rag_agent",
            description="You are a helpful delegator assistant that can answer banking product questions by delegating to specialized agents.",
            instruction="""
                <You are a helpful assistant that can answer banking product questions across checking accounts, savings accounts, auto ,mortgage ,travel and credit cards.
                You delegate credit card tasks to the creditAgent ,  checking/savings tasks to the checkingAgent , auto tasks to the autoAgent, mortgage tasks to the mortgageAgent, travel tasks to the travelAgent.
                Follow these steps:
                1. If the user asks credit card questions, delegate to the creditAgent.
                2. If the user asks mortgage questions, delegate to the mortgageAgent.
                3. If the user asks auto loan questions, delegate to the autoAgent.
                4. If the user asks checking/savings account questions, delegate to the checkingAgent.
                5. If the user asks travel related questions, delegate to the travelAgent.
                6. Always clarify the results before proceeding.>
                """,
                tools=[self.creditAgent, self.mortgageAgent, self.autoAgent, self.checkingAgent, self.travelAgent],

            
        )


        self.session_service = InMemorySessionService()
        self.runner = Runner(
            agent=self.rootagent,
            app_name="ROOT_GPT",
            session_service=self.session_service
        )
        self.user_id = "user_1_gpt"
        self.session_id = "root_session_001_gpt"
        self.app_name = "ROOT_GPT"

    async def create_session(self):
        await self.session_service.create_session(
            app_name=self.app_name,
            user_id=self.user_id,
            session_id=self.session_id
        )

    
    async def call_agent_async(self,query: str, runner, user_id, session_id)->str:
        
        
        content = types.Content(role='user', parts=[types.Part(text=query)])

        async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
      
            if getattr(event, "content", None) and getattr(event.content, "parts", None):
                for part in event.content.parts:
                    text = getattr(part, "text", "") or ""
                
        response = text
        print(response)
        # response = await runner.run_async(
    
        #     user_id=user_id,
        #     session_id=session_id,
        #     new_message=content
        
        # )
        
        return response
    
    async def run_query(self, query: str):
        await self.create_session()
        print(f"Session created: App='{self.app_name}', User='{self.user_id}', Session='{self.session_id}'")
        print(f"Running Root agent query: {query}")
        response = await self.call_agent_async(
            query=query,
            runner=self.runner,
            user_id=self.user_id,
            session_id=self.session_id
        )
        return response

    async def invoke(self, rag_query: str):
        context = await self.run_query(rag_query)
        print("Identified SubAgent context", context)
        if context.__contains__("checkingAgent"):
            print("Delegated to checkingAgent")
            resp = await self.checkingAgent(rag_query)
            print("Final Result from checkingAgent", resp)
        elif context.__contains__("savingsAgent"):
            print("Delegated to savingsAgent")
            resp = await self.savingsAgent(rag_query)
            print("Final Result from savingsAgent", resp)
        elif context.__contains__("autoAgent"):
            print("Delegated to autoAgent")
            resp = await self.autoAgent(rag_query)
            print("Final Result from autoAgent", resp)
        elif context.__contains__("mortgageAgent"):
            print("Delegated to mortgageAgent")
            resp = await self.mortgageAgent(rag_query)
            print("Final Result from mortgageAgent", resp)
        elif context.__contains__("creditAgent"):
            print("Delegated to creditAgent")
            resp = await self.creditAgent(rag_query)
            print("Final Result from creditAgent", resp)
        elif context.__contains__("travelAgent"):
            print("Delegated to travelAgent")
            resp = await self.travelAgent(rag_query)
            print("Final Result from travelAgent", resp)
        else:
            resp= context

        return resp    
        
# Usage example
async def main():
    agent = RootAgent()
    rag_query = "what is monthly service fee for checking account?"
    response = await agent.invoke(rag_query)    
    print("Final Response from RootAgent:", response)
if __name__ == "__main__":
    asyncio.run(main())