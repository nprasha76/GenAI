# ...existing code...
import re
import json
from .adk_tools import vectorSearch
from google.genai import types # For creating message Content/Parts

async def call_agent_async(query: str, runner, user_id, session_id)->str:
  """Sends a query to the agent and prints the final response."""
  print(f"\n>>> User Query: {query}")

  # Prepare the user's message in ADK format
  content = types.Content(role='user', parts=[types.Part(text=query)])

  final_response_text = "Agent did not produce a final response." # Default
  invoked_tool_result = None
  done = False

  # Key Concept: run_async executes the agent logic and yields Events.
  # We iterate through events to find the final answer or tool_call.
  async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
      # Debug: print every event so you can inspect tool_call events in logs
      #print(f"[Event] Author:{getattr(event,'author',None)} Type:{type(event).__name__} Final:{event.is_final_response()} Actions:{getattr(event,'actions',None)} Content:{getattr(event,'content',None)}")
      #print(event)
      # Inspect any content parts for JSON tool_call or simulated text
      if getattr(event, "content", None) and getattr(event.content, "parts", None):
          for part in event.content.parts:
              text = getattr(part, "text", "") or ""
              # 1) Try to extract structured tool_call JSON embedded in assistant output
              
              jmatch = re.search(r'(\{[^}]*"tool_call"[^}]*\})', text, re.S)
              
              if jmatch:
                  try:
                      obj = json.loads(jmatch.group(1))
                      tc = obj.get("tool_call") or obj.get("toolCall") or {}
                      name = tc.get("name")
                      args = tc.get("arguments", {}) or {}
                      print(f"[Detected tool_call JSON] name={name} args={args}")
                      if name:
                          if name.lower().startswith("greet"):
                              invoked_tool_result = greetWF()
                              print(f"[Invoke] greetWF() -> {invoked_tool_result}")
                              done = True
                              break
                          if "getweather" in name.lower():
                              city = args.get("city") or (re.search(r'weather in ([\w\s]+)\??', query, re.I) or [None])[1]
                              city = city.strip() if city else query
                              tr = getweather(city)
                              invoked_tool_result = tr.get("report") if isinstance(tr, dict) and tr.get("status")=="success" else (tr.get("error_message") if isinstance(tr, dict) else str(tr))
                              print(f"[Invoke] getweather('{city}') -> {tr}")
                              done = True
                              break
                  except Exception as e:
                      print(f"[Error parsing tool_call JSON]: {e}")

              # 2) Fallback: detect simulated tool-call text like "using custom tool greetWF()"
              print("Text *** ",text)
              sm = re.search(r'using custom tool\s+([A-Za-z0-9_]+)\s*\(\)', text, re.I) \
                   or re.search(r'call the tool[:\s]+([A-Za-z0-9_]+)\b', text, re.I) \
                   or re.search(r'use the tool[:\s]+([A-Za-z0-9_]+)\b', text, re.I)\
                   or re.search(r"(?:use|call|invoke).*?['\"]?([A-Za-z0-9_]+)['\"]?\s*tool", text, re.I) \
                   or re.search(r"(?:call|use|invoke).*?\b([A-Za-z_][A-Za-z0-9_]*)\s*\(.*?\)\s*tool", text, re.I|re.S)\
                   or re.search(r"`?([A-Za-z_][A-Za-z0-9_]*)`?\s+tool", text)\
                   or re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", text)
              
              print(sm)
             
              if sm or True:
                  # tool_name = sm.group(1)
                  #always tool_name is vectorSearch for RAG agent irrespective of regex match
                  tool_name="vectorSearch"
                  print(f"[Detected simulated tool text] {tool_name}")
                  try:
                      if tool_name.lower().startswith("greet"):
                          invoked_tool_result = greetWF()
                          print(f"[Invoke] greetWF() -> {invoked_tool_result}")
                          done = True
                          break
                      if "getweather" in tool_name.lower():
                          city = (re.search(r'weather in ([\w\s]+)\??', query, re.I) or [None, None])[1]
                          city = city.strip() if city else query
                          tr = getweather(city)
                          invoked_tool_result = tr.get("report") if isinstance(tr, dict) and tr.get("status")=="success" else (tr.get("error_message") if isinstance(tr, dict) else str(tr))
                          print(f"[Invoke] getweather('{city}') -> {tr}")
                          done = True
                          break
                      
                      if "vectorsearch" in tool_name.lower():
                          invoked_tool_result = vectorSearch(query)
                          print(f"[Invoke] vectorSearch('{query}')-> {invoked_tool_result}")
                          done = True
                          break
                  except Exception as e:
                      print(f"[Fallback invoke error]: {e}")

      # If the event marks final response, capture it (unless we already ran a tool)
      if event.is_final_response():
          if invoked_tool_result:
              final_response_text = invoked_tool_result
          else:
              if event.content and event.content.parts:
                  final_response_text = event.content.parts[0].text
              elif getattr(event, "actions", None) and getattr(event.actions, "escalate", False):
                  final_response_text = f"Agent escalated: {getattr(event, 'error_message', 'No specific message.')}"
         
          # Detect "agent answered directly" patterns and invoke local tool based on query intent
          if not invoked_tool_result:
              # treat greetings / "who are you" => greetWF
              if re.search(r'\bwho are you\b', query, re.I) or re.search(r'\bwho am i talking to\b', query, re.I) \
                 or re.search(r'\bintroduc(e|tion)\b', query, re.I):
                  print("[Fallback] Model answered directly for greeting -> invoking greetWF()")
                  try:
                      final_response_text = greetWF()
                  except Exception as e:
                      print(f"[Fallback] greetWF() error: {e}")
              # treat weather questions => getweather(city)
              elif re.search(r'\bweather\b', query, re.I):
                  city_match = re.search(r'weather in ([\w\s]+)\??', query, re.I)
                  city = city_match.group(1).strip() if city_match else query
                  print(f"[Fallback] Model answered directly for weather -> invoking getweather('{city}')")
                  try:
                      tr = getweather(city)
                      if isinstance(tr, dict):
                          final_response_text = tr.get("report") if tr.get("status")=="success" else tr.get("error_message", str(tr))
                      else:
                          final_response_text = str(tr)
                  except Exception as e:
                      print(f"[Fallback] getweather() error: {e}")
         
          break

      if done:
          # we invoked a tool from a non-final event; stop and return that result
          final_response_text = invoked_tool_result or final_response_text
          break

  print(f"<<< Agent Response: {final_response_text}")
  return final_response_text
