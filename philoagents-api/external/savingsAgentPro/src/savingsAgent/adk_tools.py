
# @title Define the get_weather Tool
from chase_embeddings.src.faissembeddings import load_faiss_index
from chase_embeddings.src.embeddings import load_embeddings_from_meta
from chase_embeddings.src.faissembeddings import query_embedding
import numpy as np
from sentence_transformers import SentenceTransformer


def getweatherCustom(city: str) -> dict:
    """Retrieves the current weather report for a specified city.

    Args:
        city (str): The name of the city (e.g., "New York", "London", "Tokyo").

    Returns:
        dict: A dictionary containing the weather information.
              Includes a 'status' key ('success' or 'error').
              If 'success', includes a 'report' key with weather details.
              If 'error', includes an 'error_message' key.
    """
    print(f"--- Tool: get_weather called for city: {city} ---") # Log tool execution
    city_normalized = city.lower().replace(" ", "") # Basic normalization

    # Mock weather data
    mock_weather_db = {
        "newyork": {"status": "success", "report": "The weather in New York is sunny with a temperature of 25°C."},
        "london": {"status": "success", "report": "It's cloudy in London with a temperature of 15°C."},
        "tokyo": {"status": "success", "report": "Tokyo is experiencing light rain and a temperature of 18°C."},
    }

    if city_normalized in mock_weather_db:
        return mock_weather_db[city_normalized]
    else:
        return {"status": "error", "error_message": f"Sorry, I don't have weather information for '{city}'."}

# ...existing code...
def getweather(city: str) -> dict:
    """Wrapper with the exact tool name 'getweather' that the model will call."""
    print(f"--- Tool: getweather called for city: {city} ---")
    return getweatherCustom(city)

def greetWF() -> str:
    print("--- Tool: greetWF called ---")
    return "Hello! I am your friendly WFassistant. How can I help you today?"

def vectorSearch(query: str)-> str:
    print(f"--- Tool: vectorSearch called for query: {query} ---")
    # Placeholder implementation
    
    index=load_faiss_index('/home/pnanda/chase_embeddings/savings_faiss.index')
    print("Index ntotal chunks:", index.ntotal)  # Should be num_chunks)
    embeddings, chunks, meta = load_embeddings_from_meta('/home/pnanda/chase_embeddings/savingsembeddings_meta.json')
    #print ('Chunks 1***',chunks[0])
    
    #print ('Chunks 2***',chunks[1])

    #print ('Chunks 3***',chunks[2])
    
    #print ('Chunks 4***',chunks[3])
    
    
    if embeddings is None or len(chunks) == 0:
            return []
    #query="What are the savings account options?" 
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    query_emb = embedder.encode(query)
    print("Query embedding shape:", query_emb.shape)

    # Alternatively, use the existing function
    q = query_embedding(query, "all-MiniLM-L6-v2").astype(np.float32).reshape(1, -1)
    # Normalize query vector as FAISS IndexFlatIP expects normalized vectors 
    q_norm = np.linalg.norm(q, axis=1, keepdims=True)
    q_norm[q_norm == 0] = 1.0
    q = q / q_norm
    
    try:
        ntotal = getattr(index, "ntotal", None)
        idx_dim = getattr(index, "d", None)
        print(f"Index ntotal={ntotal} d={idx_dim}, query vector shape={q.shape}, dtype={q.dtype}")
        if idx_dim is not None and q.shape[1] != idx_dim:
            print("Embedding dim mismatch: query dim", q.shape[1], "!= index dim", idx_dim)
            return []
    except Exception:
        pass

    # pick top_k safely
    #pick top 1 for this demo
    top_k =3
    if getattr(index, "ntotal", 0) is not None:
        top_k = min(top_k, max(1, int(getattr(index, "ntotal", top_k))))


    print("Pass Normalised Query embedding shape:", q.shape)
    # run search (Index built with IndexFlatIP expects normalized vectors)
    try:
        scores, idxs = index.search(q, top_k)
    except Exception as e:
        print("FAISS search failed:", e)
        return []

    scores = scores[0] if scores.ndim == 2 else scores
    print("Returned chunk idxs",idxs)
    idxs = idxs[0] if idxs.ndim == 2 else idxs

    results = []
    
    for i, s in zip(idxs, scores):
        if int(i) == -1:
            continue
        # guard against out-of-range
        if int(i) < 0 or int(i) >= len(chunks):
            continue
        results.append((int(i), float(s), chunks[int(i)]))
    #print(f"FAISS search results: {results}")
    return results


# ...existing code...
# Example tool usage (optional test)
#print(get_weather("New York"))
#print(get_weather("Paris"))

#vectorSearch("How is Chase Mobile app available?")