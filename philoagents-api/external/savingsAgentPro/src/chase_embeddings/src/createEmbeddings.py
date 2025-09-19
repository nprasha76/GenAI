from crawler import downloadcontent_withmarkdown

from .embeddings import create_embeddings,save_embeddings,create_chunks,load_embeddings_from_meta
from .faissembeddings import build_index_from_meta
import os
import json

# Personal Banking https://personal.chase.com/personal/savings
## Credit Card https://creditcards.chase.com/?jp_ltg=chsecate_featured&CELL=6TKV
# Auto https://autofinance.chase.com/?offercode=WDXDPXXX03
# Mortgage https://www.chase.com/personal/mortgage
# Travel https://www.chase.com/travel/the-edit

def main_creditcard():
    # Step 1: Download content from the specified URL
    url = "https://creditcards.chase.com/?jp_ltg=chsecate_featured&CELL=6TKV"
    content = downloadcontent_withmarkdown(url)

    # Step 2: Create vector embeddings from the markdown content
    # save the embeddings (npy) ,chunks(.json) and metadata (json)
    chunks = create_chunks(content)
    embeddings = create_embeddings(chunks)
    metadata=save_embeddings(embeddings, chunks, out_prefix="creditembeddings")
    print(f"Embeddings metadata {metadata}")

    #Step 3: Load the metadata for emebeddings,chunks and metadata
    embeddings,chunks,meta=load_embeddings_from_meta('/home/pnanda/chase_embeddings/src/creditembeddings_meta.json')
    #print(embeddings)
    #print (chunks)
    #print (meta)

def main_travel():
    # Step 1: Download content from the specified URL
    url = "https://www.chase.com/travel/the-edit"
    content = downloadcontent_withmarkdown(url)

    # Step 2: Create vector embeddings from the markdown content
    # save the embeddings (npy) ,chunks(.json) and metadata (json)
    chunks = create_chunks(content)
    embeddings = create_embeddings(chunks)
    metadata=save_embeddings(embeddings, chunks, out_prefix="travelembeddings")
    print(f"Embeddings metadata {metadata}")

    #Step 3: Load the metadata for emebeddings,chunks and metadata
    embeddings,chunks,meta=load_embeddings_from_meta('/home/pnanda/chase_embeddings/travelembeddings_meta.json')
    #print(embeddings)
    #print (chunks)
    #print (meta)


def main_savings():
    # Step 1: Download content from the specified URL
    url = "https://personal.chase.com/personal/savings"
    content = downloadcontent_withmarkdown(url)

    # Step 2: Create vector embeddings from the markdown content
    # save the embeddings (npy) ,chunks(.json) and metadata (json)
    chunks = create_chunks(content)
    embeddings = create_embeddings(chunks)
    metadata=save_embeddings(embeddings, chunks, out_prefix="savingsembeddings")
    print(f"Embeddings metadata {metadata}")

    #Step 3: Load the metadata for emebeddings,chunks and metadata
    embeddings,chunks,meta=load_embeddings_from_meta('/home/pnanda/chase_embeddings/savingsembeddings_meta.json')
    #print(embeddings)
    #print (chunks)
    #print (meta)

def main_mortgage():
    # Step 1: Download content from the specified URL
    url = "https://www.chase.com/personal/mortgage"
    content = downloadcontent_withmarkdown(url)

    # Step 2: Create vector embeddings from the markdown content
    # save the embeddings (npy) ,chunks(.json) and metadata (json)
    chunks = create_chunks(content)
    embeddings = create_embeddings(chunks)
    metadata=save_embeddings(embeddings, chunks, out_prefix="mortgageembeddings")
    print(f"Embeddings metadata {metadata}")

    #Step 3: Load the metadata for emebeddings,chunks and metadata
    embeddings,chunks,meta=load_embeddings_from_meta('/home/pnanda/chase_embeddings/mortgageembeddings_meta.json')
    #print(embeddings)
    #print (chunks)
    #print (meta)
   

def main_auto():
    # Step 1: Download content from the specified URL
    url = "https://autofinance.chase.com/?offercode=WDXDPXXX03"
    content = downloadcontent_withmarkdown(url)

    # Step 2: Create vector embeddings from the markdown content
    # save the embeddings (npy) ,chunks(.json) and metadata (json)
    chunks = create_chunks(content)
    embeddings = create_embeddings(chunks)
    metadata=save_embeddings(embeddings, chunks, out_prefix="autoembeddings")
    print(f"Embeddings metadata {metadata}")

    #Step 3: Load the metadata for emebeddings,chunks and metadata
    embeddings,chunks,meta=load_embeddings_from_meta('/home/pnanda/chase_embeddings/autoembeddings_meta.json')
    #print(embeddings)
    #print (chunks)
    #print (meta)
   

def main():
    # Step 1: Download content from the specified URL
    url = "https://www.chase.com/travel/the-edit"
    content = downloadcontent_withmarkdown(url)

    # Step 2: Create vector embeddings from the markdown content
    # save the embeddings (npy) ,chunks(.json) and metadata (json)
    chunks = create_chunks(content)
    print(len(chunks))
    
    embeddings = create_embeddings(chunks)
    metadata=save_embeddings(embeddings, chunks, out_prefix="travelembeddings")
    print(f"Embeddings metadata {metadata}")

    #Step 3: Load the metadata.json for embeddings,chunks and metadata
    embeddings,chunks,meta=load_embeddings_from_meta('/home/pnanda/chase_embeddings/travelembeddings_meta.json')
    #print(embeddings)
    #print (chunks)
    #print (meta)

    #Step 4: Build the FAISS index from the embeddings whose path is in metadata file
    
def buildFaiss(metapath:str,indexpath:str):
    #Step 4: Build the FAISS index from the embeddings whose path is in metadata file
    build_index_from_meta(metapath, indexpath)


if __name__ == "__main__":
      #main()
      #main_savings()
      main_mortgage()
      main_auto()    
      main_creditcard()  
      main_travel()    

    
    #buildFaiss('/home/pnanda/chase_embeddings/savingsembeddings_meta.json', indexpath="savings_faiss.index")
      buildFaiss('/home/pnanda/chase_embeddings/mortgageembeddings_meta.json', indexpath="mortgage_faiss.index")
      buildFaiss('/home/pnanda/chase_embeddings/autoembeddings_meta.json', indexpath="auto_faiss.index")
      buildFaiss('/home/pnanda/chase_embeddings/creditembeddings_meta.json', indexpath="credit_faiss.index")
      buildFaiss('/home/pnanda/chase_embeddings/travelembeddings_meta.json', indexpath="travel_faiss.index")  