import json
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings

# --- Configuration ---
INPUT_JSON_FILE = "parsed_report.json"  # The output from your parser
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "timing_paths"

def main():
    # --- Step 1: Load the Local Embedding Model ---
    print(f"[1/5] Loading the local embedding model: {EMBEDDING_MODEL_NAME}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Model loaded successfully.\n")

    # --- Step 2: Initialize the Local Vector DB (Chroma) ---
    print(f"[2/5] Initializing ChromaDB at: {CHROMA_DB_PATH}")
    client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=Settings(allow_reset=True, anonymized_telemetry=False)
    )
    
    # Delete existing collection if it exists to avoid storage issues
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection: {COLLECTION_NAME}")
    except:
        print(f"No existing collection found, creating new one: {COLLECTION_NAME}")
    
    # Create new collection. 'cosine' similarity is best for text.
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    print("ChromaDB collection created.\n")

    # --- Step 3: Load your parsed data ---
    print(f"[3/5] Loading data from: {INPUT_JSON_FILE}")
    with open(INPUT_JSON_FILE, 'r') as f:
        data = json.load(f)
    
    # Store session metadata for reference
    session_metadata = data.get("session_metadata", {})
    
    # Assuming your JSON has a top-level "paths" key with a list of path objects
    paths = data.get("paths", [])
    print(f"Found {len(paths)} paths to process.\n")

    # --- Step 4: Process each path and prepare for upsert ---
    print("[4/5] Processing paths and generating embeddings...")
    
    ids = []
    embeddings = []
    metadatas = []
    documents = []  # Store the embedding content as documents for retrieval
    
    for path in paths:
        # This is your golden input to the model
        text_to_embed = path["embedding_content"] 
        
        # Generate the vector
        vector = embedding_model.encode(text_to_embed).tolist() # Convert to list for Chroma
        
        # Get the metadata dict for this path
        path_metadata = path["metadata"]
        
        # Get full details for comprehensive storage
        full_details = path.get("full_details", {})
        
        # Prepare the record for the Vector DB with ALL necessary data
        record_metadata = {                       
            "slack": float(path_metadata["slack_value"]) if path_metadata["slack_value"] and path_metadata["slack_value"].strip() != '' else 0.0,
            "violation_status": path_metadata["violation_status"],
            "clock_domain": path_metadata["clock_domain"],
            "design_block": path_metadata["design_block"],
            "json_pointer": path_metadata["json_pointer"],
            "report_id": path_metadata["session_id"], # Crucial for filtering later
            "path_id": path_metadata["path_id"],
            "path_uid": path_metadata["path_uid"],
            "path_group": path_metadata["path_group"],
            "path_type": path_metadata["path_type"],
            "sigma": path_metadata["sigma"],
            "report_timestamp": path_metadata["report_timestamp"],
            "startpoint": full_details.get("startpoint", ""),
            "endpoint": full_details.get("endpoint", ""),
            "session_id": session_metadata.get("session_id", ""),
            "total_paths": session_metadata.get("total_paths", 0),
            "violated_paths": session_metadata.get("violated_paths", 0),
            "original_json_file": INPUT_JSON_FILE  # Pointer back to original JSON
        }
        
        # Append the data for batch upsert
        ids.append(path_metadata["path_uid"]) # Use the unique ID as the Chroma ID
        embeddings.append(vector)
        metadatas.append(record_metadata)
        documents.append(text_to_embed)  # Store the embedding content as document
    
    # --- Step 5: Upsert to Vector DB ---
    print("[5/5] Uploading all vectors to the database...")
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents
    )
    print("All vectors uploaded successfully!")
    
    # --- Verification ---
    final_count = collection.count()
    print(f"\nIngestion complete! The database now contains {final_count} entries.")
    
    # Show sample of what was stored
    if final_count > 0:
        sample_result = collection.get(ids=[ids[0]])
        print(f"\nSample stored metadata keys: {list(sample_result['metadatas'][0].keys())}")
        print(f"Sample document length: {len(sample_result['documents'][0])} characters")

if __name__ == "__main__":
    main()