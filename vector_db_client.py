# retriever_optimized.py
import json
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings
import ollama
import argparse
import os
import time
from typing import List, Dict, Any

# --- Configuration ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_DB_PATH = "./chroma_db"
OLLAMA_MODEL = "llama3.2"
MAX_PATHS_FOR_LLM = 3
MIN_SIMILARITY_THRESHOLD = 0.5
DEFAULT_QUERY = "show me the worst timing violations"  # Added default query

def get_collections(client):
    """Get available collections quickly"""
    return {coll.name: client.get_collection(coll.name) for coll in client.list_collections()}

def retrieve_original_path_data_fast(json_file_path: str, json_pointer: str) -> Dict[str, Any]:
    """Fast JSON retrieval for a SINGLE path"""
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        pointer_parts = [p for p in json_pointer.strip('/').split('/') if p and p != '']
        current = data
        for part in pointer_parts:
            if part.isdigit() and isinstance(current, list):
                current = current[int(part)]
            elif part in current:
                current = current[part]
            else:
                return None
        return current
    except Exception as e:
        print(f"Retrieval error for {json_pointer}: {e}")
        return None

def get_worst_violated_paths(collection, limit: int = 3) -> List[Dict]:
    """Pre-filter using metadata BEFORE semantic search"""
    try:
        results = collection.query(
            query_texts=[""],
            n_results=limit,
            where={"violation_status": "VIOLATED"},
            include=["metadatas", "documents"],
        )
        
        violated_paths = []
        for i in range(len(results['ids'][0])):
            violated_paths.append({
                'id': results['ids'][0][i],
                'metadata': results['metadatas'][0][i],
                'document': results['documents'][0][i],
                'similarity': 1.0,
            })
        
        violated_paths.sort(key=lambda x: x['metadata']['slack'])
        return violated_paths[:limit]
        
    except Exception as e:
        print(f"Metadata filtering failed: {e}")
        return []

def semantic_search_with_filtering(query_text: str, collection, embedding_model, n_results: int = 5) -> List[Dict]:
    """Combined semantic + metadata filtering"""
    try:
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results * 3,
            where={"violation_status": "VIOLATED"},
            include=["metadatas", "documents", "distances"],
        )
        
        filtered_paths = []
        for i in range(len(results['ids'][0])):
            similarity = 1 - results['distances'][0][i]
            if similarity >= MIN_SIMILARITY_THRESHOLD:
                filtered_paths.append({
                    'id': results['ids'][0][i],
                    'distance': results['distances'][0][i],
                    'similarity': similarity,
                    'metadata': results['metadatas'][0][i],
                    'document': results['documents'][0][i],
                })
        
        filtered_paths.sort(key=lambda x: (-x['similarity'], x['metadata']['slack']))
        return filtered_paths[:n_results]
        
    except Exception as e:
        print(f"Semantic search failed: {e}")
        return []

def format_focused_context(retrieved_paths: List[Dict], total_paths_count: int, query: str) -> str:
    """Extreme focus + validation instructions"""
    if not retrieved_paths:
        return "No relevant timing paths found."
    
    context = f"TIMING ANALYSIS CONTEXT - Focus ONLY on these {len(retrieved_paths)} most relevant paths:\n\n"
    context += "IMPORTANT: VERIFY all facts against this data before answering. Do NOT hallucinate.\n\n"
    
    if "worst" in query.lower() or "violat" in query.lower():
        paths_to_include = retrieved_paths[:MAX_PATHS_FOR_LLM]
    else:
        paths_to_include = retrieved_paths[:2]
    
    for i, path in enumerate(paths_to_include, 1):
        context += f"--- PATH {i} ---\n"
        context += f"ID: {path['id']}\n"
        context += f"Slack: {path['metadata']['slack']:.6f}\n"
        context += f"Status: {path['metadata']['violation_status']}\n"
        context += f"Clock: {path['metadata']['clock_domain']}\n"
        context += f"Design: {path['metadata']['design_block']}\n"
        
        if 'similarity' in path:
            context += f"Relevance: {path['similarity']:.3f}\n"
        
        context += f"Preview: {path['document'][:100]}...\n\n"
    
    violated_count = sum(1 for p in retrieved_paths if p['metadata']['violation_status'] == 'VIOLATED')
    context += f"SUMMARY: {violated_count} violated paths of {total_paths_count} total\n"
    
    if violated_count > 0:
        worst_slack = min(p['metadata']['slack'] for p in retrieved_paths if p['metadata']['violation_status'] == 'VIOLATED')
        context += f"Worst slack: {worst_slack:.6f}\n"
    
    return context

def generate_validated_response(query: str, context: str, model_name: str = OLLAMA_MODEL) -> str:
    """Structured prompting with validation requirements"""
    
    prompt = f"""**CRITICAL INSTRUCTIONS:**
1. Answer the question DIRECTLY and CONCISELY first
2. Then provide supporting evidence ONLY from the paths below
3. VERIFY that your evidence matches your answer exactly
4. If paths contradict each other, explain the discrepancy
5. Keep the entire response under 300 words

**CONTEXT DATA:**
{context}

**USER QUESTION:**
{query}

**STRUCTURED RESPONSE:**
Direct Answer:"""

    try:
        start_time = time.time()
        response = ollama.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt}],
            options={
                'temperature': 0.1,
                'num_predict': 500,
                'top_k': 20
            }
        )
        llm_time = time.time() - start_time
        print(f"LLM processed in {llm_time:.1f}s")
        return response['message']['content']
    except Exception as e:
        return f"LLM Error: {str(e)}"

def hybrid_search_worst_slack(query_text: str, collection, embedding_model, n_results: int = 5) -> List[Dict]:
    """HYBRID SEARCH: Semantic + Numerical filtering"""
    
    # Step 1: Semantic understanding - what does the user want?
    if "worst" in query_text.lower() or "most negative" in query_text.lower() or "lowest" in query_text.lower():
        # Step 2: Numerical filtering - get actually worst slack values
        results = collection.query(
            query_texts=["timing violation critical path"],  # Semantic boost
            n_results=n_results * 2,  # Get extra for filtering
            where={"violation_status": "VIOLATED"},  # Must be violated
            include=["metadatas", "documents", "distances"],
        )
        
        # Step 3: Numerical sorting by actual slack values
        paths = []
        for i in range(len(results['ids'][0])):
            paths.append({
                'id': results['ids'][0][i],
                'distance': results['distances'][0][i],
                'similarity': 1 - results['distances'][0][i],
                'metadata': results['metadatas'][0][i],
                'document': results['documents'][0][i],
            })
        
        # CRITICAL: Sort by actual slack value (numerical), not similarity
        paths.sort(key=lambda x: x['metadata']['slack'])  # Most negative first
        return paths[:n_results]
    
    else:
        # For non-numerical queries, use regular semantic search
        return semantic_search_with_filtering(query_text, collection, embedding_model, n_results)

def main():
    parser = argparse.ArgumentParser(description="Optimized Timing Analysis")
    parser.add_argument("query", nargs='?', default=DEFAULT_QUERY, help="Your question about timing analysis (default: show worst violations)")  # Made optional with default
    parser.add_argument("--collection", default="timing_paths_all", help="Collection to search")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM analysis")
    parser.add_argument("--strategy", choices=['semantic', 'worst', 'hybrid', 'all'], default='worst', help="Search strategy")  # FIXED: Added 'hybrid'
    args = parser.parse_args()

    print(f"Query: '{args.query}'")
    print(f"Strategy: {args.strategy}")
    print(f"Collection: {args.collection}")
    
    print(f"[1/4] Initializing...")
    start_time = time.time()
    
    if args.strategy == 'semantic' or args.strategy == 'hybrid':  # FIXED: Need embedding model for hybrid too
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    else:
        embedding_model = None
    
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collections = get_collections(client)
    
    if args.collection not in collections:
        print(f"Available collections: {list(collections.keys())}")
        return
    
    collection = collections[args.collection]
    total_paths = collection.count()
    print(f"Found {total_paths} paths in '{args.collection}'")
    
    print(f"[2/4] Retrieving paths with '{args.strategy}' strategy...")
    retrieval_start = time.time()
    
    if args.strategy == 'worst':
        retrieved_paths = get_worst_violated_paths(collection, limit=MAX_PATHS_FOR_LLM)
    elif args.strategy == 'semantic':
        retrieved_paths = semantic_search_with_filtering(args.query, collection, embedding_model, n_results=MAX_PATHS_FOR_LLM)
    elif args.strategy == 'hybrid':  # FIXED: Changed from args.startegy to args.strategy
        retrieved_paths = hybrid_search_worst_slack(args.query, collection, embedding_model, n_results=MAX_PATHS_FOR_LLM)
    else:
        results = collection.get(limit=MAX_PATHS_FOR_LLM, include=["metadatas", "documents"])
        retrieved_paths = [{
            'id': results['ids'][i],
            'metadata': results['metadatas'][i],
            'document': results['documents'][i],
            'similarity': 1.0
        } for i in range(min(MAX_PATHS_FOR_LLM, len(results['ids'])))]
    
    retrieval_time = time.time() - retrieval_start
    print(f"Retrieved {len(retrieved_paths)} paths in {retrieval_time:.1f}s")
    
    print(f"\n📊 TOP PATHS:")
    print("=" * 80)
    for i, path in enumerate(retrieved_paths, 1):
        print(f"{i}. {path['id']}")
        print(f"   Slack: {path['metadata']['slack']:.6f} ({path['metadata']['violation_status']})")
        print(f"   Clock: {path['metadata']['clock_domain']}")
        if 'similarity' in path:
            print(f"   Relevance: {path['similarity']:.3f}")
        print(f"   Preview: {path['document'][:80]}...")
        print("-" * 60)
    
    if not args.no_llm and retrieved_paths:
        print(f"\n[3/4] Generating focused analysis...")
        context = format_focused_context(retrieved_paths, total_paths, args.query)
        analysis = generate_validated_response(args.query, context)
        
        print(f"\n🤖 VALIDATED ANALYSIS:")
        print("=" * 80)
        print(analysis)
        print("=" * 80)
    
    total_time = time.time() - start_time
    print(f"\n✅ Total time: {total_time:.1f}s")

# Add test function for direct execution without command line
def test_retrieval():
    """Test function that can be called directly without command line arguments"""
    print("🧪 Running test retrieval with default query...")
    
    # Test the worst violations strategy
    test_args = type('Args', (), {
        'query': DEFAULT_QUERY,
        'collection': 'timing_paths_all',
        'no_llm': False,
        'strategy': 'worst'
    })()
    
    # Simulate the main function logic
    print(f"Testing with query: '{test_args.query}'")
    
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collections = get_collections(client)
    
    if 'timing_paths_all' not in collections:
        print("Error: timing_paths_all collection not found")
        return
    
    collection = collections['timing_paths_all']
    total_paths = collection.count()
    print(f"Found {total_paths} total paths")
    
    # Test worst violations retrieval
    retrieved_paths = get_worst_violated_paths(collection, limit=3)
    print(f"Retrieved {len(retrieved_paths)} worst paths")
    
    for i, path in enumerate(retrieved_paths, 1):
        print(f"{i}. Slack: {path['metadata']['slack']:.6f} - {path['id']}")
    
    return retrieved_paths

if __name__ == "__main__":
    # Check if no arguments provided, run test instead
    import sys
    if len(sys.argv) == 1:
        print("No arguments provided - running test mode...")
        test_retrieval()
    else:
        main()