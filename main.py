import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings
import ollama

# --- Configuration ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "timing_paths"
OLLAMA_MODEL = "llama3.2"

def initialize_components():
    """Initialize all necessary components"""
    print("Loading embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    print("Connecting to vector database...")
    client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=Settings(anonymized_telemetry=False)
    )
    
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
        total_paths = collection.count()
        print(f"✅ Database connected. It contains {total_paths} timing paths.")
    except:
        print("❌ Error: Vector database not found. Run the ingestion script first.")
        return None, None
    
    return embedding_model, collection

def get_all_timing_paths(collection):
    """Retrieve ALL timing paths from the database"""
    try:
        # Get all documents from the collection
        results = collection.get(include=["metadatas", "documents", "embeddings"])
        
        # Format the results like Chroma query output
        formatted_results = {
            'ids': [results['ids']],
            'metadatas': [results['metadatas']],
            'documents': [results['documents']],
            'embeddings': [results['embeddings']],  # Always include
            'distances': [[0.0] * len(results['ids'])]  # Placeholder distances
        }
        
        return formatted_results
        
    except Exception as e:
        print(f"❌ Error retrieving all paths: {e}")
        return {'ids': [[]], 'metadatas': [[]], 'documents': [[]], 'distances': [[]]}

def search_timing_paths(query, collection, embedding_model):
    """Search for relevant timing paths - REMOVED top_k constraint"""
    # First get ALL paths from the database
    all_results = get_all_timing_paths(collection)
    
    if not all_results['ids'][0]:
        return all_results  # Return empty results if no paths found
    
    # If there's a query, calculate similarity for ALL paths
    if query and query.strip():
        query_embedding = embedding_model.encode(query).tolist()
        
        # Calculate similarity for each path
        distances = []
        for i, path_embedding in enumerate(all_results['embeddings'][0]):
            try:
                # Calculate cosine distance manually
                from numpy import dot
                from numpy.linalg import norm
                import numpy as np
                
                cosine_similarity = dot(query_embedding, path_embedding) / (norm(query_embedding) * norm(path_embedding))
                distance = 1 - cosine_similarity
                distances.append(distance)
            except:
                distances.append(1.0)  # Default distance if calculation fails
    else:
        # If no query, use placeholder distances
        distances = [0.0] * len(all_results['ids'][0])
    
    # Update the results with calculated distances
    all_results['distances'] = [distances]
    
    return all_results

def print_intermediate_results(search_results):
    """Print debug-friendly intermediate results for inspection"""
    if not search_results['ids'] or not search_results['ids'][0]:
        print("⚠ No timing paths found.")
        return
    
    total_paths = len(search_results['ids'][0])
    print(f"\n🔍 Retrieved ALL {total_paths} Paths:")
    print("=" * 80)
    
    for i in range(total_paths):
        metadata = search_results['metadatas'][0][i]
        document = search_results['documents'][0][i]
        distance = search_results['distances'][0][i]
        
        print(f"Result {i+1}")
        print("-" * 80)
        print(f"ID: {search_results['ids'][0][i]}")
        print(f"Similarity Score: {1 - distance:.4f}")  # Convert distance to similarity
        print(f"Startpoint: {metadata.get('startpoint', 'N/A')}")
        print(f"Endpoint: {metadata.get('endpoint', 'N/A')}")
        print(f"Slack: {metadata.get('slack', 'N/A')}")
        print(f"Violation Status: {metadata.get('violation_status', 'N/A')}")
        print(f"Clock Domain: {metadata.get('clock_domain', 'N/A')}")
        print(f"Design Block: {metadata.get('design_block', 'N/A')}")
        print(f"Content (preview): {document[:200]}...\n")  # Preview first 200 chars

def format_rag_context(search_results):
    """Format search results for the LLM context - now includes ALL paths"""
    if not search_results['ids'] or not search_results['ids'][0]:
        return "No timing paths found in the database."
    
    total_paths = len(search_results['ids'][0])
    context = f"COMPLETE TIMING REPORT ANALYSIS - {total_paths} PATHS TOTAL:\n\n"
    
    # Group by violation status for better organization
    violated_paths = []
    met_paths = []
    
    for i in range(total_paths):
        metadata = search_results['metadatas'][0][i]
        document = search_results['documents'][0][i]
        distance = search_results['distances'][0][i]
        
        path_info = {
            'metadata': metadata,
            'document': document,
            'similarity': 1 - distance
        }
        
        if metadata.get('violation_status') == 'VIOLATED':
            violated_paths.append(path_info)
        else:
            met_paths.append(path_info)
    
    # Add violated paths first (usually more important)
    context += f"VIOLATED PATHS: {len(violated_paths)}\n"
    context += "=" * 50 + "\n"
    for i, path in enumerate(violated_paths, 1):
        context += f"\n--- Violated Path {i} ---\n"
        context += f"Similarity: {path['similarity']:.3f}\n"
        context += f"Slack: {path['metadata'].get('slack', 'N/A')}\n"
        context += f"Clock Domain: {path['metadata'].get('clock_domain', 'N/A')}\n"
        context += f"Design Block: {path['metadata'].get('design_block', 'N/A')}\n"
        context += f"Content: {path['document']}\n"
    
    # Add met paths
    context += f"\n\nPATHS MEETING TIMING: {len(met_paths)}\n"
    context += "=" * 50 + "\n"
    for i, path in enumerate(met_paths, 1):
        context += f"\n--- Met Path {i} ---\n"
        context += f"Similarity: {path['similarity']:.3f}\n"
        context += f"Slack: {path['metadata'].get('slack', 'N/A')}\n"
        context += f"Clock Domain: {path['metadata'].get('clock_domain', 'N/A')}\n"
        context += f"Design Block: {path['metadata'].get('design_block', 'N/A')}\n"
        context += f"Content: {path['document']}\n"
    
    # Add summary statistics
    context += f"\n\nSUMMARY STATISTICS:\n"
    context += "=" * 50 + "\n"
    context += f"Total Paths: {total_paths}\n"
    context += f"Violated Paths: {len(violated_paths)}\n"
    context += f"Paths Meeting Timing: {len(met_paths)}\n"
    if total_paths > 0:
        context += f"Violation Rate: {(len(violated_paths)/total_paths*100):.1f}%\n"
    
    return context

def ask_llm(question, context):
    """Ask the LLM with COMPLETE RAG context"""
    prompt = f"""You are a senior timing analysis expert. Analyze the COMPLETE timing report below and provide a comprehensive analysis focused on answering the user's specific question.

USER'S QUESTION: {question}

TIMING REPORT DATA:
{context}

ANALYSIS STRUCTURE:

1. **DIRECT ANSWER TO USER'S QUESTION**:
   - Start by directly addressing "{question}"
   - Provide a clear, concise answer based on the timing data
   - Use specific numbers and facts from the report

2. **DETAILED SUPPORTING EVIDENCE**:
   - Present the data that supports your answer
   - Reference specific paths, slack values, and violations
   - Explain why this answers the user's question

3. **OVERALL TIMING STATUS**:
   - Total paths analyzed: [number]
   - Violation percentage: [percentage]
   - Overall timing health assessment

4. **CRITICAL FINDINGS RELATED TO THE QUERY**:
   - Worst timing violations relevant to the question
   - Patterns or trends that answer the user's query
   - Key insights specifically addressing "{question}"

5. **ROOT CAUSE ANALYSIS**:
   - Potential causes behind the findings related to the question
   - Common characteristics of paths mentioned in your answer
   - Technical reasons for the observed behavior

6. **ACTIONABLE RECOMMENDATIONS**:
   - Specific fixes addressing the user's concerns
   - Design optimizations relevant to the question
   - Next steps that directly help with "{question}"

7. **ADDITIONAL CONTEXT**:
   - Other important findings from the full report
   - Positive aspects worth mentioning
   - Any limitations or caveats in the analysis

CRITICAL INSTRUCTIONS:
- **START YOUR RESPONSE BY DIRECTLY ANSWERING THE USER'S QUESTION**
- Use concrete data: slack values, path counts, violation percentages
- Reference specific path IDs when discussing examples
- Keep the focus on addressing "{question}" throughout
- Provide technical insights that would help a design engineer
- Use bullet points and clear organization for readability

DIRECT ANSWER TO "{question}":"""
    
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{
                'role': 'user',
                'content': prompt
            }],
            options={
                'temperature': 0.01,
                'num_predict': 2048
            }
        )
        return response['message']['content']
    except Exception as e:
        return f"Error: {str(e)}. Make sure Ollama is running with 'ollama serve'"

def main():
    print("🔧 Initializing Timing Report RAG System...")
    embedding_model, collection = initialize_components()
    
    if embedding_model is None:
        return
    
    print("\n" + "="*50)
    print("🤖 Timing Report Assistant Ready!")
    print("Ask questions about timing paths, slack, violations, etc.")
    print("Type 'quit' or 'exit' to end the session")
    print("="*50 + "\n")
    
    while True:
        try:
            user_query = input("\n💬 Your question: ").strip()
            
            if user_query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye! 👋")
                break
            
            if not user_query:
                continue
            
            print("\n🔍 Retrieving ALL timing paths...")
            search_results = search_timing_paths(user_query, collection, embedding_model)
            
            # ✅ Show intermediate debug results
            print_intermediate_results(search_results)
            
            # ✅ Prepare context for LLM with ALL paths
            context = format_rag_context(search_results)
            
            print("\n🧠 Analyzing with LLM...")
            response = ask_llm(user_query, context)
            
            print("\n" + "="*50)
            print("📋 COMPREHENSIVE LLM ANALYSIS:")
            print("="*50)
            print(response)
            print("="*50)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()