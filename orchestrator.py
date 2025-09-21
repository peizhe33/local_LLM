import ollama
import json
from typing import Dict, Any
import nsql_modified

def process_query(user_query: str) -> Dict[str, Any]:
    """
    Process a user query through the LLM to extract SQL and vector components
    """
    print(f"\nProcessing Query: '{user_query}'\n")
    
    # Create the prompt
    prompt = f"""You are an STA analysis orchestrator. Analyze the user's query and split it into two distinct parts:

1. DATA QUERY: The specific numerical/data extraction request (for SQL conversion)
2. THEORETICAL QUERY: The explanatory/contextual request (for vector DB semantic search)

For the user query: "{user_query}"

Respond with ONLY a JSON object with this structure:
{{
  "natural_language_for_sql": "only the data extraction part in concise natural language, one sentence",
  "vector_question": "only the theoretical explanation part for vector DB search", 
  "reasoning": "brief explanation of the split"
}}

Important: The vector_question should be a standalone question about concepts, problems, and solutions - not about specific data points.
"""

    print("Sending request to Ollama...")
    print("=" * 50)

    try:
        # Send request using the ollama library to your local deepseek-r1 model
        response = ollama.generate(
            model="deepseek-r1:8b",
            prompt=prompt,
            format="json",
            options={
                "temperature": 0.1
            }
        )
        
        llm_output = response['response']
        
        print("Raw LLM Output:")
        print("=" * 50)
        print(llm_output)
        print("=" * 50)
        
        # Try to parse the JSON response
        try:
            analysis = json.loads(llm_output)
            print("\n✅ Successfully parsed JSON response!")
            print("\nParsed Instructions:")
            print(f"Natural Language for SQL: '{analysis.get('natural_language_for_sql', 'N/A')}'")
            print(f"Vector Question: '{analysis.get('vector_question', 'N/A')}'")
            print(f"Reasoning: {analysis.get('reasoning', 'N/A')}")
            
            # Extract both parts
            sql_natural_language = analysis.get('natural_language_for_sql', '')
            vector_query = analysis.get('vector_question', '')
            print(f"\n📋 For SQL Converter: '{sql_natural_language}'")
            print(f"🔍 For Vector DB Search: '{vector_query}'")
            
            # Prepare output data
            output_data = {
                "original_query": user_query,
                "sql_query": sql_natural_language,
                "vector_query": vector_query,
                "reasoning": analysis.get('reasoning', ''),
                "success": True,
                "timestamp": "2025-09-11T12:00:00Z"  # You might want to use datetime.now().isoformat()
            }
            
            # Save to JSON file
            with open("orchestrator_output.json", "w") as json_file:
                json.dump(output_data, json_file, indent=2)
            
            print(f"\n💾 Output saved to 'orchestrator_output.json'")
            
            return output_data
            
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse JSON response from LLM: {e}"
            print(f"\n❌ {error_msg}")
            print("The raw output was:", llm_output)
            
            # Save error to JSON file
            error_data = {
                "original_query": user_query,
                "success": False,
                "error": error_msg,
                "raw_output": llm_output,
                "timestamp": "2025-09-11T12:00:00Z"
            }
            
            with open("orchestrator_output.json", "w") as json_file:
                json.dump(error_data, json_file, indent=2)
                
            return error_data
            
    except Exception as e:
        error_msg = f"Error communicating with Ollama: {e}"
        print(error_msg)
        print("Make sure Ollama is running and the model name is correct.")
        
        # Save error to JSON file
        error_data = {
            "original_query": user_query,
            "success": False,
            "error": error_msg,
            "timestamp": "2025-09-11T12:00:00Z"
        }
        
        with open("orchestrator_output.json", "w") as json_file:
            json.dump(error_data, json_file, indent=2)
            
        return error_data

def main():
    # Default query
    default_query = "What is the start and endpoint for the path with worst slack? How to improve this path?"
    
    print("STA Analysis Query Processor")
    print("=" * 40)
    print("Enter your query or press Enter to use the default query.\n")
    
    while True:
        # Get user input
        user_input = input("Enter your query ('q' to quit): ").strip()
        
        # Check if user wants to quit
        if user_input.lower() == 'q':
            print("Exiting program. Goodbye!")
            break
            
        # Use default if input is empty, otherwise use user input
        query_to_process = default_query if user_input == "" else user_input
        
        # Process the query
        result = process_query(query_to_process)
        nsql_modified.main()

        print("\n" + "="*50)
        print("Ready for next query...")
        print("="*50 + "\n")

if __name__ == "__main__":
    main()