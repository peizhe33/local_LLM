import sqlite3
import pandas as pd
import requests
import json
import re
from typing import List, Dict, Any

class TimingReportQuerySystem:
    def __init__(self, db_path: str, ollama_host: str = "http://localhost:11434"):
        """
        Initialize the SQLite + Ollama LLM query system
        """
        self.db_path = db_path
        self.ollama_host = ollama_host
        self.conn = sqlite3.connect(db_path)
        
        # Test Ollama connection
        self.test_ollama_connection()
        
        # Get database schema for context
        self.schema_info = self.get_database_schema()
        
    def test_ollama_connection(self):
        """Test if Ollama is running and available"""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags")
            if response.status_code == 200:
                print("✓ Ollama connection successful")
                models = response.json().get('models', [])
                if models:
                    print(f"Available models: {[m['name'] for m in models]}")
                else:
                    print("No models found in Ollama")
            else:
                raise Exception(f"Ollama API returned status {response.status_code}")
        except Exception as e:
            raise Exception(f"Ollama connection failed: {str(e)}. Make sure Ollama is running.")
    
    def get_database_schema(self) -> str:
        """
        Extract database schema information for LLM context
        """
        schema = []
        
        # Get table information
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        for table in tables:
            table_name = table[0]
            schema.append(f"Table: {table_name}")
            
            # Get column information
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            
            for col in columns:
                col_name, col_type = col[1], col[2]
                schema.append(f"  - {col_name}: {col_type}")
            
            schema.append("")  # Empty line for readability
        
        return "\n".join(schema)
    
    def generate_sql_with_ollama(self, natural_language_query: str) -> str:
        """
        Use Ollama to convert natural language to SQL query
        """
        prompt = f"""You are an expert SQL developer. Given the following database schema:

{self.schema_info}

IMPORTANT RULES:
1. Return ALL information requested in the natural language query
2. If query asks for "what corresponds to" or "which X has Y", include identifying information
3. For maximum/minimum values, include the row that contains that extreme value
4. Use proper JOINs when data spans multiple tables
5. Include relevant context columns (point_name, path_id, startpoint, endpoint when relevant)
6. NULL values represent missing/unknown data, not numerical values
7. When querying for extreme values (minimum, maximum, top, bottom, highest, lowest, 
  best, worst, largest, smallest, or any ranking/ordering operation), you MUST exclude NULL values

NULL EXCLUSION RULE:
For ANY query that involves ordering, ranking, or finding extreme values, add:
WHERE numerical_column IS NOT NULL

CRITICAL SEMANTIC UNDERSTANDING:
- "slack" is a NUMERICAL VALUE (can be positive or negative)
- "worst slack" = most NEGATIVE slack value = MIN(slack)
- "best slack" = most POSITIVE slack value = MAX(slack)  
- "violated paths" = paths with slack < 0 OR status = 'VIOLATED'
- "met paths" = paths with slack >= 0 OR status = 'MET'

STATUS VALUES (CATEGORICAL):
- 'VIOLATED' = timing violation (slack < 0)
- 'MET' = timing requirement met (slack >= 0)
- There is NO 'Worst' status - "worst" describes numerical slack values

MATHEMATICAL DEFINITIONS:
- Worst slack = minimum slack value (most negative number)
- Best slack = maximum slack value (least negative or most positive number)
- Negative slack = timing violation
- Positive slack = timing margin

QUERY MAPPING - NEVER CONFUSE NUMERICAL vs CATEGORICAL:
- "what is the worst slack?" → SELECT MIN(slack) FROM timing_paths;
- "show me violated paths" → SELECT * FROM timing_paths WHERE status = 'VIOLATED';
- "paths with worst slack" → SELECT * FROM timing_paths ORDER BY slack ASC LIMIT 10;
- "what is the best slack?" → SELECT MAX(slack) FROM timing_paths;

CRITICAL DOMAIN KNOWLEDGE:
- "trans" means TRANSITION TIME (found in path_points.trans column)
- "cap" means CAPACITANCE (found in path_points.capacitance column)  
- "slack" means TIMING SLACK (found in timing_paths.slack column)
- "point_name" identifies specific points in the design hierarchy
- path_points contains detailed timing data for each point in a path
- timing_paths contains overall path metadata

IMPORTANT SQL BEHAVIORS:
- NULL values sort first in ASCENDING order (use WHERE column IS NOT NULL to exclude NULLs)
- NULL values sort last in DESCENDING order
- For meaningful min/max calculations, EXCLUDE NULL values using WHERE column IS NOT NULL

COMPARATIVE TERM MAPPING:
- "highest", "maximum", "largest", "greatest" → Use MAX() or ORDER BY DESC
- "lowest", "minimum", "smallest", "least" → Use MIN() or ORDER BY ASC  
- "worst" slack → Use MIN() [most negative value]
- "best" slack → Use MAX() [least negative or positive value]

QUERY CONSTRUCTION RULES:
1. For extreme values: Use MAX() for highest, MIN() for lowest
2. EXCLUDE NULL VALUES: Add WHERE column IS NOT NULL for meaningful min/max
3. When asking "what point corresponds to": Include path_points.point_name
4. Use JOINs when data spans multiple tables
5. Use LIMIT 1 when looking for extreme values

EXAMPLE QUERIES:
- "minimum fanout and point name": SELECT pp.point_name, pp.fanout FROM path_points pp WHERE pp.fanout IS NOT NULL ORDER BY pp.fanout ASC LIMIT 1;
- "maximum fanout and point name": SELECT pp.point_name, pp.fanout FROM path_points pp WHERE pp.fanout IS NOT NULL ORDER BY pp.fanout DESC LIMIT 1;
- "lowest cap with path info": SELECT pp.point_name, pp.capacitance, tp.startpoint, tp.endpoint FROM path_points pp JOIN timing_paths tp ON pp.path_id = tp.path_id WHERE pp.capacitance IS NOT NULL ORDER BY pp.capacitance ASC LIMIT 1;
- "highest trans with path info": SELECT pp.point_name, pp.trans, tp.startpoint, tp.endpoint FROM path_points pp JOIN timing_paths tp ON pp.path_id = tp.path_id WHERE pp.trans IS NOT NULL ORDER BY pp.trans DESC LIMIT 1;

Convert this natural language query to SQL:
"{natural_language_query}"

Return ONLY the SQL query without any explanation. Use proper JOINs if needed.
The SQL query should be syntactically correct for SQLite.

SQL Query:"""
        
        # Ollama API request
        payload = {
            "model": "llama3.2",  # Use the model you have in Ollama
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "max_tokens": 200
            }
        }
        
        try:
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json=payload,
                timeout=600  # 30 second timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                sql_query = result['response'].strip()
                
                # Clean up the SQL query
                sql_query = re.sub(r'^```sql\s*', '', sql_query, flags=re.IGNORECASE)
                sql_query = re.sub(r'```$', '', sql_query)
                sql_query = sql_query.strip().rstrip(';') + ';'
                
                return sql_query
            else:
                raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
                
        except requests.exceptions.Timeout:
            raise Exception("Ollama request timed out. Is the model loaded?")
        except Exception as e:
            raise Exception(f"Ollama request failed: {str(e)}")
    
    def execute_query(self, sql_query: str) -> pd.DataFrame:
        """
        Execute SQL query and return results as DataFrame
        """
        try:
            df = pd.read_sql_query(sql_query, self.conn)
            return df
        except Exception as e:
            raise Exception(f"SQL execution error: {str(e)}")
    
    def validate_and_fix_sql(self, sql_query: str, natural_query: str) -> str:
        """
        Validate SQL and use Ollama for correction if needed
        """
        max_attempts = 3
        attempt = 0
        original_sql = sql_query
        
        while attempt < max_attempts:
            try:
                # Test the query
                self.execute_query(sql_query)
                return sql_query  # Query is valid
                
            except Exception as e:
                attempt += 1
                if attempt >= max_attempts:
                    raise Exception(f"Failed to generate valid SQL after {max_attempts} attempts. Last error: {str(e)}")
                
                # Ask Ollama to fix the query
                fix_prompt = f"""The SQL query you generated has an error: {str(e)}

Original natural language query: "{natural_query}"
Generated SQL: {original_sql}

Common issues to check:
1. If query needs data from multiple tables, use JOIN: 
   - JOIN path_points pp ON pp.path_id = tp.path_id
   - JOIN clock_data cd ON cd.path_id = tp.path_id
2. Use table aliases: timing_paths tp, path_points pp, clock_data cd
3. Qualify column names with table aliases in JOIN queries

Please fix the SQL query. Return ONLY the corrected SQL without any explanation.

Corrected SQL:"""
                
                payload = {
                    "model": "llama3.2",
                    "prompt": fix_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "max_tokens": 200
                    }
                }
                
                response = requests.post(
                    f"{self.ollama_host}/api/generate",
                    json=payload,
                    timeout=600
                )
                
                if response.status_code == 200:
                    result = response.json()
                    fixed_sql = result['response'].strip()
                    fixed_sql = re.sub(r'^```sql\s*', '', fixed_sql, flags=re.IGNORECASE)
                    fixed_sql = re.sub(r'```$', '', fixed_sql)
                    fixed_sql = fixed_sql.strip().rstrip(';') + ';'
                    
                    sql_query = fixed_sql
                else:
                    raise Exception(f"Ollama fix request failed: {response.status_code}")
    
    def query_database(self, natural_language_query: str) -> Dict[str, Any]:
        """
        Main method to process natural language query and return results
        """
        try:
            # Step 1: Generate SQL from natural language
            print("Generating SQL query with Ollama...")
            sql_query = self.generate_sql_with_ollama(natural_language_query)
            print(f"Generated SQL: {sql_query}")
            
            # Step 2: Validate and fix if needed
            sql_query = self.validate_and_fix_sql(sql_query, natural_language_query)
            print(f"Validated SQL: {sql_query}")
            
            # Step 3: Execute query
            print("Executing query...")
            results_df = self.execute_query(sql_query)
            
            # Step 4: Format results
            results = {
                "query": natural_language_query,
                "sql_query": sql_query,
                "results": results_df.to_dict('records'),
                "row_count": len(results_df),
                "columns": list(results_df.columns)
            }
            
            return results
            
        except Exception as e:
            return {
                "error": str(e),
                "query": natural_language_query
            }
    
    def close(self):
        """Close database connection"""
        self.conn.close()

# Example usage
def main():
    # Initialize the system
    query_system = TimingReportQuerySystem("timing_report.db")
    
    print("Timing Report Query System Initialized with Ollama!")
    print("Reading natural language query from 'orchestrator_output.json'...\n")
    
    try:
        # Read the JSON file
        with open('orchestrator_output.json', 'r') as f:
            data = json.load(f)
        
        # Extract the natural language query from the "sql_query" field
        natural_language_query = data.get("sql_query")
        
        if not natural_language_query:
            print("Error: 'sql_query' field not found or empty in orchestrator_output.json")
            query_system.close()
            return
        
        print(f"Found natural language query: '{natural_language_query}'")
        
        # Process the query from the file
        result = query_system.query_database(natural_language_query)
        
        # Display results on screen
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"\nResults ({result['row_count']} rows):")
            print("-" * 50)
            
            # Display as table
            df = pd.DataFrame(result['results'])
            if not df.empty:
                print(df.to_string(index=False))
            else:
                print("No results found")
            
            print(f"\nGenerated SQL: {result['sql_query']}")
            print("-" * 50)
        
        # Write the complete result dictionary to a JSON file
        output_filename = "query_results.json"
        with open(output_filename, 'w') as outfile:
            json.dump(result, outfile, indent=4)
        
        print(f"\n✓ Full results have been saved to '{output_filename}'")
            
    except FileNotFoundError:
        print("Error: File 'orchestrator_output.json' not found.")
    except json.JSONDecodeError:
        print("Error: File 'orchestrator_output.json' contains invalid JSON.")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
    finally:
        query_system.close()

if __name__ == "__main__":
    main()