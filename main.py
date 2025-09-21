
import sqparser
import orchestrator

def main():
    session_id = sqparser.parse_timing_report()
    print(f"Successfully parsed & stored timing report, generated session: {session_id}")
    orchestrator.main()
    
if __name__ == "__main__":
    main()
    