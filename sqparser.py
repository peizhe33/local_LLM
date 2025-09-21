import json
import re
import sqlite3
from datetime import datetime
import uuid

def parse_timing_report(input_file="start_endpoints.txt", aliases_file="aliases.json"):

    # Load aliases from aliases.json
    with open("aliases.json") as f:
        aliases = json.load(f)

    # Extract regex patterns
    start_regex = "|".join(aliases["startpoint"])
    end_regex = "|".join(aliases["endpoint"])
    path_group = "|".join(aliases["path_group"])
    path_type = "|".join(aliases["path_type"])
    sigma = "|".join(aliases["sigma"])

    # Read timing report text
    with open("start_endpoints.txt") as f:
        text = f.read()

    # Generate unique session ID with timestamp
    session_id = f"timing_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    report_timestamp = datetime.now().isoformat()

    # Initialize SQLite database
    conn = sqlite3.connect('timing_report.db')
    cursor = conn.cursor()

    # Create tables
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS timing_paths (
        path_id INTEGER PRIMARY KEY,
        session_id TEXT,
        report_timestamp TEXT,
        startpoint TEXT,
        endpoint TEXT,
        clock_start TEXT,
        clock_end TEXT,
        path_group TEXT,
        path_type TEXT,
        sigma REAL,
        slack REAL,
        status TEXT,
        design_block TEXT,
        clock_domain TEXT
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS path_points (
        point_id INTEGER PRIMARY KEY,
        path_id INTEGER,
        point_name TEXT,
        fanout INTEGER,
        capacitance REAL,
        dtrans REAL,
        trans REAL,
        derate REAL,
        delta REAL,
        mean_incr REAL,
        sensit_incr REAL,
        corner_incr REAL,
        value_incr REAL,
        mean_path REAL,
        sensit_path REAL,
        value_path REAL,
        edge_type TEXT,
        FOREIGN KEY (path_id) REFERENCES timing_paths(path_id)
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS clock_data (
        clock_id INTEGER PRIMARY KEY,
        path_id INTEGER,
        clock_name TEXT,
        trans REAL,
        mean_delay REAL,
        corner_delay REAL,
        value_delay REAL,
        clock_type TEXT,
        FOREIGN KEY (path_id) REFERENCES timing_paths(path_id)
    )
    ''')

    conn.commit()

    def parse_header(header_line):
        """
        Parse the header line to determine column positions dynamically.
        Returns a dictionary of column names and their (start, end) positions.
        """
        columns = {}
        pos = 0
        
        # Track duplicate column names
        column_count = {}
        
        # Process Point column
        if header_line.startswith("Point"):
            # Count spaces after "Point" until next non-space character
            spaces_after_point = 0
            temp_pos = 5  # Start after "Point" (5 characters)
            while temp_pos < len(header_line) and header_line[temp_pos] == ' ':
                spaces_after_point += 1
                temp_pos += 1
            
            point_width = 5 + spaces_after_point
            columns["Point"] = (0, point_width)
            pos = point_width
        
        # Process Fanout column
        if pos < len(header_line) and header_line.startswith("Fanout", pos):
            columns["Fanout"] = (pos, pos + 6)
            pos = pos + 6
        
        # Process remaining dynamic columns
        while pos < len(header_line) and header_line[pos] != '\n':
            # Count leading spaces
            leading_spaces = 0
            while pos < len(header_line) and header_line[pos] == ' ':
                leading_spaces += 1
                pos += 1
            
            if pos >= len(header_line):
                break
            
            # Extract column name
            name_start = pos
            column_name = ""
            while pos < len(header_line) and header_line[pos] != ' ' and header_line[pos] != '\n':
                column_name += header_line[pos]
                pos += 1
            
            if column_name:
                column_width = leading_spaces + len(column_name)
                
                # Handle duplicate column names with simple numbering
                if column_name in column_count:
                    column_count[column_name] += 1
                    final_column_name = f"{column_name}_{column_count[column_name]}"
                else:
                    column_count[column_name] = 0  # First occurrence keeps original name
                    final_column_name = column_name
                
                columns[final_column_name] = (name_start - leading_spaces, name_start - leading_spaces + column_width)
        
        return columns

    def extract_design_context(startpoint):
        """Extract design block from startpoint hierarchy"""
        if not startpoint:
            return "unknown"
        parts = startpoint.split('/')
        for part in parts:
            if part.startswith('u0_'):
                return part
        return parts[2] if len(parts) > 2 else "unknown"

    def extract_clock_domain(block):
        """Extract clock domain from block content - removes trailing punctuation"""
        clock_match = re.search(r"clocked by (\S+)", block, re.I)
        if clock_match:
            clock_name = clock_match.group(1)
            # Remove any trailing punctuation: ) ] } > , . ; : etc.
            clock_name = re.sub(r'[^\w]+$', '', clock_name)
            return clock_name
        return "unknown"

    def extract_slack_value(rows):
        """Extract slack value from rows"""
        for row in reversed(rows):  # Slack is usually at the end
            if row.get("Point", "").startswith("slack"):
                return row.get("Path", "").strip() or row.get("Value_1", "").strip()
        return None

    def parse_value(value_str):
        """Parse a string value to appropriate type (int, float, or string)"""
        if not value_str or value_str == "":
            return None
        try:
            return int(value_str)
        except ValueError:
            try:
                return float(value_str)
            except ValueError:
                return value_str

    # Split text into blocks using startpoint regex pattern
    blocks = re.split(rf"(?={start_regex})", text)
    blocks = [b.strip() for b in blocks if b.strip()]

    path_counter = 1

    for block in blocks:
        # Extract startpoint using regex
        start_match = re.search(rf"(?:{start_regex}):\s+(.*)", block, re.I)
        startpoint = start_match.group(1).strip() if start_match else None

        # Extract endpoint using regex
        end_match = re.search(rf"(?:{end_regex}):\s+(.*)", block, re.I)
        endpoint = end_match.group(1).strip() if end_match else None

        # Extract path_group using regex
        pg_match = re.search(rf"(?:{path_group}):\s+(.*)", block, re.I)
        pg = pg_match.group(1).strip() if pg_match else None

        # Extract path_type using regex
        pt_match = re.search(rf"(?:{path_type}):\s+(.*)", block, re.I)
        pt = pt_match.group(1).strip() if pt_match else None

        # Extract sigma using regex
        sigma_match = re.search(rf"(?:{sigma}):\s+(.*)", block, re.I)
        sigma_value = parse_value(sigma_match.group(1).strip()) if sigma_match else None

        # Extract clock information
        clock_start_match = re.search(r"clocked by (\S+)", block.split('\n')[0] if '\n' in block else block, re.I)
        clock_start = clock_start_match.group(1).strip() if clock_start_match else None
        if clock_start:
            clock_start = re.sub(r'[^\w]+$', '', clock_start)  # Remove trailing punctuation

        # Find header line in this block and parse columns
        header_line = None
        for line in block.splitlines():
            if line.strip().startswith("Point"):
                header_line = line
                break
        
        if header_line:
            columns = parse_header(header_line)
            print(f"Detected columns for block: {columns.keys()}")
        else:
            # Fallback to fixed columns if header not found in this block
            columns = {
                "Point": (0, 32),
                "Fanout": (32, 40),
                "Cap": (40, 48),
                "DTrans": (48, 56),
                "Trans": (56, 64),
                "Derate": (64, 72),
                "Delta": (72, 80),
                "Incr_Value": (124, 132),
                "Path_Value": (154, 161),
            }
            print("Header not found in block, using fallback columns")

        # Initialize rows list to store only data rows (from clock to slack)
        rows = []
        found_clock_line = False
        found_slack = False
        clock_data_rows = []
        
        for line in block.splitlines():
            line_stripped = line.strip()
            
            # Skip header lines, separator lines, and metadata lines
            if (line_stripped.startswith("Point") or 
                line_stripped.startswith("---") or 
                re.match(rf"(?:{start_regex}):", line_stripped, re.I) or
                re.match(rf"(?:{end_regex}):", line_stripped, re.I) or
                re.match(rf"(?:{path_group}):", line_stripped, re.I) or
                re.match(rf"(?:{path_type}):", line_stripped, re.I) or
                re.match(rf"(?:{sigma}):", line_stripped, re.I) or
                not line_stripped):
                continue
            
            # Check for clock lines
            clock_match = re.match(r"^clock\s+(\S+)", line_stripped, re.I)
            if clock_match:
                found_clock_line = True
                clock_name = clock_match.group(1)
                clock_data_rows.append({"type": "clock", "name": clock_name, "line": line})
                continue
            
            # Check for clock network delay lines
            clock_network_match = re.match(r"^clock_network_delay", line_stripped, re.I)
            if clock_network_match:
                clock_data_rows.append({"type": "network_delay", "line": line})
                continue
            
            # Only parse data rows after finding the first clock line and before finding slack
            if found_clock_line and not found_slack:
                # Skip rows that start with "-" (separator lines)
                if line_stripped.startswith("-"):
                    continue
                
                # Parse data row using column positions
                parsed_row = {}
                for name, (start, end) in columns.items():
                    if start < len(line):
                        end_pos = min(end, len(line))
                        parsed_row[name] = line[start:end_pos].strip()
                    else:
                        parsed_row[name] = ""
                
                # Add row to rows list
                rows.append(parsed_row)
                
                # Check if we've reached "slack" row
                if parsed_row.get("Point", "").startswith("slack"):
                    found_slack = True
                    break
        
        # Extract slack value
        slack_value = extract_slack_value(rows)
        
        # Create path entry in database if startpoint and endpoint found
        if startpoint and endpoint:
            design_block = extract_design_context(startpoint)
            clock_domain = extract_clock_domain(block)
            
            # Insert into timing_paths table
            cursor.execute('''
            INSERT INTO timing_paths 
            (session_id, report_timestamp, startpoint, endpoint, clock_start, clock_end, 
            path_group, path_type, sigma, slack, status, design_block, clock_domain)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (session_id, report_timestamp, startpoint, endpoint, clock_start, None,
                pg, pt, sigma_value, parse_value(slack_value), 
                "VIOLATED" if slack_value and parse_value(slack_value) and parse_value(slack_value) < 0 else "MET",
                design_block, clock_domain))
            
            path_id = cursor.lastrowid
            
            # Process clock data
            for clock_row in clock_data_rows:
                if clock_row["type"] == "clock":
                    # Parse clock line using column positions
                    parsed_clock = {}
                    for name, (start, end) in columns.items():
                        if start < len(clock_row["line"]):
                            end_pos = min(end, len(clock_row["line"]))
                            parsed_clock[name] = clock_row["line"][start:end_pos].strip()
                        else:
                            parsed_clock[name] = ""
                    
                    cursor.execute('''
                    INSERT INTO clock_data 
                    (path_id, clock_name, trans, mean_delay, corner_delay, value_delay, clock_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (path_id, clock_row["name"], 
                        parse_value(parsed_clock.get("Trans")),
                        parse_value(parsed_clock.get("Mean")),
                        parse_value(parsed_clock.get("Corner")),
                        parse_value(parsed_clock.get("Value")),
                        "start" if clock_row == clock_data_rows[0] else "end"))
            
            # Process path points
            for row in rows:
                if not row.get("Point") or row["Point"].startswith("slack"):
                    continue
                    
                # Determine edge type (r/f)
                edge_type = None
                point_name = row.get("Point", "")
                if point_name.endswith('r'):
                    edge_type = 'r'
                elif point_name.endswith('f'):
                    edge_type = 'f'
                
                cursor.execute('''
                INSERT INTO path_points 
                (path_id, point_name, fanout, capacitance, dtrans, trans, derate, delta,
                mean_incr, sensit_incr, corner_incr, value_incr,
                mean_path, sensit_path, value_path, edge_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (path_id, 
                    row.get("Point"), 
                    parse_value(row.get("Fanout")),
                    parse_value(row.get("Cap")),
                    parse_value(row.get("DTrans")),
                    parse_value(row.get("Trans")),
                    parse_value(row.get("Derate")),
                    parse_value(row.get("Delta")),
                    parse_value(row.get("Mean")),
                    parse_value(row.get("Sensit")),
                    parse_value(row.get("Corner")),
                    parse_value(row.get("Value")),
                    parse_value(row.get("Mean_1")),
                    parse_value(row.get("Sensit_1")),
                    parse_value(row.get("Value_1")),
                    edge_type))
            
            path_counter += 1
            conn.commit()

    print(f"Successfully parsed {path_counter-1} paths into SQLite database")
    print(f"Session ID: {session_id}")
    # Close database connection
    conn.close()

    return session_id 


if __name__ == "__main__":
    parse_timing_report()