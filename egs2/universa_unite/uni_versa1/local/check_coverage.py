import json
import re
from collections import Counter

def analyze_field_coverage(file_path):
    # Initialize counters
    total_lines = 0
    field_counter = Counter()
    all_fields = set()
    
    # Regular expression to extract the JSON part
    pattern = r'^.+?\s+(\{.+\})$'
    
    # Process the file
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
                
            total_lines += 1
            
            # Extract the JSON part
            match = re.match(pattern, line)
            if match:
                try:
                    json_str = match.group(1)
                    data = json.loads(json_str)
                    
                    # Count fields
                    for field in data.keys():
                        field_counter[field] += 1
                        all_fields.add(field)
                        
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse JSON in line: {line}")
            else:
                print(f"Warning: Line format not recognized: {line}")
    
    # Calculate coverage percentages
    if total_lines > 0:
        coverage = {}
        for field in all_fields:
            coverage[field] = (field_counter[field] / total_lines) * 100
            
        # Sort by coverage percentage (descending)
        sorted_coverage = {k: v for k, v in sorted(coverage.items(), key=lambda item: item[1], reverse=True)}
        
        return {
            "total_lines": total_lines,
            "field_coverage": sorted_coverage
        }
    else:
        return {
            "total_lines": 0,
            "field_coverage": {}
        }

# Example usage
if __name__ == "__main__":
    file_path = "dump/raw/overall_base/metric.scp"  # Change this to your file path
    result = analyze_field_coverage(file_path)
    
    print(f"Total number of entries: {result['total_lines']}")
    print("\nField coverage percentages:")
    for field, coverage in result['field_coverage'].items():
        print(f"{field}: {coverage:.2f}%")
