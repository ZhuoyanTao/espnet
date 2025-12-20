import argparse
import re
from pathlib import Path

def extract_metrics(log_file, output_file=None, compress=False):
    """
    Extract metrics from log lines containing "INFO: best hypo:" and maintain their original order.
    
    Args:
        log_file (str): Path to input log file
        output_file (str, optional): Path to output file. If None, print to console.
        compress (bool): If True, output metrics as a single space-separated line
    """
    # List to store all metric occurrences with their positions
    metric_positions = {}
    occurrences = 0
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                if 'INFO: best hypo:' in line:
                    occurrences += 1
                    # Extract content after "INFO: best hypo:"
                    match = re.search(r'INFO: best hypo:(.*)', line)
                    if match:
                        content = match.group(1).strip()
                        tokens = content.split()
                        
                        # Track positions of meta_label metrics
                        for position, token in enumerate(tokens):
                            if '@meta_label' in token:
                                if token not in metric_positions:
                                    metric_positions[token] = []
                                metric_positions[token].append(position)
        
        # Calculate average positions
        avg_positions = {}
        for metric, positions in metric_positions.items():
            avg_positions[metric] = sum(positions) / len(positions)
        
        # Sort metrics by their average position to maintain original sequence
        ordered_metrics = sorted(avg_positions.keys(), key=lambda x: avg_positions[x])
        
        # Generate statistics
        stats = []
        for metric in ordered_metrics:
            avg_pos = avg_positions[metric]
            count = len(metric_positions[metric])
            stats.append(f"{metric}: avg_pos={avg_pos:.2f}, count={count}/{occurrences}")
        
        # Prepare output
        if compress:
            output_text = " ".join(ordered_metrics)
        else:
            output_text = "\n".join(ordered_metrics)
            
            # Add statistics at the end if not compressed
            if not compress:
                output_text += "\n\n# Statistics (avg position in sequence, occurrence count)\n"
                output_text += "\n".join(stats)
        
        # Write to file or print to console
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(output_text)
            print(f"Results written to {output_file}")
        else:
            print(output_text)
        
        return ordered_metrics
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return []

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Extract metrics from log files.')
    parser.add_argument('log_file', help='Path to the log file')
    parser.add_argument('-o', '--output', help='Output file path (optional)')
    parser.add_argument('--compress', action='store_true', 
                        help='Output metrics as a single space-separated line')
    parser.add_argument('--stats', action='store_true',
                        help='Include statistics about metric positions')
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.log_file).exists():
        print(f"Error: File not found: {args.log_file}")
        return
    
    # Process the file
    extract_metrics(args.log_file, args.output, args.compress)

if __name__ == "__main__":
    main()
