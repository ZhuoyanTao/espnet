import json
import numpy as np
from typing import Dict, List, Tuple, Any
import math

class MetricTokenizer:
    def __init__(self, tokenizer_config: Dict):
        """
        Initialize the metric tokenizer with the given configuration.
        
        Args:
            tokenizer_config: Dictionary containing 'tokenizer', 'VOCAB', and 'offset' keys
        """
        self.tokenizer_config = tokenizer_config['tokenizer']
        self.vocab = tokenizer_config['VOCAB']
        self.offset = tokenizer_config['offset']
        
        # Create reverse mapping for decoding
        self.vocab_to_idx = {token: idx for idx, token in enumerate(self.vocab)}
        
    def encode_metric(self, metric_name: str, value: float) -> int:
        """
        Encode a metric value to a token index.
        
        Args:
            metric_name: Name of the metric
            value: Numerical value to encode
            
        Returns:
            Token index
        """
        if metric_name not in self.tokenizer_config:
            raise ValueError(f"Metric {metric_name} not found in tokenizer config")
            
        # Get the discretization bins for this metric
        bins = self.tokenizer_config[metric_name]
        
        # Find the closest bin
        closest_bin = min(bins, key=lambda x: abs(x - value))
        closest_idx = bins.index(closest_bin)

        # Get the token string
        token_str = f"{metric_name}@{closest_idx}"

        
        # Return the vocabulary index
        return self.vocab_to_idx.get(token_str, -1)
    
    def decode_metric(self, metric_name: str, token_idx: int) -> float:
        """
        Decode a token index back to a metric value.
        
        Args:
            metric_name: Name of the metric
            token_idx: Token index to decode
            
        Returns:
            Reconstructed numerical value
        """
        if token_idx >= len(self.vocab):
            raise ValueError(f"Token index {token_idx} out of vocabulary range")
            
        token_str = self.vocab[token_idx]
        
        # Parse the token string to get the value
        if token_str.startswith(f"{metric_name}@"):
            value_str = token_str.split('@')[1]
            if value_str == "meta_label":
                return 0.0  # Default for meta labels
            try:
                bins = self.tokenizer_config[metric_name]
                return bins[int(value_str)]
            except ValueError:
                return 0.0
        
        return 0.0
    
    def reconstruct_metric(self, metric_name: str, value: float) -> float:
        """
        Perform full encode-decode cycle to get reconstructed value.
        
        Args:
            metric_name: Name of the metric
            value: Original value
            
        Returns:
            Reconstructed value
        """
        try:
            token_idx = self.encode_metric(metric_name, value)
            # if token_idx == -1:
            #     return value  # Return original if encoding fails
          
            reconstructed = self.decode_metric(metric_name, token_idx)
            return reconstructed
        except:
            return value  # Return original if reconstruction fails

def parse_metric_scp(file_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Parse the metric.scp file.
    
    Args:
        file_path: Path to the metric.scp file
        
    Returns:
        Dictionary mapping utterance IDs to their metrics
    """
    data = {}
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # Split on first space to separate ID from JSON
            parts = line.split(' ', 1)
            if len(parts) != 2:
                continue
                
            utt_id = parts[0]
            json_str = parts[1]
            
            try:
                metrics = json.loads(json_str)
                data[utt_id] = metrics
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON for {utt_id}: {e}")
                continue
    
    return data

def calculate_reconstruction_errors(original_data: Dict[str, Dict[str, Any]], 
                                  tokenizer: MetricTokenizer) -> Dict[str, Dict[str, float]]:
    """
    Calculate reconstruction errors for all numerical metrics.
    
    Args:
        original_data: Original metric data
        tokenizer: Configured metric tokenizer
        
    Returns:
        Dictionary containing RMSE, MAE, and MSE for each metric
    """
    # Collect all numerical metrics
    all_metrics = set()
    for utt_metrics in original_data.values():
        for key, value in utt_metrics.items():
            if isinstance(value, (int, float)):
                all_metrics.add(key)
    
    results = {}
    
    for metric_name in all_metrics:
        original_values = []
        reconstructed_values = []
        
        for utt_id, utt_metrics in original_data.items():
            if metric_name in utt_metrics:
                original_val = utt_metrics[metric_name]
                if isinstance(original_val, (int, float)):
                    # Reconstruct the value
                    reconstructed_val = tokenizer.reconstruct_metric(metric_name, original_val)
                    
                    original_values.append(original_val)
                    reconstructed_values.append(reconstructed_val)
        
        if len(original_values) > 0:
            original_array = np.array(original_values)
            reconstructed_array = np.array(reconstructed_values)
            
            # Calculate errors
            errors = original_array - reconstructed_array
            mse = np.mean(errors ** 2)
            mae = np.mean(np.abs(errors))
            rmse = np.sqrt(mse)
            
            results[metric_name] = {
                'MSE': mse,
                'MAE': mae,
                'RMSE': rmse,
                'num_samples': len(original_values),
                'original_mean': np.mean(original_array),
                'original_std': np.std(original_array),
                'reconstructed_mean': np.mean(reconstructed_array),
                'reconstructed_std': np.std(reconstructed_array)
            }
    
    return results

def print_results(results: Dict[str, Dict[str, float]]):
    """Print the reconstruction error results in a formatted table."""
    print("=" * 120)
    print(f"{'Metric Name':<25} {'RMSE':<10} {'MAE':<10} {'MSE':<10} {'Samples':<8} {'Orig Mean':<10} {'Recon Mean':<11}")
    print("=" * 120)
    
    # Sort by RMSE for better readability
    sorted_metrics = sorted(results.items(), key=lambda x: x[1]['RMSE'], reverse=True)
    
    for metric_name, stats in sorted_metrics:
        print(f"{metric_name:<25} {stats['RMSE']:<10.4f} {stats['MAE']:<10.4f} {stats['MSE']:<10.4f} "
              f"{stats['num_samples']:<8} {stats['original_mean']:<10.4f} {stats['reconstructed_mean']:<11.4f}")
    
    print("=" * 120)
    
    # Summary statistics
    all_rmse = [stats['RMSE'] for stats in results.values()]
    all_mae = [stats['MAE'] for stats in results.values()]
    all_mse = [stats['MSE'] for stats in results.values()]
    
    print(f"\nSummary Statistics:")
    print(f"Average RMSE across all metrics: {np.mean(all_rmse):.4f}")
    print(f"Average MAE across all metrics: {np.mean(all_mae):.4f}")
    print(f"Average MSE across all metrics: {np.mean(all_mse):.4f}")
    print(f"Total number of metrics evaluated: {len(results)}")

# Example usage
def main():
    import json
    tokenizer_config = json.load(open("data/token_list/metric_500_percentile_overall_scale_temp_w-numerical/tokens.json", "r")) 
    # Initialize tokenizer
    tokenizer = MetricTokenizer(tokenizer_config)
    
    # Parse metric file
    print("Parsing metric.scp file...")
    metric_data = parse_metric_scp("dump/raw/generation_test/metric.scp")  # Replace with your file path
    print(f"Loaded {len(metric_data)} utterances")
    
    # Calculate reconstruction errors
    print("Calculating reconstruction errors...")
    results = calculate_reconstruction_errors(metric_data, tokenizer)
    
    # Print results
    print_results(results)

if __name__ == "__main__":
    main()
