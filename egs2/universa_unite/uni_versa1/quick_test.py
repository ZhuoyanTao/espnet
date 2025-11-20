import numpy as np
import torch
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional
from espnet2.bin.universa_inference import UniversaInference


class AudioProcessor:
    """Audio processing utility for ESPnet Universa inference."""
    
    def __init__(self, model_name: str = "espnet/arecho_base_v0", device: str = "cuda"):
        """
        Initialize the audio processor.
        
        Args:
            model_name: Pre-trained model name
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.target_sr = 16000
        self.device = device
        self.universa_inference = UniversaInference.from_pretrained(
            model_name, device=device
        )
    
    def preprocess_audio(self, audio_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess audio file for inference.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (audio_tensor, audio_lengths)
            
        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValueError: If audio file is invalid
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            # Load audio file
            audio, orig_sr = sf.read(str(audio_path))
            
            # Handle stereo audio by taking first channel
            if audio.ndim > 1:
                audio = audio[:, 0]
            
            # Resample if necessary
            if orig_sr != self.target_sr:
                audio = librosa.resample(
                    audio, 
                    orig_sr=orig_sr, 
                    target_sr=self.target_sr
                )
            
            # Convert to float32 and normalize
            audio = audio.astype(np.float32)
            
            # Convert to torch tensor
            audio_tensor = torch.from_numpy(audio).unsqueeze(0)
            audio_lengths = torch.tensor([audio_tensor.shape[1]])
            
            return audio_tensor, audio_lengths
            
        except Exception as e:
            raise ValueError(f"Error processing audio file {audio_path}: {e}")
    
    def create_reference_audio(self, length: int = 8000) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create reference audio tensor (zeros).
        
        Args:
            length: Length of reference audio
            
        Returns:
            Tuple of (ref_audio_tensor, ref_audio_lengths)
        """
        ref_audio = torch.zeros(1, length, dtype=torch.float32)
        ref_audio_lengths = torch.tensor([length])
        return ref_audio, ref_audio_lengths
    
    def inference(
        self, 
        audio_path: str, 
        ref_audio_length: int = 8000,
        verbose: bool = True
    ) -> dict:
        """
        Run inference on audio file.
        
        Args:
            audio_path: Path to input audio file
            ref_audio_length: Length of reference audio
            verbose: Whether to print results
            
        Returns:
            Inference results dictionary
        """
        try:
            # Preprocess input audio
            audio, audio_lengths = self.preprocess_audio(audio_path)
            
            # Create reference audio
            ref_audio, ref_audio_lengths = self.create_reference_audio(ref_audio_length)
            
            # Move tensors to device if using CUDA
            if self.device == "cuda" and torch.cuda.is_available():
                audio = audio.cuda()
                audio_lengths = audio_lengths.cuda()
                ref_audio = ref_audio.cuda()
                ref_audio_lengths = ref_audio_lengths.cuda()
            
            # Run inference
            result = self.universa_inference(
                audio.float(),
                audio_lengths,
                ref_audio=ref_audio.float(),
                ref_audio_lengths=ref_audio_lengths
            )
            
            if verbose:
                print("Inference completed successfully!")
                print(f"Result: {result}")
            
            return result
            
        except Exception as e:
            print(f"Error during inference: {e}")
            raise


def main():
    """Main function to run audio processing."""
    # Initialize processor
    processor = AudioProcessor(device="cuda")
    
    # Define audio file path
    audio_file = "versa_demo_egs/examples/normal_speech/codec/encodec/1.wav"
    
    # Check if file exists
    if not Path(audio_file).exists():
        print(f"Warning: Audio file {audio_file} not found!")
        print("Please update the path to your audio file.")
        return
    
    # Run inference
    try:
        result = processor.inference(audio_file)
        return result
    except Exception as e:
        print(f"Failed to process audio: {e}")
        return None


if __name__ == "__main__":
    main()
