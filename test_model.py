
import torch
import whisper

print("Checking Deep Learning environment...")

print("PyTorch Version:", torch.__version__)

# Use CPU for reliable Whisper model loading
device = "cpu"

print("Selected Device:", device)

print("\nLoading Whisper Base Model...")

model = whisper.load_model("base", device=device)

print("\nWhisper model loaded successfully!")
print("Model Device:", device)

print("\nLINGUA-AI environment is ready!")
