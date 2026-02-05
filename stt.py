import os

import numpy as np
from faster_whisper import WhisperModel

# Load once at startup
# device="cuda" will put the model into GPU/VRAM
model_path = os.path.expandvars("$HOME/models/faster-whisper/faster-whisper-medium/")
stt_model = WhisperModel(
    model_path,
    device="cuda",  # or "mps" for Apple GPU
    compute_type="float16"  # keeps it smaller and faster
)

def pcm16_to_float32(pcm_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(pcm_bytes, dtype=np.int16)
    return arr.astype(np.float32) / 32768.0

def transcribe_segment(segment: bytes):
    audio_float32 = pcm16_to_float32(segment)

    # Faster Whisper expects audio in float32 numpy
    segments, _ = stt_model.transcribe(audio_float32, beam_size=5, vad_filter=True)
    
    # Collect the text
    text = " ".join(seg.text for seg in segments)
    return text
