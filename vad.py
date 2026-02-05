"""
VAD engine that captures microphone audio via sounddevice, uses webrtcvad
to detect speech, and yields speech segments as PCM16 bytes (16kHz mono).

Improvements:
- Captures at 16kHz natively to avoid resampling bugs.
- Fixed CFFI buffer error in audio callback.
- Enforced strict frame-length for webrtcvad compatibility.
"""

import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
import webrtcvad

# ---- Configuration defaults ----
# We use 16000Hz natively. Most Linux sound systems (PipeWire/Pulse) 
# will handle the hardware conversion from 48k to 16k more reliably than Python.
SAMPLE_RATE = 16000
VAD_SAMPLE_RATE = 16000  
VAD_AGGRESSIVENESS = 3       # 0..3 (3 is most restrictive)
FRAME_DURATION_MS = 30       # webrtcvad supports 10, 20, or 30ms
PADDING_DURATION_MS = 300    # pre/post speech padding (unused in simple loop below)
MAX_QUEUE_SIZE = 100         # Prevent memory leaks if processing falls behind

@dataclass
class Frame:
    """Represents a single window of audio data."""
    bytes: bytes
    timestamp: float
    duration: float

class VADAudio:
    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        frame_duration_ms: int = FRAME_DURATION_MS,
        vad_aggressiveness: int = VAD_AGGRESSIVENESS,
        device: Optional[int | str] = None,
    ):
        self.sample_rate = sample_rate
        self.vad_sample_rate = sample_rate # Kept identical to simplify logic
        self.frame_ms = frame_duration_ms
        
        # webrtcvad MUST have exactly this many samples per call
        self.frame_samples = int(sample_rate * frame_duration_ms / 1000)
        self.expected_bytes = self.frame_samples * 2 # 2 bytes per sample (int16)
        
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        self.audio_queue: "queue.Queue[bytes]" = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        self.running = threading.Event()
        self.device = device
        self._stream = None

    def audio_callback(self, indata, frames, time_info, status):
        """Standard sounddevice callback."""
        if status:
            print(f"ALSA/Pulse Status: {status}", file=sys.stderr)
        try:
            # FIX: Convert CFFI buffer to bytes immediately to avoid attribute errors
            self.audio_queue.put_nowait(bytes(indata))
        except queue.Full:
            pass 

    def start_stream(self):
        """Initializes and starts the raw audio input stream."""
        if self.device is None:
            # Fallback to system default if no index provided
            self.device = sd.default.device[0]
            
        self._stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=self.frame_samples, # Capture in VAD-sized chunks
            device=self.device,
            dtype="int16",
            channels=1,
            callback=self.audio_callback,
        )
        self._stream.start()
        self.running.set()
        print(f"Audio stream started on device {self.device} at {self.sample_rate}Hz")

    def stop_stream(self):
        """Stops the stream and clears the buffer."""
        self.running.clear()
        if self._stream:
            self._stream.stop()
            self._stream.close()
        # Flush the queue
        while not self.audio_queue.empty():
            self.audio_queue.get_nowait()
        print("Audio stream stopped")

    def _yield_frames(self) -> Iterator[Frame]:
        """Internal generator to split stream data into exact VAD-sized frames."""
        while self.running.is_set():
            try:
                data = self.audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            # Ensure we only process chunks of the exact required byte length
            offset = 0
            while offset + self.expected_bytes <= len(data):
                chunk = data[offset : offset + self.expected_bytes]
                yield Frame(
                    bytes=chunk, 
                    timestamp=time.time(), 
                    duration=self.frame_ms / 1000.0
                )
                offset += self.expected_bytes

    def vad_collected_segments(self) -> Iterator[bytes]:
        """
        Main logic: Collects voiced frames and yields them as a single segment 
        once a period of silence is detected.
        """
        triggered = False
        voiced_frames = []
        
        # Max silence before we cut the segment (e.g., 1.5 seconds)
        max_silence_frames = int(1500 / self.frame_ms) 
        silence_counter = 0

        for frame in self._yield_frames():
            # Double check length (Safety Guard for webrtcvad)
            if len(frame.bytes) != self.expected_bytes:
                continue

            is_speech = self.vad.is_speech(frame.bytes, self.vad_sample_rate)

            if not triggered:
                if is_speech:
                    triggered = True
                    voiced_frames.append(frame.bytes)
            else:
                voiced_frames.append(frame.bytes)
                if not is_speech:
                    silence_counter += 1
                else:
                    silence_counter = 0

                # If we've hit the silence threshold, yield the whole block
                if silence_counter >= max_silence_frames:
                    yield b"".join(voiced_frames)
                    # Reset state
                    triggered = False
                    voiced_frames = []
                    silence_counter = 0

def save_pcm16_to_wav(pcm_bytes: bytes, sample_rate: int, filename: str):
    """Utility to save segments to disk for debugging."""
    arr = np.frombuffer(pcm_bytes, dtype=np.int16)
    os.makedirs("data", exist_ok=True)
    sf.write(os.path.join("data", filename), arr, sample_rate, subtype='PCM_16')

def main():
    # 1. Initialize VAD (Device 11 is your SteelSeries Arctis 7)
    # If it fails, try device=None to use system default
    vad = VADAudio(device=11, vad_aggressiveness=3)
    
    print("--- VAD CLI TESTER ---")
    print("Recording at 16kHz. Speak into your mic...")
    print("Press Ctrl+C to stop.\n")

    try:
        vad.start_stream()
        clip_count = 0
        
        # 2. Listen for complete speech segments
        # vad_collected_segments() yields when silence is detected
        for i, segment_bytes in enumerate(vad.vad_collected_segments()):
            clip_count += 1
            filename = f"test_clip_{clip_count}.wav"
            
            print(f"DEBUG: Detected Segment {clip_count} ({len(segment_bytes)} bytes)")
            
            # 3. Save to disk to verify it sounds correct (not 3x slow/fast)
            save_pcm16_to_wav(segment_bytes, 16000, filename)
            print(f"SAVED: data/{filename}")
            print("Listening for next segment...")

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        vad.stop_stream()

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    if not os.path.exists("data"):
        os.makedirs("data")
    main()
