# STT Module

A high-performance, real-time **Speech-to-Text (STT)** engine designed for Linux. This module combines **WebRTC VAD** for voice activity detection and **Faster-Whisper** for near-instant transcription, optimized specifically for NVIDIA Blackwell (RTX 50-series) hardware.

---

## ✨ Features

- **Low Latency:** Uses `faster-whisper` (CTranslate2) for high-speed inference.
- **Robust VAD:** Integrated Google WebRTC Voice Activity Detection to segment speech and ignore background noise.
- **Linux Optimized:** Designed to work with ALSA, PulseAudio, and PipeWire at a native 16kHz sample rate.
- **Blackwell Ready:** Pre-configured for RTX 5070+ GPUs with CUDA 12/13 compatibility fixes.

---

## 🛠️ Architecture

The module operates in three distinct stages:

1. **Capture:** Streams mono audio from your hardware using `sounddevice`.
2. **Detection:** WebRTC VAD monitors 30ms frames. It buffers voiced segments until a silence threshold is reached.
3. **Transcription:** The collected audio segment is converted to `float16` and processed by the Whisper model on the GPU.

---

## 🚀 Installation

### Recommended

```
git clone https://github.com/PhoeniX5971/STT-Module.git

cd STT-Module

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

python main.py
```

Assuming you will be using:

1. Git CLI
2. Bash
3. Basic python virtual environment

> Below is the Manual Instructions, not recommended.

### 1. System Requirements

- **OS:** Linux
- **Hardware:** NVIDIA GPU (RTX 5070/Blackwell tested)
- **Audio:** ALSA, PulseAudio, or PipeWire

### 2. Dependencies

Install the core Python packages:

```bash
pip install faster-whisper webrtcvad sounddevice numpy scipy

```

### 3. GPU Setup (Blackwell / CUDA 13 Fix)

Since `faster-whisper` looks for CUDA 12 libraries, you must install the redistributable wheels to run on an RTX 50-series card without breaking your CUDA 13 system drivers:

```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12

```

---

## 📦 Usage

### Basic Implementation

```python
from VAD import VADAudio
from stt import transcribe_segment

# Initialize VAD on your specific device (e.g., index 11)
vad = VADAudio(device=<device-index>, vad_aggressiveness=3)
vad.start_stream()

try:
    for segment_bytes in vad.vad_collected_segments():
        # Transcribe the detected speech segment
        text = transcribe_segment(segment_bytes)
        print(f"Transcription: {text}")
finally:
    vad.stop_stream()

```

---

## 🔧 Configuration

In `VAD.py`, you can fine-tune the following parameters:

| Variable             | Default   | Description                                               |
| -------------------- | --------- | --------------------------------------------------------- |
| `SAMPLE_RATE`        | 16000     | Native recording rate. Matches Whisper requirements.      |
| `VAD_AGGRESSIVENESS` | 3         | 0-3 scale. 3 is the most strict at filtering noise.       |
| `max_silence_frames` | 50 (1.5s) | How long to wait after speech before finishing a segment. |

---

Contributions are welcome!
