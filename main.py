import asyncio
from functools import partial

from stt import transcribe_segment
from vad import VADAudio


async def main():
    vad = VADAudio(device=11)
    vad.start_stream()
    print("VAD running — press Ctrl+C to stop")

    try:
        async for segment in async_vad_segments(vad):
            # Schedule transcription without blocking
            asyncio.create_task(handle_segment(segment))
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        vad.stop_stream()


async def async_vad_segments(vad):
    """Async generator wrapping the blocking vad_collected_segments()"""
    while vad.running.is_set():
        loop = asyncio.get_running_loop()
        # Run next_segment in executor to avoid blocking event loop
        segment = await loop.run_in_executor(None, partial(next_segment, vad))
        if segment:  # Only yield real segments
            yield segment


def next_segment(vad):
    """Helper to get next segment from blocking generator"""
    try:
        return next(vad.vad_collected_segments())
    except StopIteration:
        return None


async def handle_segment(segment: bytes):
    loop = asyncio.get_running_loop()
    # Run transcribe_segment in executor to avoid blocking
    text = await loop.run_in_executor(None, partial(transcribe_segment, segment))
    print(text)


if __name__ == "__main__":
    asyncio.run(main())
