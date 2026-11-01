"""Echo voice engines (Phase 5) — ears (STT), endpointing (VAD), wake word.

Every engine here imports WITHOUT the optional voice extras installed
(faster-whisper, torch, silero-vad, openwakeword): heavy libraries are
lazy-imported inside the engines (docs/phase5-contract.md, FROZEN) and
tests inject fakes through the constructor seams.

NOTE: the voice pipeline and TTS live in their own modules —
``app.voice.pipeline`` and ``app.voice.tts`` — and are imported from
there, never from this package init, so importing ``app.voice`` stays
cheap and dependency-free.
"""

from app.voice.stt import WhisperSTT, pcm16_to_float32
from app.voice.vad import EnergyVAD, SileroVAD, wait_for_silence
from app.voice.wake import WakeWordEngine, normalize_wake_word

__all__ = [
    "EnergyVAD",
    "SileroVAD",
    "WakeWordEngine",
    "WhisperSTT",
    "normalize_wake_word",
    "pcm16_to_float32",
    "wait_for_silence",
]
