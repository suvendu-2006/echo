"""Speech-to-text engine (PLAN Phase 5.1) — faster-whisper behind a narrow seam.

Contract (docs/phase5-contract.md, FROZEN): ``WhisperSTT.transcribe(pcm16,
sample_rate) -> str`` for TURN-level STT — one call per finished utterance,
no partial results. The audio arrives as mono 16-bit little-endian PCM; the
class converts it to floats in [-1, 1] with the stdlib ``struct`` module so
THIS module imports without numpy — only the default lazy path touches
numpy, and faster-whisper pulls that in itself.

Heavy dependencies are OPTIONAL and never imported at module level: the
default path lazy-imports faster-whisper inside :meth:`WhisperSTT._load`
(cached — the model is loaded once) and raises ``RuntimeError`` carrying
the "pip install -r requirements-voice.txt" hint when it is missing. Tests
and the pipeline inject a ``transcriber`` callable instead, so the whole
voice suite runs offline with fakes.

Errors from the real model PROPAGATE out of :meth:`WhisperSTT.transcribe` —
the voice pipeline catches them and turns them into generic ``error``
events; empty or truncated audio never raises (it yields ``""``).
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import struct
from typing import Any, Awaitable, Callable, Union

LOGGER = logging.getLogger("echo.voice")

#: The one install hint every lazy-import failure in stt.py must carry.
VOICE_EXTRAS_HINT = "voice extras missing — pip install -r requirements-voice.txt"

#: The injected seam: ``transcriber(audio_float32, sample_rate) -> str``.
#: Async per the contract; a plain sync callable is tolerated too.
Transcriber = Callable[[Any, int], Union[str, Awaitable[str]]]


def pcm16_to_float32(pcm16: bytes) -> list[float]:
    """Convert mono 16-bit little-endian PCM to floats in [-1, 1] (stdlib only).

    ``struct`` with an explicit ``<`` format guarantees little-endian
    decoding on every platform (DECISIONS D-009: Windows today, macOS
    later). Truncated trailing bytes (odd lengths from chunked transports)
    are dropped, never a crash. ``0x0001`` maps to 1/32768 (tiny positive),
    ``-32768`` maps to exactly ``-1.0``.
    """
    usable = len(pcm16) - (len(pcm16) % 2)
    if usable <= 0:
        return []
    count = usable // 2
    samples = struct.unpack("<%dh" % count, pcm16[:usable])
    scale = 1.0 / 32768.0
    return [sample * scale for sample in samples]


class WhisperSTT:
    """Turn-level speech-to-text: PCM16 bytes in, plain text out (5.1).

    ``model_size`` names the faster-whisper model for the default path
    ("base" — the $0 local default). ``transcriber`` is the offline seam:
    an injected ``callable(audio_float32, sample_rate) -> str`` (async per
    the contract; sync tolerated). When it is None, the default
    faster-whisper path is used — lazy-imported and cached so the model
    loads at most once per engine.
    """

    def __init__(
        self, model_size: str = "base", transcriber: Transcriber | None = None
    ) -> None:
        self.model_size = model_size
        self._transcriber = transcriber
        self._model: Any | None = None  # the cached faster-whisper WhisperModel

    async def transcribe(self, pcm16: bytes, sample_rate: int = 16000) -> str:
        """Transcribe one finished utterance of mono 16-bit LE PCM.

        Empty or truncated audio returns ``""`` without ever touching the
        model (silence transcribes to ``""`` naturally on the default
        path). Injected-transcriber results are stripped; errors from the
        real model propagate — the pipeline catches them.
        """
        if not pcm16 or len(pcm16) < 2:
            return ""
        audio = pcm16_to_float32(pcm16)
        if not audio:
            return ""
        if self._transcriber is not None:
            result = self._transcriber(audio, sample_rate)
            if inspect.isawaitable(result):
                result = await result
            return "" if result is None else str(result).strip()
        # Default path: run the blocking model off the event loop so a long
        # transcription never stalls the WS streams (the Phase 2 lesson).
        return await asyncio.to_thread(self._transcribe_blocking, audio)

    def _load(self) -> Any:
        """Lazy, cached faster-whisper load — the ONLY place it is imported."""
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError(VOICE_EXTRAS_HINT) from exc
            self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
            LOGGER.info("loaded faster-whisper model %r", self.model_size)
        return self._model

    def _transcribe_blocking(self, audio: list[float]) -> str:
        """The real model call (runs in a worker thread via asyncio.to_thread).

        faster-whisper takes the waveform straight from memory
        (from_buffer-style): numpy array in, segment generator out. Audio
        must already be 16 kHz — the browser mic downsamples per the
        contract before the bytes ever get here.
        """
        model = self._load()  # first: missing extras must raise the friendly hint
        import numpy as np  # real path only; faster-whisper pulls numpy itself

        segments, _info = model.transcribe(
            np.asarray(audio, dtype=np.float32), language=None
        )
        return "".join(segment.text for segment in segments).strip()
