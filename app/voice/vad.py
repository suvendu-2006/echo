"""Voice activity detection (PLAN Phase 5.2) — endpointing without timers.

Contract (docs/phase5-contract.md, FROZEN): ``is_speech(pcm16,
sample_rate) -> bool`` plus the :func:`wait_for_silence` endpointing
helper. Speech ENDS on ~600 ms of silence — never a fixed speech timer
(plan 5.2).

Two backends behind one seam:

- :class:`EnergyVAD` — pure-stdlib RMS over frames; always available, the
  floor every machine gets.
- :class:`SileroVAD` — the real neural VAD, lazy-imported; **when torch or
  the silero-vad wheel is missing it logs a warning and DELEGATES to an
  internal EnergyVAD** — voice must never hard-fail for missing wheels
  (contract law). Tests inject a ``frame_scorer`` callable instead.

Both classes (and the module-level helper) expose ``wait_for_silence``,
which consumes an async iterator of PCM chunks and returns the accumulated
audio once ``silence_ms`` of CONTINUOUS non-speech has been consumed (or
the stream ends). Silence is measured in AUDIO TIME — samples/sample_rate
— so endpointing is deterministic offline: no wall-clock sleeps anywhere.
"""

from __future__ import annotations

import functools
import logging
import math
import struct
from collections.abc import AsyncIterator
from typing import Any, Callable

LOGGER = logging.getLogger("echo.voice")

#: Default endpointing (plan 5.2): speech is over after this much quiet.
DEFAULT_SILENCE_MS = 600

#: A speech check: ``chunk -> bool`` (the test seam for wait_for_silence).
SpeechCheck = Callable[[bytes], bool]


def _iter_frames(pcm16: bytes, frame_samples: int) -> Any:
    """Yield fixed-size byte frames; a short tail frame is yielded too.

    Truncated single trailing bytes are dropped (a 16-bit sample is 2
    bytes); a final partial frame is still analysed — natural speech ends
    mid-frame all the time.
    """
    step = max(2, frame_samples * 2)  # bytes per frame, at least one sample
    for start in range(0, len(pcm16) - step + 1, step):
        yield pcm16[start : start + step]
    tail = len(pcm16) % step
    if tail >= 2:
        yield pcm16[-tail:]


def _rms(pcm16: bytes) -> float:
    """RMS amplitude of a 16-bit LE PCM frame, in raw sample units."""
    usable = len(pcm16) - (len(pcm16) % 2)
    if usable <= 0:
        return 0.0
    count = usable // 2
    samples = struct.unpack("<%dh" % count, pcm16[:usable])
    return math.sqrt(math.fsum(sample * sample for sample in samples) / count)


class _EndpointingMixin:
    """Shared ``wait_for_silence`` seam; subclasses provide ``is_speech``."""

    def is_speech(self, pcm16: bytes, sample_rate: int = 16000) -> bool:
        raise NotImplementedError

    async def wait_for_silence(
        self,
        chunks: AsyncIterator[bytes],
        sample_rate: int = 16000,
        silence_ms: int = DEFAULT_SILENCE_MS,
        check: SpeechCheck | None = None,
    ) -> bytes:
        """Method twin of the module-level :func:`wait_for_silence`.

        The only difference: with no ``check`` given, THIS VAD's
        ``is_speech`` is the check (the module function defaults to a fresh
        EnergyVAD instead).
        """
        if check is None:
            check = functools.partial(self.is_speech, sample_rate=sample_rate)
        return await wait_for_silence(
            chunks, sample_rate=sample_rate, silence_ms=silence_ms, check=check
        )


class EnergyVAD(_EndpointingMixin):
    """RMS-threshold VAD — pure stdlib, always available (the floor backend).

    ``rms_threshold`` is in raw 16-bit sample units (a quiet room sits well
    under 100; normal speech peaks in the thousands — 300 is the default).
    ``frame_ms`` is the analysis frame. Speech = ANY frame whose RMS is
    strictly above the threshold.
    """

    def __init__(self, rms_threshold: float = 300.0, frame_ms: int = 30) -> None:
        self.rms_threshold = float(rms_threshold)
        self.frame_ms = int(frame_ms)

    def is_speech(self, pcm16: bytes, sample_rate: int = 16000) -> bool:
        frame_samples = max(1, int(sample_rate * self.frame_ms / 1000))
        for frame in _iter_frames(pcm16, frame_samples):
            if _rms(frame) > self.rms_threshold:
                return True
        return False


def _load_silero() -> Any:
    """The single lazy-import seam for the real silero backend (tests patch it).

    Uses the small ``silero-vad`` package (weights bundled, offline-friendly)
    on top of torch. Raises ImportError when either wheel is missing — the
    CALLER decides to fall back; this function only reports.
    """
    import torch  # noqa: F401 — the heavy wheel; proves it before loading weights
    from silero_vad import load_silero_vad

    return load_silero_vad()


class SileroVAD(_EndpointingMixin):
    """The real (neural) VAD — with a guaranteed stdlib fallback (5.2).

    ``model``: an already-loaded silero model (injected; called as
    ``model(tensor, sample_rate)``). ``frame_scorer``: the test seam —
    ``callable(frame_bytes) -> float`` probability; a frame is speech when
    its score is strictly above ``threshold``. When neither is given, the
    real backend is lazy-imported; **ImportError → a logged warning and an
    internal :class:`EnergyVAD` takes over** — voice never hard-fails for
    missing wheels (contract law).
    """

    def __init__(
        self,
        model: Any | None = None,
        threshold: float = 0.5,
        frame_scorer: Callable[[bytes], float] | None = None,
    ) -> None:
        self.threshold = float(threshold)
        self._model = model
        self._frame_scorer = frame_scorer
        self._fallback: EnergyVAD | None = None
        if self._frame_scorer is None and self._model is None:
            try:
                self._model = _load_silero()
            except ImportError:
                LOGGER.warning(
                    "silero-vad/torch missing — falling back to EnergyVAD "
                    "(pip install -r requirements-voice.txt for the neural VAD)"
                )
                self._fallback = EnergyVAD()

    def is_speech(self, pcm16: bytes, sample_rate: int = 16000) -> bool:
        if self._fallback is not None:
            return self._fallback.is_speech(pcm16, sample_rate)
        frame_samples = max(1, int(sample_rate * 32 / 1000))  # silero's ~32 ms frame
        for frame in _iter_frames(pcm16, frame_samples):
            if self._score_frame(frame, sample_rate) > self.threshold:
                return True
        return False

    def _score_frame(self, frame: bytes, sample_rate: int) -> float:
        """One frame → speech probability, via the injected or real backend."""
        if self._frame_scorer is not None:
            return float(self._frame_scorer(frame))
        import torch  # real path only; guaranteed present past _load_silero()

        tensor = torch.frombuffer(bytearray(frame), dtype=torch.int16)
        return float(self._model(tensor.float() / 32768.0, sample_rate))


async def wait_for_silence(
    chunks: AsyncIterator[bytes],
    sample_rate: int = 16000,
    silence_ms: int = DEFAULT_SILENCE_MS,
    check: SpeechCheck | None = None,
) -> bytes:
    """Consume an async PCM stream and return the audio up to the endpoint.

    Every chunk is accumulated. The silence clock — measured in AUDIO time
    (samples/sample_rate, exact integer math) — resets on any chunk the
    ``check`` callable calls speech and otherwise accumulates; once
    ``silence_ms`` of CONTINUOUS non-speech has been consumed the stream is
    abandoned and the accumulated bytes — INCLUDING the trailing quiet —
    are returned. If the stream ends first, everything consumed is
    returned: this helper never hangs waiting for silence that never comes.

    ``check`` defaults to a fresh :class:`EnergyVAD`; pass any ``chunk ->
    bool`` callable in tests. A non-positive ``sample_rate`` is a caller
    bug and raises ValueError.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if check is None:
        check = functools.partial(EnergyVAD().is_speech, sample_rate=sample_rate)
    # Exact endpoint math in samples: silence_ms * sample_rate / 1000.
    endpoint_samples = max(0, int(silence_ms)) * sample_rate // 1000
    buffer = bytearray()
    silent_samples = 0
    async for chunk in chunks:
        if chunk:
            buffer += chunk
            if check(chunk):
                silent_samples = 0
            else:
                silent_samples += len(chunk) // 2
        if silent_samples >= endpoint_samples:
            break
    return bytes(buffer)
