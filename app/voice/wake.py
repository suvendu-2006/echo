"""Wake word engine (PLAN Phase 5.3) — "Hey Echo" plus THE ECHO GUARD.

Two of the phase's hard laws live here:

- **Never wake itself** (plan 5.3): while Echo is speaking, the pipeline
  calls :meth:`WakeWordEngine.mute` and EVERY chunk — including audio that
  literally contains the wake word (Echo's own TTS bleeding through the
  mic) — is swallowed before the detector is ever consulted. Detection
  state does not advance while muted, and :meth:`WakeWordEngine.unmute`
  reopens from a clean slate, so nothing heard during playback can leak
  into the next listening state.
- **Edge triggering**: :meth:`WakeWordEngine.process` returns True only on
  a detection edge (the detector crossing from quiet to wake) and never
  twice within :data:`REFRACTORY_S` of audio, so one utterance cannot
  double-fire.

The default detector lazy-imports openWakeWord inside
:meth:`WakeWordEngine._load` and raises ``RuntimeError`` with the
requirements hint when it is missing — unlike the VAD, a missing wake
model is a real setup gap, BUT it is the PIPELINE that catches it and
degrades voice to push-to-talk; this module raises the error, it never
crashes the app. (Plan note: "Echo" is an official Alexa wake word and
openWakeWord ships no pretrained "hey echo" — a custom-trained model goes
under that name; DECISIONS D-007 records the fallback decision.)

Word matching is case/spacing tolerant: "Hey Echo" == "hey echo" — see
:func:`normalize_wake_word`.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

LOGGER = logging.getLogger("echo.voice")

#: The hint the default detector's lazy import failure must carry.
WAKE_EXTRAS_HINT = (
    "wake word extras missing — pip install -r requirements-voice.txt "
    "(voice falls back to push-to-talk until then)"
)

#: Seconds of (unmuted) AUDIO after a detection during which new edges are
#: ignored — the anti-double-fire guard, measured in audio time so tests
#: stay deterministic (no wall clock).
REFRACTORY_S = 1.5

#: The injected seam: ``detector(pcm16, sample_rate) -> bool`` per chunk.
Detector = Callable[[bytes, int], bool]


def normalize_wake_word(word: str) -> str:
    """Case/spacing-tolerant wake word form: "Hey  Echo" == "hey echo"."""
    return " ".join(str(word).lower().split())


class WakeWordEngine:
    """Wake detection with the echo guard built in from day one (5.3).

    ``word`` — the wake phrase (normalized case/spacing-tolerant).
    ``sensitivity`` — the score the default openWakeWord path must exceed.
    ``detector`` — the offline seam: an injected ``callable(pcm16,
    sample_rate) -> bool`` consulted once per chunk; when it is None the
    default openWakeWord path is used, lazily and cached.
    """

    def __init__(
        self,
        word: str = "hey echo",
        sensitivity: float = 0.5,
        detector: Detector | None = None,
    ) -> None:
        self.word = normalize_wake_word(word)
        self.sensitivity = float(sensitivity)
        self.refractory_s = REFRACTORY_S
        self._detector = detector
        self._muted = False
        self._model: Any | None = None  # the cached openwakeword Model
        self._audio_s = 0.0  # unmuted audio consumed, seconds (exact enough)
        self._prev_detected = False
        self._last_detect_s: float | None = None

    # ------------------------------------------------- the echo guard (5.3)

    def mute(self) -> None:
        """THE ECHO GUARD: swallow every chunk until unmute() — wake word or not.

        The pipeline calls this the moment Echo starts speaking (and during
        barge-in transitions, per the contract). While muted, process()
        returns False immediately and the detector is never consulted.
        """
        self._muted = True
        self._reset_detection_state()

    def unmute(self) -> None:
        """Reopen wake detection from a clean slate (no stale edge state)."""
        self._muted = False
        self._reset_detection_state()

    @property
    def muted(self) -> bool:
        """Whether the echo guard currently swallows everything."""
        return self._muted

    def _reset_detection_state(self) -> None:
        self._prev_detected = False
        self._last_detect_s = None

    # -------------------------------------------------------- the loop seam

    def process(self, pcm16: bytes, sample_rate: int = 16000) -> bool:
        """Feed one mic chunk; True ONLY on a wake detection edge.

        Muted → False immediately, detector never consulted, state never
        advanced — for ANY input, including audio that contains the wake
        word. Empty chunks are always False and never raise. Errors from
        the default openwakeword path propagate (the pipeline catches them
        and degrades to push-to-talk).
        """
        if self._muted:
            return False
        if not pcm16:
            return False
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        self._audio_s += (len(pcm16) // 2) / sample_rate
        detected = bool(self._detect(pcm16, sample_rate))
        fire = False
        if detected and not self._prev_detected:
            if (
                self._last_detect_s is None
                or (self._audio_s - self._last_detect_s) >= self.refractory_s
            ):
                fire = True
                self._last_detect_s = self._audio_s
        self._prev_detected = detected
        return fire

    # ----------------------------------------------------- detector backends

    def _detect(self, pcm16: bytes, sample_rate: int) -> bool:
        """One chunk → detected? via the injected or default detector."""
        if self._detector is not None:
            return bool(self._detector(pcm16, sample_rate))
        return self._default_detect(pcm16)

    def _load(self) -> Any:
        """Lazy, cached openwakeword load — the ONLY place it is imported."""
        if self._model is None:
            try:
                from openwakeword import Model
            except ImportError as exc:
                # A real setup gap (not a fallback case like the VAD): the
                # pipeline catches this and voice degrades to push-to-talk.
                raise RuntimeError(WAKE_EXTRAS_HINT) from exc
            model_name = self.word.replace(" ", "_")  # "hey echo" → "hey_echo"
            self._model = Model(wake_words=[model_name])
            LOGGER.info("loaded openwakeword model for %r", self.word)
        return self._model

    def _default_detect(self, pcm16: bytes) -> bool:
        """Score the chunk with openWakeWord; speech of ANY frame counts.

        openWakeWord predicts per-frame scores over its native 80 ms frames
        (audio is 16 kHz per the contract, so this path ignores
        ``sample_rate`` beyond the pipeline's guarantee). The chunk wakes
        Echo when its best score crosses ``sensitivity``; the edge +
        refractory logic lives in process(), shared with injected
        detectors.
        """
        model = self._load()
        frame_bytes = 1280 * 2  # openwakeword's native 80 ms frame at 16 kHz
        best = 0.0
        for start in range(0, len(pcm16) - frame_bytes + 1, frame_bytes):
            scores = model.predict(pcm16[start : start + frame_bytes])
            if scores:
                best = max(best, max(float(score) for score in scores.values()))
        return best > self.sensitivity
