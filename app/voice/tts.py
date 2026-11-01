"""Text-to-speech (PLAN Phase 5.4) — sentence-chunked Piper, fake-friendly.

Echo speaks in sentences: :func:`split_sentences` chunks a reply at ``. ! ? …``
and newlines (keeping the punctuation, merging abbreviation false-positives
like ``Dr.`` / ``p.m.`` / ``e.g.``), so the FIRST sentence can be spoken while
the rest is still being synthesized. That chunking is what keeps the plan's
mouth-to-voice budget realistic without waiting for the whole reply.

:class:`PiperTTS` wraps piper-tts. Heavy deps are OPTIONAL (the frozen
contract): piper is lazy-imported inside ``_load_voice`` — never at module
level — and a missing install raises RuntimeError with the pip hint. Tests
and the offline suite inject a ``synthesizer`` callable instead and touch no
wheels.

:func:`text_to_speech_events` turns one reply into contract events — one
``{"type": "tts", "audio": <base64 wav>}`` per sentence — and honors a
``cancel_event`` (the pipeline's barge-in hook): set between sentences, the
remaining sentences are dropped on the spot.
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import io
import logging
import wave
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("echo.voice")

#: The only dependency-failure text a caller ever sees (frozen contract hint).
VOICE_EXTRAS_MESSAGE = "voice extras missing — pip install -r requirements-voice.txt"

#: Words that never end a sentence (checked lowercase, without the final dot).
#: Multi-part forms ("p.m", "e.g", "u.s") get a smarter rule below: they DO end
#: a sentence when a capitalized one follows ("at 5 p.m. Then we left.").
_ABBREVIATIONS = frozenset(
    {
        "dr", "mr", "mrs", "ms", "mx", "st", "jr", "sr", "prof", "rev", "hon",
        "sgt", "capt", "col", "gen", "lt", "sen", "rep", "gov",
        "vs", "etc", "eg", "ie", "al", "no",
        "a.m", "p.m", "e.g", "i.e", "ph.d", "u.s", "u.k", "u.n",
        "inc", "ltd", "corp", "dept", "est", "fig", "vol", "pp",
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept",
        "oct", "nov", "dec",
    }
)


def split_sentences(text: str) -> list[str]:
    """Chunk text into speakable sentences (pure stdlib, total).

    Splits on ``. ! ? …`` and newlines, keeps the punctuation attached, drops
    empty fragments, and merges the minimal abbreviation false-positives:
    ``Dr.`` / ``Mr.`` / ``e.g.`` / ``i.e.`` / single-letter initials
    ("John F. Kennedy") and decimals ("3.14") never end a sentence.
    Empty or whitespace-only text yields ``[]``.
    """
    if not text:
        return []
    sentences: list[str] = []
    start = 0
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in "\r\n":
            end = i
            i += 1
            if ch == "\r" and i < n and text[i] == "\n":
                i += 1
            _append_sentence(sentences, text[start:end])
            start = i
            continue
        if ch in "!?…":
            _append_sentence(sentences, text[start : i + 1])
            i += 1
            start = i
            continue
        if ch == ".":
            run_end = i
            while run_end < n and text[run_end] == ".":
                run_end += 1
            # Two or more dots are an ellipsis — always a boundary. A single
            # dot needs the abbreviation/decimal/initial vetting.
            if run_end - i >= 2 or _ends_sentence(text, start, i, run_end):
                _append_sentence(sentences, text[start:run_end])
                start = run_end
            i = run_end
            continue
        i += 1
    _append_sentence(sentences, text[start:])
    return sentences


def _append_sentence(sentences: list[str], piece: str) -> None:
    """Append a stripped, non-empty sentence fragment (keeps the scanner flat)."""
    stripped = piece.strip()
    if stripped:
        sentences.append(stripped)


def _ends_sentence(text: str, start: int, dot: int, run_end: int) -> bool:
    """Decide whether the single dot at ``text[dot]`` really ends the sentence.

    The word immediately before the dot decides: known abbreviations and
    single-letter initials do not; a multi-part form ("p.m", "e.g", "u.s")
    does when a capitalized sentence follows it; a digit right AFTER the dot
    means a decimal ("3.14") and never splits.
    """
    if dot + 1 < len(text) and text[dot + 1].isdigit():
        return False  # decimal like 3.14 — never a boundary
    word = _word_before(text, start, dot)
    if not word:
        return True
    lowered = word.lower()
    if "." in lowered:
        # "at 5 p.m. Then" splits; "tools, e.g. hammers" does not.
        return _next_nonspace_is_upper(text, run_end)
    if lowered in _ABBREVIATIONS:
        return False
    if len(lowered) == 1 and lowered.isalpha():
        return False  # an initial: "John F. Kennedy"
    return True


def _word_before(text: str, start: int, dot: int) -> str:
    """The word ending at the dot, including internal dots ("p.m" stays one)."""
    j = dot
    while j > start and (text[j - 1].isalnum() or text[j - 1] == "."):
        j -= 1
    return text[j:dot]


def _next_nonspace_is_upper(text: str, from_index: int) -> bool:
    """Whether the next non-space character starts a capitalized sentence."""
    k = from_index
    n = len(text)
    while k < n and text[k].isspace():
        k += 1
    if k >= n:
        return True  # the text ends here — so does the sentence
    return text[k].isupper()


class PiperTTS:
    """Piper-backed TTS with an injected-synthesizer seam for offline tests.

    ``synthesize(text)`` returns WAV bytes. With a ``synthesizer`` given it is
    the whole story (``async def synth(text) -> bytes``; a plain sync callable
    works too). Without one, piper-tts is lazy-imported on first use and the
    voice model is loaded ONCE and cached; the raw PCM is wrapped in a WAV
    header via the stdlib ``wave`` + ``io`` modules. Empty text returns
    ``b""`` without ever consulting the synthesizer.
    """

    def __init__(
        self,
        voice: str = "en_US-amy-medium",
        synthesizer: Callable[[str], Awaitable[bytes]] | Callable[[str], bytes] | None = None,
    ) -> None:
        self._voice_name = voice
        self._synthesizer = synthesizer
        self._voice: Any | None = None  # the cached PiperVoice (lazy)

    async def synthesize(self, text: str) -> bytes:
        """One sentence (or any text) in, WAV bytes out. Empty text -> b""."""
        if not text or not text.strip():
            return b""
        if self._synthesizer is not None:
            produced = self._synthesizer(text)
            if inspect.isawaitable(produced):
                produced = await produced
            return produced  # duck-typed: whatever the injected seam returns
        # piper's synthesis is CPU-bound sync work — run it in a thread so
        # the pipeline's barge-in watcher keeps pulling mic chunks meanwhile.
        return await asyncio.to_thread(self._synthesize_blocking, text)

    # ------------------------------------------------------------ internals

    def _synthesize_blocking(self, text: str) -> bytes:
        """Load piper (once), synthesize, wrap the PCM in a WAV header."""
        voice = self._load_voice()
        buf = io.BytesIO()
        synth_wav = getattr(voice, "synthesize_wav", None)
        if callable(synth_wav):
            # piper >= 1.2 writes straight into a wave file object.
            with wave.open(buf, "wb") as wav:
                synth_wav(text, wav)
            return buf.getvalue()
        rate, width, channels = 22050, 2, 1
        pcm = bytearray()
        for chunk in voice.synthesize(text):
            audio = getattr(chunk, "audio", None)
            pcm.extend(bytes(audio if audio is not None else chunk))
            rate = int(getattr(chunk, "sample_rate", rate) or rate)
            width = int(getattr(chunk, "sample_width", width) or width)
            channels = int(getattr(chunk, "channels", channels) or channels)
        with wave.open(buf, "wb") as wav:
            wav.setnchannels(channels)
            wav.setsampwidth(width)
            wav.setframerate(rate)
            wav.writeframes(bytes(pcm))
        return buf.getvalue()

    def _load_voice(self) -> Any:
        """Lazy, cached piper voice load — never at module level (contract)."""
        if self._voice is not None:
            return self._voice
        try:
            from piper import PiperVoice  # lazy: the OPTIONAL voice extra
        except ImportError as exc:
            raise RuntimeError(VOICE_EXTRAS_MESSAGE) from exc
        model, config = self._model_paths()
        loader = getattr(PiperVoice, "load", None)
        if callable(loader):
            try:
                self._voice = loader(model) if config is None else loader(model, config)
            except TypeError:
                self._voice = loader(model)  # older signature without config
        else:  # pragma: no cover - every piper release ships load()
            self._voice = PiperVoice(model)
        if self._voice is None:
            raise RuntimeError(VOICE_EXTRAS_MESSAGE)
        LOGGER.info("piper voice loaded: %s", self._voice_name)
        return self._voice

    def _model_paths(self) -> tuple[Path, Path | None]:
        """Locate the downloaded voice model (onnx + its json sidecar)."""
        name = self._voice_name
        candidates = [
            Path("voices") / f"{name}.onnx",
            Path.cwd() / f"{name}.onnx",
            Path.home() / ".local" / "share" / "piper" / f"{name}.onnx",
        ]
        for candidate in candidates:
            if candidate.is_file():
                config = candidate.with_name(candidate.name + ".json")
                return candidate, (config if config.is_file() else None)
        raise RuntimeError(
            f"piper voice '{name}' not found — run `python -m piper.download_voices "
            f"{name}` (or pass synthesizer= to PiperTTS)"
        )


async def text_to_speech_events(
    text: str,
    tts: Any,
    cancel_event: asyncio.Event | None = None,
) -> AsyncIterator[dict[str, str]]:
    """Yield one ``{"type": "tts", "audio": <b64 wav>}`` event per sentence.

    The pipeline consumes this through a producer task so sentence N speaks
    while sentence N+1 synthesizes (plan 5.4). When ``cancel_event`` (an
    asyncio.Event — the pipeline's barge-in hook) is set between sentences,
    the remaining sentences are dropped immediately. Empty text yields
    nothing at all.
    """
    for sentence in split_sentences(text):
        if cancel_event is not None and cancel_event.is_set():
            return
        produced = tts.synthesize(sentence)
        if inspect.isawaitable(produced):
            produced = await produced
        if not produced:
            continue  # a silent sentence is no event at all
        yield {
            "type": "tts",
            "audio": base64.b64encode(bytes(produced)).decode("ascii"),
        }
