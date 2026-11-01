"""The voice loop (PLAN Phase 5) — Echo listens, thinks, speaks, never wakes itself.

:class:`VoicePipeline` is the speaking/listening cycle wired through the
harness door: mic chunks in, wake word, VAD endpointing, STT, then
``chat.stream_chat`` — and the reply spoken back sentence by sentence. Two
unbreakable laws (docs/phase5-contract.md, FROZEN):

1. **Never wake itself** (plan 5.3): wake detection is muted the moment a
   wake fires and stays muted through thinking and speaking — the pipeline
   ALSO skips ``wake.process()`` entirely while muted, so even an engine
   that ignores its own guard cannot hear Echo talk.
2. **Barge-in ≤300 ms** (plan 5.5): while speaking, the mic is drained
   concurrently; one VAD-speech chunk (or a ``request_stop()``) cancels the
   remaining sentences via a zero-sleep ``asyncio.wait`` race — no polling,
   no fixed timers — then listening reopens, with wake held muted for a
   300 ms cooldown (measured on the injected clock) before it unmutes.

Every event flows through ``on_event`` (the server forwards it on the WS);
``run()`` also yields the very same dicts so a caller may iterate instead —
consume ONE channel, never both. ``run()`` performs ONE full loop iteration
(listening → wake → thinking → speaking → back to listening) and then
RETURNS — the server re-enters ``run()`` for the next conversation round,
which keeps the recursion flat and every state reachable in tests. The
pipeline NEVER raises out of ``run()``: any internal failure collapses to a
single generic ``error`` event (details to the ``echo.voice`` log, SEC
posture). All engines are duck-typed, so the whole suite runs offline with
fakes.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, Any

from app.voice.tts import text_to_speech_events

if TYPE_CHECKING:  # the real door — duck-typed so tests inject fakes
    from app.chatservice import ChatService

LOGGER = logging.getLogger("echo.voice")

#: The ONLY internal-failure text a client ever sees (chat ``error`` payloads
#: are forwarded as-is — ChatService already made those generic).
VOICE_SNAG_MESSAGE = "voice hit a snag — try again"

#: Plan 5.2: an utterance ends after ~600 ms of continuous silence (no fixed
#: utterance timers — this is the VAD's silence window, not a deadline).
UTTERANCE_SILENCE_SECONDS = 0.6
UTTERANCE_SILENCE_MS = int(UTTERANCE_SILENCE_SECONDS * 1000)

#: Plan 5.5: after barge-in, wake stays muted this long before unmuting, so
#: the tail of Echo's own audio cannot re-trigger detection.
BARGE_COOLDOWN_SECONDS = 0.3


def _final_content(payload: dict[str, Any]) -> str:
    """Extract the assistant text from a chat ``final`` payload."""
    message = payload.get("message")
    if isinstance(message, dict):
        return str(message.get("content") or "")
    return str(message or "")


def _as_audio(result: Any) -> bytes:
    """Normalize a VAD helper's return (bytes | chunk list | None) to bytes."""
    if isinstance(result, (bytes, bytearray, memoryview)):
        return bytes(result)
    if isinstance(result, (list, tuple)):
        return b"".join(
            bytes(chunk)
            for chunk in result
            if isinstance(chunk, (bytes, bytearray, memoryview))
        )
    return b""


class _TeeStream:
    """Mic view handed to the VAD helper: forwards chunks, keeps a copy.

    The pipeline owns the mic stream; ``wait_for_silence`` consumes whatever
    it pulls — through this tee every chunk is ALSO appended to the
    pipeline's buffer, so the utterance survives no matter what the helper
    returns (or whether it returns audio at all). The underlying stream is
    only advanced when the helper asks, so the pipeline resumes it cleanly
    afterwards.
    """

    def __init__(self, source: AsyncIterator[bytes], buffer: bytearray) -> None:
        self._source = source
        self._buffer = buffer

    def __aiter__(self) -> "_TeeStream":
        return self

    async def __anext__(self) -> bytes:
        chunk = await self._source.__anext__()
        self._buffer.extend(chunk)
        return chunk


class VoicePipeline:
    """The full voice loop over duck-typed engines and the chat door.

    Args:
        chat: the :class:`~app.chatservice.ChatService` door (anything with
            ``async stream_chat(text, session_id)`` and ``interrupt()``).
        stt: ``async transcribe(pcm16, sample_rate=16000) -> str``.
        tts: ``async synthesize(text) -> bytes`` (WAV).
        wake: ``process(pcm16) -> bool`` plus ``mute()`` / ``unmute()``.
        vad: ``is_speech(pcm16) -> bool`` and (optionally)
            ``wait_for_silence(chunks, silence_ms=...) -> bytes``.
        on_event: async callable receiving every contract event dict — the
            server forwards these to the WS.
        clock: monotonic seconds, injected so tests measure the barge-in
            cooldown without sleeping.
    """

    def __init__(
        self,
        chat: "ChatService | Any",
        stt: Any,
        tts: Any,
        wake: Any,
        vad: Any,
        on_event: Callable[[dict[str, Any]], Awaitable[None]] | None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._chat = chat
        self._stt = stt
        self._tts = tts
        self._wake = wake
        self._vad = vad
        self._on_event = on_event
        self._clock = clock
        self._stop_event = asyncio.Event()  # set by request_stop()
        self._muted = False  # the pipeline-side echo guard (belt to the wake engine's braces)
        self._last_status: str | None = None  # own-state dedupe (no double listening)
        self._speak_end = 0.0  # clock reading at the last speech activity

    # ------------------------------------------------------------- control

    def request_stop(self) -> None:
        """Public: end the current activity (the server's ``voice_stop``).

        Listening → nothing to end; collecting → the utterance is discarded;
        thinking → the chat turn is interrupted (``chat.interrupt()``) and the
        partial reply is not spoken; speaking → the barge-in path.
        """
        self._stop_event.set()

    # ------------------------------------------------------------ the loop

    async def run(
        self, mic_stream: AsyncIterator[bytes], session_id: str | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        """ONE loop iteration: listening → thinking → speaking → listening.

        Every event is passed to ``on_event`` AND yielded; drive with
        ``async for _ in pipeline.run(mic, sid): pass`` and wire on_event —
        never both. run() RETURNS after one full conversation round (the
        server re-enters it for the next round), or earlier when the mic
        stream ends with no wake. Never raises: internal failures become
        one generic ``error`` event.
        """
        try:
            event = await self._emit_status("listening")
            if event is not None:
                yield event

            # LISTENING: wake per chunk — skipped entirely while muted
            # (the echo guard, plan 5.3), which also gives the wake
            # refractory: the loud audio that follows a detection lands
            # in the muted window and can never double-beep.
            while True:
                if self._stop_event.is_set():
                    self._stop_event.clear()
                try:
                    chunk = await mic_stream.__anext__()
                except StopAsyncIteration:
                    return  # user walked away — clean idle
                if self._muted:
                    continue
                detect = getattr(self._wake, "process", None)
                if not callable(detect):
                    continue
                try:
                    detected = bool(detect(chunk))
                except Exception:
                    LOGGER.exception("wake engine failed on a chunk")
                    detected = False
                if detected:
                    break

            # WAKE FIRED → one full turn (beep → transcript → chat → speak),
            # then back to listening and RETURN — the server re-enters run()
            # for the next round. The barge-in path already reopened the mic
            # before its cooldown, so the status dedupe swallows this second
            # listening there (no duplicate on the wire).
            async for event in self._turn(mic_stream, session_id):
                yield event
            event = await self._emit_status("listening")
            if event is not None:
                yield event
        except asyncio.CancelledError:
            raise
        except Exception:
            LOGGER.exception("voice pipeline crashed")
            try:
                yield await self._emit({"type": "error", "message": VOICE_SNAG_MESSAGE})
                event = await self._emit_status("listening")
                if event is not None:
                    yield event
            except Exception:
                LOGGER.exception("failed to emit the pipeline crash error")
        finally:
            self._set_muted(False)

    async def _turn(
        self, mic_stream: AsyncIterator[bytes], session_id: str | None
    ) -> AsyncIterator[dict[str, Any]]:
        """One wake→speak cycle: beep, collect, transcribe, chat, speak."""
        try:
            yield await self._emit({"type": "beep"})
            event = await self._emit_status("thinking")
            if event is not None:
                yield event
            self._set_muted(True)  # echo guard from the first collected chunk

            audio = await self._collect_utterance(mic_stream)
            if audio is None:  # stop requested mid-collection: discard, relisten
                return

            text = await self._transcribe(audio)
            yield await self._emit({"type": "transcript", "text": text})
            if not text.strip():
                return  # nothing said — straight back to listening, no chat

            content = ""
            errored = False
            watcher = asyncio.ensure_future(self._watch_stop(self._chat))
            try:
                # THINKING: the harness door. Status frames (tool_running
                # included) and tool activity pass through unchanged; deltas
                # are skipped (voice speaks the final); final is consumed.
                async for payload in self._chat.stream_chat(text, session_id):
                    if not isinstance(payload, dict):
                        LOGGER.warning("chat yielded a non-dict payload; ignored")
                        continue
                    if self._stop_event.is_set():
                        self._interrupt_chat(self._chat)
                    kind = payload.get("type")
                    if kind == "delta":
                        continue
                    if kind == "final":
                        if not errored:
                            content = _final_content(payload)
                        continue
                    if kind == "error":
                        errored = True
                    yield await self._emit(payload)  # status/tool/permission/error as-is
            finally:
                watcher.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await watcher

            if errored or self._stop_event.is_set() or not content.strip():
                return  # chat error / stop during thinking → listening, no speech

            # SPEAKING (with the barge-in watcher inside).
            async for event in self._speak(content, mic_stream):
                yield event
        except asyncio.CancelledError:
            raise
        except Exception:
            LOGGER.exception("voice turn failed")
            yield await self._emit({"type": "error", "message": VOICE_SNAG_MESSAGE})
        finally:
            # Unmute exactly once per turn (no-op when _speak already did) so
            # a failure mid-turn can never leave the echo guard stuck on.
            self._set_muted(False)

    # ------------------------------------------------------------ speaking

    async def _speak(
        self, text: str, mic_stream: AsyncIterator[bytes]
    ) -> AsyncIterator[dict[str, Any]]:
        """Producer/consumer TTS with the ≤300 ms barge-in race (plan 5.4/5.5).

        A producer task synthesizes sentences into a queue (sentence N+1
        generates while sentence N emits); the emitter below drains it and
        races each wait against barge-in mic speech and ``request_stop`` —
        ``asyncio.wait(FIRST_COMPLETED)``, zero sleeps.
        """
        event = await self._emit_status("speaking")
        if event is not None:
            yield event

        cancel_event = asyncio.Event()  # text_to_speech_events' early stop
        barge = asyncio.Event()  # mic speech detected while speaking
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        producer = asyncio.ensure_future(self._produce_sentences(text, cancel_event, queue))
        drain = asyncio.ensure_future(self._drain_for_barge(mic_stream, barge))
        interrupted = False
        try:
            while True:
                if barge.is_set() or self._stop_event.is_set():
                    interrupted = True
                    break
                get_task = asyncio.ensure_future(queue.get())
                barge_wait = asyncio.ensure_future(barge.wait())
                stop_wait = asyncio.ensure_future(self._stop_event.wait())
                done, pending = await asyncio.wait(
                    {get_task, barge_wait, stop_wait},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)
                if barge_wait in done or stop_wait in done:
                    interrupted = True  # a queued sentence dies unspoken
                    break
                item = get_task.result()
                if item is None:
                    break  # producer sentinel — spoke everything
                if isinstance(item, dict) and item.get("type") == "error":
                    yield await self._emit(item)  # synthesis failed mid-reply
                    continue
                yield await self._emit(item)  # {"type": "tts", ...}
        except asyncio.CancelledError:
            raise
        finally:
            producer.cancel()
            drain.cancel()
            await asyncio.gather(producer, drain, return_exceptions=True)

        if interrupted:
            cancel_event.set()
            self._speak_end = self._clock()
            event = await self._emit_status("listening")  # mic reopens NOW
            if event is not None:
                yield event
            await self._cool_down()  # wake stays muted for the 300 ms tail
        self._set_muted(False)  # natural completion unmutes immediately (contract)

    async def _produce_sentences(
        self, text: str, cancel_event: asyncio.Event, queue: "asyncio.Queue[dict[str, Any] | None]"
    ) -> None:
        """Producer: synthesize sentence events into the queue, then a sentinel."""
        try:
            async for event in text_to_speech_events(
                text, self._tts, cancel_event=cancel_event
            ):
                await queue.put(event)
        except asyncio.CancelledError:
            raise
        except Exception:
            LOGGER.exception("sentence synthesis failed")
            await queue.put({"type": "error", "message": VOICE_SNAG_MESSAGE})
        finally:
            await queue.put(None)

    async def _drain_for_barge(self, mic_stream: AsyncIterator[bytes], barge: asyncio.Event) -> None:
        """Consume mic chunks while speaking; speech → barge-in, immediately."""
        try:
            while True:
                chunk = await mic_stream.__anext__()
                try:
                    speech = bool(self._vad.is_speech(chunk))
                except Exception:
                    LOGGER.exception("vad failed during the barge-in watch")
                    speech = False
                if speech:
                    barge.set()
                    return
        except asyncio.CancelledError:
            raise
        except StopAsyncIteration:
            return
        except Exception:
            LOGGER.exception("barge-in mic drain failed")

    # ----------------------------------------------------------- collecting

    async def _collect_utterance(self, mic_stream: AsyncIterator[bytes]) -> bytes | None:
        """Collect the command via the VAD's silence endpointing.

        Ends on ~600 ms of silence (the VAD helper), the mic ending, or a
        ``request_stop`` (→ ``None``: discard). The tee buffer is ground
        truth, so any ``wait_for_silence`` return shape works; a missing or
        failing helper falls back to the pipeline's own chunk loop.
        """
        buffer = bytearray()
        helper = getattr(self._vad, "wait_for_silence", None)
        helper_result: Any = None
        if callable(helper):
            tee = _TeeStream(mic_stream, buffer)
            collect = asyncio.ensure_future(self._call_wait_for_silence(helper, tee))
            stop_wait = asyncio.ensure_future(self._stop_event.wait())
            done, pending = await asyncio.wait(
                {collect, stop_wait}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            if self._stop_event.is_set():
                return None
            if collect in done:
                error = collect.exception()
                if error is not None:
                    LOGGER.warning(
                        "vad.wait_for_silence failed (%s: %s) — using the fallback loop",
                        type(error).__name__,
                        error,
                    )
                else:
                    helper_result = collect.result()
        if bytes(buffer):
            return bytes(buffer)
        normalized = _as_audio(helper_result)
        if normalized:
            return normalized  # a helper that returns audio without consuming
        return await self._manual_collect(mic_stream, buffer)

    async def _call_wait_for_silence(self, helper: Callable[..., Any], tee: _TeeStream) -> Any:
        """Call the VAD helper in its native dialect (real engine first)."""
        try:
            return await helper(tee, silence_ms=UTTERANCE_SILENCE_MS)
        except TypeError:
            return await helper(tee, UTTERANCE_SILENCE_SECONDS)

    async def _manual_collect(self, mic_stream: AsyncIterator[bytes], buffer: bytearray) -> bytes | None:
        """Fallback endpointing when no ``wait_for_silence`` helper exists.

        Ends after UTTERANCE_SILENCE_SECONDS without speech (wall clock — the
        injected clock governs the barge cooldown), on mic end, or on stop.
        Note: the wait_for timeout cancels ``__anext__`` mid-wait, which
        closes one-shot async-generator mic streams early — queue-backed
        streams (the server's) are unaffected.
        """
        started = False
        last_sound = time.monotonic()
        while True:
            if self._stop_event.is_set():
                return None
            remaining = UTTERANCE_SILENCE_SECONDS - (time.monotonic() - last_sound)
            if started and remaining <= 0:
                break
            timeout = max(remaining, 0.05) if started else None
            try:
                chunk = await asyncio.wait_for(mic_stream.__anext__(), timeout)
            except StopAsyncIteration:
                break
            except TimeoutError:
                break  # the silence window elapsed with no chunk arriving
            buffer.extend(chunk)
            try:
                speech = bool(self._vad.is_speech(chunk))
            except Exception:
                LOGGER.exception("vad failed during endpointing")
                speech = False
            if speech:
                started = True
                last_sound = time.monotonic()
        return bytes(buffer)

    async def _transcribe(self, audio: bytes) -> str:
        """STT over the collected utterance (turn-level, no partials needed)."""
        transcribe = getattr(self._stt, "transcribe", None)
        if not callable(transcribe):
            raise RuntimeError(
                "the stt engine has no transcribe() — inject one or install "
                "the voice extras"
            )
        return str(await transcribe(audio) or "")

    # ---------------------------------------------------------- chat door

    async def _watch_stop(self, chat: Any) -> None:
        """Interrupt the running chat turn the moment a stop is requested —
        even while the stream is blocked with no payloads flowing."""
        try:
            await self._stop_event.wait()
        except asyncio.CancelledError:
            raise
        self._interrupt_chat(chat)

    def _interrupt_chat(self, chat: Any) -> None:
        """Best-effort ``chat.interrupt()`` (the partial reply is kept)."""
        interrupt = getattr(chat, "interrupt", None)
        if not callable(interrupt):
            return
        try:
            interrupt()
        except Exception:
            LOGGER.exception("chat interrupt failed")

    # ------------------------------------------------------------- plumbing

    async def _cool_down(self) -> None:
        """Hold wake muted until BARGE_COOLDOWN_SECONDS after the last speech."""
        delay = (self._speak_end + BARGE_COOLDOWN_SECONDS) - self._clock()
        if delay > 0:
            await asyncio.sleep(delay)

    def _set_muted(self, muted: bool) -> None:
        """Sync the pipeline-side guard AND the wake engine (exactly-once)."""
        if muted == self._muted:
            return
        self._muted = muted
        action = getattr(self._wake, "mute" if muted else "unmute", None)
        if callable(action):
            try:
                action()
            except Exception:
                LOGGER.exception("wake %s failed", "mute" if muted else "unmute")

    async def _emit_status(self, state: str) -> dict[str, str] | None:
        """Emit an own status once per transition (no duplicate ``listening``)."""
        if state == self._last_status:
            return None
        self._last_status = state
        return await self._emit({"type": "status", "state": state})

    async def _emit(self, event: dict[str, Any]) -> dict[str, Any]:
        """Hand one event to the server's sink; a dead sink never kills the loop."""
        if self._on_event is not None:
            try:
                await self._on_event(event)
            except Exception:
                LOGGER.exception("voice event sink failed")
        return event
