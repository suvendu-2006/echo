/* Echo voice (Phase 5) — floating mic button, streaming to /ws/voice.
   Self-contained: no dependencies, createElement/textContent only (SEC-2),
   browser echo cancellation (getUserMedia default) covers speaker pickup. */
(function () {
  'use strict';

  var state = {
    on: false,
    ws: null,
    ctx: null,
    processor: null,
    source: null,
    stream: null,
    audioQueue: [],
    playing: false,
    label: null,
  };

  function $(id) { return document.getElementById(id); }

  function b64FromInt16(int16) {
    var bytes = new Uint8Array(int16.buffer, int16.byteOffset, int16.byteLength);
    var binary = '';
    var CHUNK = 0x8000;
    for (var i = 0; i < bytes.length; i += CHUNK) {
      binary += String.fromCharCode.apply(null, bytes.subarray(i, i + CHUNK));
    }
    return btoa(binary);
  }

  function floatToInt16(floats) {
    var out = new Int16Array(floats.length);
    for (var i = 0; i < floats.length; i++) {
      var s = Math.max(-1, Math.min(1, floats[i]));
      out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    return out;
  }

  function setLabel(text) {
    if (!state.label) {
      state.label = document.createElement('div');
      state.label.className = 'voice-chip';
      document.body.appendChild(state.label);
    }
    state.label.textContent = text;
    state.label.hidden = !text;
  }

  function connectWS() {
    var proto = location.protocol === 'https:' ? 'wss://' : 'ws://';
    var sock = new WebSocket(proto + location.host + '/ws/voice');
    state.ws = sock;
    sock.onmessage = function (ev) {
      var data = null;
      try { data = JSON.parse(ev.data); } catch (err) { return; }
      if (!data || typeof data !== 'object') return;
      if (data.type === 'tts' && typeof data.audio === 'string' && data.audio) {
        state.audioQueue.push(data.audio);
        playNext();
      } else if (data.type === 'transcript' && data.text) {
        setLabel('You said: ' + String(data.text).slice(0, 60));
      } else if (data.type === 'status' && data.state === 'listening') {
        setLabel(state.on ? 'Listening for "Hey Echo"' : '');
      } else if (data.type === 'status' && data.state === 'speaking') {
        setLabel('Speaking… (speak to interrupt)');
      } else if (data.type === 'error' && typeof data.message === 'string') {
        setLabel('Voice: ' + data.message);
      }
    };
    sock.onclose = function () {
      if (state.on) { stopVoice(); setLabel('Voice connection lost'); }
    };
    return sock;
  }

  function playNext() {
    if (state.playing || state.audioQueue.length === 0) return;
    state.playing = true;
    var b64 = state.audioQueue.shift();
    try {
      var bytes = Uint8Array.from(atob(b64), function (c) { return c.charCodeAt(0); });
      var blob = new Blob([bytes], { type: 'audio/wav' });
      var audio = new Audio(URL.createObjectURL(blob));
      audio.onended = function () {
        URL.revokeObjectURL(audio.src);
        state.playing = false;
        playNext();
      };
      audio.onerror = function () { state.playing = false; playNext(); };
      var p = audio.play();
      if (p && p.catch) p.catch(function () { state.playing = false; playNext(); });
    } catch (err) {
      state.playing = false;
      playNext();
    }
  }

  function startVoice() {
    var sock = connectWS();
    sock.onopen = function () {
      sock.send(JSON.stringify({ type: 'voice_start' }));
      navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true }
      }).then(function (stream) {
        state.stream = stream;
        state.ctx = new AudioContext({ sampleRate: 16000 });
        state.source = state.ctx.createMediaStreamSource(stream);
        state.processor = state.ctx.createScriptProcessor(4096, 1, 1);
        state.processor.onaudioprocess = function (e) {
          if (!state.on || !state.ws || state.ws.readyState !== 1) return;
          var int16 = floatToInt16(e.inputBuffer.getChannelData(0));
          if (int16.length) {
            state.ws.send(JSON.stringify({ type: 'audio', data: b64FromInt16(int16) }));
          }
        };
        state.source.connect(state.processor);
        state.processor.connect(state.ctx.destination); // ScriptProcessor needs a destination
        setLabel('Listening for "Hey Echo"');
      }).catch(function () {
        setLabel('Microphone access denied');
        stopVoice();
      });
    };
  }

  function stopVoice() {
    state.on = false;
    if (state.ws) { try { state.ws.send(JSON.stringify({ type: 'voice_stop' })); } catch (e) {} try { state.ws.close(); } catch (e) {} }
    state.ws = null;
    if (state.processor) { try { state.processor.disconnect(); } catch (e) {} }
    if (state.source) { try { state.source.disconnect(); } catch (e) {} }
    if (state.ctx) { try { state.ctx.close(); } catch (e) {} }
    if (state.stream) { state.stream.getTracks().forEach(function (t) { t.stop(); }); }
    state.processor = state.source = state.ctx = state.stream = null;
    state.audioQueue = [];
    setLabel('');
    button().textContent = '🎙';
  }

  function button() {
    var b = $('echo-voice-btn');
    return b;
  }

  function toggle() {
    var b = button();
    if (state.on) {
      stopVoice();
    } else {
      state.on = true;
      b.textContent = '⏹';
      setLabel('Connecting…');
      startVoice();
    }
  }

  function init() {
    var b = document.createElement('button');
    b.id = 'echo-voice-btn';
    b.type = 'button';
    b.title = 'Talk to Echo (wake word: "Hey Echo")';
    b.textContent = '🎙';
    b.addEventListener('click', toggle);
    document.body.appendChild(b);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
