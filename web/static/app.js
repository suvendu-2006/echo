/* ============================================================
   Echo — Phase 2 frontend
   Vanilla JS (ES2020), no dependencies, no build step.
   All dynamic content is rendered with createElement/textContent —
   user text and model output are never injected as markup.
   API: see docs/phase2-contract.md (frozen).
   ============================================================ */
(function () {
  'use strict';

  /* ---------- helpers ---------- */

  function $(id) { return document.getElementById(id); }

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined && text !== null) n.textContent = text; // safe by construction
    return n;
  }

  function clearChildren(n) {
    while (n.firstChild) n.removeChild(n.firstChild);
  }

  function flash(node, ms) {
    if (!node) return;
    node.hidden = false;
    if (node._flashT) clearTimeout(node._flashT);
    node._flashT = setTimeout(function () { node.hidden = true; }, ms || 2400);
  }

  /* ---------- state ---------- */

  var state = {
    view: 'chat',
    currentSessionId: null,
    sessions: [],
    // websocket
    ws: null,
    wsOpen: false,
    wsAttempt: 0,
    wsTimer: null,
    // turn lifecycle
    turnActive: false,
    assistantEl: null,
    pendingEl: null,
    turnTimer: null,
    interruptTimer: null,
    lastInterruptAt: 0,
    // settings
    settings: null,
    failoverOrder: [],
    modelCache: {}
  };

  /* ---------- dom refs (script runs at end of body) ---------- */

  var dom = {
    statusDot: $('status-dot'),
    statusLabel: $('status-label'),
    reconnecting: $('reconnecting'),
    serverBanner: $('server-banner'),
    bannerRetry: $('banner-retry'),
    bannerDismiss: $('banner-dismiss'),
    errorBar: $('error-bar'),
    errorText: $('error-text'),
    errorDismiss: $('error-dismiss'),
    viewChat: $('view-chat'),
    viewSettings: $('view-settings'),
    tabChat: $('tab-chat'),
    tabSettings: $('tab-settings'),
    sidebarToggle: $('sidebar-toggle'),
    backdrop: $('backdrop'),
    newChat: $('new-chat'),
    sessionList: $('session-list'),
    chatTitle: $('chat-title'),
    thread: $('thread'),
    threadEmpty: $('thread-empty'),
    composer: $('composer'),
    input: $('msg-input'),
    send: $('send'),
    interrupt: $('interrupt'),
    activeStrip: $('active-strip'),
    settingsError: $('settings-error'),
    providerCards: $('provider-cards'),
    failoverList: $('failover-list'),
    saveFailover: $('save-failover'),
    failoverSaved: $('failover-saved')
  };

  /* ---------- errors & banners ---------- */

  function showServerBanner() { dom.serverBanner.hidden = false; }
  function hideServerBanner() { dom.serverBanner.hidden = true; }

  function showErrorBar(msg) {
    // Empty message = hide the bar entirely (an empty strip with an × is noise).
    if (!msg || !String(msg).trim()) { hideErrorBar(); return; }
    dom.errorText.textContent = String(msg); // textContent only
    dom.errorBar.hidden = false;
  }
  function hideErrorBar() { dom.errorBar.hidden = true; }

  function showInline(node, msg, isError) {
    node.textContent = msg;
    node.classList.toggle('error', !!isError);
    node.hidden = false;
  }

  /* Every fetch failure surfaces visibly: unreachable -> server banner
     (plus optional inline note), API error -> inline node or error bar. */
  function handleFetchError(err, inlineNode) {
    var msg = err && err.message ? err.message : 'Something went wrong.';
    if (inlineNode) {
      showInline(inlineNode, msg, !err.unreachable);
    } else if (!err.unreachable) {
      showErrorBar(msg);
    }
  }

  /* ---------- REST client (JSON, contract paths) ---------- */

  function api(path, options) {
    var opts = options || {};
    var init = { method: opts.method || 'GET', headers: {} };
    if (opts.body !== undefined) {
      init.headers['Content-Type'] = 'application/json';
      init.body = JSON.stringify(opts.body);
    }
    return fetch(path, init).then(function (res) {
      hideServerBanner(); // any successful call proves the server is back
      if (res.status === 204) return null;
      return res.json().catch(function () { return null; }).then(function (data) {
        if (!res.ok) {
          var detail = (data && typeof data.detail === 'string' && data.detail)
            ? data.detail
            : 'Request failed (HTTP ' + res.status + ').';
          var e = new Error(detail);
          e.status = res.status;
          throw e;
        }
        return data;
      });
    }, function () {
      showServerBanner();
      var e = new Error('Echo server unreachable — is it running? python -m app.serve');
      e.unreachable = true;
      throw e;
    });
  }

  /* ---------- hash routing (#chat default, #settings) ---------- */

  function applyRoute() {
    var h = location.hash.replace(/^#/, '');
    var view = null;
    if (h === 'settings') view = 'settings';
    else if (h === 'chat' || h === '') view = 'chat';
    else return; // unknown anchor (e.g. in-page) — keep current view

    state.view = view;
    dom.viewChat.hidden = view !== 'chat';
    dom.viewSettings.hidden = view !== 'settings';
    if (view === 'chat') {
      dom.tabChat.setAttribute('aria-current', 'page');
      dom.tabSettings.removeAttribute('aria-current');
    } else {
      dom.tabSettings.setAttribute('aria-current', 'page');
      dom.tabChat.removeAttribute('aria-current');
    }
    closeSidebar();
    if (view === 'settings') loadSettingsView();
  }

  /* ---------- sidebar (mobile) ---------- */

  function openSidebar() {
    document.body.classList.add('sidebar-open');
    dom.sidebarToggle.setAttribute('aria-expanded', 'true');
    dom.backdrop.hidden = false;
  }
  function closeSidebar() {
    document.body.classList.remove('sidebar-open');
    dom.sidebarToggle.setAttribute('aria-expanded', 'false');
    dom.backdrop.hidden = true;
  }

  /* ---------- sessions ---------- */

  function refreshSessions() {
    return api('/api/sessions').then(function (list) {
      state.sessions = Array.isArray(list) ? list : [];
      renderSessions();
      if (state.currentSessionId) {
        var cur = null;
        for (var i = 0; i < state.sessions.length; i++) {
          if (state.sessions[i].id === state.currentSessionId) { cur = state.sessions[i]; break; }
        }
        if (cur) dom.chatTitle.textContent = cur.title || 'Untitled';
      }
    }).catch(function (err) { handleFetchError(err); });
  }

  function renderSessions() {
    clearChildren(dom.sessionList);
    if (!state.sessions.length) {
      dom.sessionList.appendChild(el('li', 'session-empty', 'No sessions yet — say hello.'));
      return;
    }
    state.sessions.forEach(function (s) {
      var li = el('li', 'session-item' + (s.id === state.currentSessionId ? ' active' : ''));
      li.dataset.id = s.id || '';

      var main = el('button', 'session-main');
      main.type = 'button';
      var count = (s.message_count === 1) ? '1 message' : (s.message_count || 0) + ' messages';
      main.appendChild(el('span', 's-title', s.title || 'Untitled'));
      main.appendChild(el('span', 's-count', count));
      main.addEventListener('click', function () { openSession(s.id); });

      var actions = el('span', 'session-actions');
      var ren = el('button', 'btn tiny', 'Rename');
      ren.type = 'button';
      ren.setAttribute('aria-label', 'Rename session');
      ren.addEventListener('click', function () { startRename(li, s); });
      var del = el('button', 'btn tiny danger', 'Delete');
      del.type = 'button';
      del.setAttribute('aria-label', 'Delete session');
      del.addEventListener('click', function () { armDelete(li, s); });
      actions.appendChild(ren);
      actions.appendChild(del);

      li.appendChild(main);
      li.appendChild(actions);
      dom.sessionList.appendChild(li);
    });
  }

  function openSession(id) {
    if (state.turnActive) {
      showErrorBar('Finish the current reply first (or press Interrupt).');
      return;
    }
    api('/api/sessions/' + encodeURIComponent(id)).then(function (data) {
      if (!data) return;
      state.currentSessionId = data.id || id;
      clearThread();
      var msgs = Array.isArray(data.messages) ? data.messages : [];
      msgs.forEach(function (m) {
        appendMessage(m.role === 'user' ? 'user' : 'assistant', m.content == null ? '' : String(m.content));
      });
      dom.threadEmpty.hidden = msgs.length > 0;
      dom.chatTitle.textContent = data.title || 'Untitled';
      renderSessions();
      scrollToBottom();
      closeSidebar();
      dom.input.focus();
    }).catch(function (err) { handleFetchError(err); });
  }

  function startRename(li, session) {
    if (li.classList.contains('renaming')) return;
    li.classList.add('renaming');
    var main = li.querySelector('.session-main');
    var actions = li.querySelector('.session-actions');
    if (main) main.hidden = true;
    if (actions) actions.hidden = true;

    var input = el('input', 'rename-input');
    input.type = 'text';
    input.value = session.title || '';
    input.maxLength = 200;
    input.setAttribute('aria-label', 'Session title');
    li.insertBefore(input, li.firstChild);
    input.focus();
    input.select();

    var done = false;
    var finish = function (save) {
      if (done) return;
      done = true;
      var val = input.value.trim();
      if (input.parentNode) input.parentNode.removeChild(input);
      if (save && val && val !== (session.title || '')) {
        api('/api/sessions/' + encodeURIComponent(session.id), { method: 'PATCH', body: { title: val } })
          .then(function () { return refreshSessions(); })
          .catch(function (err) { handleFetchError(err); renderSessions(); });
      } else {
        renderSessions();
      }
    };
    input.addEventListener('keydown', function (ev) {
      if (ev.key === 'Enter') { ev.preventDefault(); finish(true); }
      else if (ev.key === 'Escape') { ev.preventDefault(); finish(false); }
    });
    input.addEventListener('blur', function () { finish(true); });
  }

  function armDelete(li, session) {
    var actions = li.querySelector('.session-actions');
    if (!actions) return;
    clearChildren(actions);
    var yes = el('button', 'btn tiny danger', 'Confirm');
    yes.type = 'button';
    var no = el('button', 'btn tiny', 'Keep');
    no.type = 'button';
    yes.addEventListener('click', function () {
      api('/api/sessions/' + encodeURIComponent(session.id), { method: 'DELETE' }).then(function () {
        if (state.currentSessionId === session.id) {
          state.currentSessionId = null;
          clearThread();
          dom.chatTitle.textContent = 'New chat';
        }
        return refreshSessions();
      }).catch(function (err) {
        handleFetchError(err);
        renderSessions();
      });
    });
    no.addEventListener('click', function () { renderSessions(); });
    actions.appendChild(yes);
    actions.appendChild(no);
    yes.focus();
  }

  function newChat() {
    if (state.turnActive) {
      showErrorBar('Finish the current reply first (or press Interrupt).');
      return;
    }
    api('/api/sessions', { method: 'POST' }).then(function (created) {
      state.currentSessionId = (created && created.id) ? created.id : null;
      clearThread();
      dom.chatTitle.textContent = (created && created.title) ? created.title : 'New chat';
      return refreshSessions();
    }).then(function () {
      closeSidebar();
      dom.input.focus();
    }).catch(function (err) { handleFetchError(err); });
  }

  /* ---------- thread rendering (textContent only — XSS-safe) ---------- */

  function clearThread() {
    var nodes = dom.thread.querySelectorAll('.msg, .typing');
    Array.prototype.forEach.call(nodes, function (n) { n.parentNode.removeChild(n); });
    state.assistantEl = null;
    state.pendingEl = null;
    dom.threadEmpty.hidden = false;
  }

  function nearBottom() {
    return dom.thread.scrollHeight - dom.thread.scrollTop - dom.thread.clientHeight < 140;
  }

  function scrollToBottom() {
    dom.thread.scrollTop = dom.thread.scrollHeight;
  }

  function appendMessage(role, content) {
    var wrap = el('div', 'msg ' + (role === 'user' ? 'user' : 'assistant'));
    var bubble = el('div', 'bubble', content);
    wrap.appendChild(bubble);
    dom.thread.appendChild(wrap);
    dom.threadEmpty.hidden = true;
    return bubble;
  }

  function showPending() {
    if (state.pendingEl) return;
    var t = el('div', 'typing');
    t.appendChild(el('span', 'typing-text', 'thinking'));
    t.appendChild(el('span', 'tdot'));
    t.appendChild(el('span', 'tdot'));
    t.appendChild(el('span', 'tdot'));
    dom.thread.appendChild(t);
    state.pendingEl = t;
    scrollToBottom();
  }

  function clearPending() {
    if (state.pendingEl) {
      if (state.pendingEl.parentNode) state.pendingEl.parentNode.removeChild(state.pendingEl);
      state.pendingEl = null;
    }
  }

  function dropEmptyAssistant() {
    if (state.assistantEl && state.assistantEl.textContent === '') {
      var wrap = state.assistantEl.parentNode;
      if (wrap) wrap.parentNode.removeChild(wrap);
    }
    state.assistantEl = null;
  }

  /* ---------- composer ---------- */

  function autoGrow() {
    dom.input.style.height = 'auto';
    dom.input.style.height = Math.min(dom.input.scrollHeight, 180) + 'px';
  }

  function updateComposer() {
    dom.send.disabled = state.turnActive || dom.input.value.trim().length === 0;
    dom.interrupt.disabled = !state.turnActive || !state.wsOpen;
  }

  function sendMessage() {
    if (state.turnActive) return;
    var text = dom.input.value.trim();
    if (!text) return;
    if (!state.wsOpen || !state.ws) {
      showErrorBar('Not connected — retrying. Try again in a moment.');
      connectWS();
      return;
    }
    try {
      state.ws.send(JSON.stringify({ type: 'chat', session_id: state.currentSessionId, text: text }));
    } catch (err) {
      showErrorBar('Could not send — the connection dropped. Retrying…');
      connectWS();
      return;
    }
    dom.input.value = '';
    autoGrow();
    state.turnActive = true;
    dom.thread.setAttribute('aria-busy', 'true');
    appendMessage('user', text);
    showPending();
    scrollToBottom();
    updateComposer();
    if (state.turnTimer) clearTimeout(state.turnTimer);
    state.turnTimer = setTimeout(function () {
      if (state.turnActive) failTurn('The server did not respond in time.');
    }, 120000);
  }

  function endTurn() {
    state.turnActive = false;
    state.assistantEl = null;
    if (state.turnTimer) { clearTimeout(state.turnTimer); state.turnTimer = null; }
    if (state.interruptTimer) { clearTimeout(state.interruptTimer); state.interruptTimer = null; }
    dom.thread.removeAttribute('aria-busy');
    updateComposer();
  }

  function failTurn(reason) {
    clearPending();
    dropEmptyAssistant();
    endTurn();
    setDot('idle');
    showErrorBar(reason);
  }

  function onInterruptClick() {
    if (!state.turnActive || !state.ws || !state.wsOpen) return;
    var now = Date.now();
    if (now - state.lastInterruptAt < 500) return;
    state.lastInterruptAt = now;
    try {
      state.ws.send(JSON.stringify({ type: 'interrupt' }));
    } catch (err) { /* the close handler will surface the drop */ }
    if (state.interruptTimer) clearTimeout(state.interruptTimer);
    state.interruptTimer = setTimeout(function () {
      if (state.turnActive) failTurn('Interrupt sent — no confirmation received from the server.');
    }, 30000);
  }

  /* ---------- websocket ---------- */

  function setDot(stateName) {
    var cls = (stateName === 'thinking' || stateName === 'streaming') ? stateName : 'idle';
    dom.statusDot.className = 'dot ' + cls;
    dom.statusLabel.textContent = cls;
  }

  function setReconnecting(on) {
    dom.reconnecting.hidden = !on;
  }

  function connectWS() {
    if (state.ws) {
      var rs = state.ws.readyState;
      if (rs === WebSocket.OPEN || rs === WebSocket.CONNECTING) return;
    }
    var proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    var sock;
    try {
      sock = new WebSocket(proto + '//' + location.host + '/ws/chat');
    } catch (err) {
      scheduleReconnect();
      return;
    }
    state.ws = sock;

    sock.onopen = function () {
      if (state.ws !== sock) return;
      state.wsOpen = true;
      state.wsAttempt = 0;
      setReconnecting(false);
      updateComposer();
    };

    sock.onmessage = function (ev) {
      var data = null;
      try { data = JSON.parse(ev.data); } catch (err) { return; }
      if (data && typeof data === 'object') handleWSMessage(data);
    };

    sock.onclose = function () {
      if (state.ws !== sock) return;
      state.ws = null;
      state.wsOpen = false;
      if (state.turnActive) {
        failTurn('Connection lost mid-reply. Send your message again to continue.');
      }
      setReconnecting(true);
      scheduleReconnect();
      updateComposer();
    };

    sock.onerror = function () { /* the close event follows */ };
  }

  function scheduleReconnect() {
    if (state.wsTimer) return;
    var delay = Math.min(8000, 600 * Math.pow(2, state.wsAttempt));
    state.wsAttempt = Math.min(state.wsAttempt + 1, 6);
    state.wsTimer = setTimeout(function () {
      state.wsTimer = null;
      connectWS();
    }, delay);
  }

  /* Server events. The status dot is driven ONLY by {"type":"status"};
     idle is restored when a turn ends (final/error/close). */
  function handleWSMessage(msg) {
    switch (msg.type) {
      case 'status':
        if (msg.state === 'thinking' || msg.state === 'streaming' || msg.state === 'idle') {
          setDot(msg.state);
        }
        break;
      case 'delta':
        if (typeof msg.text === 'string' && msg.text) onDelta(msg.text);
        break;
      case 'final':
        hideToolActivity();
        onFinal(msg);
        break;
      case 'error':
        hideToolActivity();
        clearPending();
        dropEmptyAssistant();
        endTurn();
        setDot('idle');
        showErrorBar(typeof msg.message === 'string' && msg.message ? msg.message : 'The server reported an error.');
        break;
      case 'tool':
        showToolActivity('Running tool: ' + String(msg.name || '?'));
        break;
      case 'tool_done':
        showToolActivity('Tool finished: ' + String(msg.name || '?') +
          (msg.status ? ' (' + String(msg.status) + ')' : ''));
        break;
      case 'permission':
        showPermissionModal(msg);
        break;
      case 'pong':
      default:
        break; // unknown server types are ignored client-side
    }
  }

  /* Harness activity + permission popup (Phase 3 H7-3/H4-4).
     Everything is built with createElement/textContent — XSS-safe by rule. */
  var toolChip = null;
  function showToolActivity(text) {
    if (!toolChip) {
      toolChip = document.createElement('div');
      toolChip.className = 'tool-chip';
      document.body.appendChild(toolChip);
    }
    toolChip.textContent = text; // textContent only — markup injection forbidden
    toolChip.hidden = false;
  }
  function hideToolActivity() {
    if (toolChip) toolChip.hidden = true;
  }

  function showPermissionModal(msg) {
    var overlay = document.createElement('div');
    overlay.className = 'permission-overlay';
    var card = document.createElement('div');
    card.className = 'permission-card';
    var title = document.createElement('h3');
    title.textContent = 'Permission needed';
    var toolLine = document.createElement('p');
    toolLine.textContent = 'Echo wants to run: ' + String(msg.tool || '?');
    var argsPre = document.createElement('pre');
    try {
      argsPre.textContent = JSON.stringify(msg.arguments || {}, null, 2);
    } catch (err) {
      argsPre.textContent = String(msg.arguments || '');
    }
    var row = document.createElement('div');
    row.className = 'permission-row';
    function answer(allow) {
      overlay.remove();
      if (state.ws && state.wsOpen) {
        state.ws.send(JSON.stringify({
          type: 'permission_answer',
          id: String(msg.id || ''),
          allow: !!allow
        }));
      }
    }
    var denyBtn = document.createElement('button');
    denyBtn.type = 'button';
    denyBtn.textContent = 'Deny';
    denyBtn.addEventListener('click', function () { answer(false); });
    var allowBtn = document.createElement('button');
    allowBtn.type = 'button';
    allowBtn.className = 'primary';
    allowBtn.textContent = 'Allow';
    allowBtn.addEventListener('click', function () { answer(true); });
    row.appendChild(denyBtn);
    row.appendChild(allowBtn);
    card.appendChild(title);
    card.appendChild(toolLine);
    card.appendChild(argsPre);
    card.appendChild(row);
    overlay.appendChild(card);
    document.body.appendChild(overlay);
  }

  function onDelta(text) {
    var stick = nearBottom();
    dom.threadEmpty.hidden = true;
    if (!state.assistantEl) {
      clearPending();
      state.assistantEl = appendMessage('assistant', '');
    }
    state.assistantEl.appendChild(document.createTextNode(text)); // safe append
    if (stick) scrollToBottom();
  }

  function onFinal(msg) {
    clearPending();
    if (state.assistantEl && msg && msg.message && typeof msg.message.content === 'string') {
      state.assistantEl.textContent = msg.message.content; // sync bubble to saved content
    }
    if (msg && msg.session_id) state.currentSessionId = msg.session_id;
    endTurn();
    setDot('idle');
    scrollToBottom();
    refreshSessions(); // sidebar counts + (new) session appears; nothing stored locally
  }

  /* ---------- settings ---------- */

  function activeProvider() {
    return (state.settings && state.settings.active && state.settings.active.provider) || null;
  }

  function deriveFailover(providers) {
    return providers.map(function (p, i) {
      var n = Number(p.failover_order);
      return { name: p.name, key: Number.isFinite(n) ? n : i };
    }).sort(function (a, b) { return a.key - b.key; }).map(function (p) { return p.name; });
  }

  function loadSettingsView() {
    return api('/api/settings').then(function (s) {
      state.settings = (s && typeof s === 'object') ? s : {};
      var providers = Array.isArray(state.settings.providers) ? state.settings.providers : [];
      state.failoverOrder = deriveFailover(providers);
      dom.settingsError.hidden = true;
      renderSettings();
      renderFailover();
      providers.forEach(function (p) { refreshModels(p.name, null); });
    }).catch(function (err) { handleFetchError(err, dom.settingsError); });
  }

  function updateActiveStrip() {
    var a = state.settings && state.settings.active ? state.settings.active : null;
    if (a && a.provider) {
      dom.activeStrip.textContent = 'Active provider: ' + a.provider + (a.model ? ' · ' + a.model : '');
    } else {
      dom.activeStrip.textContent = 'Active provider: none yet — pick a model below and press Set active.';
    }
  }

  function updateActiveBadges() {
    var active = activeProvider();
    Array.prototype.forEach.call(dom.providerCards.children, function (card) {
      var is = card.dataset.provider === active;
      card.classList.toggle('active-card', is);
      var head = card.querySelector('.card-head');
      if (!head) return;
      var b = head.querySelector('.badge.active');
      if (is && !b) {
        b = el('span', 'badge active', 'active');
        head.appendChild(b);
      } else if (!is && b) {
        b.parentNode.removeChild(b);
      }
    });
  }

  function renderSettings() {
    clearChildren(dom.providerCards);
    var providers = Array.isArray(state.settings.providers) ? state.settings.providers : [];
    providers.forEach(function (p) {
      dom.providerCards.appendChild(buildProviderCard(p));
    });
    updateActiveStrip();
  }

  function buildProviderCard(p) {
    var name = String(p.name || '');
    var isCustom = name === 'openai-compatible';
    var card = el('section', 'card provider-card' + (activeProvider() === name ? ' active-card' : ''));
    card.dataset.provider = name;

    /* header: name + configured badge (+ active badge) */
    var head = el('div', 'card-head');
    head.appendChild(el('h3', 'card-title', name));
    var badge = el('span', 'badge ' + (p.configured ? 'ok' : 'off'), p.configured ? 'configured ✓' : 'not configured');
    head.appendChild(badge);
    if (activeProvider() === name) head.appendChild(el('span', 'badge active', 'active'));
    card.appendChild(head);

    /* base_url display (from the server — never hardcoded here) */
    var url = el('p', 'base-url');
    url.appendChild(document.createTextNode('base_url: '));
    url.appendChild(el('code', null, p.base_url ? String(p.base_url) : '—'));
    card.appendChild(url);

    var cardMsg = el('p', 'card-msg');
    cardMsg.hidden = true;

    /* --- API key row: save / remove / test --- */
    var keyRow = el('div', 'key-row');
    var keyLabel = el('label', 'lbl', 'API key');
    keyLabel.htmlFor = 'key-' + name;
    var keyInput = el('input', 'text-input key-input');
    keyInput.type = 'password';
    keyInput.id = 'key-' + name;
    keyInput.setAttribute('autocomplete', 'new-password');
    keyInput.placeholder = p.configured ? '•••• saved' : 'paste API key';
    var savedMsg = el('span', 'saved-msg', 'saved ✓');
    savedMsg.hidden = true;

    var saveBtn = el('button', 'btn', 'Save key');
    saveBtn.type = 'button';
    var removeBtn = el('button', 'btn danger', 'Remove key');
    removeBtn.type = 'button';
    removeBtn.disabled = !p.configured;
    var testBtn = el('button', 'btn', 'Test');
    testBtn.type = 'button';

    keyRow.appendChild(keyLabel);
    keyRow.appendChild(keyInput);
    keyRow.appendChild(saveBtn);
    keyRow.appendChild(removeBtn);
    keyRow.appendChild(testBtn);
    keyRow.appendChild(savedMsg);
    card.appendChild(keyRow);
    card.appendChild(cardMsg);

    saveBtn.addEventListener('click', function () {
      cardMsg.hidden = true;
      var key = keyInput.value;
      if (!key.trim()) {
        showInline(cardMsg, 'Type a key first — it is stored in the OS keychain, never shown again.', true);
        return;
      }
      saveBtn.disabled = true;
      api('/api/settings/providers/' + encodeURIComponent(name) + '/key', { method: 'PUT', body: { key: key } })
        .then(function () {
          p.configured = true;
          badge.className = 'badge ok';
          badge.textContent = 'configured ✓';
          keyInput.value = '';
          keyInput.placeholder = '•••• saved';
          removeBtn.disabled = false;
          flash(savedMsg);
        })
        .catch(function (err) { handleFetchError(err, cardMsg); })
        .then(function () { saveBtn.disabled = false; });
    });

    removeBtn.addEventListener('click', function () {
      cardMsg.hidden = true;
      removeBtn.disabled = true;
      api('/api/settings/providers/' + encodeURIComponent(name) + '/key', { method: 'DELETE' })
        .then(function () {
          p.configured = false;
          badge.className = 'badge off';
          badge.textContent = 'not configured';
          keyInput.placeholder = 'paste API key';
        })
        .catch(function (err) {
          handleFetchError(err, cardMsg);
          removeBtn.disabled = !p.configured;
        });
    });

    var testResult = el('p', 'test-result');
    testResult.hidden = true;
    testBtn.addEventListener('click', function () {
      testResult.hidden = true;
      testBtn.disabled = true;
      testBtn.textContent = 'Testing…';
      api('/api/settings/providers/' + encodeURIComponent(name) + '/test', { method: 'POST' })
        .then(function (res) {
          if (res && res.ok) {
            var count = Array.isArray(res.models) ? res.models.length : 0;
            showInline(testResult, 'ok — ' + count + (count === 1 ? ' model available' : ' models available'), false);
          } else {
            showInline(testResult, (res && res.error) ? String(res.error) : 'Test failed.', true);
          }
        })
        .catch(function (err) {
          showInline(testResult, err.message || 'Test failed.', true);
        })
        .then(function () {
          testBtn.disabled = false;
          testBtn.textContent = 'Test';
        });
    });
    card.appendChild(testResult);

    /* --- model picker + Set active --- */
    var modelRow = el('div', 'model-row');
    var modelLabel = el('label', 'lbl', 'Model');
    modelLabel.htmlFor = 'model-' + name;
    var select = el('select', 'text-input select');
    select.id = 'model-' + name;
    select.dataset.provider = name;
    modelRow.appendChild(modelLabel);
    modelRow.appendChild(select);

    if (name === 'ollama') {
      var refreshBtn = el('button', 'btn', 'Refresh');
      refreshBtn.type = 'button';
      refreshBtn.setAttribute('aria-label', 'Refresh model list');
      refreshBtn.addEventListener('click', function () { refreshModels(name, cardMsg); });
      modelRow.appendChild(refreshBtn);
    }

    var activeSaved = el('span', 'saved-msg', 'saved ✓');
    activeSaved.hidden = true;
    var activeBtn = el('button', 'btn primary', 'Set active');
    activeBtn.type = 'button';
    modelRow.appendChild(activeBtn);
    modelRow.appendChild(activeSaved);
    card.appendChild(modelRow);

    activeBtn.addEventListener('click', function () {
      cardMsg.hidden = true;
      var model = select.value;
      if (!model) {
        showInline(cardMsg, 'No models available yet — save a key, then use Test (or Refresh for Ollama).', true);
        return;
      }
      activeBtn.disabled = true;
      api('/api/settings/active', { method: 'PUT', body: { provider: name, model: model } })
        .then(function (res) {
          state.settings.active = (res && res.active) ? res.active : { provider: name, model: model };
          if (isCustom) {
            // keep the typed base_url as a local note only (no key implication;
            // the frozen contract has no endpoint that stores it server-side)
            try { localStorage.setItem('echo.openai_compatible_base_url', customUrlInput.value.trim()); } catch (err) { /* private mode */ }
            applyCustomNoteMode(true);
          }
          updateActiveStrip();
          updateActiveBadges();
          flash(activeSaved);
        })
        .catch(function (err) { handleFetchError(err, cardMsg); })
        .then(function () { activeBtn.disabled = false; });
    });

    /* --- custom base_url note (openai-compatible only) --- */
    var customUrlInput = null;
    var customEditBtn = null;
    if (isCustom) {
      var noteRow = el('div', 'key-row');
      var noteLabel = el('label', 'lbl', 'Base URL');
      noteLabel.htmlFor = 'baseurl-openai-compatible';
      customUrlInput = el('input', 'text-input baseurl-input');
      customUrlInput.type = 'text';
      customUrlInput.id = 'baseurl-openai-compatible';
      customUrlInput.placeholder = 'server URL, e.g. host:8000/v1';
      var stored = '';
      try { stored = localStorage.getItem('echo.openai_compatible_base_url') || ''; } catch (err) { stored = ''; }
      customUrlInput.value = p.base_url ? String(p.base_url) : stored;
      customUrlInput.addEventListener('change', function () {
        try { localStorage.setItem('echo.openai_compatible_base_url', customUrlInput.value.trim()); } catch (err) { /* private mode */ }
      });
      customEditBtn = el('button', 'btn tiny', 'Edit');
      customEditBtn.type = 'button';
      customEditBtn.addEventListener('click', function () {
        applyCustomNoteMode(false);
        customUrlInput.focus();
      });
      noteRow.appendChild(noteLabel);
      noteRow.appendChild(customUrlInput);
      noteRow.appendChild(customEditBtn);
      card.appendChild(noteRow);
      card.appendChild(el('p', 'hint', 'Note only — kept in this browser. The current API contract has no endpoint to store a custom base_url server-side.'));
    }

    function applyCustomNoteMode(locked) {
      if (!customUrlInput || !customEditBtn) return;
      customUrlInput.disabled = locked;
      customEditBtn.hidden = !locked;
    }
    if (isCustom) {
      var wasCurrent = activeProvider() === 'openai-compatible';
      var hasValue = !!(p.base_url || stored);
      applyCustomNoteMode(wasCurrent && hasValue);
    }

    return card;
  }

  function updateModelSelect(name) {
    var sel = null;
    var selects = dom.providerCards.querySelectorAll('select[data-provider="' + (window.CSS && CSS.escape ? CSS.escape(name) : name) + '"]');
    if (selects.length) sel = selects[0];
    if (!sel) return;

    var models = state.modelCache[name] || [];
    var active = state.settings && state.settings.active ? state.settings.active : null;
    var desired = sel.value || (active && active.provider === name && active.model ? active.model : '');
    clearChildren(sel);

    if (!models.length) {
      var none = el('option', null, 'no models found');
      none.value = '';
      sel.appendChild(none);
      sel.disabled = true;
      return;
    }
    sel.disabled = false;
    models.forEach(function (m) {
      var o = el('option', null, String(m));
      o.value = String(m);
      sel.appendChild(o);
    });
    var matched = false;
    Array.prototype.forEach.call(sel.options, function (o) { if (o.value === desired) matched = true; });
    if (desired && !matched) {
      var extra = el('option', null, desired);
      extra.value = desired;
      sel.appendChild(extra);
    }
    if (desired) sel.value = desired;
  }

  function refreshModels(name, inlineNode) {
    return api('/api/providers/available?provider=' + encodeURIComponent(name)).then(function (data) {
      state.modelCache[name] = (data && Array.isArray(data.models)) ? data.models.map(String) : [];
    }).catch(function (err) {
      state.modelCache[name] = [];
      if (inlineNode) showInline(inlineNode, err.message || 'Could not fetch models.', !err.unreachable);
    }).then(function () {
      updateModelSelect(name);
    });
  }

  /* --- failover priority: up/down arrows (dependency-free) --- */

  function swap(arr, i, d) {
    var j = i + d;
    if (j < 0 || j >= arr.length) return;
    var t = arr[i]; arr[i] = arr[j]; arr[j] = t;
  }

  function renderFailover() {
    clearChildren(dom.failoverList);
    var order = state.failoverOrder.slice();
    order.forEach(function (name, i) {
      var li = el('li', 'failover-row');
      li.appendChild(el('span', 'fo-name', name));
      var ctrls = el('span', 'fo-ctrls');
      var up = el('button', 'btn tiny', '↑');
      up.type = 'button';
      up.setAttribute('aria-label', 'Move ' + name + ' up');
      up.disabled = i === 0;
      var down = el('button', 'btn tiny', '↓');
      down.type = 'button';
      down.setAttribute('aria-label', 'Move ' + name + ' down');
      down.disabled = i === order.length - 1;
      up.addEventListener('click', function () {
        swap(state.failoverOrder, i, -1);
        renderFailover();
      });
      down.addEventListener('click', function () {
        swap(state.failoverOrder, i, 1);
        renderFailover();
      });
      ctrls.appendChild(up);
      ctrls.appendChild(down);
      li.appendChild(ctrls);
      dom.failoverList.appendChild(li);
    });
  }

  function saveFailoverOrder() {
    api('/api/settings/failover', { method: 'PUT', body: { order: state.failoverOrder } })
      .then(function () {
        flash(dom.failoverSaved);
        return loadSettingsView(); // pick up the canonical order from the server
      })
      .catch(function (err) { handleFetchError(err, dom.settingsError); });
  }

  /* ---------- wiring ---------- */

  function init() {
    dom.sidebarToggle.addEventListener('click', function () {
      if (document.body.classList.contains('sidebar-open')) closeSidebar();
      else openSidebar();
    });
    dom.backdrop.addEventListener('click', closeSidebar);
    dom.newChat.addEventListener('click', newChat);

    dom.composer.addEventListener('submit', function (ev) {
      ev.preventDefault();
      sendMessage();
    });
    dom.input.addEventListener('input', function () {
      autoGrow();
      updateComposer();
    });
    dom.input.addEventListener('keydown', function (ev) {
      if (ev.key === 'Enter' && !ev.shiftKey) {
        ev.preventDefault();
        sendMessage();
      }
    });
    dom.interrupt.addEventListener('click', onInterruptClick);

    dom.errorDismiss.addEventListener('click', hideErrorBar);
    dom.bannerDismiss.addEventListener('click', function () { dom.serverBanner.hidden = true; });
    dom.bannerRetry.addEventListener('click', function () {
      dom.serverBanner.hidden = true;
      hideErrorBar();
      refreshSessions();
      connectWS();
      if (state.view === 'settings') loadSettingsView();
    });
    document.addEventListener('keydown', function (ev) {
      if (ev.key === 'Escape') hideErrorBar();
    });

    dom.saveFailover.addEventListener('click', saveFailoverOrder);

    window.addEventListener('hashchange', applyRoute);
    if (!location.hash) history.replaceState(null, '', '#chat'); // view persists across reload via hash

    applyRoute();
    autoGrow();
    updateComposer();
    connectWS();
    refreshSessions();
  }

  init();
})();
