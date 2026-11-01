/* Echo memory manager (Phase 6.4) — Settings → "What Echo remembers".
   Self-contained: createElement/textContent only (SEC-2), zero dependencies. */
(function () {
  'use strict';

  var listEl = null;
  var searchEl = null;

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined && text !== null) n.textContent = text;
    return n;
  }

  async function api(path, options) {
    var response = await fetch(path, options);
    if (!response.ok) {
      var detail = '';
      try { detail = (await response.json()).detail || ''; } catch (e) {}
      throw new Error(detail || ('HTTP ' + response.status));
    }
    if (response.status === 204) return null;
    return response.json();
  }

  function row(mem, refresh) {
    var item = el('div', 'memory-row');
    var text = el('span', 'memory-text', (mem.kind || 'fact') + ': ' + (mem.text || ''));
    if (mem.in_wallet) {
      var badge = el('span', 'memory-badge', 'core');
      item.appendChild(badge);
    }
    item.appendChild(text);
    var pin = el('button', 'memory-btn');
    pin.textContent = mem.in_wallet ? 'Unpin' : 'Pin';
    pin.addEventListener('click', async function () {
      if (mem.in_wallet) {
        // Unpin = demote to archival (importance drops out of the wallet) —
        // the memory itself is NEVER deleted by pinning controls.
        await api('/api/memory/' + mem.id + '/demote', { method: 'POST' })
          .catch(function () {});
        refresh();
        return;
      }
      await api('/api/memory/' + mem.id + '/core', { method: 'POST' }).catch(function () {});
      refresh();
    });
    var edit = el('button', 'memory-btn');
    edit.textContent = 'Edit';
    edit.addEventListener('click', async function () {
      var updated = window.prompt('Edit this memory:', mem.text || '');
      if (updated === null || !updated.trim()) return;
      await api('/api/memory/' + mem.id, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: updated.trim() })
      });
      refresh();
    });
    var del = el('button', 'memory-btn danger');
    del.textContent = 'Forget';
    del.addEventListener('click', async function () {
      await api('/api/memory/' + mem.id, { method: 'DELETE' }).catch(function () {});
      refresh();
    });
    item.appendChild(pin);
    item.appendChild(edit);
    item.appendChild(del);
    return item;
  }

  async function refresh() {
    if (!listEl) return;
    try {
      var q = searchEl ? searchEl.value.trim() : '';
      var data = await api('/api/memory' + (q ? '?q=' + encodeURIComponent(q) : ''));
      listEl.textContent = '';
      if (!data.memories || data.memories.length === 0) {
        listEl.appendChild(el('p', 'memory-empty',
          'Nothing remembered yet — chat with Echo and it will remember what matters.'));
        return;
      }
      data.memories.forEach(function (mem) {
        listEl.appendChild(row(mem, refresh));
      });
    } catch (err) {
      listEl.textContent = '';
      listEl.appendChild(el('p', 'memory-empty', 'Memory unavailable: ' + err.message));
    }
  }

  function buildCard() {
    var card = el('div', 'memory-card');
    var heading = el('h3', null, 'What Echo remembers');
    var note = el('p', 'memory-note',
      'Echo files facts, preferences and decisions from your chats. Pin the ones that should always be in its head; everything else is searched when needed. Nothing leaves this machine.');
    searchEl = el('input', 'memory-search');
    searchEl.type = 'search';
    searchEl.placeholder = 'Search memories…';
    searchEl.addEventListener('input', function () { refresh(); });
    listEl = el('div', 'memory-list');
    card.appendChild(heading);
    card.appendChild(note);
    card.appendChild(searchEl);
    card.appendChild(listEl);
    return card;
  }

  function mount() {
    var view = document.getElementById((document.currentScript && document.currentScript.dataset.mount) || 'view-settings');
    if (!view) { view = document.getElementById('view-settings'); }
    if (!view || view.querySelector('.memory-card')) return;
    view.appendChild(buildCard());
    refresh();
  }

  window.addEventListener('echo:section', function () { if (location.hash.indexOf('#settings') === 0) refresh(); });
  window.addEventListener('hashchange', function () {
    if (location.hash === '#settings') mount();
  });
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () {
      if (location.hash === '#settings') mount();
    });
  } else if (location.hash === '#settings') {
    mount();
  }
})();
