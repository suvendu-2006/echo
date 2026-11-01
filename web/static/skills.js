/* Echo skills (Phase 7.4) — Settings → "Skills" (tools Echo forged itself).
   Self-contained: createElement/textContent only (SEC-2), zero dependencies. */
(function () {
  'use strict';

  var listEl = null;

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

  async function refresh() {
    if (!listEl) return;
    try {
      var data = await api('/api/skills');
      listEl.textContent = '';
      if (!data.skills || data.skills.length === 0) {
        listEl.appendChild(el('p', 'memory-empty',
          'No learned skills yet. Ask Echo to script something (it will show you the code first), then tell it to save it as a skill.'));
        return;
      }
      data.skills.forEach(function (skill) {
        var item = el('div', 'memory-row');
        item.appendChild(el('span', 'memory-text',
          (skill.enabled ? '✅ ' : '⏸ ') + (skill.name || '') +
          ' — ' + String(skill.description || '').slice(0, 70) +
          ' (runs: ' + (skill.runs || 0) + ')'));
        var toggle = el('button', 'memory-btn');
        toggle.textContent = skill.enabled ? 'Disable' : 'Enable';
        toggle.addEventListener('click', async function () {
          await api('/api/skills/' + encodeURIComponent(skill.name) + '/' +
            (skill.enabled ? 'disable' : 'enable'), { method: 'POST' }).catch(function () {});
          refresh();
        });
        var del = el('button', 'memory-btn danger');
        del.textContent = 'Forget';
        del.addEventListener('click', async function () {
          await api('/api/skills/' + encodeURIComponent(skill.name), { method: 'DELETE' }).catch(function () {});
          refresh();
        });
        item.appendChild(toggle);
        item.appendChild(del);
        listEl.appendChild(item);
      });
    } catch (err) {
      listEl.textContent = '';
      listEl.appendChild(el('p', 'memory-empty', 'Skills unavailable: ' + err.message));
    }
  }

  function buildCard() {
    var card = el('div', 'memory-card');
    card.appendChild(el('h3', null, 'Skills'));
    card.appendChild(el('p', 'memory-note',
      'Scripts Echo wrote for you and saved as reusable tools. Every run shows you the code and asks permission first.'));
    listEl = el('div', 'memory-list');
    card.appendChild(listEl);
    return card;
  }

  function mount() {
    var view = document.getElementById((document.currentScript && document.currentScript.dataset.mount) || 'view-settings');
    if (!view) { view = document.getElementById('view-settings'); }
    if (!view || view.querySelector('.skills-mount')) return;
    var mountPoint = el('div', 'skills-mount');
    mountPoint.appendChild(buildCard());
    view.appendChild(mountPoint);
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
