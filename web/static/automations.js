/* Echo automations (Phase 7.3) — Settings → "Automations" (the night shift).
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

  function row(job, refresh) {
    var item = el('div', 'memory-row');
    item.appendChild(el('span', 'memory-text',
      (job.enabled === false ? '⏸ ' : '▶ ') + (job.name || '') +
      ' — ' + (job.schedule || '') + ' → ' + String(job.task || '').slice(0, 60)));
    var run = el('button', 'memory-btn');
    run.textContent = 'Run now';
    run.addEventListener('click', async function () {
      setNote('Running "' + job.name + '"… (a full agent turn — this can take a while)');
      try {
        var result = await api('/api/automations/' + encodeURIComponent(job.name) + '/run-now', { method: 'POST' });
        setNote('"' + job.name + '" finished: ' + String(result.result || '').slice(0, 140));
      } catch (err) { setNote('Run failed: ' + err.message); }
      refresh();
    });
    var pause = el('button', 'memory-btn');
    pause.textContent = job.enabled === false ? 'Resume' : 'Pause';
    pause.addEventListener('click', async function () {
      var action = job.enabled === false ? 'resume' : 'pause';
      await api('/api/automations/' + encodeURIComponent(job.name) + '/' + action, { method: 'POST' }).catch(function () {});
      refresh();
    });
    var del = el('button', 'memory-btn danger');
    del.textContent = 'Delete';
    del.addEventListener('click', async function () {
      await api('/api/automations/' + encodeURIComponent(job.name), { method: 'DELETE' }).catch(function () {});
      refresh();
    });
    item.appendChild(run);
    item.appendChild(pause);
    item.appendChild(del);
    return item;
  }

  function setNote(text) {
    if (listEl) {
      listEl.textContent = '';
      listEl.appendChild(el('p', 'memory-empty', text));
      setTimeout(refresh, 1500);
    }
  }

  async function refresh() {
    if (!listEl) return;
    try {
      var data = await api('/api/automations');
      listEl.textContent = '';
      if (!data.automations || data.automations.length === 0) {
        listEl.appendChild(el('p', 'memory-empty',
          'No automations yet. Example: name "Morning briefing", schedule "daily 08:00", task "brief me on my emails and today\'s calendar".'));
        return;
      }
      data.automations.forEach(function (job) {
        listEl.appendChild(row(job, refresh));
      });
    } catch (err) {
      listEl.textContent = '';
      listEl.appendChild(el('p', 'memory-empty', 'Automations unavailable: ' + err.message));
    }
  }

  function buildCard() {
    var card = el('div', 'memory-card');
    card.appendChild(el('h3', null, 'Automations'));
    card.appendChild(el('p', 'memory-note',
      'Scheduled agent turns — the 8 AM briefing, price watchers. Echo runs them through the full harness (tools, memory, voice) and notifies you. Schedule formats: "daily 08:00" · "weekly MON 09:00" · "every 30 minutes".'));
    var nameEl = el('input', 'memory-search');
    nameEl.placeholder = 'Name (e.g. Morning briefing)';
    var schedEl = el('input', 'memory-search');
    schedEl.placeholder = 'Schedule (daily 08:00)';
    var taskEl = el('input', 'memory-search');
    taskEl.placeholder = 'Task — what Echo should do';
    var add = el('button', null);
    add.textContent = 'Add automation';
    add.className = 'primary';
    add.addEventListener('click', async function () {
      try {
        await api('/api/automations', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ title: nameEl.value, schedule: schedEl.value, task: taskEl.value })
        });
        nameEl.value = schedEl.value = taskEl.value = '';
        refresh();
      } catch (err) {
        setNote('Add failed: ' + err.message);
      }
    });
    card.appendChild(nameEl);
    card.appendChild(schedEl);
    card.appendChild(taskEl);
    card.appendChild(add);
    listEl = el('div', 'memory-list');
    card.appendChild(listEl);
    return card;
  }

  function mount() {
    var view = document.getElementById((document.currentScript && document.currentScript.dataset.mount) || 'view-settings');
    if (!view || view.querySelector('.automations-mount')) return;
    var mountPoint = el('div', 'automations-mount');
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
