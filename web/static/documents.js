/* Echo document library (Phase 6.5 RAG) — Settings → "Document library".
   Self-contained: createElement/textContent only (SEC-2), zero dependencies. */
(function () {
  'use strict';

  var listEl = null;
  var pathEl = null;

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
      var data = await api('/api/rag/documents');
      listEl.textContent = '';
      if (!data.documents || data.documents.length === 0) {
        listEl.appendChild(el('p', 'memory-empty',
          'No documents yet. Ingest a file or folder (txt, md, html, pdf) and Echo can search and cite it.'));
        return;
      }
      data.documents.forEach(function (doc) {
        var item = el('div', 'memory-row');
        item.appendChild(el('span', 'memory-text', (doc.name || doc.doc_id) + ' — ' + doc.n_chunks + ' chunks'));
        var del = el('button', 'memory-btn danger');
        del.textContent = 'Remove';
        del.addEventListener('click', async function () {
          await api('/api/rag/documents/' + doc.doc_id, { method: 'DELETE' }).catch(function () {});
          refresh();
        });
        item.appendChild(del);
        listEl.appendChild(item);
      });
    } catch (err) {
      listEl.textContent = '';
      listEl.appendChild(el('p', 'memory-empty', 'Document library unavailable: ' + err.message));
    }
  }

  async function ingest(target) {
    if (!target || !target.trim()) return;
    setBusy('Ingesting… (large folders can take a while)');
    try {
      var result = await api('/api/rag/ingest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: target.trim() })
      });
      setBusy('Ingested ' + (result.documents ?? 1) + ' document(s), ' +
        (result.chunks ?? result.n_chunks ?? 0) + ' chunks.');
    } catch (err) {
      setBusy('Ingest failed: ' + err.message);
    }
    refresh();
  }

  function setBusy(text) {
    if (listEl) {
      var note = el('p', 'memory-empty', text);
      listEl.textContent = '';
      listEl.appendChild(note);
    }
  }

  function buildCard() {
    var card = el('div', 'memory-card');
    card.appendChild(el('h3', null, 'Document library'));
    card.appendChild(el('p', 'memory-note',
      'Ingest files or whole folders — Echo chunks them, indexes them twice (keyword + semantic) and cites them as [S1], [S2]… in answers. All local.'));
    pathEl = el('input', 'memory-search');
    pathEl.type = 'text';
    pathEl.placeholder = 'C:\\path\\to\\file.pdf or a whole folder';
    var row = el('div', 'permission-row');
    var go = el('button', null);
    go.textContent = 'Ingest';
    go.className = 'primary';
    go.addEventListener('click', function () { ingest(pathEl.value); });
    row.appendChild(go);
    card.appendChild(pathEl);
    card.appendChild(row);
    listEl = el('div', 'memory-list');
    card.appendChild(listEl);
    return card;
  }

  function mount() {
    var view = document.getElementById((document.currentScript && document.currentScript.dataset.mount) || 'view-settings');
    if (!view) { view = document.getElementById('view-settings'); }
    if (!view || view.querySelector('.rag-card-mount')) return;
    var mountPoint = el('div', 'rag-card-mount');
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
