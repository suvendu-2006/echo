/* Echo settings rail (Phase 9 UI adoption) — Atticus-style section switching.
   Self-contained, no dependencies; plain display toggling + hash deep-links. */
(function () {
  'use strict';

  function activate(sectionId) {
    var view = document.getElementById('view-settings');
    if (!view) return;
    var navs = view.querySelectorAll('.snav');
    var sections = view.querySelectorAll('.ssection');
    navs.forEach(function (btn) {
      btn.classList.toggle('on', btn.dataset.section === sectionId);
    });
    sections.forEach(function (sec) {
      var on = sec.id === sectionId;
      sec.hidden = !on;
      sec.classList.toggle('on', on);
    });
    window.dispatchEvent(new CustomEvent('echo:section', { detail: sectionId }));
  }

  document.addEventListener('click', function (ev) {
    var btn = ev.target.closest('.snav');
    if (!btn) return;
    var id = btn.dataset.section;
    activate(id);
    if (location.hash === '#settings') {
      history.replaceState(null, '', '#settings/' + id.replace('sec-', ''));
    }
  });

  function mount() {
    // deep link: #settings/memory → sec-memory
    var sub = (location.hash.match(/^#settings\/(\w+)/) || [])[1];
    activate(sub ? 'sec-' + sub : 'sec-providers');
  }

  window.addEventListener('hashchange', function () {
    if (location.hash.indexOf('#settings') === 0) mount();
  });
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', mount);
  } else {
    mount();
  }
})();
