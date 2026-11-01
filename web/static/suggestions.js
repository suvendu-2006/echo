/* Echo suggestions (Phase 9) — hero chips fill the composer in one tap. */
(function () {
  'use strict';
  document.addEventListener('click', function (ev) {
    var chip = ev.target.closest('.chip');
    if (!chip) return;
    var input = document.getElementById('msg-input');
    if (!input) return;
    input.value = chip.dataset.fill || chip.textContent;
    input.focus();
    input.dispatchEvent(new Event('input', { bubbles: true }));
  });
})();
