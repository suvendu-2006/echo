/* Echo login (Phase 9) — the gate for cloud access. Apple-styled, minimal. */
(function () {
  'use strict';

  var form = document.getElementById('login-form');
  var password = document.getElementById('password');
  var error = document.getElementById('login-error');
  var button = document.getElementById('login-go');

  function showError(text) {
    error.textContent = text;
    error.hidden = false;
    password.focus();
  }

  form.addEventListener('submit', async function (ev) {
    ev.preventDefault();
    error.hidden = true;
    button.disabled = true;
    button.textContent = 'Checking…';
    try {
      var response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ password: password.value })
      });
      var data = await response.json().catch(function () { return {}; });
      if (response.ok && data.ok) {
        location.href = '/';
        return;
      }
      showError(data.error || 'Login failed — try again.');
    } catch (err) {
      showError('Cannot reach Echo — is the server running?');
    }
    button.disabled = false;
    button.textContent = 'Unlock Echo';
  });

  // Setup mode: Echo has no password yet — the first one creates it.
  fetch('/api/auth/status').then(function (r) { return r.json(); }).then(function (d) {
    if (d.setup) {
      document.getElementById('login-title').textContent = 'Create your password';
      var confirmWrap = document.getElementById('confirm-wrap');
      confirmWrap.hidden = false;
      form.addEventListener('submit', function (ev) {
        if (password.value !== document.getElementById('confirm').value) {
          ev.preventDefault();
          ev.stopPropagation();
          showError('Passwords do not match.');
        }
      }, true); // capture: validate before the submit handler posts
      button.textContent = 'Create password';
    }
  }).catch(function () {});
})();
