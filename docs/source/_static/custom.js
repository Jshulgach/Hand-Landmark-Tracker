(function () {
  function activatePane(root, target) {
    const panes = root.querySelectorAll('.ht-demo-pane');
    const buttons = root.querySelectorAll('[data-target]');

    panes.forEach((pane) => {
      pane.classList.toggle('is-active', pane.getAttribute('data-name') === target);
    });

    buttons.forEach((button) => {
      button.classList.toggle('is-active', button.getAttribute('data-target') === target);
    });
  }

  function initSwitchers() {
    const roots = document.querySelectorAll('[data-ht-switcher]');
    roots.forEach((root) => {
      const firstButton = root.querySelector('[data-target]');
      if (firstButton) {
        activatePane(root, firstButton.getAttribute('data-target'));
      }

      root.addEventListener('click', (event) => {
        const button = event.target.closest('[data-target]');
        if (!button) {
          return;
        }
        activatePane(root, button.getAttribute('data-target'));
      });
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initSwitchers);
  } else {
    initSwitchers();
  }
})();
