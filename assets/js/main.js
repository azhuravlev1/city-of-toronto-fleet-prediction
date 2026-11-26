(function () {
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* ---------- Figure 6: walk-forward validation diagram ---------- */
  const walk = document.getElementById('walk');
  if (walk) {
    const W = 900, rowH = 34, top = 30, L = 80, R = 20;
    const months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'];
    const x = (m) => L + (m / 12) * (W - L - R);
    // five 60-day test slices ending at year end (approximate, for illustration)
    const testStarts = [2, 4, 6, 8, 10];
    let s = `<svg viewBox="0 0 ${W} ${top + 5 * rowH + 40}" xmlns="http://www.w3.org/2000/svg">`;
    months.forEach((m, i) => { s += `<text x="${x(i + 0.5)}" y="18" font-size="12" text-anchor="middle" fill="#5A5F66">${m}</text>`; });
    testStarts.forEach((t, i) => {
      const y = top + i * rowH;
      s += `<text x="10" y="${y + 21}" font-size="13" font-weight="600" fill="#23262B">FOLD ${i + 1}</text>`;
      s += `<rect class="tr" x="${x(0)}" y="${y + 6}" width="${x(t) - x(0) - 3}" height="22" rx="4" fill="#9AA0A8" style="--d:${i * 120}ms"/>`;
      s += `<rect class="te" x="${x(t)}" y="${y + 6}" width="${x(t + 2) - x(t)}" height="22" rx="4" fill="#FFC20E" stroke="#1F2226" stroke-width="2" style="--d:${i * 120 + 200}ms"/>`;
    });
    const ly = top + 5 * rowH + 22;
    s += `<rect x="${L}" y="${ly - 11}" width="16" height="14" rx="3" fill="#9AA0A8"/><text x="${L + 22}" y="${ly}" font-size="13" fill="#23262B">TRAIN: everything before the cut-off</text>`;
    s += `<rect x="${L + 330}" y="${ly - 11}" width="16" height="14" rx="3" fill="#FFC20E" stroke="#1F2226" stroke-width="2"/><text x="${L + 352}" y="${ly}" font-size="13" fill="#23262B">TEST: the next ~60 days</text>`;
    s += '</svg>';
    walk.innerHTML = s;
  }

  /* ---------- Results: baseline vs gradient boosting ---------- */
  const SCORES = [
    { name: 'ROC-AUC', dir: 'higher is better', max: 0.75, base: [0.637, 0.022], gbm: [0.609, 0.016], better: 'high' },
    { name: 'PR-AUC', dir: 'higher is better', max: 0.2, base: [0.145, 0.017], gbm: [0.149, 0.008], better: 'high' },
    { name: 'Brier', dir: 'lower is better', max: 0.12, base: [0.091, 0.003], gbm: [0.094, 0.005], better: 'low' },
  ];
  const scores = document.getElementById('scores');
  if (scores) {
    scores.innerHTML = SCORES.map((m) => {
      const row = (lbl, cls, [v, sd]) => {
        const w = (v / m.max) * 100, a = ((v - sd) / m.max) * 100, b = ((v + sd) / m.max) * 100;
        return `<div class="srow"><span class="lbl">${lbl}</span><span class="strack"><span class="sfill ${cls}" data-w="${w}"></span><span class="serr" style="left:${a}%;width:${b - a}%"></span></span><span class="sval">${v.toFixed(3)}<small> ±${sd.toFixed(3)}</small></span></div>`;
      };
      const baseWins = m.better === 'high' ? m.base[0] > m.gbm[0] : m.base[0] < m.gbm[0];
      return `<div class="score"><h3>${m.name}</h3><span class="dir">${m.dir}</span>${row('Baseline', 'base', m.base)}${row('GBM', 'gbm', m.gbm)}<span class="win">BETTER: ${baseWins ? 'BASELINE' : 'GBM (SLIGHTLY)'}</span></div>`;
    }).join('');
  }

  /* ---------- reveal, checklist ticks, bar fills ---------- */
  const items = document.querySelectorAll('.item');
  if (reduceMotion || !('IntersectionObserver' in window)) {
    items.forEach((i) => i.classList.add('done'));
    document.querySelectorAll('.sfill').forEach((f) => { f.style.width = f.dataset.w + '%'; });
    if (scores) scores.classList.add('show');
    return;
  }
  items.forEach((i) => i.classList.add('reveal'));
  const io = new IntersectionObserver((es) => es.forEach((e) => {
    if (!e.isIntersecting) return;
    e.target.classList.add('in');
    setTimeout(() => e.target.classList.add('done'), 300);
    io.unobserve(e.target);
  }), { rootMargin: '0px 0px -12% 0px' });
  items.forEach((i) => io.observe(i));

  if (scores) {
    const sio = new IntersectionObserver((es) => {
      if (!es.some((e) => e.isIntersecting)) return;
      scores.querySelectorAll('.sfill').forEach((f, k) => setTimeout(() => { f.style.width = f.dataset.w + '%'; }, k * 120));
      scores.classList.add('show');
      sio.disconnect();
    }, { threshold: 0.3 });
    sio.observe(scores);
  }

  const walkSvg = walk && walk.querySelector('svg');
  if (walkSvg) {
    walkSvg.querySelectorAll('rect.tr, rect.te').forEach((r) => {
      r.style.transformOrigin = 'left center';
      r.style.transformBox = 'fill-box';
      r.style.transform = 'scaleX(0)';
      r.style.transition = `transform 0.7s cubic-bezier(0.22,1,0.36,1) ${r.style.getPropertyValue('--d')}`;
    });
    const wio = new IntersectionObserver((es) => {
      if (!es.some((e) => e.isIntersecting)) return;
      walkSvg.querySelectorAll('rect.tr, rect.te').forEach((r) => { r.style.transform = 'scaleX(1)'; });
      wio.disconnect();
    }, { threshold: 0.4 });
    wio.observe(walk);
  }
})();
