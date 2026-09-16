/* T090 (C5) -- the RENDERED measurement of the CONSCIOUSNESS plate (2.2, 2.3 (v1)/(v2)).

   A BROWSER SCRIPT. IT IS NEVER RUN BY CI AND NO TEST IMPORTS IT: a browser is not CI, and the
   thing it measures -- where a character actually lands after layout -- has no representation in
   the source that a pytest could read. Paste it into the live cockpit page's console (or run it
   through the built-in pane's javascript_tool) at the pane's own width and again at an emulated
   1920 x 1080. Its printed output is the arc's record 3.x. tests/test_conclusion_plate.py carries
   what CI CAN see, and says in its own docstring that it is a ratchet of source text (obs #2195).

   Read-only: no click, no fetch, no mutation. `<details>` is measured in whatever state it is in
   and is never opened by this script -- 2.5 makes the collapsed state a control, because the
   sentences must be in view WITHOUT expanding anything.

   Extends scratchpad/t082-panel/refute/measure_access_plate.js: the ACCESS census below is that
   script's shape, kept so the BEFORE numbers (fact 2: 0 visible sentence characters on every
   charged row, at both widths) and the AFTER are read by the same instrument.

   The barred quantity is `insideChars` vs `chars` per sentence: EVERY character inside the plate's
   visible box, at both widths, for every charged row. One clipped character is NOT MET. */
(() => {
  const round = (r) => ({w: Math.round(r.width), h: Math.round(r.height),
                         l: Math.round(r.left), t: Math.round(r.top),
                         r: Math.round(r.right), b: Math.round(r.bottom)});
  const inside = (r, p) => r.left >= p.left && r.right <= p.right &&
                           r.top >= p.top && r.bottom <= p.bottom;

  /* Per-CHARACTER census of one text node against the plate's visible box. A box test on the
     whole range is not the same measurement: a range that starts inside and ends outside still
     reports a rect, and half a sentence is not the sentence. */
  function charCensus(node, p) {
    if (!node || node.nodeType !== 3) return {chars: 0, insideChars: 0, missing: true};
    const rng = document.createRange();
    let vis = 0, firstOut = null;
    for (let i = 0; i < node.length; i++) {
      rng.setStart(node, i); rng.setEnd(node, i + 1);
      const rr = rng.getBoundingClientRect();
      if (rr.width === 0 && rr.height === 0) { vis++; continue; }  /* a collapsed space */
      if (inside(rr, p)) vis++;
      else if (firstOut === null) firstOut = i;
    }
    rng.setStart(node, 0); rng.setEnd(node, node.length);
    return {chars: node.length, insideChars: vis, firstOutsideAt: firstOut,
            allInside: firstOut === null, box: round(rng.getBoundingClientRect()),
            text: node.textContent.trim().slice(0, 60)};
  }

  /* the licensed sentence is the text node following the <i> label directly inside .c5; the
     refused sentence is the text node following the <i> label inside .rf */
  const afterLabel = (el) => {
    if (!el) return null;
    const lab = el.querySelector(':scope > i');
    return lab ? lab.nextSibling : null;
  };

  const out = {viewport: {w: innerWidth, h: innerHeight}, c5: null, access: null};

  const wrap = document.getElementById('wC5');
  if (!wrap) { out.c5 = {error: 'no #wC5 -- the plate is not on this page'}; }
  else {
    const plate = wrap.closest('.plate') || wrap.parentElement;
    const pr = plate.getBoundingClientRect();
    const rows = [...wrap.querySelectorAll(':scope > .row')];
    const blocks = [...wrap.querySelectorAll(':scope > .c5')];
    out.c5 = {
      plate: {...round(pr), overflow: getComputedStyle(plate).overflow,
              scrollH: plate.scrollHeight, clipped: plate.scrollHeight > Math.round(pr.height)},
      servedRows: rows.length,
      insideBottom: !!plate.closest('#bottom'),
      entries: rows.map((rowEl, i) => {
        const b = blocks[i] || null;
        const v = rowEl.querySelector('.v');
        const rf = b ? b.querySelector('.rf') : null;
        const det = b ? b.querySelector('details') : null;
        const vr = v ? v.getBoundingClientRect() : null;
        return {
          i,
          instrument: (rowEl.querySelector('.k') || {}).textContent,
          valueText: v ? v.textContent : null,
          valueBox: vr ? round(vr) : null,
          valueInside: vr ? inside(vr, pr) : null,
          valueWhiteSpace: v ? getComputedStyle(v).whiteSpace : null,
          blockWhiteSpace: b ? getComputedStyle(b).whiteSpace : null,
          licensed: charCensus(afterLabel(b), pr),
          refused: charCensus(afterLabel(rf), pr),
          detailsPresent: !!det,
          detailsOpen: det ? det.open : null,
          detailsBox: det ? round(det.getBoundingClientRect()) : null,
          provChars: det ? (det.querySelector('div') || {textContent: ''}).textContent.length : 0,
          provWhiteSpace: det && det.querySelector('div')
            ? getComputedStyle(det.querySelector('div')).whiteSpace : null,
        };
      }),
    };
    /* (v2): the refusal holds -- no verdict word anywhere on the plate. Kept in the instrument so
       the count is read off the RENDERED text, not off the source the gate already scans. */
    const words = ['is conscious', 'is sentient', 'has experience', 'phenomenally conscious',
                   'there is something it is like', 'subjective experience'];
    const low = (wrap.textContent || '').toLowerCase();
    out.c5.verdictWords = words.filter((w) => low.includes(w));
  }

  /* the ACCESS plate, in the panel script's own shape: the BEFORE numbers and, after this arc,
     the control that its remaining rows did not change */
  const acc = document.getElementById('bAcc');
  if (!acc) { out.access = {error: 'no #bAcc'}; }
  else {
    const ap = acc.closest('.plate') || acc.parentElement;
    const ar = ap.getBoundingClientRect();
    out.access = {
      plate: {...round(ar), overflow: getComputedStyle(ap).overflow, scrollH: ap.scrollHeight},
      rows: [...acc.querySelectorAll('.row')].map((el, i) => {
        const r = el.getBoundingClientRect();
        const k = el.querySelector('.k'), v = el.querySelector('.v');
        const tg = el.querySelector('.tag');
        return {i, k: k ? k.textContent.trim() : '',
                topRel: Math.round(r.top - ar.top), botRel: Math.round(r.bottom - ar.top),
                visible: r.top >= ar.top && r.bottom <= ar.bottom,
                vWidth: v ? Math.round(v.getBoundingClientRect().width) : null,
                vChars: v ? v.textContent.length : null,
                tagChars: tg ? tg.textContent.length : 0,
                visibleTagChars: tg ? charCensus(tg.firstChild, ar).insideChars : 0};
      }),
    };
  }
  return out;
})()
