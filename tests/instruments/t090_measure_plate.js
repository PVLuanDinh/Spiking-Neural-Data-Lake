/* T090 (C5) -- the RENDERED measurement of the CONSCIOUSNESS plate, EXTENDED for the SECOND
   reading (record 3.2, E1-E8). STRICTLY HARDER THAN THE INSTRUMENT THAT TOOK THE FIRST READING,
   NEVER DIFFERENT. The barred quantity is unchanged and is quoted at 3.2.0: every character of
   the licensed sentence and of the refused sentence, and the row's live value, inside the plate's
   VISIBLE box, at both widths; one clipped character is NOT MET.

   A BROWSER SCRIPT. IT IS NEVER RUN BY CI AND NO TEST IMPORTS IT: a browser is not CI, and the
   thing it measures -- where a character actually lands after layout -- has no representation in
   the source that a pytest could read. tests/test_conclusion_plate.py carries what CI CAN see and
   says in its own docstring that it is a ratchet of source text (obs #2195).

   HOW TO RUN IT. The file defines ONE function, exposes it and calls it, so both carriers work:
   pasted into the live cockpit page's console the last expression IS the result, and loaded as a
   <script> element (its committed BYTES, off a local read-only static server over
   tests/instruments/) the entry point is called on load. There is no options argument and no flag
   that turns a check off: a bypass inside an instrument is a hazard, not a convenience. Re-running
   is `window.t090MeasurePlate()`; re-LOADING the same file into one document throws on the `const`,
   so reload the page first.

   Read-only: no click, no fetch, no style write, no <details> opened. 2.5 makes the collapsed
   state a control, because the sentences must be in view WITHOUT expanding anything.

   Extends scratchpad/t082-panel/refute/measure_access_plate.js: the ACCESS census at the foot is
   that script's shape, kept so the BEFORE numbers (fact 2: 0 visible sentence characters on every
   charged row, at both widths) and the AFTER are read by the same instrument. It is the control
   that the fixed-height bottom strip did not change; it now gets the same visible-box treatment.

   WHAT THE EXTENSION ADDED (E1-E8; every one of them can only LOWER a count, never raise it):
     E1 THE STRICT VISIBLE BOX. The census runs against the plate's border rect intersected with
        the CLIENT (padding) box of every ancestor to the root whose computed overflow-x or
        overflow-y is not `visible`, and with the viewport. Client boxes, not border boxes: a box
        with overflow != visible clips at its padding edge, and its scrollbar gutter lies outside
        that edge. Because the walk starts from the border rect and only intersects,
        `insideChars` <= `insideCharsPlate` is structural, not a hope. The chain ACTUALLY walked is
        printed at `c5.clipChain` by selector with each member's rect and computed overflow -- data,
        never an enumeration assumed in advance.
     E2 SCROLL IS A PRECONDITION. Any non-zero scrollTop/scrollLeft on a chain member or on the
        document REFUSES with no numbers. Browsers restore a scroll container's offset across
        reloads, and a silent wrong number here is indistinguishable from a pass. The positions are
        printed at `scroll` even when green -- the precondition's evidence, not its assumption.
     E3 THE COLLAPSED <details> IS A PRECONDITION TOO, with the same refusal rather than a reported
        field.
     E4 INVISIBLE TEXT IS NO LONGER IN VIEW. A sentence whose parent computes `visibility` other
        than visible, or opacity 0, or font-size 0, is REFUSED with `hidden:true` and 0 inside; and
        the zero-area branch now counts only actual whitespace, so a glyph with no box is out. The
        committed census counted both as in view, which made the barred quantity satisfiable by
        text nobody can see.
     E5/E6 ONE BOX FOR ONE CRITERION. The sentences and the live value are tested against the SAME
        strict box. `insideCharsPlate` / `valueInsidePlate` keep the committed plate-only predicate
        beside them, code path otherwise unchanged, so the continuity control (R8) can compare.
        Per sentence the script also reads `lines` (the wrap is READ, never assumed), `minMargin`,
        `advancePx` (the character advance, measured for the first time rather than assumed) and,
        when the two censuses differ, `divergence` with `attributedTo` naming the single clipper
        the first divergent character falls outside of.
     E7 `clipped` is scrollHeight > clientHeight, two integers off the same box. REPORTED, never
        barred.
     E8 THE COLUMN AND EVERY PLATE IN IT, in DOM order, with rowsTotal / rowsInView /
        rowsInPlateOnly and each plate's content-vs-box -- the reachability ledger, read as a count
        by this same script at both heads. `column.clientW` IS the measured scrollbar gutter.

   HONEST NOTES, STATED RATHER THAN CLAIMED:
     - `#app` sets no overflow and is therefore not a clipper; it will not appear in the chain.
     - `body` COMPUTES overflow `hidden` but its USED value is `visible` -- the root propagates
       body's overflow to the viewport -- so including it costs nothing here, its client box
       coinciding with the viewport. It is kept because a future non-100vh layout is the case it
       guards, not because it clips anything today.
     - THE VIEWPORT TERM CAN NEVER FIRE ALONE ON THIS PAGE, and it is explicitly NOT presented as a
       conjunct with a red of its own. The grid is height:100vh with 10px padding and body is
       overflow:hidden, so the widget column's box is inside the viewport at every viewport size
       (44+910 = 954 < 1080; 44+925 = 969 < 1321) and a short viewport shrinks the 1fr column
       before it can clip it. It is kept because it costs nothing.
     - NOT CARRIED: `column.scrollHAgain`, a second column scrollHeight read about a second later.
       It cannot be taken without making this entry point async (the only synchronous way to wait
       is a busy loop, which blocks the 1 Hz repaint it is meant to observe and would return the
       same number by construction). The entry point is called TWICE instead, a second apart, and
       the two `column.scrollH` values compared -- the column's scrollHeight really does vary at
       1 Hz now, and that is measured rather than narrated.
     - A missing plate is reported as `c5.error` exactly as the committed script reported it; the
       two REFUSING preconditions are E2 and E3 and no others. */
const t090MeasurePlate = () => {
  const VERSION = 't090-measure-2';

  const round = (r) => ({w: Math.round(r.width), h: Math.round(r.height),
                         l: Math.round(r.left), t: Math.round(r.top),
                         r: Math.round(r.right), b: Math.round(r.bottom)});
  const box2 = (x) => ({w: Math.round(x.right - x.left), h: Math.round(x.bottom - x.top),
                        l: Math.round(x.left), t: Math.round(x.top),
                        r: Math.round(x.right), b: Math.round(x.bottom)});
  const inside = (r, p) => r.left >= p.left && r.right <= p.right &&
                           r.top >= p.top && r.bottom <= p.bottom;

  /* E1: the strict visible box. CLIENT boxes for the clippers, because a box with
     overflow != visible clips at its PADDING edge and its scrollbar gutter lies outside it. */
  const clientBox = (el) => {
    const r = el.getBoundingClientRect(), cs = getComputedStyle(el);
    const bl = parseFloat(cs.borderLeftWidth) || 0, bt = parseFloat(cs.borderTopWidth) || 0;
    return {left: r.left + bl, top: r.top + bt,
            right: r.left + bl + el.clientWidth, bottom: r.top + bt + el.clientHeight};
  };
  const isect = (a, b) => ({left: Math.max(a.left, b.left), top: Math.max(a.top, b.top),
                            right: Math.min(a.right, b.right), bottom: Math.min(a.bottom, b.bottom)});
  const sel = (el) => el.id ? '#' + el.id
    : (el.tagName.toLowerCase() + '.' + String(el.className || '').split(' ')[0]);
  const viewportBox = () => ({left: 0, top: 0, right: innerWidth, bottom: innerHeight});
  function clipChain(el) {
    const chain = [];
    for (let n = el; n && n.nodeType === 1; n = n.parentElement) {
      const cs = getComputedStyle(n);
      if (cs.overflowX !== 'visible' || cs.overflowY !== 'visible') chain.push(n);
    }
    return chain;
  }
  function visibleBox(el) {
    const r = el.getBoundingClientRect();
    let b = {left: r.left, top: r.top, right: r.right, bottom: r.bottom};
    for (const n of clipChain(el)) b = isect(b, clientBox(n));  /* includes el itself */
    return isect(b, viewportBox());
  }
  const chainOf = (el) => clipChain(el).map((n) => {
    const cs = getComputedStyle(n);
    return {el: n, sel: sel(n), rect: n.getBoundingClientRect(), box: clientBox(n),
            overflowX: cs.overflowX, overflowY: cs.overflowY};
  });
  const reportChain = (chain) => chain.map((m) => ({sel: m.sel, rect: round(m.rect),
    clientBox: box2(m.box), overflowX: m.overflowX, overflowY: m.overflowY}));
  /* which single clipper's client box the first divergent character falls outside of */
  const attributeOutside = (rr, chain) => {
    for (const m of chain) if (!inside(rr, m.box)) return m.sel;
    if (!inside(rr, viewportBox())) return 'viewport';
    return 'none -- inside every named box';
  };

  /* Per-CHARACTER census of one text node, in ONE pass, against BOTH boxes: `insideChars` is the
     strict visible box (E1) and `insideCharsPlate` is the committed plate-border predicate. A box
     test on the whole range is not the same measurement: a range that starts inside and ends
     outside still reports a rect, and half a sentence is not the sentence. */
  function charCensus(node, strictBox, plateBorderRect) {
    if (!node || node.nodeType !== 3)
      return {chars: 0, insideChars: 0, insideCharsPlate: 0, missing: true};
    /* E4: text nobody can see is not in view. visibility:hidden preserves layout, so its rects are
       real, correctly positioned, and would pass even the intersected test. */
    const pcs = getComputedStyle(node.parentElement);
    if (pcs.visibility !== 'visible' || parseFloat(pcs.opacity) === 0 ||
        parseFloat(pcs.fontSize) === 0)
      return {chars: node.length, insideChars: 0, insideCharsPlate: 0, hidden: true,
              why: 'visibility ' + pcs.visibility + ' opacity ' + pcs.opacity +
                   ' font-size ' + pcs.fontSize};
    const rng = document.createRange();
    const tops = new Set();
    let vis = 0, visP = 0, firstOut = null, firstOutP = null;
    let advancePx = null, minMargin = null, firstDivergentAt = null, firstDivergentRect = null;
    for (let i = 0; i < node.length; i++) {
      rng.setStart(node, i); rng.setEnd(node, i + 1);
      const rr = rng.getBoundingClientRect();
      const ws = /\s/.test(node.data[i]);
      if (rr.width === 0 && rr.height === 0) {
        if (ws) { vis++; visP++; continue; }   /* a collapsed space, the only legitimate case */
        if (firstOut === null) firstOut = i;   /* a glyph with no box is not in view */
        if (firstOutP === null) firstOutP = i;
        continue;
      }
      if (!ws) {
        tops.add(Math.round(rr.top));
        if (advancePx === null) advancePx = Math.round(rr.width * 100) / 100;
      }
      const inS = inside(rr, strictBox), inP = inside(rr, plateBorderRect);
      if (inS) {
        vis++;
        const m = Math.min(rr.left - strictBox.left, strictBox.right - rr.right,
                           rr.top - strictBox.top, strictBox.bottom - rr.bottom);
        if (minMargin === null || m < minMargin) minMargin = m;
      } else if (firstOut === null) firstOut = i;
      if (inP) visP++;
      else if (firstOutP === null) firstOutP = i;
      if (inP && !inS && firstDivergentAt === null) { firstDivergentAt = i; firstDivergentRect = rr; }
    }
    rng.setStart(node, 0); rng.setEnd(node, node.length);
    return {chars: node.length, insideChars: vis, insideCharsPlate: visP,
            firstOutsideAt: firstOut, firstOutsidePlateAt: firstOutP,
            allInside: firstOut === null, allInsidePlate: firstOutP === null,
            lines: tops.size, advancePx: advancePx,
            minMargin: minMargin === null ? null : Math.round(minMargin * 100) / 100,
            firstDivergentAt: firstDivergentAt, _firstDivergentRect: firstDivergentRect,
            box: round(rng.getBoundingClientRect()),
            text: node.textContent.trim().slice(0, 60)};
  }

  /* E5's per-sentence divergence, itemised and attributed to a NAMED box term. A stricter
     instrument is entitled to read lower; it is not entitled to read lower for an unnamed
     reason, and it may never read higher. */
  const censusOf = (node, strictBox, plateBorderRect, chain) => {
    const c = charCensus(node, strictBox, plateBorderRect);
    c.divergence = (c.insideCharsPlate || 0) - (c.insideChars || 0);
    if (c._firstDivergentRect) {
      c.firstDivergentBox = round(c._firstDivergentRect);
      c.attributedTo = attributeOutside(c._firstDivergentRect, chain);
    }
    delete c._firstDivergentRect;
    return c;
  };

  /* the licensed sentence is the text node following the <i> label directly inside .c5; the
     refused sentence is the text node following the <i> label inside .rf */
  const afterLabel = (el) => {
    if (!el) return null;
    const lab = el.querySelector(':scope > i');
    return lab ? lab.nextSibling : null;
  };

  const out = {version: VERSION, viewport: {w: innerWidth, h: innerHeight},
               scroll: null, detailsOpen: null, column: null, plates: null,
               c5: null, access: null};

  const wrap = document.getElementById('wC5');
  const plate = wrap ? (wrap.closest('.plate') || wrap.parentElement) : null;
  const chain = plate ? chainOf(plate) : [];

  /* E2 + E3: the two preconditions, read BEFORE any measurement and printed even when green. */
  const scrollSeen = [];
  const noteScroll = (n, why) => {
    if (n && n.nodeType === 1 && !scrollSeen.some((s) => s.el === n)) scrollSeen.push({el: n, why});
  };
  for (const m of chain) noteScroll(m.el, 'clip chain');
  noteScroll(document.scrollingElement, 'scrollingElement');
  noteScroll(document.documentElement, 'documentElement');
  noteScroll(document.body, 'body');
  const members = scrollSeen.map((s) => ({sel: sel(s.el), why: s.why,
                                          scrollTop: s.el.scrollTop, scrollLeft: s.el.scrollLeft}));
  out.scroll = {members: members,
                nonZero: members.filter((m) => m.scrollTop !== 0 || m.scrollLeft !== 0)
                                .map((m) => m.sel + ' (top ' + m.scrollTop + ', left ' + m.scrollLeft + ')')};
  out.detailsOpen = wrap ? [...wrap.querySelectorAll('details')].filter((d) => d.open).length : 0;

  if (out.scroll.nonZero.length) {
    return {version: VERSION, refused: 'scrolled: ' + out.scroll.nonZero.join('; '),
            scroll: out.scroll, detailsOpen: out.detailsOpen, viewport: out.viewport};
  }
  if (out.detailsOpen > 0) {
    return {version: VERSION,
            refused: 'details open: ' + out.detailsOpen +
                     ' -- 2.5 makes the collapsed state a control',
            scroll: out.scroll, detailsOpen: out.detailsOpen, viewport: out.viewport};
  }

  if (!wrap) { out.c5 = {error: 'no #wC5 -- the plate is not on this page'}; }
  else {
    const pr = plate.getBoundingClientRect();
    const strict = visibleBox(plate);
    const rows = [...wrap.querySelectorAll(':scope > .row')];
    const blocks = [...wrap.querySelectorAll(':scope > .c5')];
    out.c5 = {
      plate: {...round(pr), overflow: getComputedStyle(plate).overflow,
              scrollH: plate.scrollHeight, clientH: plate.clientHeight,
              clipped: plate.scrollHeight > plate.clientHeight,  /* E7 */
              strictBox: box2(strict)},
      clipChain: reportChain(chain),                             /* E1: walked, not assumed */
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
          valueInside: vr ? inside(vr, strict) : null,          /* E6: the same strict box */
          valueInsidePlate: vr ? inside(vr, pr) : null,
          valueWhiteSpace: v ? getComputedStyle(v).whiteSpace : null,
          blockWhiteSpace: b ? getComputedStyle(b).whiteSpace : null,
          blockInline: b ? {scrollW: b.scrollWidth, clientW: b.clientWidth,
                            overflowsInline: b.scrollWidth > b.clientWidth} : null,
          licensed: censusOf(afterLabel(b), strict, pr, chain),
          refused: censusOf(afterLabel(rf), strict, pr, chain),
          detailsPresent: !!det,
          detailsOpen: det ? det.open : null,
          detailsBox: det ? round(det.getBoundingClientRect()) : null,
          provChars: det ? (det.querySelector('div') || {textContent: ''}).textContent.length : 0,
          provWhiteSpace: det && det.querySelector('div')
            ? getComputedStyle(det.querySelector('div')).whiteSpace : null,
        };
      }),
    };
    /* the record's numbers are READ off the instrument, never re-added by hand */
    const es = out.c5.entries;
    const sum = (f) => es.reduce((s, e) => s + (e.licensed[f] || 0) + (e.refused[f] || 0), 0);
    out.c5.totals = {
      chars: sum('chars'),
      insideChars: sum('insideChars'),
      insideCharsPlate: sum('insideCharsPlate'),
      valuesInside: es.filter((e) => e.valueInside === true).length,
      valuesInsidePlate: es.filter((e) => e.valueInsidePlate === true).length,
      cellsComplete: es.filter((e) => e.valueInside === true && e.licensed.allInside === true &&
                                      e.refused.allInside === true).length,
    };
    /* (v2): the refusal holds -- no verdict word anywhere on the plate. Kept in the instrument so
       the count is read off the RENDERED text, not off the source the gate already scans. */
    const words = ['is conscious', 'is sentient', 'has experience', 'phenomenally conscious',
                   'there is something it is like', 'subjective experience'];
    const low = (wrap.textContent || '').toLowerCase();
    out.c5.verdictWords = words.filter((w) => low.includes(w));
  }

  /* E8: the column and every plate in it -- the P3/P4c reachability ledger as a COUNT. */
  const col = document.getElementById('widgets');
  if (!col) { out.column = {error: 'no #widgets'}; out.plates = []; }
  else {
    const cr = col.getBoundingClientRect(), ccs = getComputedStyle(col), cb = clientBox(col);
    out.column = {box: round(cr), clientW: col.clientWidth, clientH: col.clientHeight,
                  scrollW: col.scrollWidth, scrollH: col.scrollHeight, scrollTop: col.scrollTop,
                  overflowX: ccs.overflowX, overflowY: ccs.overflowY};
    out.plates = [...col.children].map((el, i) => {
      const r = el.getBoundingClientRect(), s = visibleBox(el);
      const pRows = [...el.querySelectorAll('.row')];
      return {i, h2: ((el.querySelector('h2') || {}).textContent || '').trim(),
              box: round(r), clientH: el.clientHeight, scrollH: el.scrollHeight,
              clipped: el.scrollHeight > el.clientHeight,
              topInCol: Math.round(r.top - cr.top + col.scrollTop),
              visiblePx: Math.max(0, Math.round(Math.min(r.bottom, cb.bottom) -
                                                Math.max(r.top, cb.top))),
              rowsTotal: pRows.length,
              rowsInView: pRows.filter((q) => inside(q.getBoundingClientRect(), s)).length,
              rowsInPlateOnly: pRows.filter((q) => inside(q.getBoundingClientRect(), r)).length};
    });
  }

  /* the ACCESS plate, in the panel script's own shape: the BEFORE numbers and, after this arc,
     the control that its remaining rows did not change. Same visibleBox treatment, nothing else. */
  const acc = document.getElementById('bAcc');
  if (!acc) { out.access = {error: 'no #bAcc'}; }
  else {
    const ap = acc.closest('.plate') || acc.parentElement;
    const ar = ap.getBoundingClientRect();
    const aStrict = visibleBox(ap);
    const aChain = chainOf(ap);
    out.access = {
      plate: {...round(ar), overflow: getComputedStyle(ap).overflow, scrollH: ap.scrollHeight,
              strictBox: box2(aStrict)},
      clipChain: reportChain(aChain),
      rows: [...acc.querySelectorAll('.row')].map((el, i) => {
        const r = el.getBoundingClientRect();
        const k = el.querySelector('.k'), v = el.querySelector('.v');
        const tg = el.querySelector('.tag');
        const tc = tg ? censusOf(tg.firstChild, aStrict, ar, aChain) : null;
        return {i, k: k ? k.textContent.trim() : '',
                topRel: Math.round(r.top - ar.top), botRel: Math.round(r.bottom - ar.top),
                visible: inside(r, aStrict),
                visiblePlateOnly: r.top >= ar.top && r.bottom <= ar.bottom,
                vWidth: v ? Math.round(v.getBoundingClientRect().width) : null,
                vChars: v ? v.textContent.length : null,
                tagChars: tg ? tg.textContent.length : 0,
                visibleTagChars: tc ? tc.insideChars : 0,
                visibleTagCharsPlate: tc ? tc.insideCharsPlate : 0};
      }),
    };
  }
  return out;
};
window.t090MeasurePlate = t090MeasurePlate;
t090MeasurePlate()
