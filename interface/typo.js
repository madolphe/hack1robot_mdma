/* Moteur de distorsion typographique, repris de typolab/index.html (branche fontlab) pour l'interface finale.
   État de la typo = neutre + réponse(α filtré) × vecteur de la drogue ; chaque mot garde l'état de sa naissance.
     TYPO.setDrug("lsd")                    drogue courante (vecteurs et réponses par défaut de typolab)
     TYPO.update(alpha, dt)                 à chaque image : α filtré par paramètre (retards ↑/↓, hystérésis)
     const st = TYPO.state()                état typographique courant
     TYPO.birth(el, i, st, clock)           à l'écriture du mot i (élément .w contenant une <span> par lettre)
     TYPO.style(el, now, ctx)               à chaque image ; ctx = {ent, box, basePx} (entropie 0..1, conteneur, corps de base)
   Les réglages par drogue sont ceux de typolab, sans les modifications faites dans son panneau.            */
const TYPO = (() => {
// ------------------------------------------------------------------ espace de paramètres
// [clé, libellé, min, max, neutre, groupe]
const SPACE = [
  ["mix", "part Roboto Flex", 0, 1, 0.35, "Mélange des deux polices (par lettre)"],
  ["r_wght", "graisse", 300, 1000, 420, "Recursive"], ["r_CASL", "manuscrit", 0, 1, 0, "Recursive"], ["r_slnt", "penché", -15, 0, 0, "Recursive"],
  ["r_CRSV", "cursive", 0, 1, 0.5, "Recursive"], ["r_MONO", "mono", 0, 1, 0, "Recursive"],
  ["f_wght", "graisse", 100, 1000, 400, "Roboto Flex"], ["f_wdth", "largeur", 25, 151, 100, "Roboto Flex"], ["f_opsz", "taille optique", 8, 144, 36, "Roboto Flex"],
  ["f_GRAD", "grade", -200, 150, 0, "Roboto Flex"], ["f_slnt", "penché", -10, 0, 0, "Roboto Flex"], ["f_XOPQ", "trait vertical", 27, 175, 96, "Roboto Flex"],
  ["f_YOPQ", "trait horizontal", 25, 135, 79, "Roboto Flex"], ["f_XTRA", "contreformes", 323, 603, 468, "Roboto Flex"], ["f_YTLC", "haut. minuscules", 416, 570, 514, "Roboto Flex"],
  ["f_YTUC", "haut. capitales", 528, 760, 712, "Roboto Flex"], ["f_YTAS", "montantes", 649, 854, 750, "Roboto Flex"], ["f_YTDE", "descendantes", -305, -98, -203, "Roboto Flex"],
  ["size", "corps (vw)", 1, 6, 2.4, "Mise en page"], ["track", "interlettrage (em)", -0.12, 0.8, 0.01, "Mise en page"], ["wspace", "espace mots (em)", 0, 2.5, 0.28, "Mise en page"],
  ["lead", "interlignage", 0.8, 2.6, 1.32, "Mise en page"], ["baseline", "ligne de base (em)", -0.6, 0.6, 0, "Mise en page"],
  ["rate", "vitesse d'écriture ×", 0.2, 4, 1, "Temps"], ["hesit", "hésitations", 0, 1, 0, "Temps"], ["fadein", "apparition (s)", 0, 2, 0.25, "Temps"],
  ["jitter", "tremblement (em)", 0, 0.5, 0, "Instabilité"], ["rot", "rotation (°)", 0, 45, 0, "Instabilité"], ["wave", "vague (em)", 0, 0.8, 0, "Instabilité"],
  ["axnoise", "bruit d'axes", 0, 1, 0, "Instabilité"], ["scnoise", "bruit d'échelle", 0, 0.8, 0, "Instabilité"],
  ["hue", "teinte", 0, 360, 45, "Rendu"], ["sat", "saturation %", 0, 100, 0, "Rendu"], ["light", "luminosité %", 20, 100, 92, "Rendu"],
  ["opac", "opacité", 0.05, 1, 1, "Rendu"], ["blur", "flou (px)", 0, 10, 0, "Rendu"], ["glow", "halo (px)", 0, 40, 0, "Rendu"],
  ["chroma", "chromatique (px)", 0, 12, 0, "Rendu"], ["trails", "traînées", 0, 6, 0, "Rendu"],
  ["hgrad", "dégradé dans le mot (°)", 0, 360, 0, "Couleur"], ["hspread", "écart entre lettres (°)", 0, 360, 0, "Couleur"],
  ["huespin", "rotation de teinte (°/s)", 0, 240, 0, "Couleur"], ["pulse", "pulsation lumière", 0, 1, 0, "Couleur"],
  ["bg_hue", "fond : teinte", 0, 360, 240, "Couleur"], ["bg_sat", "fond : saturation %", 0, 100, 8, "Couleur"], ["bg_light", "fond : luminosité %", 0, 60, 5, "Couleur"],
  ["curve", "courbure (em)", 0, 4, 0, "Trajectoire des lignes"], ["tilt", "inclinaison (°)", -35, 35, 0, "Trajectoire des lignes"],
  ["walk", "marche aléatoire (em/mot)", 0, 1.2, 0, "Trajectoire des lignes"], ["scatter", "dispersion (0 ligne, 1 partout)", 0, 1, 0, "Trajectoire des lignes"],
  ["fall", "chute (em/s, < 0 : monte)", -0.6, 1, 0, "Trajectoire des lignes"],
];
const K = Object.fromEntries(SPACE.map(s => [s[0], s]));
const NOISE = new Set(["jitter", "rot", "wave", "axnoise", "scnoise", "hesit", "blur", "chroma", "trails", "glow",
                       "hspread", "huespin", "pulse", "curve", "tilt", "walk", "scatter", "fall"]);

// Vecteurs des drogues : valeur visée à α = +1 (pos) et à α = −1 (neg) pour les paramètres qui bougent ;
// les autres restent au neutre. neg non précisé : opposé atténué pour la géométrie, extinction pour le bruit.
const D = {
  alcool: { pos:{r_CASL:1, r_slnt:-13, r_CRSV:1, r_wght:620, mix:0.15, track:0.06, wspace:0.6, lead:1.6, rate:0.75, hesit:0.35, jitter:0.12, rot:10, wave:0.3, axnoise:0.4, blur:1.6, trails:2, hue:25, sat:55, light:85},
            neg:{r_CASL:0, r_MONO:1, mix:0, r_wght:330, track:0.12, rate:0.6, sat:0, light:60, opac:0.7} },
  cannabis:{ pos:{r_CASL:1, r_wght:380, track:0.2, wspace:1, lead:1.9, rate:0.45, hesit:0.6, wave:0.25, blur:1.2, glow:14, hue:100, sat:40, fadein:1.4},
            neg:{rate:1.4, r_MONO:0.7, track:-0.03, light:70, opac:0.8} },
  cocaine: { pos:{mix:1, f_wght:1000, f_wdth:25, f_GRAD:150, f_opsz:144, f_XOPQ:175, f_YTLC:570, size:3.4, track:-0.07, wspace:0.12, lead:1.0, rate:3.2, fadein:0, jitter:0.04, chroma:3, light:100},
            neg:{mix:1, f_wght:120, f_wdth:60, f_GRAD:-200, size:1.8, rate:0.4, light:45, opac:0.55} },
  amphetamines:{ pos:{r_MONO:1, mix:0.5, r_wght:800, f_wdth:30, f_wght:800, size:2.1, track:-0.04, wspace:0.1, lead:1.0, rate:3.8, fadein:0, jitter:0.06, axnoise:0.3, hue:185, sat:60},
            neg:{rate:0.35, r_wght:310, f_wght:120, light:50, opac:0.6, hesit:0.6} },
  mdma:    { pos:{r_CASL:1, r_wght:800, mix:0.5, f_wght:800, f_wdth:140, f_XTRA:603, size:3.2, track:0.12, wspace:0.6, wave:0.25, glow:30, hue:310, sat:85, light:90},
            neg:{r_wght:320, f_wght:150, f_wdth:70, size:2, light:45, sat:10, hue:220, opac:0.6, rate:0.6} },
  lsd:     { pos:{mix:0.5, r_wght:900, r_CASL:1, r_slnt:-15, f_wght:900, f_wdth:151, f_XTRA:603, f_YTLC:570, f_YTAS:854, f_YTDE:-305, size:3.4, track:0.1, wave:0.7, rot:14, axnoise:0.8, scnoise:0.4, chroma:8, trails:4, hue:200, sat:100, glow:16},
            neg:{f_wdth:60, f_XTRA:323, r_MONO:1, size:2, sat:0, light:70} },
  champignons:{ pos:{r_CASL:1, r_CRSV:1, r_wght:560, f_wdth:130, f_YTLC:570, track:0.08, wspace:0.7, lead:1.8, rate:0.6, wave:0.5, axnoise:0.4, blur:0.8, trails:3, glow:16, hue:70, sat:65},
            neg:{r_CASL:0, r_MONO:0.6, light:65, sat:5, rate:0.8} },
  dmt:     { pos:{mix:0.7, f_wght:1000, f_wdth:151, f_GRAD:150, f_XTRA:603, f_YTUC:760, f_YTAS:854, size:4.4, track:0.5, wspace:1.4, wave:0.6, rot:30, axnoise:1, scnoise:0.7, chroma:12, trails:6, glow:40, hue:170, sat:100, light:100},
            neg:{size:1.6, f_wght:100, sat:0, light:40, opac:0.4} },
  ketamine:{ pos:{mix:0.8, f_wght:150, f_wdth:151, f_opsz:8, f_YTLC:416, track:0.8, wspace:2.4, lead:2.4, rate:0.5, hesit:0.5, fadein:1.6, baseline:0.4, jitter:0.2, scnoise:0.3, blur:1.5, sat:0, light:80, opac:0.75},
            neg:{track:-0.1, wspace:0.05, lead:0.9, rate:1.6, size:2.8} },
  heroine: { pos:{r_wght:300, f_wght:120, r_CASL:0.6, r_slnt:-6, size:2.1, track:0.25, wspace:0.8, lead:1.9, rate:0.3, fadein:2, blur:3.5, glow:22, hue:28, sat:30, light:78, opac:0.75},
            neg:{rate:1.8, hesit:0.5, jitter:0.15, light:96, sat:0, r_wght:700, axnoise:0.5} },
  datura:  { pos:{mix:0.5, r_wght:900, r_CASL:1, r_slnt:-15, f_wght:900, f_wdth:28, f_YTDE:-305, f_YTUC:760, size:3, track:-0.05, rate:1.6, hesit:0.8, jitter:0.4, rot:28, axnoise:1, scnoise:0.6, chroma:7, trails:3, hue:0, sat:75},
            neg:{opac:0.35, blur:4, sat:0, light:55, rate:0.4} },
};
// trajectoire des lignes et couleur, ajoutées aux vecteurs ci-dessus
const EXTRA = {
  alcool:      { pos:{light:70, curve:0.9, tilt:6, walk:0.18, fall:0.04, hgrad:40, hspread:20, bg_hue:20, bg_sat:30, bg_light:9},
                 neg:{bg_hue:210, bg_sat:5, bg_light:14} },
  cannabis:    { pos:{light:70, curve:0.6, walk:0.08, fall:-0.05, hgrad:60, huespin:8, pulse:0.2, bg_hue:120, bg_sat:25, bg_light:7},
                 neg:{bg_hue:200, bg_sat:10, bg_light:10} },
  cocaine:     { pos:{tilt:-4, walk:0.05, hspread:0, bg_hue:0, bg_sat:0, bg_light:0, pulse:0.35},
                 neg:{bg_hue:220, bg_sat:6, bg_light:16, fall:0.12} },
  amphetamines:{ pos:{light:66, tilt:3, walk:0.25, curve:0.2, hspread:30, pulse:0.5, bg_hue:185, bg_sat:40, bg_light:6},
                 neg:{fall:0.15, bg_hue:200, bg_sat:5, bg_light:12} },
  mdma:        { pos:{light:66, curve:1.2, walk:0.1, hgrad:120, hspread:40, huespin:25, pulse:0.4, bg_hue:320, bg_sat:45, bg_light:12},
                 neg:{fall:0.2, bg_hue:230, bg_sat:15, bg_light:6} },
  lsd:         { pos:{light:62, curve:2.6, tilt:10, walk:0.35, scatter:0.35, hgrad:240, hspread:140, huespin:90, pulse:0.5, bg_hue:275, bg_sat:55, bg_light:14},
                 neg:{bg_hue:210, bg_sat:0, bg_light:10} },
  champignons: { pos:{light:64, curve:1.8, walk:0.22, scatter:0.15, fall:-0.03, hgrad:90, hspread:50, huespin:15, pulse:0.3, bg_hue:95, bg_sat:35, bg_light:10},
                 neg:{bg_hue:40, bg_sat:5, bg_light:8} },
  dmt:         { pos:{light:60, curve:3.5, tilt:25, walk:0.8, scatter:1, hgrad:360, hspread:360, huespin:200, pulse:0.8, bg_hue:190, bg_sat:70, bg_light:18},
                 neg:{bg_hue:0, bg_sat:0, bg_light:0} },
  ketamine:    { pos:{walk:0.5, scatter:0.85, fall:-0.08, hspread:10, bg_hue:220, bg_sat:10, bg_light:22},
                 neg:{bg_hue:220, bg_sat:5, bg_light:4} },
  heroine:     { pos:{light:74, tilt:-3, walk:0.12, fall:0.18, hgrad:20, bg_hue:30, bg_sat:30, bg_light:10},
                 neg:{bg_hue:200, bg_sat:5, bg_light:18} },
  datura:      { pos:{light:56, curve:1.2, tilt:22, walk:0.9, scatter:0.55, fall:0.12, hspread:90, pulse:0.7, bg_hue:350, bg_sat:50, bg_light:7},
                 neg:{bg_hue:0, bg_sat:0, bg_light:2, scatter:0.2} },
};
for (const d in EXTRA) { Object.assign(D[d].pos, EXTRA[d].pos); Object.assign(D[d].neg, EXTRA[d].neg); }
const LABELS = {alcool:"Alcool", cannabis:"Cannabis", cocaine:"Cocaïne", amphetamines:"Amphétamines", mdma:"MDMA", lsd:"LSD", champignons:"Champignons", dmt:"DMT", ketamine:"Kétamine", heroine:"Héroïne", datura:"Datura"};

function defaultFor(drug) {
  const v = D[drug], rows = {};
  for (const [k, , min, max, neu] of SPACE) {
    const pos = v.pos[k] ?? neu;
    let neg = v.neg[k];
    if (neg === undefined) neg = NOISE.has(k) ? (k === "hesit" ? 0 : 0) : Math.max(min, Math.min(max, neu - (pos - neu) * 0.5));
    const isNoise = NOISE.has(k);
    rows[k] = {neu, pos, neg, resp: isNoise ? "seuil" : "lin", thr: isNoise ? 0.25 : 0, up: isNoise ? 0.6 : 0.3, down: isNoise ? 4 : 1.2};
  }
  return rows;
}

  const shape = (r, x) => {
    if (r.resp === "seuil") { const z = Math.max(0, Math.min(1, (x - r.thr) / Math.max(0.05, 1 - r.thr))); return z * z * (3 - 2 * z) * Math.max(1, x); }
    if (r.resp === "sat") return (1 - Math.exp(-3 * x)) / (1 - Math.exp(-3));
    return x;
  };
  const rnd = (a, b) => { const x = Math.sin(a * 127.1 + b * 311.7) * 43758.5453; return x - Math.floor(x); };
  const nz = (i, t, s) => Math.sin(t * 6.1 * s + i * 1.7) * 0.6 + Math.sin(t * 11.3 * s + i * 4.3) * 0.4;
  const clampAx = (k, v) => Math.max(K[k][2], Math.min(K[k][3], v));
  // couplage réseau -> typo (valeurs par défaut de typolab)
  const NET = {inst: 0.8, wght: 0.3, hue: 40, op: 0};

  let P = defaultFor("lsd"), filt = {}, walkX = 0, walkY = 0;
  return {
    SPACE, NET,
    setDrug(d) { P = defaultFor(D[d] ? d : "lsd"); },
    reset() { filt = {}; walkX = 0; walkY = 0; },
    update(a, dt) {
      for (const k in P) {
        const r = P[k], f = filt[k] ?? 0, up = Math.abs(a) > Math.abs(f) && Math.sign(a) === Math.sign(f || a);
        filt[k] = f + (a - f) * (1 - Math.exp(-dt / Math.max(0.001, up ? r.up : r.down)));
      }
    },
    state() {
      const s = {};
      for (const [k, , min, max] of SPACE) {
        const r = P[k], a = filt[k] ?? 0, m = shape(r, Math.abs(a));
        s[k] = Math.max(min, Math.min(max * 1.2, r.neu + (a >= 0 ? r.pos - r.neu : r.neg - r.neu) * m));
      }
      return s;
    },
    birth(el, i, st, clock) {
      el._st = {...st}; el._born = clock; el._i = i;
      walkY += (rnd(i, 41) - 0.5) * 2 * st.walk; walkX += (rnd(i, 43) - 0.5) * st.walk * 0.6;
      walkY *= 0.995; el._walk = [walkX, walkY];  // la marche s'accumule : la ligne dérive et ne revient pas
    },
    // true si le mot n'a plus besoin d'être restylé (figé et immobile)
    still(el, now) {
      const s = el._st;
      return now - el._born > s.fadein && s.jitter < 1e-3 && s.rot < 1e-3 && s.wave < 1e-3 && Math.abs(s.fall) < 1e-3 &&
             s.huespin < 1e-3 && s.pulse < 1e-3 && s.scatter < 1e-3;
    },
    style(el, now, ctx) {
      const st = el._st, i = el._i, e = ctx.ent ?? 0.5;
      const inst = Math.max(0, 1 + NET.inst * (e - 0.5) * 2), dW = NET.wght * (e - 0.5) * 2 * 250;
      const age = now - el._born, vis = st.fadein > 0.01 ? Math.min(1, age / st.fadein) : 1;
      const rel = st.size / 2.4, fs = rel * ctx.basePx, pw = ctx.box.clientWidth, ph = ctx.box.clientHeight;
      el.style.fontSize = rel + "em";
      el.style.marginRight = st.wspace + "em";
      el.style.letterSpacing = st.track + "em";
      el.style.opacity = (vis * st.opac * (1 - NET.op * e * 0.85)).toFixed(3);
      const hue = (st.hue + st.huespin * now + NET.hue * (e - 0.5) * 2 + 360 * 4) % 360;
      if (el._x == null) el._x = el.offsetLeft / Math.max(1, el.parentNode.clientWidth);
      const [wx, wy0] = el._walk || [0, 0];
      let dx = wx * fs, dy = wy0 * fs;
      dy += st.curve * fs * Math.sin(2 * Math.PI * el._x * 1.3 + i * 0.09);
      dy += Math.tan(st.tilt * Math.PI / 180) * (el._x - 0.5) * pw;
      dy += st.fall * fs * Math.min(60, age);
      dx += (rnd(i, 51) - 0.5) * st.scatter * pw * 0.9;
      dy += (rnd(i, 53) - 0.5) * st.scatter * ph * 0.9 + st.scatter * Math.sin(now * 0.4 + i) * fs * 0.6;
      const wr = st.tilt + (rnd(i, 57) - 0.5) * st.scatter * 120;
      el.style.transform = (dx || dy || wr) ? `translate(${dx.toFixed(1)}px, ${dy.toFixed(1)}px) rotate(${wr.toFixed(2)}deg)` : "none";
      el.style.filter = st.blur > 0.05 ? `blur(${(st.blur * (0.6 + 0.4 * inst)).toFixed(2)}px)` : "none";
      const sh = [];
      if (st.glow > 0.3) sh.push(`0 0 ${st.glow}px hsl(${hue} ${st.sat}% 60% / .75)`);
      if (st.chroma > 0.1) sh.push(`${st.chroma}px 0 0 rgba(255,0,70,.7)`, `${-st.chroma}px 0 0 rgba(0,220,255,.7)`);
      for (let t = 1; t <= Math.round(st.trails); t++) sh.push(`${-t * 4}px ${t * 2}px ${t * 1.5}px hsl(${hue} ${st.sat}% 70% / ${0.35 / t})`);
      el.style.textShadow = sh.join(", ") || "none";
      const L = el.children.length;
      [...el.children].forEach((s, j) => {
        const lh = hue + st.hgrad * (L > 1 ? j / (L - 1) - 0.5 : 0) + st.hspread * (rnd(i * 17 + j, 61) - 0.5);
        const li = Math.max(5, Math.min(100, st.light + st.pulse * 22 * Math.sin(now * 3.1 + i * 0.7 + j * 0.3)));
        s.style.color = `hsl(${(lh + 720) % 360} ${st.sat}% ${li}%)`;
        const rob = rnd(i * 31 + j, 7) < st.mix, an = st.axnoise * inst, sn = st.scnoise * inst;
        if (rob) {
          s.style.fontFamily = "'Roboto Flex'";
          s.style.fontVariationSettings =
            `'wght' ${clampAx("f_wght", st.f_wght + dW + an * 350 * (rnd(i, j) - 0.5))}, 'wdth' ${clampAx("f_wdth", st.f_wdth + an * 60 * (rnd(j, i) - 0.5))}, ` +
            `'opsz' ${st.f_opsz}, 'GRAD' ${st.f_GRAD}, 'slnt' ${st.f_slnt}, 'XOPQ' ${st.f_XOPQ}, 'YOPQ' ${st.f_YOPQ}, 'XTRA' ${st.f_XTRA}, ` +
            `'YTLC' ${st.f_YTLC}, 'YTUC' ${st.f_YTUC}, 'YTAS' ${st.f_YTAS}, 'YTDE' ${st.f_YTDE}`;
        } else {
          s.style.fontFamily = "'Recursive'";
          s.style.fontVariationSettings =
            `'wght' ${clampAx("r_wght", st.r_wght + dW + an * 300 * (rnd(i, j) - 0.5))}, 'CASL' ${clampAx("r_CASL", st.r_CASL + an * 0.6 * (rnd(j, i) - 0.5))}, ` +
            `'slnt' ${st.r_slnt}, 'CRSV' ${st.r_CRSV}, 'MONO' ${st.r_MONO}`;
        }
        const n = i * 13 + j;
        const jx = st.jitter * inst * nz(n, now, 1.9), jy = st.jitter * inst * nz(n + 77, now, 2.3);
        const wy = st.wave * Math.sin(now * 1.6 + (i * 5 + j) * 0.35) + st.baseline * (rnd(n, 3) - 0.5) * 2;
        const r = st.rot * inst * nz(n + 191, now, 0.7), sc = 1 + sn * (rnd(n, 11) - 0.5);
        s.style.transform = `translate(${jx.toFixed(3)}em, ${(jy + wy).toFixed(3)}em) rotate(${r.toFixed(2)}deg) scale(${sc.toFixed(3)})`;
      });
    },
  };
})();
