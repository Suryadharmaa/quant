import { useState, useEffect, useCallback, useRef, useMemo, Fragment } from "react";
import {
  kmeansFit, gmmFit, hmmFit,
  transitionMatrixFromLabels, stationaryDistribution, expectedDurations, nextStateForecast,
  standardize,
  type Mat,
} from "./lib/regime-models";
import {
  ScatterChart, Scatter, LineChart, Line, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  Cell, ReferenceLine, AreaChart, Area,
} from "recharts";

const C = {
  bg: "#050810", surface: "#080D1A", panel: "#0B1221", border: "#131E35",
  dimText: "#2E4870", midText: "#4A6899", bodyText: "#8AABCC", hiText: "#C4DEFF", white: "#EAF4FF",
  regimes: ["#00FFA3", "#FF3E5F", "#FFD200", "#5B8FFF", "#FF6B35", "#B44FFF"],
  green: "#00FFA3", red: "#FF3E5F", yellow: "#FFD200", blue: "#5B8FFF", purple: "#B44FFF", orange: "#FF6B35",
};

const REGIME_CHARS = [
  { name: "Bull Quiet", icon: "↗", desc: "Low vol, positive drift" },
  { name: "Bear Crisis", icon: "↘", desc: "High vol, negative returns" },
  { name: "Volatile Bull", icon: "⚡", desc: "High vol, positive trend" },
  { name: "Recovery", icon: "◎", desc: "Mean-reverting, low drawdown" },
  { name: "Sideways Chop", icon: "↔", desc: "Flat drift, moderate vol" },
  { name: "Melt-Up", icon: "★", desc: "Extreme positive momentum" },
];

const GROQ_MODELS = [
  { id: "llama-3.3-70b-versatile", label: "LLaMA 3.3 70B" },
  { id: "llama-3.1-8b-instant", label: "LLaMA 3.1 8B Fast" },
  { id: "mixtral-8x7b-32768", label: "Mixtral 8x7B" },
  { id: "gemma2-9b-it", label: "Gemma 2 9B" },
];

const ASSETS = [
  { id: "btc", label: "BTC/USD", source: "CoinGecko" },
  { id: "xauusd", label: "XAU/USD (Gold)", source: "Stooq" },
];

const FEAT_KEYS = ["return", "vol60", "drawdown", "mom20", "skewness", "kurtosis"] as const;
const FEAT_NICE = ["Return", "Vol60", "Drawdown", "Mom20", "Skewness", "Kurtosis"];
type FeatKey = typeof FEAT_KEYS[number];

export interface MarketBar {
  date: string; price: number; return: number; vol60: number; drawdown: number;
  mom20: number; skewness: number; kurtosis: number; absReturn: number;
  rsi14: number; ema20: number; ema50: number; bbWidth: number;
  regime?: number;
}

interface RawBar { date: string; price: number }

function safeMin(arr: number[], fallback = 0) { return arr.length ? Math.min(...arr) : fallback; }
function safeMax(arr: number[], fallback = 0) { return arr.length ? Math.max(...arr) : fallback; }

function computeIndicators(raw: RawBar[]): MarketBar[] {
  if (raw.length < 70) return [];
  const prices = raw.map(b => b.price);
  const dates = raw.map(b => b.date);
  const WV = 60, WM = 20, WRSI = 14, WBB = 20;
  const out: MarketBar[] = [];

  // EMA helper
  const emaArr = (period: number) => {
    const k = 2 / (period + 1);
    const arr: number[] = [];
    let prev = prices[0];
    for (let i = 0; i < prices.length; i++) {
      prev = i === 0 ? prices[0] : prices[i] * k + prev * (1 - k);
      arr.push(prev);
    }
    return arr;
  };
  const ema20 = emaArr(20);
  const ema50 = emaArr(50);

  // RSI
  const rsi: number[] = new Array(prices.length).fill(50);
  let avgGain = 0, avgLoss = 0;
  for (let i = 1; i <= WRSI && i < prices.length; i++) {
    const ch = prices[i] - prices[i - 1];
    if (ch > 0) avgGain += ch; else avgLoss -= ch;
  }
  avgGain /= WRSI; avgLoss /= WRSI;
  rsi[WRSI] = avgLoss === 0 ? 100 : 100 - 100 / (1 + avgGain / avgLoss);
  for (let i = WRSI + 1; i < prices.length; i++) {
    const ch = prices[i] - prices[i - 1];
    const g = ch > 0 ? ch : 0, l = ch < 0 ? -ch : 0;
    avgGain = (avgGain * (WRSI - 1) + g) / WRSI;
    avgLoss = (avgLoss * (WRSI - 1) + l) / WRSI;
    rsi[i] = avgLoss === 0 ? 100 : 100 - 100 / (1 + avgGain / avgLoss);
  }

  for (let i = WV; i < prices.length; i++) {
    const ret = (prices[i] - prices[i - 1]) / prices[i - 1];
    const slice: number[] = [];
    for (let j = i - WV; j < i; j++) slice.push((prices[j + 1] - prices[j]) / prices[j]);
    const mu = slice.reduce((a, b) => a + b, 0) / WV;
    const vari = slice.reduce((a, b) => a + (b - mu) ** 2, 0) / WV;
    const std = Math.sqrt(vari) || 0.0001;
    const vol = std * Math.sqrt(252);
    const skew = slice.reduce((a, b) => a + ((b - mu) / std) ** 3, 0) / WV;
    const kurt = slice.reduce((a, b) => a + ((b - mu) / std) ** 4, 0) / WV - 3;
    const window = prices.slice(Math.max(0, i - 90), i + 1);
    const maxP = safeMax(window, prices[i]);
    const dd = (prices[i] - maxP) / (maxP || 1);
    const mom20 = i >= WM ? (prices[i] - prices[i - WM]) / prices[i - WM] : 0;
    const bbSlice = prices.slice(Math.max(0, i - WBB), i + 1);
    const bbMean = bbSlice.reduce((a, b) => a + b, 0) / bbSlice.length;
    const bbStd = Math.sqrt(bbSlice.reduce((a, b) => a + (b - bbMean) ** 2, 0) / bbSlice.length);
    const bbWidth = bbMean ? (4 * bbStd) / bbMean : 0;
    out.push({
      date: dates[i], price: prices[i], return: ret,
      vol60: Math.max(0.005, Math.min(3, vol)),
      drawdown: dd, mom20,
      skewness: isFinite(skew) ? Math.max(-3, Math.min(3, skew)) : 0,
      kurtosis: isFinite(kurt) ? Math.max(-3, Math.min(10, kurt)) : 0,
      absReturn: Math.abs(ret),
      rsi14: rsi[i] || 50, ema20: ema20[i], ema50: ema50[i], bbWidth,
    });
  }
  return out;
}

function normalize(features: number[][]) {
  if (!features.length) return features;
  const n = features[0].length;
  const means = Array(n).fill(0), stds = Array(n).fill(0);
  features.forEach(f => f.forEach((v, i) => { means[i] += v; }));
  for (let i = 0; i < n; i++) means[i] /= features.length;
  features.forEach(f => f.forEach((v, i) => { stds[i] += (v - means[i]) ** 2; }));
  for (let i = 0; i < n; i++) stds[i] = Math.sqrt(stds[i] / features.length) || 1;
  return features.map(f => f.map((v, i) => (v - means[i]) / stds[i]));
}

function kmeans(features: number[][], k: number, maxIter = 150) {
  if (features.length < k) return { labels: new Array(features.length).fill(0), inertia: 0 };
  const norm = normalize(features);
  const step = Math.max(1, Math.floor(norm.length / k));
  let centers: number[][] = Array.from({ length: k }, (_, i) => [...(norm[Math.min(i * step, norm.length - 1)] || norm[0])]);
  let labels = new Array(norm.length).fill(0);
  for (let iter = 0; iter < maxIter; iter++) {
    const nl = norm.map(f => {
      let best = 0, bestD = Infinity;
      centers.forEach((c, ci) => {
        const d = Math.sqrt(f.reduce((s, v, i) => s + (v - c[i]) ** 2, 0));
        if (d < bestD) { bestD = d; best = ci; }
      });
      return best;
    });
    centers = Array.from({ length: k }, (_, c) => {
      const pts = norm.filter((_, i) => nl[i] === c);
      if (!pts.length) return centers[c];
      return pts[0].map((_, j) => pts.reduce((s, p) => s + p[j], 0) / pts.length);
    });
    if (nl.every((l, i) => l === labels[i])) break;
    labels = nl;
  }
  const inertia = norm.reduce((s, f, i) => {
    const c = centers[labels[i]] || centers[0];
    return s + f.reduce((ss, v, j) => ss + (v - c[j]) ** 2, 0);
  }, 0);
  // Reorder by avg vol of feature[1] if present
  const rs = Array.from({ length: k }, (_, c) => {
    const pts = features.filter((_, i) => labels[i] === c);
    const idx = features[0]?.length > 1 ? 1 : 0;
    return { c, avgVol: pts.length ? pts.reduce((s, p) => s + (p[idx] || 0), 0) / pts.length : 0 };
  }).sort((a, b) => a.avgVol - b.avgVol);
  const remap: Record<number, number> = {};
  rs.forEach((s, i) => { remap[s.c] = i; });
  return { labels: labels.map(l => remap[l] ?? 0), inertia };
}

function computeElbow(features: number[][], maxK = 8) {
  if (features.length < maxK) return [];
  return Array.from({ length: maxK - 1 }, (_, i) => {
    const k = i + 2;
    const { inertia } = kmeans(features, k, 80);
    return { k, inertia };
  });
}

interface RegimeStat {
  c: number; count: number; pct: string; annRet: string; annVol: string;
  sharpe: string; sortino: string; calmar: string; var95: string; cvar95: string;
  winRate: string; avgWin: string; avgLoss: string; maxDD: string; avgVol: string;
  skew: string; profitFactor: string;
}

function computeRegimeStats(data: MarketBar[], k: number): RegimeStat[] {
  const result: RegimeStat[] = [];
  for (let c = 0; c < k; c++) {
    const pts = data.filter(d => d.regime === c);
    if (!pts.length) continue;
    const rets = pts.map(d => d.return);
    const n = rets.length;
    const mean = rets.reduce((a, b) => a + b, 0) / n;
    const variance = rets.reduce((a, b) => a + (b - mean) ** 2, 0) / n;
    const std = Math.sqrt(variance) || 0.0001;
    const annRet = mean * 252, annVol = std * Math.sqrt(252);
    const sharpe = annRet / annVol;
    const dr = rets.filter(r => r < 0);
    const dStd = dr.length ? Math.sqrt(dr.reduce((a, b) => a + b ** 2, 0) / dr.length) * Math.sqrt(252) : 0.0001;
    const sortino = annRet / dStd;
    const sorted = [...rets].sort((a, b) => a - b);
    const var95 = sorted[Math.floor(n * 0.05)] || 0;
    const tailN = Math.max(1, Math.floor(n * 0.05));
    const cvar95 = sorted.slice(0, tailN).reduce((a, b) => a + b, 0) / tailN;
    const wins = rets.filter(r => r > 0), losses = rets.filter(r => r < 0);
    const winRate = wins.length / n;
    const avgWin = wins.length ? wins.reduce((a, b) => a + b, 0) / wins.length : 0;
    const avgLoss = losses.length ? losses.reduce((a, b) => a + b, 0) / losses.length : 0;
    const maxDD = safeMin(pts.map(d => d.drawdown), 0);
    const avgVol = pts.reduce((a, d) => a + d.vol60, 0) / n;
    const calmar = maxDD ? annRet / Math.abs(maxDD) : 0;
    const skew = pts.reduce((a, d) => a + d.skewness, 0) / n;
    const pf = avgLoss ? Math.abs(avgWin / avgLoss) : 99;
    result.push({
      c, count: n, pct: (n / Math.max(1, data.length) * 100).toFixed(1),
      annRet: (annRet * 100).toFixed(2), annVol: (annVol * 100).toFixed(2),
      sharpe: sharpe.toFixed(3), sortino: sortino.toFixed(3), calmar: calmar.toFixed(3),
      var95: (var95 * 100).toFixed(3), cvar95: (cvar95 * 100).toFixed(3),
      winRate: (winRate * 100).toFixed(1), avgWin: (avgWin * 100).toFixed(3),
      avgLoss: (avgLoss * 100).toFixed(3), maxDD: (maxDD * 100).toFixed(2),
      avgVol: (avgVol * 100).toFixed(1), skew: skew.toFixed(3),
      profitFactor: pf >= 99 ? "inf" : pf.toFixed(3),
    });
  }
  return result;
}

function computeTransitionMatrix(labels: number[], k: number) {
  const mat: number[][] = Array.from({ length: k }, () => Array(k).fill(0));
  for (let i = 1; i < labels.length; i++) {
    const a = labels[i - 1], b = labels[i];
    if (a >= 0 && a < k && b >= 0 && b < k) mat[a][b]++;
  }
  return mat.map(row => {
    const sum = row.reduce((a, b) => a + b, 0) || 1;
    return row.map(v => v / sum);
  });
}

function computeCumReturns(data: MarketBar[]) {
  let cum = 1;
  return data.map(d => {
    cum *= (1 + d.return);
    return { date: d.date, cum: (cum - 1) * 100, regime: d.regime };
  });
}

function computeBacktest(data: MarketBar[], activeRegimes: Set<number>) {
  let strat = 1, bh = 1;
  const res: { date: string; strategy: number; buyhold: number }[] = [];
  data.forEach((d, i) => {
    strat *= activeRegimes.has(d.regime ?? -1) ? (1 + d.return) : 1.00008;
    bh *= (1 + d.return);
    if (i % 5 === 0 || i === data.length - 1) res.push({ date: d.date, strategy: (strat - 1) * 100, buyhold: (bh - 1) * 100 });
  });
  return res;
}

function computeDurations(data: MarketBar[], k: number) {
  const dur: number[][] = Array.from({ length: k }, () => []);
  if (!data.length) return dur.map((_, c) => ({ c, count: 0, avg: "0", max: 0, min: 0 }));
  let st = 0;
  for (let i = 1; i <= data.length; i++) {
    const cur = data[st]?.regime;
    if (cur === undefined) { st = i; continue; }
    if (i === data.length || data[i]?.regime !== cur) {
      if (cur >= 0 && cur < k) dur[cur].push(i - st);
      st = i;
    }
  }
  return dur.map((d, c) => ({
    c, count: d.length,
    avg: d.length ? (d.reduce((a, b) => a + b, 0) / d.length).toFixed(1) : "0",
    max: safeMax(d, 0), min: safeMin(d, 0),
  }));
}

function corrMatrix(data: MarketBar[]) {
  if (!data.length) return Array.from({ length: 6 }, () => Array(6).fill(0));
  const feats = [
    data.map(d => d.return), data.map(d => d.vol60), data.map(d => d.drawdown),
    data.map(d => d.mom20), data.map(d => d.skewness), data.map(d => d.kurtosis),
  ];
  const means = feats.map(f => f.reduce((a, b) => a + b, 0) / f.length);
  const stds = feats.map((f, i) => Math.sqrt(f.reduce((a, b) => a + (b - means[i]) ** 2, 0) / f.length) || 1);
  return feats.map((fi, i) => feats.map((fj, j) => {
    const cov = fi.reduce((s, v, k) => s + (v - means[i]) * (fj[k] - means[j]), 0) / fi.length;
    return cov / (stds[i] * stds[j]);
  }));
}

function Dot({ color, glow, size = 8 }: { color: string; glow?: boolean; size?: number }) {
  return <span style={{ display: "inline-block", width: size, height: size, borderRadius: "50%", background: color, boxShadow: glow ? `0 0 7px ${color}` : undefined, flexShrink: 0 }} />;
}
function Tag({ label, color }: { label: string; color: string }) {
  return <span style={{ fontSize: 9, letterSpacing: 1.5, padding: "2px 7px", border: `1px solid ${color}44`, borderRadius: 3, color, background: `${color}11`, textTransform: "uppercase" }}>{label}</span>;
}
function Panel({ children, style }: { children: React.ReactNode; style?: React.CSSProperties }) {
  return <div style={{ background: C.panel, border: `1px solid ${C.border}`, borderRadius: 10, padding: "16px 18px", ...style }}>{children}</div>;
}
function PanelTitle({ children, right }: { children: React.ReactNode; right?: React.ReactNode }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12, gap: 12, flexWrap: "wrap" }}>
      <span style={{ fontSize: 9, letterSpacing: 2.5, color: C.dimText, textTransform: "uppercase" }}>{children}</span>
      {right}
    </div>
  );
}
function Btn({ children, onClick, active, color = C.blue, disabled, small }: { children: React.ReactNode; onClick?: () => void; active?: boolean; color?: string; disabled?: boolean; small?: boolean }) {
  return (
    <button onClick={onClick} disabled={disabled} style={{ padding: small ? "4px 10px" : "7px 14px", border: `1px solid ${active ? color : C.border}`, borderRadius: 5, background: active ? `${color}18` : "transparent", color: active ? color : C.bodyText, cursor: disabled ? "not-allowed" : "pointer", fontSize: small ? 9 : 11, fontFamily: "monospace", letterSpacing: small ? 1 : 0.5, opacity: disabled ? 0.4 : 1, transition: "all 0.15s" }}>{children}</button>
  );
}
function StatCell({ label, value, color, sub }: { label: string; value: React.ReactNode; color?: string; sub?: string }) {
  return (
    <div style={{ padding: "10px 12px", background: C.surface, border: `1px solid ${C.border}`, borderRadius: 7 }}>
      <div style={{ fontSize: 9, color: C.dimText, letterSpacing: 1.5, marginBottom: 4, textTransform: "uppercase" }}>{label}</div>
      <div style={{ fontSize: 16, fontWeight: 700, color: color || C.hiText, fontFamily: "monospace" }}>{value}</div>
      {sub && <div style={{ fontSize: 9, color: C.midText, marginTop: 2 }}>{sub}</div>}
    </div>
  );
}
function ScatterTip({ active, payload }: any) {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload; if (!d) return null;
  return (
    <div style={{ background: "#0B1221", border: `1px solid ${C.regimes[d.regime] || C.blue}`, borderRadius: 7, padding: "8px 12px", fontSize: 10, fontFamily: "monospace", color: C.hiText, minWidth: 140 }}>
      <div style={{ color: C.regimes[d.regime], fontWeight: 700, marginBottom: 5 }}>{REGIME_CHARS[d.regime]?.name || "R" + d.regime}</div>
      {payload.map((p: any, i: number) => (<div key={i}>{p.name}: <b>{typeof p.value === "number" ? p.value.toFixed(4) : p.value}</b></div>))}
      <div style={{ color: C.midText, marginTop: 4, fontSize: 9 }}>{d.date}</div>
    </div>
  );
}
function LineTip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: "#0B1221", border: `1px solid ${C.border}`, borderRadius: 7, padding: "8px 12px", fontSize: 10, fontFamily: "monospace", color: C.hiText }}>
      <div style={{ color: C.midText, marginBottom: 5, fontSize: 9 }}>{label}</div>
      {payload.map((p: any, i: number) => (<div key={i} style={{ color: p.color, marginBottom: 2 }}>{p.name}: <b>{typeof p.value === "number" ? p.value.toFixed(2) : p.value}</b></div>))}
    </div>
  );
}

function TimelineSVG({ data }: { data: MarketBar[] }) {
  if (!data.length) return null;
  const W = 1000, H = 200, PL = 55, PR = 15, PT = 15, PB = 28;
  const Wi = W - PL - PR, Hi = H - PT - PB;
  const prices = data.map(d => d.price);
  const minP = safeMin(prices), maxP = safeMax(prices);
  const rng = (maxP - minP) || 1;
  const xS = (i: number) => PL + (i / (data.length - 1 || 1)) * Wi;
  const yS = (p: number) => PT + Hi - ((p - minP) / rng) * Hi;
  const path = data.map((d, i) => `${i === 0 ? "M" : "L"}${xS(i).toFixed(1)},${yS(d.price).toFixed(1)}`).join(" ");
  const bands: { start: number; end: number; regime: number }[] = [];
  let st = 0;
  for (let i = 1; i <= data.length; i++) {
    const cur = data[st]?.regime;
    if (cur === undefined) { st = i; continue; }
    if (i === data.length || data[i]?.regime !== cur) {
      bands.push({ start: st, end: i - 1, regime: cur });
      st = i;
    }
  }
  const ticks: { i: number; label: string }[] = [];
  let prevYM: string | null = null;
  data.forEach((d, i) => { const ym = d.date.slice(0, 7); if (ym !== prevYM) { ticks.push({ i, label: ym }); prevYM = ym; } });
  const tf = ticks.filter((_, idx) => idx % Math.max(1, Math.floor(ticks.length / 12)) === 0);
  const pctLines = [0, 0.25, 0.5, 0.75, 1];
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", display: "block" }}>
      {bands.map((b, idx) => (
        <rect key={idx} x={xS(b.start)} y={PT} width={Math.max(1, xS(b.end) - xS(b.start) + 1.5)} height={Hi} fill={C.regimes[b.regime % C.regimes.length]} opacity={0.15} />
      ))}
      {pctLines.map(v => {
        const p = minP + v * rng; const y = yS(p);
        return (
          <g key={v}>
            <line x1={PL} y1={y} x2={PL + Wi} y2={y} stroke={C.border} strokeWidth={0.5} />
            <text x={PL - 5} y={y + 3.5} textAnchor="end" fill={C.dimText} fontSize={8} fontFamily="monospace">{p >= 1000 ? `${(p / 1000).toFixed(1)}k` : p.toFixed(1)}</text>
          </g>
        );
      })}
      <path d={path} fill="none" stroke={C.hiText} strokeWidth={1.5} opacity={0.9} />
      {tf.map((t, i) => (
        <g key={i}>
          <line x1={xS(t.i)} y1={PT + Hi} x2={xS(t.i)} y2={PT + Hi + 4} stroke={C.dimText} strokeWidth={0.5} />
          <text x={xS(t.i)} y={H - 4} textAnchor="middle" fill={C.dimText} fontSize={8} fontFamily="monospace">{t.label}</text>
        </g>
      ))}
    </svg>
  );
}

function TransitionMatrix({ matrix, k }: { matrix: number[][]; k: number }) {
  if (!matrix.length) return null;
  const max = safeMax(matrix.flat(), 1);
  return (
    <div style={{ display: "grid", gridTemplateColumns: `60px repeat(${k}, 1fr)`, gap: 2, fontFamily: "monospace", fontSize: 9 }}>
      <div />
      {Array.from({ length: k }, (_, c) => (<div key={c} style={{ textAlign: "center", color: C.regimes[c], padding: "4px 2px", letterSpacing: 1 }}>R{c}</div>))}
      {matrix.map((row, r) => (
        <Fragment key={r}>
          <div style={{ color: C.regimes[r], display: "flex", alignItems: "center", gap: 5, paddingRight: 8 }}><Dot color={C.regimes[r]} size={6} />R{r}</div>
          {row.map((val, c) => {
            const intensity = max ? val / max : 0;
            const isMain = r === c;
            return (
              <div key={c} style={{ background: `rgba(${isMain ? "0,255,163" : "91,143,255"},${intensity * 0.75})`, border: `1px solid ${C.border}`, borderRadius: 4, display: "flex", alignItems: "center", justifyContent: "center", padding: "7px 4px", color: intensity > 0.4 ? C.white : C.midText, fontWeight: isMain ? 700 : 400, fontSize: 10 }}>
                {(val * 100).toFixed(0)}%
              </div>
            );
          })}
        </Fragment>
      ))}
    </div>
  );
}

function CorrHeatmap({ data }: { data: MarketBar[] }) {
  const mat = useMemo(() => corrMatrix(data), [data]);
  const n = FEAT_NICE.length;
  return (
    <div style={{ overflowX: "auto" }}>
      <div style={{ display: "grid", gridTemplateColumns: `70px repeat(${n}, 1fr)`, gap: 2, fontSize: 8, fontFamily: "monospace", minWidth: 380 }}>
        <div />
        {FEAT_NICE.map(l => (<div key={l} style={{ textAlign: "center", color: C.midText, padding: "3px 2px", fontSize: 8 }}>{l.slice(0, 5)}</div>))}
        {Array.from({ length: n }, (_, r) => (
          <Fragment key={r}>
            <div style={{ color: C.midText, display: "flex", alignItems: "center", paddingRight: 6, fontSize: 8 }}>{FEAT_NICE[r]}</div>
            {Array.from({ length: n }, (_, c) => {
              const val = mat[r]?.[c] ?? 0;
              const abs = Math.abs(val);
              const rgb = val > 0 ? "0,255,163" : "255,62,95";
              return (
                <div key={c} style={{ background: `rgba(${rgb},${abs * 0.7})`, border: `1px solid ${C.border}`, borderRadius: 3, display: "flex", alignItems: "center", justifyContent: "center", padding: "5px 2px", color: abs > 0.5 ? C.white : C.midText, fontSize: 8 }}>{val.toFixed(2)}</div>
              );
            })}
          </Fragment>
        ))}
      </div>
    </div>
  );
}

interface ChatMsg { role: "user" | "assistant"; content: string; }

const BASE = (import.meta as any).env?.BASE_URL || "/";
const apiUrl = (path: string) => `${BASE}${path.replace(/^\/+/, "")}`.replace(/\/{2,}/g, "/").replace(":/", "://");

export default function App() {
  const [asset, setAsset] = useState<"btc" | "xauusd">("btc");
  const [k, setK] = useState(4);
  const [model, setModel] = useState<"kmeans" | "gmm" | "hmm">("kmeans");
  const [modelInfo, setModelInfo] = useState<{ logLik: number; bic: number; aic: number; iters: number } | null>(null);
  const [hmmTrans, setHmmTrans] = useState<Mat | null>(null);
  const [activeFeats, setActiveFeats] = useState<FeatKey[]>(["return", "vol60", "drawdown", "mom20"]);
  const [rawBars, setRawBars] = useState<RawBar[]>([]);
  const [data, setData] = useState<MarketBar[]>([]);
  const [tab, setTab] = useState("overview");
  const [groqKey, setGroqKey] = useState(() => localStorage.getItem("groq_key") || "");
  const [groqModel, setGroqModel] = useState(GROQ_MODELS[0].id);
  const [aiMessages, setAiMessages] = useState<ChatMsg[]>([]);
  const [aiInput, setAiInput] = useState("");
  const [aiLoading, setAiLoading] = useState(false);
  const [btRegimes, setBtRegimes] = useState<Set<number>>(new Set([0]));
  const [elbow, setElbow] = useState<{ k: number; inertia: number }[]>([]);
  const [loading, setLoading] = useState(true);
  const [computing, setComputing] = useState(false);
  const [fetchErr, setFetchErr] = useState<string>("");
  const [meta, setMeta] = useState<{ source: string; count: number; updated: string }>({ source: "", count: 0, updated: "" });
  const chatRef = useRef<HTMLDivElement>(null);

  // Fetch real data
  const fetchData = useCallback(async (sym: "btc" | "xauusd") => {
    setLoading(true);
    setFetchErr("");
    try {
      const res = await fetch(apiUrl(`api/market?symbol=${sym}`));
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      if (json.error) throw new Error(json.error);
      const bars: RawBar[] = json.bars || [];
      if (bars.length < 100) throw new Error(`Only ${bars.length} bars returned`);
      setRawBars(bars);
      setMeta({ source: json.source || "", count: bars.length, updated: bars[bars.length - 1]?.date || "" });
    } catch (e: any) {
      setFetchErr(e.message || String(e));
      setRawBars([]);
    }
    setLoading(false);
  }, []);

  useEffect(() => { fetchData(asset); }, [asset, fetchData]);

  // Compute when raw data, k, features, or model change
  useEffect(() => {
    if (!rawBars.length) { setData([]); setElbow([]); setModelInfo(null); setHmmTrans(null); return; }
    setComputing(true);
    const t = setTimeout(() => {
      const indi = computeIndicators(rawBars);
      if (indi.length < k * 5) { setData([]); setElbow([]); setComputing(false); return; }
      const feats = activeFeats.length ? activeFeats : ["return", "vol60"];
      const features = indi.map(d => feats.map(f => (d as any)[f] || 0));

      let labels: number[];
      let info: { logLik: number; bic: number; aic: number; iters: number } | null = null;
      let trans: Mat | null = null;

      if (model === "kmeans") {
        labels = kmeans(features, k).labels;
      } else {
        const { Z } = standardize(features);
        if (model === "gmm") {
          const r = gmmFit(Z, k, 80);
          labels = r.labels;
          info = { logLik: r.logLik, bic: r.bic, aic: r.aic, iters: r.iters };
        } else {
          const r = hmmFit(Z, k, 30);
          labels = r.labels;
          info = { logLik: r.logLik, bic: r.bic, aic: r.aic, iters: r.iters };
          trans = r.transMat;
        }
      }

      const withR = indi.map((d, i) => ({ ...d, regime: labels[i] ?? 0 }));
      setData(withR);
      setElbow(computeElbow(features));
      setModelInfo(info);
      setHmmTrans(trans);
      setComputing(false);
    }, 30);
    return () => clearTimeout(t);
  }, [rawBars, k, activeFeats, model]);

  useEffect(() => { localStorage.setItem("groq_key", groqKey); }, [groqKey]);

  const stats = useMemo(() => computeRegimeStats(data, k), [data, k]);
  const transMat = useMemo<Mat>(() => {
    if (!data.length) return [];
    if (model === "hmm" && hmmTrans && hmmTrans.length === k) return hmmTrans;
    return transitionMatrixFromLabels(data.map(d => d.regime ?? 0), k);
  }, [data, k, model, hmmTrans]);
  const stationary = useMemo(() => transMat.length ? stationaryDistribution(transMat) : [], [transMat]);
  const expDur = useMemo(() => transMat.length ? expectedDurations(transMat) : [], [transMat]);
  const nextForecast = useMemo(() => (transMat.length && currentRegimeValid()) ? nextStateForecast(transMat, data[data.length-1]?.regime ?? 0) : [], [transMat, data]);
  function currentRegimeValid() { const r = data[data.length-1]?.regime; return r != null && r >= 0 && r < transMat.length; }
  const cumRets = useMemo(() => computeCumReturns(data), [data]);
  const btResults = useMemo(() => computeBacktest(data, btRegimes), [data, btRegimes]);
  const durations = useMemo(() => computeDurations(data, k), [data, k]);
  const scatterData = useMemo(() => data.map(d => ({ x: d.return * 100, y: d.vol60 * 100, regime: d.regime ?? 0, date: d.date })), [data]);

  const last = data[data.length - 1];
  const currentRegime = last?.regime ?? -1;
  const currentRegimeStat = stats.find(s => s.c === currentRegime);

  const toggleFeat = (f: FeatKey) => setActiveFeats(prev => {
    const next = prev.includes(f) ? prev.filter(x => x !== f) : [...prev, f];
    return next.length < 2 ? prev : next;
  });

  const buildContext = () => {
    if (!stats.length || !last) return `Asset:${asset} (no data loaded)`;
    const st = stats.map(s => `R${s.c}(${REGIME_CHARS[s.c]?.name} ${s.pct}%): Ret=${s.annRet}% Vol=${s.annVol}% SR=${s.sharpe} Sortino=${s.sortino} WR=${s.winRate}% MaxDD=${s.maxDD}% VaR95=${s.var95}%`).join("\n");
    const tr = transMat.map((row, r) => `R${r}=>[${row.map((v, c) => `R${c}:${(v * 100).toFixed(0)}%`).join(",")}]`).join(" ");
    const stat = stationary.map((v, c) => `R${c}(${REGIME_CHARS[c]?.name}):${(v*100).toFixed(1)}%`).join(", ");
    const dur = expDur.map((d, c) => `R${c}:${d.toFixed(2)}d`).join(", ");
    const fc = nextForecast.length ? nextForecast.map((v, c) => `R${c}(${REGIME_CHARS[c]?.name}):${(v*100).toFixed(1)}%`).join(", ") : "n/a";
    const fitLine = modelInfo ? `LogLik=${modelInfo.logLik.toFixed(1)} BIC=${modelInfo.bic.toFixed(1)} AIC=${modelInfo.aic.toFixed(1)} EM_iters=${modelInfo.iters}` : "n/a (k-means is geometric, no likelihood)";
    return `Real Market Data
Asset: ${asset.toUpperCase()} (source: ${meta.source})
Bars: ${meta.count} | Last date: ${meta.updated} | Last price: ${last.price.toFixed(2)}
Current regime: R${currentRegime} (${REGIME_CHARS[currentRegime]?.name})
Current indicators: RSI14=${last.rsi14.toFixed(1)} | EMA20=${last.ema20.toFixed(2)} | EMA50=${last.ema50.toFixed(2)} | BBwidth=${(last.bbWidth*100).toFixed(2)}% | Vol60=${(last.vol60*100).toFixed(1)}% | Mom20=${(last.mom20*100).toFixed(2)}% | DD=${(last.drawdown*100).toFixed(2)}%
Model: ${model.toUpperCase()} | K=${k} clusters | Features: ${activeFeats.join(",")}
Model fit: ${fitLine}
Regime statistics:
${st}
Transition probabilities (${model === "hmm" ? "HMM Baum-Welch" : "empirical"}):
${tr}
Stationary distribution (long-run regime mass): ${stat}
Expected duration per regime (1/(1-Pii)): ${dur}
Next-day regime forecast from current state: ${fc}`;
  };

  const sendAI = async (prompt: string) => {
    if (!groqKey.trim()) {
      setAiMessages(m => [...m, { role: "user", content: prompt }, { role: "assistant", content: "Please enter your Groq API key (free at console.groq.com) in the field above." }]);
      return;
    }
    const newMsg: ChatMsg = { role: "user", content: prompt };
    const history = [...aiMessages, newMsg];
    setAiMessages(history);
    setAiInput("");
    setAiLoading(true);
    try {
      const resp = await fetch("https://api.groq.com/openai/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${groqKey}` },
        body: JSON.stringify({
          model: groqModel, max_tokens: 1500, temperature: 0.3,
          messages: [
            { role: "system", content: `You are an elite quantitative analyst working with REAL market data from a regime clustering model. Be precise, actionable, and reference the actual indicators and regime stats provided.\n\n${buildContext()}` },
            ...history.map(m => ({ role: m.role, content: m.content })),
          ],
        }),
      });
      const d = await resp.json();
      if (d.error) throw new Error(d.error.message || JSON.stringify(d.error));
      setAiMessages(m => [...m, { role: "assistant", content: d.choices?.[0]?.message?.content || "No response" }]);
    } catch (e: any) {
      setAiMessages(m => [...m, { role: "assistant", content: `Error: ${e.message}` }]);
    }
    setAiLoading(false);
  };

  useEffect(() => { if (chatRef.current) chatRef.current.scrollTop = chatRef.current.scrollHeight; }, [aiMessages]);

  const TABS = [
    { id: "overview", label: "Overview" }, { id: "analytics", label: "Analytics" },
    { id: "risk", label: "Risk Metrics" }, { id: "indicators", label: "Indicators" },
    { id: "features", label: "Features" }, { id: "backtest", label: "Backtest" },
    { id: "ai", label: "AI Console" },
  ];
  const btFinal = btResults.length ? (+btResults[btResults.length - 1].strategy).toFixed(1) : "--";
  const bhFinal = btResults.length ? (+btResults[btResults.length - 1].buyhold).toFixed(1) : "--";
  const alphaNum = btResults.length ? btResults[btResults.length - 1].strategy - btResults[btResults.length - 1].buyhold : 0;
  const alpha = btResults.length ? alphaNum.toFixed(1) : "--";

  const showLoader = loading || computing;
  const noData = !showLoader && !data.length;

  return (
    <div style={{ minHeight: "100vh", background: C.bg, color: C.bodyText, fontSize: 12 }}>
      {/* Header */}
      <div style={{ background: C.surface, borderBottom: `1px solid ${C.border}`, padding: "0 24px", display: "flex", alignItems: "stretch", height: 46, flexWrap: "wrap" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, paddingRight: 24, borderRight: `1px solid ${C.border}` }}>
          <Dot color={C.green} glow size={7} />
          <span style={{ fontSize: 11, color: C.hiText, fontWeight: 700, letterSpacing: 0.5 }}>QUANT REGIME</span>
          <Tag label="LIVE" color={C.green} />
        </div>
        <div style={{ display: "flex", alignItems: "center", paddingLeft: 8, flex: 1, overflowX: "auto" }}>
          {TABS.map(t => (
            <button key={t.id} onClick={() => setTab(t.id)} style={{ padding: "0 16px", height: 46, border: "none", background: "transparent", borderBottom: tab === t.id ? `2px solid ${C.blue}` : "2px solid transparent", color: tab === t.id ? C.blue : C.midText, cursor: "pointer", fontSize: 10, fontFamily: "monospace", letterSpacing: 1, textTransform: "uppercase", whiteSpace: "nowrap" }}>{t.label}</button>
          ))}
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8, paddingLeft: 24, borderLeft: `1px solid ${C.border}`, fontSize: 9, color: C.dimText }}>
          <span>{asset.toUpperCase()}</span><span>K={k}</span><span>{data.length}D</span>
          {meta.source && <Tag label={meta.source} color={C.purple} />}
        </div>
      </div>

      {/* Controls */}
      <div style={{ background: C.surface, borderBottom: `1px solid ${C.border}`, padding: "10px 24px", display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
        <div style={{ display: "flex", gap: 4 }}>
          {ASSETS.map(a => (<Btn key={a.id} active={asset === a.id} color={C.green} onClick={() => setAsset(a.id as any)}>{a.label}</Btn>))}
        </div>
        <div style={{ width: 1, height: 24, background: C.border }} />
        <span style={{ fontSize: 9, color: C.dimText }}>K:</span>
        {[2, 3, 4, 5, 6].map(n => (<Btn key={n} small active={k === n} color={C.regimes[n - 2]} onClick={() => setK(n)}>{String(n)}</Btn>))}
        <div style={{ width: 1, height: 24, background: C.border }} />
        <span style={{ fontSize: 9, color: C.dimText }}>Model:</span>
        {([
          { id: "kmeans", label: "K-Means" },
          { id: "gmm", label: "GMM" },
          { id: "hmm", label: "HMM" },
        ] as const).map(m => (
          <Btn key={m.id} small active={model === m.id} color={C.purple} onClick={() => setModel(m.id)}>{m.label}</Btn>
        ))}
        <div style={{ width: 1, height: 24, background: C.border }} />
        <span style={{ fontSize: 9, color: C.dimText }}>Features:</span>
        {FEAT_KEYS.map((f, i) => (<Btn key={f} small active={activeFeats.includes(f)} color={C.yellow} onClick={() => toggleFeat(f)}>{FEAT_NICE[i]}</Btn>))}
        <div style={{ marginLeft: "auto", display: "flex", gap: 8, alignItems: "center" }}>
          {meta.updated && <span style={{ fontSize: 9, color: C.midText }}>Updated: {meta.updated}</span>}
          <Btn onClick={() => fetchData(asset)} color={C.blue} small>Refresh Data</Btn>
        </div>
      </div>

      <div style={{ padding: "16px 24px 40px" }}>
        {fetchErr && (
          <div style={{ background: `${C.red}15`, border: `1px solid ${C.red}`, borderRadius: 8, padding: "12px 16px", marginBottom: 16, color: C.red, fontSize: 11 }}>
            <b>Data fetch failed:</b> {fetchErr}<br />
            <span style={{ color: C.midText, fontSize: 10 }}>Click Refresh Data to retry. The free providers (CoinGecko / Stooq) occasionally rate-limit.</span>
          </div>
        )}
        {showLoader ? (
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: 80, gap: 12 }}>
            <div style={{ fontSize: 11, color: C.dimText, letterSpacing: 3 }}>{loading ? "FETCHING REAL MARKET DATA..." : "COMPUTING REGIME CLUSTERS..."}</div>
            <div style={{ width: 200, height: 2, background: C.border, borderRadius: 1, overflow: "hidden" }}>
              <div style={{ height: "100%", background: C.blue, animation: "slide 1.2s linear infinite", width: "40%" }} />
            </div>
          </div>
        ) : noData ? (
          <Panel><div style={{ padding: 40, textAlign: "center", color: C.midText }}>No data available. Try Refresh Data.</div></Panel>
        ) : (
          <>
            {tab === "overview" && (
              <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                {/* Live snapshot */}
                {last && currentRegimeStat && (
                  <Panel style={{ background: `linear-gradient(135deg, ${C.regimes[currentRegime]}10, transparent)`, borderColor: `${C.regimes[currentRegime]}50` }}>
                    <div style={{ display: "flex", justifyContent: "space-between", flexWrap: "wrap", gap: 16 }}>
                      <div>
                        <div style={{ fontSize: 9, color: C.dimText, letterSpacing: 2, marginBottom: 6 }}>CURRENT MARKET STATE — {asset.toUpperCase()}</div>
                        <div style={{ display: "flex", gap: 14, alignItems: "baseline" }}>
                          <span style={{ fontSize: 28, fontWeight: 700, color: C.hiText }}>${last.price.toFixed(2)}</span>
                          <span style={{ fontSize: 14, color: last.return >= 0 ? C.green : C.red }}>{last.return >= 0 ? "+" : ""}{(last.return * 100).toFixed(2)}%</span>
                          <span style={{ fontSize: 10, color: C.midText }}>{last.date}</span>
                        </div>
                      </div>
                      <div style={{ display: "flex", gap: 10 }}>
                        <StatCell label="Regime" value={REGIME_CHARS[currentRegime]?.name || `R${currentRegime}`} color={C.regimes[currentRegime]} />
                        <StatCell label="RSI 14" value={last.rsi14.toFixed(1)} color={last.rsi14 > 70 ? C.red : last.rsi14 < 30 ? C.green : C.yellow} />
                        <StatCell label="60D Vol" value={`${(last.vol60 * 100).toFixed(1)}%`} color={C.yellow} />
                        <StatCell label="Drawdown" value={`${(last.drawdown * 100).toFixed(2)}%`} color={C.red} />
                        <StatCell label="Regime Sharpe" value={currentRegimeStat.sharpe} color={parseFloat(currentRegimeStat.sharpe) > 0 ? C.green : C.red} />
                      </div>
                    </div>
                  </Panel>
                )}

                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 10 }}>
                  {stats.map(s => (
                    <div key={s.c} style={{ background: C.panel, border: `1px solid ${C.regimes[s.c]}30`, borderRadius: 9, padding: "12px 14px", borderLeft: `3px solid ${C.regimes[s.c]}` }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 8 }}>
                        <Dot color={C.regimes[s.c]} glow size={7} />
                        <span style={{ fontSize: 9, color: C.regimes[s.c], fontWeight: 700, letterSpacing: 1 }}>{REGIME_CHARS[s.c]?.name || "R" + s.c}</span>
                      </div>
                      <div style={{ fontSize: 18, fontWeight: 700, color: parseFloat(s.annRet) >= 0 ? C.green : C.red, marginBottom: 3 }}>{parseFloat(s.annRet) >= 0 ? "+" : ""}{s.annRet}%</div>
                      <div style={{ fontSize: 9, color: C.dimText, marginBottom: 6 }}>Ann. Return</div>
                      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 4, fontSize: 9 }}>
                        <div><span style={{ color: C.dimText }}>Vol: </span><b style={{ color: C.yellow }}>{s.annVol}%</b></div>
                        <div><span style={{ color: C.dimText }}>SR: </span><b style={{ color: C.blue }}>{s.sharpe}</b></div>
                        <div><span style={{ color: C.dimText }}>Time: </span><b style={{ color: C.midText }}>{s.pct}%</b></div>
                        <div><span style={{ color: C.dimText }}>WR: </span><b style={{ color: C.midText }}>{s.winRate}%</b></div>
                      </div>
                    </div>
                  ))}
                </div>

                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(360px, 1fr))", gap: 16 }}>
                  <Panel>
                    <PanelTitle right={<Tag label="K-Means" color={C.blue} />}>Return vs Volatility</PanelTitle>
                    <ResponsiveContainer width="100%" height={240}>
                      <ScatterChart margin={{ top: 5, right: 5, bottom: 0, left: -15 }}>
                        <CartesianGrid strokeDasharray="2 4" stroke={C.border} />
                        <XAxis dataKey="x" name="Return" unit="%" tick={{ fill: C.dimText, fontSize: 8 }} />
                        <YAxis dataKey="y" name="Vol" unit="%" tick={{ fill: C.dimText, fontSize: 8 }} />
                        <Tooltip content={<ScatterTip />} />
                        <ReferenceLine x={0} stroke={C.midText} strokeDasharray="3 2" strokeWidth={1} opacity={0.5} />
                        <Scatter data={scatterData} isAnimationActive={false}>
                          {scatterData.map((d, i) => <Cell key={i} fill={C.regimes[d.regime % C.regimes.length]} opacity={0.65} />)}
                        </Scatter>
                      </ScatterChart>
                    </ResponsiveContainer>
                  </Panel>
                  <Panel>
                    <PanelTitle right={<Tag label="Rolling" color={C.purple} />}>Cumulative Returns</PanelTitle>
                    <ResponsiveContainer width="100%" height={240}>
                      <AreaChart data={cumRets.filter((_, i) => i % 2 === 0)} margin={{ top: 5, right: 5, bottom: 0, left: -15 }}>
                        <defs><linearGradient id="cg" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor={C.green} stopOpacity={0.3} /><stop offset="95%" stopColor={C.green} stopOpacity={0} /></linearGradient></defs>
                        <CartesianGrid strokeDasharray="2 4" stroke={C.border} />
                        <XAxis dataKey="date" tick={false} />
                        <YAxis tick={{ fill: C.dimText, fontSize: 8 }} unit="%" />
                        <Tooltip content={<LineTip />} />
                        <ReferenceLine y={0} stroke={C.midText} strokeDasharray="3 2" strokeWidth={1} />
                        <Area type="monotone" dataKey="cum" name="Cum Return %" stroke={C.green} fill="url(#cg)" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                      </AreaChart>
                    </ResponsiveContainer>
                  </Panel>
                </div>

                <Panel>
                  <PanelTitle right={
                    <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
                      {Array.from({ length: k }, (_, c) => (
                        <div key={c} style={{ display: "flex", alignItems: "center", gap: 5 }}>
                          <Dot color={C.regimes[c]} size={6} /><span style={{ fontSize: 8, color: C.regimes[c] }}>{REGIME_CHARS[c]?.name || "R" + c}</span>
                        </div>
                      ))}
                    </div>
                  }>{asset.toUpperCase()} Price Regime Timeline ({meta.count}d real data)</PanelTitle>
                  <TimelineSVG data={data} />
                </Panel>
              </div>
            )}

            {tab === "analytics" && (
              <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(360px, 1fr))", gap: 16 }}>
                  <Panel>
                    <PanelTitle right={<Tag label={model === "hmm" ? "HMM (Baum-Welch)" : "Empirical"} color={C.orange} />}>Regime Transition Matrix</PanelTitle>
                    <TransitionMatrix matrix={transMat} k={k} />
                    <div style={{ marginTop: 10, fontSize: 9, color: C.dimText }}>
                      Rows = FROM regime · Cols = TO regime · {model === "hmm" ? "HMM-learned via forward-backward" : "Empirical from label sequence"}
                    </div>
                  </Panel>
                  <Panel>
                    <PanelTitle>Regime Duration Distribution</PanelTitle>
                    <ResponsiveContainer width="100%" height={200}>
                      <BarChart data={durations} margin={{ top: 5, right: 5, bottom: 0, left: -15 }}>
                        <CartesianGrid strokeDasharray="2 4" stroke={C.border} />
                        <XAxis dataKey="c" tickFormatter={v => `R${v}`} tick={{ fill: C.dimText, fontSize: 9 }} />
                        <YAxis tick={{ fill: C.dimText, fontSize: 8 }} />
                        <Tooltip content={({ active, payload }: any) => {
                          if (!active || !payload?.length) return null;
                          const d = payload[0]?.payload;
                          return (<div style={{ background: C.panel, border: `1px solid ${C.border}`, borderRadius: 6, padding: "8px 12px", fontSize: 9 }}>
                            <div style={{ color: C.regimes[d.c], marginBottom: 4 }}>{REGIME_CHARS[d.c]?.name}</div>
                            <div>Avg: <b>{d.avg}d</b></div><div>Max: <b>{d.max}d</b></div><div>Episodes: <b>{d.count}</b></div>
                          </div>);
                        }} />
                        <Bar dataKey="avg" name="Avg Duration (d)" radius={[3, 3, 0, 0]} isAnimationActive={false}>
                          {durations.map((d, i) => <Cell key={i} fill={C.regimes[d.c]} />)}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </Panel>
                </div>

                <Panel>
                  <PanelTitle right={<Tag label={`${model.toUpperCase()} · k=${k}`} color={C.purple} />}>
                    Markov Math &mdash; Stationary, Expected Duration, Next-Step Forecast
                  </PanelTitle>
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 12, marginTop: 4 }}>
                    {/* Stationary distribution */}
                    <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8, padding: 12 }}>
                      <div style={{ fontSize: 9, color: C.dimText, letterSpacing: 1.5, marginBottom: 8 }}>STATIONARY DISTRIBUTION  &pi;</div>
                      <div style={{ fontSize: 8, color: C.midText, marginBottom: 8, fontFamily: "monospace" }}>&pi; = &pi;&middot;P (long-run regime mass)</div>
                      {Array.from({ length: k }, (_, c) => {
                        const v = stationary[c] || 0;
                        return (
                          <div key={c} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 5 }}>
                            <Dot color={C.regimes[c]} size={6} />
                            <span style={{ fontSize: 9, color: C.regimes[c], minWidth: 70 }}>{REGIME_CHARS[c]?.name || `R${c}`}</span>
                            <div style={{ flex: 1, height: 5, background: C.border, borderRadius: 2, overflow: "hidden" }}>
                              <div style={{ height: "100%", width: `${v * 100}%`, background: C.regimes[c] }} />
                            </div>
                            <span style={{ fontSize: 10, color: C.hiText, fontFamily: "monospace", minWidth: 42, textAlign: "right" }}>{(v * 100).toFixed(1)}%</span>
                          </div>
                        );
                      })}
                    </div>
                    {/* Expected duration */}
                    <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8, padding: 12 }}>
                      <div style={{ fontSize: 9, color: C.dimText, letterSpacing: 1.5, marginBottom: 8 }}>EXPECTED DURATION  E[D]</div>
                      <div style={{ fontSize: 8, color: C.midText, marginBottom: 8, fontFamily: "monospace" }}>E[D&#7522;] = 1 / (1 &minus; P&#7522;&#7522;)</div>
                      {Array.from({ length: k }, (_, c) => {
                        const d = expDur[c] || 0;
                        const stay = transMat[c]?.[c] || 0;
                        return (
                          <div key={c} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6, fontSize: 10 }}>
                            <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
                              <Dot color={C.regimes[c]} size={6} />
                              <span style={{ color: C.regimes[c] }}>{REGIME_CHARS[c]?.name || `R${c}`}</span>
                            </span>
                            <span style={{ fontFamily: "monospace", color: C.midText, fontSize: 9 }}>P&#7522;&#7522;={(stay * 100).toFixed(1)}%</span>
                            <b style={{ fontFamily: "monospace", color: C.hiText }}>{d.toFixed(2)}d</b>
                          </div>
                        );
                      })}
                    </div>
                    {/* Next-step forecast */}
                    <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8, padding: 12 }}>
                      <div style={{ fontSize: 9, color: C.dimText, letterSpacing: 1.5, marginBottom: 8 }}>
                        NEXT-DAY FORECAST &middot; FROM <span style={{ color: C.regimes[currentRegime] }}>{REGIME_CHARS[currentRegime]?.name || `R${currentRegime}`}</span>
                      </div>
                      <div style={{ fontSize: 8, color: C.midText, marginBottom: 8, fontFamily: "monospace" }}>P(s&#7596;&#8330;&#8321; | s&#7596;) &mdash; row of P</div>
                      {nextForecast.length ? Array.from({ length: k }, (_, c) => {
                        const v = nextForecast[c] || 0;
                        return (
                          <div key={c} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 5 }}>
                            <Dot color={C.regimes[c]} size={6} />
                            <span style={{ fontSize: 9, color: C.regimes[c], minWidth: 70 }}>{REGIME_CHARS[c]?.name || `R${c}`}</span>
                            <div style={{ flex: 1, height: 5, background: C.border, borderRadius: 2, overflow: "hidden" }}>
                              <div style={{ height: "100%", width: `${v * 100}%`, background: C.regimes[c] }} />
                            </div>
                            <span style={{ fontSize: 10, color: C.hiText, fontFamily: "monospace", minWidth: 42, textAlign: "right" }}>{(v * 100).toFixed(1)}%</span>
                          </div>
                        );
                      }) : <div style={{ fontSize: 10, color: C.dimText }}>n/a</div>}
                    </div>
                    {/* Model fit */}
                    <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8, padding: 12 }}>
                      <div style={{ fontSize: 9, color: C.dimText, letterSpacing: 1.5, marginBottom: 8 }}>MODEL FIT &middot; {model.toUpperCase()}</div>
                      {model === "kmeans" ? (
                        <div style={{ fontSize: 10, color: C.midText, lineHeight: 1.6 }}>
                          K-Means is geometric &mdash; no likelihood. Switch to <b style={{ color: C.purple }}>GMM</b> or <b style={{ color: C.purple }}>HMM</b> for log-likelihood, BIC, AIC and to compare models.
                        </div>
                      ) : modelInfo ? (
                        <div style={{ fontFamily: "monospace", fontSize: 11, lineHeight: 1.7 }}>
                          <div style={{ display: "flex", justifyContent: "space-between" }}>
                            <span style={{ color: C.dimText }}>log-Likelihood</span>
                            <b style={{ color: C.green }}>{modelInfo.logLik.toFixed(1)}</b>
                          </div>
                          <div style={{ display: "flex", justifyContent: "space-between" }}>
                            <span style={{ color: C.dimText }}>BIC (lower=better)</span>
                            <b style={{ color: C.blue }}>{modelInfo.bic.toFixed(1)}</b>
                          </div>
                          <div style={{ display: "flex", justifyContent: "space-between" }}>
                            <span style={{ color: C.dimText }}>AIC (lower=better)</span>
                            <b style={{ color: C.yellow }}>{modelInfo.aic.toFixed(1)}</b>
                          </div>
                          <div style={{ display: "flex", justifyContent: "space-between" }}>
                            <span style={{ color: C.dimText }}>EM iterations</span>
                            <b style={{ color: C.midText }}>{modelInfo.iters}</b>
                          </div>
                          <div style={{ marginTop: 8, fontSize: 8, color: C.dimText, lineHeight: 1.5 }}>
                            Try k=2..6 to see which has the lowest BIC &mdash; that&apos;s the most parsimonious model.
                          </div>
                        </div>
                      ) : <div style={{ fontSize: 10, color: C.dimText }}>fitting...</div>}
                    </div>
                  </div>
                  <div style={{ marginTop: 10, fontSize: 9, color: C.dimText, lineHeight: 1.5 }}>
                    <b style={{ color: C.midText }}>K-Means</b>: geometric clustering, sequence-agnostic.
                    &nbsp;<b style={{ color: C.midText }}>GMM</b>: probabilistic mixture, full covariance, EM-fit.
                    &nbsp;<b style={{ color: C.midText }}>HMM</b>: time-aware, transitions learned via Baum-Welch, decoded with Viterbi.
                    &nbsp;Stationary distribution from raising the transition matrix to the 200th power. Expected duration uses geometric-distribution mean.
                  </div>
                </Panel>

                <Panel>
                  <PanelTitle>Daily Return Histogram per Regime</PanelTitle>
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 12 }}>
                    {stats.map(s => {
                      const pts = data.filter(d => d.regime === s.c);
                      if (!pts.length) return null;
                      const retsArr = pts.map(d => d.return * 100);
                      const mn = safeMin(retsArr), mx = safeMax(retsArr);
                      const bw = ((mx - mn) / 20) || 0.1;
                      const buckets = Array(20).fill(0);
                      retsArr.forEach(r => { const b = Math.min(19, Math.max(0, Math.floor((r - mn) / bw))); buckets[b]++; });
                      const histData = buckets.map((v, i) => ({ x: +(mn + i * bw).toFixed(2), v }));
                      return (
                        <div key={s.c} style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8, padding: "10px" }}>
                          <div style={{ display: "flex", gap: 6, alignItems: "center", marginBottom: 8 }}>
                            <Dot color={C.regimes[s.c]} size={6} />
                            <span style={{ fontSize: 9, color: C.regimes[s.c] }}>{REGIME_CHARS[s.c]?.name}</span>
                          </div>
                          <ResponsiveContainer width="100%" height={80}>
                            <BarChart data={histData} margin={{ top: 0, right: 0, bottom: 0, left: -30 }}>
                              <YAxis tick={false} />
                              <XAxis dataKey="x" tick={{ fill: C.dimText, fontSize: 7 }} interval={4} />
                              <Bar dataKey="v" fill={C.regimes[s.c]} opacity={0.7} isAnimationActive={false} />
                            </BarChart>
                          </ResponsiveContainer>
                        </div>
                      );
                    })}
                  </div>
                </Panel>
              </div>
            )}

            {tab === "risk" && (
              <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                <Panel>
                  <PanelTitle right={<Tag label="Comprehensive" color={C.red} />}>Full Risk Metrics Matrix</PanelTitle>
                  <div style={{ overflowX: "auto" }}>
                    <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 10, fontFamily: "monospace" }}>
                      <thead>
                        <tr style={{ borderBottom: `1px solid ${C.border}` }}>
                          {["Regime", "Time%", "AnnRet", "AnnVol", "Sharpe", "Sortino", "Calmar", "MaxDD", "VaR95", "CVaR95", "WinRate", "AvgWin", "AvgLoss", "ProfitFactor", "Skew"].map(h => (
                            <th key={h} style={{ padding: "8px 10px", color: C.dimText, fontWeight: 400, textAlign: h === "Regime" ? "left" : "right", fontSize: 9, letterSpacing: 0.5, whiteSpace: "nowrap" }}>{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {stats.map(s => (
                          <tr key={s.c} style={{ borderBottom: `1px solid ${C.border}30` }}>
                            <td style={{ padding: "9px 10px" }}>
                              <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
                                <Dot color={C.regimes[s.c]} glow size={7} />
                                <div>
                                  <div style={{ color: C.regimes[s.c], fontSize: 10, fontWeight: 700 }}>{REGIME_CHARS[s.c]?.name}</div>
                                  <div style={{ color: C.dimText, fontSize: 8 }}>{s.count}d</div>
                                </div>
                              </div>
                            </td>
                            <td style={{ textAlign: "right", padding: "9px 10px", color: C.midText }}>{s.pct}%</td>
                            <td style={{ textAlign: "right", padding: "9px 10px", color: parseFloat(s.annRet) >= 0 ? C.green : C.red, fontWeight: 700 }}>{parseFloat(s.annRet) >= 0 ? "+" : ""}{s.annRet}%</td>
                            <td style={{ textAlign: "right", padding: "9px 10px", color: C.yellow }}>{s.annVol}%</td>
                            <td style={{ textAlign: "right", padding: "9px 10px", color: parseFloat(s.sharpe) >= 1 ? C.green : parseFloat(s.sharpe) >= 0 ? C.yellow : C.red, fontWeight: 700 }}>{s.sharpe}</td>
                            <td style={{ textAlign: "right", padding: "9px 10px", color: parseFloat(s.sortino) >= 1 ? C.green : parseFloat(s.sortino) >= 0 ? C.yellow : C.red }}>{s.sortino}</td>
                            <td style={{ textAlign: "right", padding: "9px 10px", color: C.blue }}>{s.calmar}</td>
                            <td style={{ textAlign: "right", padding: "9px 10px", color: C.red }}>{s.maxDD}%</td>
                            <td style={{ textAlign: "right", padding: "9px 10px", color: C.orange }}>{s.var95}%</td>
                            <td style={{ textAlign: "right", padding: "9px 10px", color: C.red }}>{s.cvar95}%</td>
                            <td style={{ textAlign: "right", padding: "9px 10px", color: C.midText }}>{s.winRate}%</td>
                            <td style={{ textAlign: "right", padding: "9px 10px", color: C.green }}>{s.avgWin}%</td>
                            <td style={{ textAlign: "right", padding: "9px 10px", color: C.red }}>{s.avgLoss}%</td>
                            <td style={{ textAlign: "right", padding: "9px 10px", color: parseFloat(s.profitFactor) >= 1.5 ? C.green : C.yellow }}>{s.profitFactor}</td>
                            <td style={{ textAlign: "right", padding: "9px 10px", color: parseFloat(s.skew) >= 0 ? C.blue : C.orange }}>{s.skew}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </Panel>

                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(360px, 1fr))", gap: 16 }}>
                  <Panel>
                    <PanelTitle>Sharpe vs Sortino</PanelTitle>
                    <ResponsiveContainer width="100%" height={200}>
                      <BarChart data={stats.map(s => ({ name: "R" + s.c, sharpe: +s.sharpe, sortino: +s.sortino }))} margin={{ top: 5, right: 5, bottom: 0, left: -15 }}>
                        <CartesianGrid strokeDasharray="2 4" stroke={C.border} />
                        <XAxis dataKey="name" tick={{ fill: C.dimText, fontSize: 9 }} />
                        <YAxis tick={{ fill: C.dimText, fontSize: 8 }} />
                        <Tooltip content={<LineTip />} />
                        <ReferenceLine y={0} stroke={C.midText} strokeWidth={1} />
                        <ReferenceLine y={1} stroke={C.green} strokeDasharray="3 2" strokeWidth={1} opacity={0.5} />
                        <Bar dataKey="sharpe" name="Sharpe" fill={C.blue} opacity={0.8} radius={[3, 3, 0, 0]} isAnimationActive={false} />
                        <Bar dataKey="sortino" name="Sortino" fill={C.purple} opacity={0.8} radius={[3, 3, 0, 0]} isAnimationActive={false} />
                      </BarChart>
                    </ResponsiveContainer>
                  </Panel>
                  <Panel>
                    <PanelTitle>Drawdown and VaR</PanelTitle>
                    <ResponsiveContainer width="100%" height={200}>
                      <BarChart data={stats.map(s => ({ name: "R" + s.c, maxDD: Math.abs(+s.maxDD), var95: Math.abs(+s.var95), cvar95: Math.abs(+s.cvar95) }))} margin={{ top: 5, right: 5, bottom: 0, left: -15 }}>
                        <CartesianGrid strokeDasharray="2 4" stroke={C.border} />
                        <XAxis dataKey="name" tick={{ fill: C.dimText, fontSize: 9 }} />
                        <YAxis tick={{ fill: C.dimText, fontSize: 8 }} unit="%" />
                        <Tooltip content={<LineTip />} />
                        <Bar dataKey="maxDD" name="Max DD" fill={C.red} opacity={0.7} radius={[3, 3, 0, 0]} isAnimationActive={false} />
                        <Bar dataKey="var95" name="VaR 95%" fill={C.orange} opacity={0.7} radius={[3, 3, 0, 0]} isAnimationActive={false} />
                        <Bar dataKey="cvar95" name="CVaR 95%" fill={C.yellow} opacity={0.5} radius={[3, 3, 0, 0]} isAnimationActive={false} />
                      </BarChart>
                    </ResponsiveContainer>
                  </Panel>
                </div>
              </div>
            )}

            {tab === "indicators" && (
              <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                <Panel>
                  <PanelTitle right={<Tag label="Price + EMA20/50" color={C.blue} />}>Price with Moving Averages</PanelTitle>
                  <ResponsiveContainer width="100%" height={260}>
                    <LineChart data={data.filter((_, i) => i % 2 === 0)} margin={{ top: 5, right: 5, bottom: 0, left: 0 }}>
                      <CartesianGrid strokeDasharray="2 4" stroke={C.border} />
                      <XAxis dataKey="date" tick={false} />
                      <YAxis tick={{ fill: C.dimText, fontSize: 8 }} domain={["auto", "auto"]} />
                      <Tooltip content={<LineTip />} />
                      <Line type="monotone" dataKey="price" name="Price" stroke={C.hiText} strokeWidth={1.5} dot={false} isAnimationActive={false} />
                      <Line type="monotone" dataKey="ema20" name="EMA 20" stroke={C.blue} strokeWidth={1} dot={false} isAnimationActive={false} />
                      <Line type="monotone" dataKey="ema50" name="EMA 50" stroke={C.orange} strokeWidth={1} dot={false} isAnimationActive={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </Panel>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(360px, 1fr))", gap: 16 }}>
                  <Panel>
                    <PanelTitle right={<Tag label="Momentum" color={C.purple} />}>RSI 14</PanelTitle>
                    <ResponsiveContainer width="100%" height={200}>
                      <LineChart data={data.filter((_, i) => i % 2 === 0)} margin={{ top: 5, right: 5, bottom: 0, left: -15 }}>
                        <CartesianGrid strokeDasharray="2 4" stroke={C.border} />
                        <XAxis dataKey="date" tick={false} />
                        <YAxis tick={{ fill: C.dimText, fontSize: 8 }} domain={[0, 100]} />
                        <Tooltip content={<LineTip />} />
                        <ReferenceLine y={70} stroke={C.red} strokeDasharray="3 2" />
                        <ReferenceLine y={30} stroke={C.green} strokeDasharray="3 2" />
                        <Line type="monotone" dataKey="rsi14" name="RSI" stroke={C.purple} strokeWidth={1.2} dot={false} isAnimationActive={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </Panel>
                  <Panel>
                    <PanelTitle right={<Tag label="Volatility" color={C.yellow} />}>60D Annualized Vol & BB Width</PanelTitle>
                    <ResponsiveContainer width="100%" height={200}>
                      <LineChart data={data.filter((_, i) => i % 2 === 0).map(d => ({ date: d.date, vol: d.vol60 * 100, bb: d.bbWidth * 100 }))} margin={{ top: 5, right: 5, bottom: 0, left: -15 }}>
                        <CartesianGrid strokeDasharray="2 4" stroke={C.border} />
                        <XAxis dataKey="date" tick={false} />
                        <YAxis tick={{ fill: C.dimText, fontSize: 8 }} unit="%" />
                        <Tooltip content={<LineTip />} />
                        <Line type="monotone" dataKey="vol" name="Vol60" stroke={C.yellow} strokeWidth={1.2} dot={false} isAnimationActive={false} />
                        <Line type="monotone" dataKey="bb" name="BB Width" stroke={C.orange} strokeWidth={1} dot={false} isAnimationActive={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </Panel>
                </div>
                <Panel>
                  <PanelTitle right={<Tag label="Risk" color={C.red} />}>Drawdown Curve</PanelTitle>
                  <ResponsiveContainer width="100%" height={180}>
                    <AreaChart data={data.filter((_, i) => i % 2 === 0).map(d => ({ date: d.date, dd: d.drawdown * 100 }))} margin={{ top: 5, right: 5, bottom: 0, left: -15 }}>
                      <defs><linearGradient id="ddg" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor={C.red} stopOpacity={0.4} /><stop offset="95%" stopColor={C.red} stopOpacity={0} /></linearGradient></defs>
                      <CartesianGrid strokeDasharray="2 4" stroke={C.border} />
                      <XAxis dataKey="date" tick={false} />
                      <YAxis tick={{ fill: C.dimText, fontSize: 8 }} unit="%" />
                      <Tooltip content={<LineTip />} />
                      <Area type="monotone" dataKey="dd" name="Drawdown %" stroke={C.red} fill="url(#ddg)" strokeWidth={1.2} dot={false} isAnimationActive={false} />
                    </AreaChart>
                  </ResponsiveContainer>
                </Panel>
              </div>
            )}

            {tab === "features" && (
              <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(360px, 1fr))", gap: 16 }}>
                  <Panel>
                    <PanelTitle right={<Tag label="K-Selection" color={C.yellow} />}>Elbow Chart</PanelTitle>
                    <ResponsiveContainer width="100%" height={200}>
                      <LineChart data={elbow} margin={{ top: 5, right: 5, bottom: 0, left: 0 }}>
                        <CartesianGrid strokeDasharray="2 4" stroke={C.border} />
                        <XAxis dataKey="k" tick={{ fill: C.dimText, fontSize: 9 }} />
                        <YAxis tick={{ fill: C.dimText, fontSize: 8 }} />
                        <Tooltip content={<LineTip />} />
                        <ReferenceLine x={k} stroke={C.yellow} strokeDasharray="3 2" strokeWidth={1.5} />
                        <Line type="monotone" dataKey="inertia" name="Inertia" stroke={C.yellow} strokeWidth={2} dot={{ fill: C.yellow, r: 4 }} isAnimationActive={false} />
                      </LineChart>
                    </ResponsiveContainer>
                    <div style={{ fontSize: 9, color: C.dimText, marginTop: 6 }}>Pick K at the elbow where inertia drop flattens. Selected K={k}.</div>
                  </Panel>
                  <Panel>
                    <PanelTitle right={<Tag label="Pearson" color={C.blue} />}>Feature Correlation Matrix</PanelTitle>
                    {data.length > 0 && <CorrHeatmap data={data} />}
                  </Panel>
                </div>

                <Panel>
                  <PanelTitle>Feature Toggle (min 2 active)</PanelTitle>
                  <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
                    {FEAT_KEYS.map((f, i) => {
                      const active = activeFeats.includes(f);
                      const descs: Record<string, string> = {
                        "return": "Daily log return", "vol60": "60d annualized vol", "drawdown": "90d max drawdown",
                        "mom20": "20d momentum", "skewness": "Return skewness", "kurtosis": "Excess kurtosis",
                      };
                      return (
                        <div key={f} onClick={() => toggleFeat(f)} style={{ padding: "10px 16px", border: `1px solid ${active ? C.yellow : C.border}`, borderRadius: 8, cursor: "pointer", background: active ? `${C.yellow}15` : "transparent", minWidth: 120 }}>
                          <div style={{ fontSize: 11, color: active ? C.yellow : C.midText, fontWeight: 700 }}>{FEAT_NICE[i]}</div>
                          <div style={{ fontSize: 8, color: C.dimText, marginTop: 3 }}>{descs[f]}</div>
                          <div style={{ marginTop: 6, fontSize: 8, color: active ? C.yellow : C.dimText }}>{active ? "Active" : "Inactive"}</div>
                        </div>
                      );
                    })}
                  </div>
                </Panel>

                <Panel>
                  <PanelTitle>Feature Variance Contribution per Regime</PanelTitle>
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 10 }}>
                    {stats.map(s => {
                      const pts = data.filter(d => d.regime === s.c);
                      if (!pts.length) return null;
                      const featVars = FEAT_KEYS.map(f => {
                        const vals = pts.map(d => (d as any)[f] || 0);
                        const m = vals.reduce((a, b) => a + b, 0) / vals.length;
                        return { f, v: vals.reduce((a, b) => a + (b - m) ** 2, 0) / vals.length };
                      });
                      const total = featVars.reduce((a, b) => a + b.v, 0) || 1;
                      return (
                        <div key={s.c} style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8, padding: "10px 12px" }}>
                          <div style={{ display: "flex", gap: 6, alignItems: "center", marginBottom: 10 }}>
                            <Dot color={C.regimes[s.c]} glow size={6} />
                            <span style={{ fontSize: 9, color: C.regimes[s.c], fontWeight: 700 }}>{REGIME_CHARS[s.c]?.name}</span>
                          </div>
                          {[...featVars].sort((a, b) => b.v - a.v).map(fv => (
                            <div key={fv.f} style={{ marginBottom: 6 }}>
                              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 2 }}>
                                <span style={{ fontSize: 8, color: C.dimText }}>{fv.f}</span>
                                <span style={{ fontSize: 8, color: C.midText }}>{(fv.v / total * 100).toFixed(1)}%</span>
                              </div>
                              <div style={{ height: 3, background: C.border, borderRadius: 2 }}>
                                <div style={{ height: "100%", background: C.regimes[s.c], borderRadius: 2, width: `${fv.v / total * 100}%`, opacity: 0.8 }} />
                              </div>
                            </div>
                          ))}
                        </div>
                      );
                    })}
                  </div>
                </Panel>
              </div>
            )}

            {tab === "backtest" && (
              <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                <Panel>
                  <PanelTitle right={<Tag label="Long/Cash" color={C.green} />}>Select Regimes to Trade LONG</PanelTitle>
                  <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginBottom: 12 }}>
                    {Array.from({ length: k }, (_, c) => {
                      const active = btRegimes.has(c);
                      const stat = stats.find(s => s.c === c);
                      return (
                        <div key={c} onClick={() => setBtRegimes(prev => { const next = new Set(prev); if (next.has(c)) next.delete(c); else next.add(c); return next; })} style={{ padding: "10px 16px", border: `1px solid ${active ? C.regimes[c] : C.border}`, borderRadius: 8, cursor: "pointer", background: active ? `${C.regimes[c]}18` : "transparent" }}>
                          <div style={{ display: "flex", gap: 7, alignItems: "center", marginBottom: 4 }}>
                            <Dot color={C.regimes[c]} glow={active} size={7} />
                            <span style={{ fontSize: 10, color: C.regimes[c], fontWeight: 700 }}>{REGIME_CHARS[c]?.name}</span>
                          </div>
                          <div style={{ fontSize: 8, color: C.dimText }}>SR: {stat?.sharpe || "--"} Ret: {stat?.annRet || "--"}%</div>
                          <div style={{ marginTop: 4, fontSize: 9, color: active ? C.green : C.dimText }}>{active ? "LONG" : "CASH"}</div>
                        </div>
                      );
                    })}
                  </div>
                  <div style={{ fontSize: 9, color: C.dimText }}>Long selected regimes. Cash earns 0.008%/day. No slippage or costs.</div>
                </Panel>

                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 10 }}>
                  <StatCell label="Strategy Return" value={`${btFinal}%`} color={parseFloat(btFinal) >= 0 ? C.green : C.red} />
                  <StatCell label="Buy and Hold" value={`${bhFinal}%`} color={C.midText} />
                  <StatCell label="Alpha" value={`${alpha}%`} color={alphaNum >= 0 ? C.blue : C.red} />
                  <StatCell label="Active Regimes" value={btRegimes.size} sub={`of ${k} total`} />
                </div>

                <Panel>
                  <PanelTitle>Strategy vs Buy and Hold</PanelTitle>
                  <ResponsiveContainer width="100%" height={260}>
                    <LineChart data={btResults} margin={{ top: 5, right: 5, bottom: 0, left: -10 }}>
                      <CartesianGrid strokeDasharray="2 4" stroke={C.border} />
                      <XAxis dataKey="date" tick={false} />
                      <YAxis tick={{ fill: C.dimText, fontSize: 8 }} unit="%" />
                      <Tooltip content={<LineTip />} />
                      <ReferenceLine y={0} stroke={C.midText} strokeWidth={1} />
                      <Line type="monotone" dataKey="strategy" name="Regime Strategy" stroke={C.green} strokeWidth={2} dot={false} isAnimationActive={false} />
                      <Line type="monotone" dataKey="buyhold" name="Buy and Hold" stroke={C.midText} strokeWidth={1.5} dot={false} strokeDasharray="4 3" isAnimationActive={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </Panel>
              </div>
            )}

            {tab === "ai" && (
              <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                <Panel>
                  <PanelTitle right={<Tag label="Groq API" color={C.purple} />}>AI Configuration (your key, stored locally)</PanelTitle>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: 12, marginBottom: 14 }}>
                    <div>
                      <div style={{ fontSize: 9, color: C.dimText, marginBottom: 5, letterSpacing: 1 }}>GROQ API KEY — free at console.groq.com</div>
                      <input type="password" value={groqKey} onChange={e => setGroqKey(e.target.value)} placeholder="gsk_xxxxxxxxxxxxxxxxxxxx" style={{ width: "100%", background: C.surface, border: `1px solid ${groqKey ? C.purple : C.border}`, borderRadius: 6, padding: "8px 12px", color: C.hiText, fontSize: 11, fontFamily: "monospace", outline: "none", boxSizing: "border-box" }} />
                    </div>
                    <div>
                      <div style={{ fontSize: 9, color: C.dimText, marginBottom: 5, letterSpacing: 1 }}>MODEL</div>
                      <select value={groqModel} onChange={e => setGroqModel(e.target.value)} style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 6, padding: "8px 12px", color: C.bodyText, fontSize: 10, fontFamily: "monospace", outline: "none" }}>
                        {GROQ_MODELS.map(m => <option key={m.id} value={m.id}>{m.label}</option>)}
                      </select>
                    </div>
                  </div>
                  <div style={{ fontSize: 9, color: C.dimText, marginBottom: 8 }}>Quick Prompts (real {asset.toUpperCase()} data injected automatically):</div>
                  <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                    {[
                      { label: "Live Market Read", p: `Given the current ${asset.toUpperCase()} regime and indicators, what does the market say RIGHT NOW? Bullish/bearish bias, key levels to watch, and conviction.` },
                      { label: "Full Regime Report", p: "Give a complete regime characterization. For each regime: market character, risk profile, alpha opportunities, position sizing guidance." },
                      { label: "Transition Signals", p: "Based on the transition matrix, what early-warning signals should I monitor to predict regime changes? What's the most likely next regime from current?" },
                      { label: "Risk Deep Dive", p: "Deep risk analysis: which regime has the greatest tail risk? How to adjust hedging across regimes?" },
                      { label: "Strategy Design", p: "Design an optimal regime-aware systematic trading strategy with entry/exit rules, position sizing, and risk controls." },
                      { label: "Trade Setup", p: `Based on current regime, RSI, EMA cross, and volatility, give me a concrete trade setup for ${asset.toUpperCase()}: entry, stop, target, size relative to account.` },
                    ].map(q => (
                      <button key={q.label} onClick={() => sendAI(q.p)} disabled={aiLoading || !groqKey} style={{ padding: "6px 12px", border: `1px solid ${C.purple}40`, borderRadius: 5, background: `${C.purple}10`, color: C.purple, cursor: groqKey && !aiLoading ? "pointer" : "not-allowed", fontSize: 9, fontFamily: "monospace", opacity: groqKey && !aiLoading ? 1 : 0.4 }}>{q.label}</button>
                    ))}
                  </div>
                </Panel>

                <Panel>
                  <PanelTitle>Quant AI Conversation</PanelTitle>
                  <div ref={chatRef} style={{ height: 380, overflowY: "auto", marginBottom: 12, display: "flex", flexDirection: "column", gap: 12 }}>
                    {aiMessages.length === 0 && (
                      <div style={{ padding: 24, textAlign: "center", color: C.dimText, fontSize: 11 }}>
                        Enter your Groq API key and use a preset prompt or type your own quant question.
                        <br />
                        <span style={{ fontSize: 9, display: "block", marginTop: 8 }}>Real {asset.toUpperCase()} regime context ({k} regimes, {data.length}d, source: {meta.source}) sent automatically with every prompt.</span>
                      </div>
                    )}
                    {aiMessages.map((m, i) => (
                      <div key={i} style={{ display: "flex", gap: 10, flexDirection: m.role === "user" ? "row-reverse" : "row" }}>
                        <div style={{ width: 28, height: 28, borderRadius: "50%", flexShrink: 0, background: m.role === "user" ? `${C.blue}30` : `${C.purple}30`, border: `1px solid ${m.role === "user" ? C.blue : C.purple}`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 9, color: m.role === "user" ? C.blue : C.purple }}>{m.role === "user" ? "U" : "AI"}</div>
                        <div style={{ maxWidth: "80%", padding: "10px 14px", borderRadius: 8, fontSize: 10, lineHeight: 1.7, background: m.role === "user" ? `${C.blue}12` : `${C.purple}10`, border: `1px solid ${m.role === "user" ? C.blue + "30" : C.purple + "30"}`, color: C.bodyText, whiteSpace: "pre-wrap", fontFamily: "monospace" }}>{m.content}</div>
                      </div>
                    ))}
                    {aiLoading && (
                      <div style={{ display: "flex", gap: 10, alignItems: "flex-start" }}>
                        <div style={{ width: 28, height: 28, borderRadius: "50%", background: `${C.purple}30`, border: `1px solid ${C.purple}`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 9, color: C.purple }}>AI</div>
                        <div style={{ padding: "10px 14px", background: `${C.purple}10`, border: `1px solid ${C.purple}30`, borderRadius: 8, display: "flex", gap: 4, alignItems: "center" }}>
                          {[0, 1, 2].map(i => (<div key={i} style={{ width: 6, height: 6, borderRadius: "50%", background: C.purple, animation: `bounce 0.8s ${i * 0.15}s infinite` }} />))}
                        </div>
                      </div>
                    )}
                  </div>
                  <div style={{ display: "flex", gap: 8 }}>
                    <input value={aiInput} onChange={e => setAiInput(e.target.value)} onKeyDown={e => e.key === "Enter" && !e.shiftKey && aiInput.trim() && sendAI(aiInput)} placeholder="Ask about regime dynamics, risk, strategy..." disabled={aiLoading || !groqKey} style={{ flex: 1, background: C.surface, border: `1px solid ${C.border}`, borderRadius: 6, padding: "9px 14px", color: C.hiText, fontSize: 11, fontFamily: "monospace", outline: "none" }} />
                    <button onClick={() => aiInput.trim() && sendAI(aiInput)} disabled={aiLoading || !groqKey || !aiInput.trim()} style={{ padding: "9px 18px", border: `1px solid ${C.purple}`, borderRadius: 6, background: `${C.purple}20`, color: C.purple, cursor: "pointer", fontSize: 11, fontFamily: "monospace", opacity: aiLoading || !groqKey || !aiInput.trim() ? 0.4 : 1 }}>Send</button>
                    {aiMessages.length > 0 && (<button onClick={() => setAiMessages([])} style={{ padding: "9px 12px", border: `1px solid ${C.border}`, borderRadius: 6, background: "transparent", color: C.dimText, cursor: "pointer", fontSize: 10, fontFamily: "monospace" }}>Clear</button>)}
                  </div>
                </Panel>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
