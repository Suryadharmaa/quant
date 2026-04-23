import { Router, type IRouter } from "express";

const router: IRouter = Router();

interface Bar { date: string; price: number }

const cache = new Map<string, { ts: number; data: Bar[] }>();
const TTL_MS = 10 * 60 * 1000;

const SYMBOLS: Record<string, { yf: string; label: string }> = {
  btc: { yf: "BTC-USD", label: "Bitcoin / USD" },
  xauusd: { yf: "GC=F", label: "Gold Futures / USD (XAU/USD proxy)" },
};

async function fetchYahoo(yfSymbol: string): Promise<Bar[]> {
  const url = `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(yfSymbol)}?interval=1d&range=2y`;
  const res = await fetch(url, {
    headers: {
      accept: "application/json",
      "user-agent":
        "Mozilla/5.0 (compatible; QuantRegime/1.0)",
    },
  });
  if (!res.ok) throw new Error(`Yahoo HTTP ${res.status}`);
  const json = (await res.json()) as {
    chart: {
      error?: { description: string };
      result?: Array<{
        timestamp: number[];
        indicators: { quote: Array<{ close: (number | null)[] }> };
      }>;
    };
  };
  if (json.chart.error) throw new Error(`Yahoo: ${json.chart.error.description}`);
  const r = json.chart.result?.[0];
  if (!r) throw new Error("Yahoo returned no result");
  const ts = r.timestamp || [];
  const closes = r.indicators?.quote?.[0]?.close || [];
  const out: Bar[] = [];
  for (let i = 0; i < ts.length; i++) {
    const c = closes[i];
    if (c == null || !Number.isFinite(c)) continue;
    out.push({
      date: new Date(ts[i] * 1000).toISOString().split("T")[0],
      price: c,
    });
  }
  return out;
}

router.get("/market", async (req, res) => {
  try {
    const symbol = String(req.query["symbol"] || "btc").toLowerCase();
    const meta = SYMBOLS[symbol];
    if (!meta) {
      res.status(400).json({ error: "symbol must be 'btc' or 'xauusd'" });
      return;
    }
    const cached = cache.get(symbol);
    const now = Date.now();
    if (cached && now - cached.ts < TTL_MS) {
      res.json({
        symbol,
        label: meta.label,
        source: "Yahoo Finance (cached)",
        count: cached.data.length,
        bars: cached.data,
      });
      return;
    }
    const bars = await fetchYahoo(meta.yf);
    if (bars.length < 100) throw new Error(`Only ${bars.length} bars returned`);
    cache.set(symbol, { ts: now, data: bars });
    res.json({
      symbol,
      label: meta.label,
      source: "Yahoo Finance",
      count: bars.length,
      bars,
    });
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    res.status(502).json({ error: msg });
  }
});

export default router;
