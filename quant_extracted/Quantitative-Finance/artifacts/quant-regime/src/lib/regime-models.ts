// Quant Regime Models: GMM, Gaussian HMM, Markov utilities.
// Pure TypeScript, no external deps. Adapted from the math in
// the Kaggle notebook "ML 5a Market regimes prediction using Clustering"
// (Gaussian HMM + GMM with BIC/AIC selection + stationary distribution).

export type Vec = number[];
export type Mat = number[][];

// ---------- Linear algebra (small matrices, full covariance) ----------

export function matMul(A: Mat, B: Mat): Mat {
  const n = A.length, m = B[0].length, p = B.length;
  const out: Mat = Array.from({ length: n }, () => new Array(m).fill(0));
  for (let i = 0; i < n; i++)
    for (let k = 0; k < p; k++) {
      const a = A[i][k];
      for (let j = 0; j < m; j++) out[i][j] += a * B[k][j];
    }
  return out;
}

export function matPow(A: Mat, n: number): Mat {
  const d = A.length;
  let result: Mat = Array.from({ length: d }, (_, i) =>
    Array.from({ length: d }, (_, j) => (i === j ? 1 : 0))
  );
  let base = A.map((r) => r.slice());
  let e = n;
  while (e > 0) {
    if (e & 1) result = matMul(result, base);
    base = matMul(base, base);
    e = Math.floor(e / 2);
  }
  return result;
}

// LU decomposition: returns {L,U,P,sign,singular}. Solve / det / inverse via this.
function lu(A: Mat) {
  const n = A.length;
  const M = A.map((r) => r.slice());
  const P = Array.from({ length: n }, (_, i) => i);
  let sign = 1;
  for (let i = 0; i < n; i++) {
    let maxV = Math.abs(M[i][i]), maxR = i;
    for (let r = i + 1; r < n; r++) {
      if (Math.abs(M[r][i]) > maxV) { maxV = Math.abs(M[r][i]); maxR = r; }
    }
    if (maxV < 1e-14) return { M, P, sign, singular: true };
    if (maxR !== i) {
      [M[i], M[maxR]] = [M[maxR], M[i]];
      [P[i], P[maxR]] = [P[maxR], P[i]];
      sign = -sign;
    }
    for (let r = i + 1; r < n; r++) {
      M[r][i] /= M[i][i];
      for (let c = i + 1; c < n; c++) M[r][c] -= M[r][i] * M[i][c];
    }
  }
  return { M, P, sign, singular: false };
}

function logDet(A: Mat): number {
  const d = A.length;
  // Add tiny ridge for numerical stability
  const R: Mat = A.map((r, i) => r.map((v, j) => v + (i === j ? 1e-9 : 0)));
  const { M, sign, singular } = lu(R);
  if (singular) return -Infinity;
  let s = 0;
  for (let i = 0; i < d; i++) s += Math.log(Math.abs(M[i][i]));
  return s + (sign < 0 ? Math.log(1) : 0); // |det|; log of |det|
}

function solve(A: Mat, b: Vec): Vec | null {
  const n = A.length;
  const R: Mat = A.map((r, i) => r.map((v, j) => v + (i === j ? 1e-9 : 0)));
  const { M, P, singular } = lu(R);
  if (singular) return null;
  const y = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    let s = b[P[i]];
    for (let j = 0; j < i; j++) s -= M[i][j] * y[j];
    y[i] = s;
  }
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let s = y[i];
    for (let j = i + 1; j < n; j++) s -= M[i][j] * x[j];
    x[i] = s / M[i][i];
  }
  return x;
}

// ---------- Multivariate Normal log-pdf ----------

export function logPdfGaussian(x: Vec, mean: Vec, cov: Mat): number {
  const d = x.length;
  const diff = x.map((v, i) => v - mean[i]);
  const sol = solve(cov, diff);
  if (!sol) return -1e10;
  let quad = 0;
  for (let i = 0; i < d; i++) quad += diff[i] * sol[i];
  const ld = logDet(cov);
  if (!isFinite(ld)) return -1e10;
  return -0.5 * (d * Math.log(2 * Math.PI) + ld + quad);
}

// ---------- Standardisation ----------

export function standardize(X: Mat): { Z: Mat; mean: Vec; std: Vec } {
  const n = X.length, d = X[0]?.length || 0;
  const mean = new Array(d).fill(0);
  const std = new Array(d).fill(0);
  for (const r of X) for (let j = 0; j < d; j++) mean[j] += r[j];
  for (let j = 0; j < d; j++) mean[j] /= n || 1;
  for (const r of X) for (let j = 0; j < d; j++) std[j] += (r[j] - mean[j]) ** 2;
  for (let j = 0; j < d; j++) std[j] = Math.sqrt(std[j] / (n || 1)) || 1;
  const Z = X.map((r) => r.map((v, j) => (v - mean[j]) / std[j]));
  return { Z, mean, std };
}

// ---------- K-means++ (used as init for GMM/HMM) ----------

function kmeansPlusPlusInit(X: Mat, k: number, seed = 1): Mat {
  // Simple deterministic seeded RNG (mulberry32)
  let s = seed >>> 0;
  const rand = () => {
    s = (s + 0x6D2B79F5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const n = X.length;
  const centers: Mat = [];
  centers.push(X[Math.floor(rand() * n)].slice());
  while (centers.length < k) {
    const d2 = X.map((p) => {
      let best = Infinity;
      for (const c of centers) {
        let s = 0;
        for (let j = 0; j < p.length; j++) s += (p[j] - c[j]) ** 2;
        if (s < best) best = s;
      }
      return best;
    });
    const sum = d2.reduce((a, b) => a + b, 0);
    let r = rand() * sum;
    let idx = 0;
    for (; idx < n; idx++) { r -= d2[idx]; if (r <= 0) break; }
    centers.push(X[Math.min(idx, n - 1)].slice());
  }
  return centers;
}

export function kmeansFit(X: Mat, k: number, maxIter = 150, seed = 1) {
  const n = X.length;
  if (n < k) return { labels: new Array(n).fill(0), inertia: 0, centers: [] as Mat };
  let centers = kmeansPlusPlusInit(X, k, seed);
  const labels = new Array(n).fill(0);
  for (let it = 0; it < maxIter; it++) {
    let changed = false;
    for (let i = 0; i < n; i++) {
      let best = 0, bd = Infinity;
      for (let c = 0; c < k; c++) {
        let s = 0;
        for (let j = 0; j < X[i].length; j++) s += (X[i][j] - centers[c][j]) ** 2;
        if (s < bd) { bd = s; best = c; }
      }
      if (labels[i] !== best) { labels[i] = best; changed = true; }
    }
    const newC: Mat = Array.from({ length: k }, () => new Array(X[0].length).fill(0));
    const cnt = new Array(k).fill(0);
    for (let i = 0; i < n; i++) {
      cnt[labels[i]]++;
      for (let j = 0; j < X[i].length; j++) newC[labels[i]][j] += X[i][j];
    }
    for (let c = 0; c < k; c++) {
      if (cnt[c] === 0) newC[c] = centers[c];
      else for (let j = 0; j < newC[c].length; j++) newC[c][j] /= cnt[c];
    }
    centers = newC;
    if (!changed) break;
  }
  let inertia = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < X[i].length; j++) inertia += (X[i][j] - centers[labels[i]][j]) ** 2;
  }
  return { labels, inertia, centers };
}

// ---------- GMM (full covariance, EM) ----------

export interface GMMResult {
  labels: number[];
  resp: Mat;          // n x k responsibilities
  weights: Vec;       // k
  means: Mat;         // k x d
  covs: Mat[];        // k x (d x d)
  logLik: number;
  bic: number;
  aic: number;
  iters: number;
}

export function gmmFit(X: Mat, k: number, maxIter = 100, tol = 1e-4, seed = 1): GMMResult {
  const n = X.length, d = X[0].length;
  // Init from kmeans
  const km = kmeansFit(X, k, 60, seed);
  const weights = new Array(k).fill(1 / k);
  const means: Mat = km.centers.length === k
    ? km.centers.map((c) => c.slice())
    : Array.from({ length: k }, (_, i) => X[i % n].slice());
  const covs: Mat[] = Array.from({ length: k }, () =>
    Array.from({ length: d }, (_, i) =>
      Array.from({ length: d }, (_, j) => (i === j ? 1 : 0))
    )
  );
  // Init covs from kmeans clusters
  for (let c = 0; c < k; c++) {
    const pts = X.filter((_, i) => km.labels[i] === c);
    if (pts.length > d) {
      const m = means[c];
      const cov: Mat = Array.from({ length: d }, () => new Array(d).fill(0));
      for (const p of pts) {
        for (let i = 0; i < d; i++)
          for (let j = 0; j < d; j++)
            cov[i][j] += (p[i] - m[i]) * (p[j] - m[j]);
      }
      for (let i = 0; i < d; i++)
        for (let j = 0; j < d; j++)
          cov[i][j] = cov[i][j] / pts.length + (i === j ? 1e-4 : 0);
      covs[c] = cov;
      weights[c] = pts.length / n;
    }
  }

  let prevLL = -Infinity;
  let resp: Mat = Array.from({ length: n }, () => new Array(k).fill(0));
  let it = 0;
  for (it = 0; it < maxIter; it++) {
    // E-step
    let ll = 0;
    for (let i = 0; i < n; i++) {
      const lp = new Array(k);
      for (let c = 0; c < k; c++) lp[c] = Math.log(weights[c] + 1e-300) + logPdfGaussian(X[i], means[c], covs[c]);
      const mx = Math.max(...lp);
      let s = 0;
      for (let c = 0; c < k; c++) s += Math.exp(lp[c] - mx);
      const logSum = mx + Math.log(s);
      ll += logSum;
      for (let c = 0; c < k; c++) resp[i][c] = Math.exp(lp[c] - logSum);
    }
    if (Math.abs(ll - prevLL) < tol) { prevLL = ll; break; }
    prevLL = ll;
    // M-step
    const Nk = new Array(k).fill(0);
    for (let c = 0; c < k; c++) for (let i = 0; i < n; i++) Nk[c] += resp[i][c];
    for (let c = 0; c < k; c++) {
      weights[c] = Nk[c] / n;
      const m = new Array(d).fill(0);
      for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) m[j] += resp[i][c] * X[i][j];
      for (let j = 0; j < d; j++) m[j] /= Nk[c] || 1;
      means[c] = m;
      const cov: Mat = Array.from({ length: d }, () => new Array(d).fill(0));
      for (let i = 0; i < n; i++) {
        const w = resp[i][c];
        for (let p = 0; p < d; p++)
          for (let q = 0; q < d; q++)
            cov[p][q] += w * (X[i][p] - m[p]) * (X[i][q] - m[q]);
      }
      for (let p = 0; p < d; p++)
        for (let q = 0; q < d; q++)
          cov[p][q] = cov[p][q] / (Nk[c] || 1) + (p === q ? 1e-6 : 0);
      covs[c] = cov;
    }
  }
  // Labels
  const labels = resp.map((r) => {
    let best = 0, bv = -Infinity;
    for (let c = 0; c < r.length; c++) if (r[c] > bv) { bv = r[c]; best = c; }
    return best;
  });
  // Free params: k-1 weights + k*d means + k*d*(d+1)/2 covariance entries
  const nparams = (k - 1) + k * d + k * (d * (d + 1)) / 2;
  const bic = -2 * prevLL + nparams * Math.log(n);
  const aic = -2 * prevLL + 2 * nparams;
  return { labels, resp, weights, means, covs, logLik: prevLL, bic, aic, iters: it };
}

// ---------- Gaussian HMM (Baum-Welch + Viterbi) ----------

export interface HMMResult {
  labels: number[];        // viterbi-decoded most likely state path
  startProb: Vec;          // k
  transMat: Mat;           // k x k
  means: Mat;              // k x d
  covs: Mat[];
  logLik: number;
  bic: number;
  aic: number;
  iters: number;
}

export function hmmFit(X: Mat, k: number, maxIter = 50, tol = 1e-3, seed = 1): HMMResult {
  const n = X.length, d = X[0].length;
  // Init emissions from GMM (good starting parameters)
  const init = gmmFit(X, k, 40, 1e-3, seed);
  const means = init.means.map((m) => m.slice());
  const covs = init.covs.map((c) => c.map((r) => r.slice()));
  const startProb = init.weights.slice();
  // Init transition matrix from GMM hard labels
  const transMat: Mat = Array.from({ length: k }, () => new Array(k).fill(1 / k));
  {
    const counts: Mat = Array.from({ length: k }, () => new Array(k).fill(1));
    for (let t = 1; t < n; t++) counts[init.labels[t - 1]][init.labels[t]] += 1;
    for (let i = 0; i < k; i++) {
      const s = counts[i].reduce((a, b) => a + b, 0);
      for (let j = 0; j < k; j++) transMat[i][j] = counts[i][j] / s;
    }
  }

  let prevLL = -Infinity;
  let it = 0;
  // Precomputed log-emissions buffer
  const logB: Mat = Array.from({ length: n }, () => new Array(k).fill(0));

  for (it = 0; it < maxIter; it++) {
    // Emissions log-prob
    for (let t = 0; t < n; t++)
      for (let s = 0; s < k; s++)
        logB[t][s] = logPdfGaussian(X[t], means[s], covs[s]);

    // Forward (scaled, log-space)
    const logAlpha: Mat = Array.from({ length: n }, () => new Array(k).fill(-Infinity));
    for (let s = 0; s < k; s++)
      logAlpha[0][s] = Math.log(startProb[s] + 1e-300) + logB[0][s];
    for (let t = 1; t < n; t++) {
      for (let j = 0; j < k; j++) {
        const arr = new Array(k);
        for (let i = 0; i < k; i++) arr[i] = logAlpha[t - 1][i] + Math.log(transMat[i][j] + 1e-300);
        const mx = Math.max(...arr);
        let s = 0;
        for (let i = 0; i < k; i++) s += Math.exp(arr[i] - mx);
        logAlpha[t][j] = mx + Math.log(s) + logB[t][j];
      }
    }
    const finalArr = logAlpha[n - 1];
    const fmx = Math.max(...finalArr);
    let fsum = 0;
    for (const v of finalArr) fsum += Math.exp(v - fmx);
    const ll = fmx + Math.log(fsum);
    if (Math.abs(ll - prevLL) < tol) { prevLL = ll; break; }
    prevLL = ll;

    // Backward
    const logBeta: Mat = Array.from({ length: n }, () => new Array(k).fill(-Infinity));
    for (let s = 0; s < k; s++) logBeta[n - 1][s] = 0;
    for (let t = n - 2; t >= 0; t--) {
      for (let i = 0; i < k; i++) {
        const arr = new Array(k);
        for (let j = 0; j < k; j++) arr[j] = Math.log(transMat[i][j] + 1e-300) + logB[t + 1][j] + logBeta[t + 1][j];
        const mx = Math.max(...arr);
        let s = 0;
        for (let j = 0; j < k; j++) s += Math.exp(arr[j] - mx);
        logBeta[t][i] = mx + Math.log(s);
      }
    }

    // Posteriors gamma, xi
    const gamma: Mat = Array.from({ length: n }, () => new Array(k).fill(0));
    for (let t = 0; t < n; t++) {
      const arr = new Array(k);
      for (let s = 0; s < k; s++) arr[s] = logAlpha[t][s] + logBeta[t][s];
      const mx = Math.max(...arr);
      let s = 0;
      for (const v of arr) s += Math.exp(v - mx);
      const lz = mx + Math.log(s);
      for (let i = 0; i < k; i++) gamma[t][i] = Math.exp(arr[i] - lz);
    }
    const xiSum: Mat = Array.from({ length: k }, () => new Array(k).fill(0));
    for (let t = 0; t < n - 1; t++) {
      const arr2: number[] = [];
      const idx: [number, number][] = [];
      for (let i = 0; i < k; i++)
        for (let j = 0; j < k; j++) {
          arr2.push(logAlpha[t][i] + Math.log(transMat[i][j] + 1e-300) + logB[t + 1][j] + logBeta[t + 1][j]);
          idx.push([i, j]);
        }
      const mx = Math.max(...arr2);
      let s = 0;
      for (const v of arr2) s += Math.exp(v - mx);
      const lz = mx + Math.log(s);
      for (let z = 0; z < arr2.length; z++) {
        const [i, j] = idx[z];
        xiSum[i][j] += Math.exp(arr2[z] - lz);
      }
    }

    // Re-estimate
    for (let s = 0; s < k; s++) startProb[s] = gamma[0][s];
    for (let i = 0; i < k; i++) {
      const denom = xiSum[i].reduce((a, b) => a + b, 0) || 1;
      for (let j = 0; j < k; j++) transMat[i][j] = xiSum[i][j] / denom;
    }
    for (let s = 0; s < k; s++) {
      let g = 0;
      const m = new Array(d).fill(0);
      for (let t = 0; t < n; t++) { g += gamma[t][s]; for (let j = 0; j < d; j++) m[j] += gamma[t][s] * X[t][j]; }
      for (let j = 0; j < d; j++) m[j] /= g || 1;
      means[s] = m;
      const cov: Mat = Array.from({ length: d }, () => new Array(d).fill(0));
      for (let t = 0; t < n; t++) {
        const w = gamma[t][s];
        for (let p = 0; p < d; p++)
          for (let q = 0; q < d; q++)
            cov[p][q] += w * (X[t][p] - m[p]) * (X[t][q] - m[q]);
      }
      for (let p = 0; p < d; p++)
        for (let q = 0; q < d; q++)
          cov[p][q] = cov[p][q] / (g || 1) + (p === q ? 1e-6 : 0);
      covs[s] = cov;
    }
  }

  // Viterbi decoding
  const labels = viterbi(X, startProb, transMat, means, covs);
  // Free params: (k-1) start + k*(k-1) trans + k*d means + k*d*(d+1)/2 covs
  const nparams = (k - 1) + k * (k - 1) + k * d + k * (d * (d + 1)) / 2;
  const bic = -2 * prevLL + nparams * Math.log(n);
  const aic = -2 * prevLL + 2 * nparams;
  return { labels, startProb, transMat, means, covs, logLik: prevLL, bic, aic, iters: it };
}

export function viterbi(X: Mat, startProb: Vec, transMat: Mat, means: Mat, covs: Mat[]): number[] {
  const n = X.length, k = startProb.length;
  const delta: Mat = Array.from({ length: n }, () => new Array(k).fill(-Infinity));
  const psi: number[][] = Array.from({ length: n }, () => new Array(k).fill(0));
  for (let s = 0; s < k; s++) delta[0][s] = Math.log(startProb[s] + 1e-300) + logPdfGaussian(X[0], means[s], covs[s]);
  for (let t = 1; t < n; t++) {
    const lb = new Array(k);
    for (let s = 0; s < k; s++) lb[s] = logPdfGaussian(X[t], means[s], covs[s]);
    for (let j = 0; j < k; j++) {
      let bestV = -Infinity, bestI = 0;
      for (let i = 0; i < k; i++) {
        const v = delta[t - 1][i] + Math.log(transMat[i][j] + 1e-300);
        if (v > bestV) { bestV = v; bestI = i; }
      }
      delta[t][j] = bestV + lb[j];
      psi[t][j] = bestI;
    }
  }
  const path = new Array(n).fill(0);
  let bestV = -Infinity, bestI = 0;
  for (let s = 0; s < k; s++) if (delta[n - 1][s] > bestV) { bestV = delta[n - 1][s]; bestI = s; }
  path[n - 1] = bestI;
  for (let t = n - 2; t >= 0; t--) path[t] = psi[t + 1][path[t + 1]];
  return path;
}

// ---------- Markov utilities ----------

export function transitionMatrixFromLabels(labels: number[], k: number): Mat {
  const M: Mat = Array.from({ length: k }, () => new Array(k).fill(0));
  for (let i = 1; i < labels.length; i++) M[labels[i - 1]][labels[i]]++;
  for (let i = 0; i < k; i++) {
    const s = M[i].reduce((a, b) => a + b, 0);
    if (s > 0) for (let j = 0; j < k; j++) M[i][j] /= s;
    else M[i][i] = 1;
  }
  return M;
}

// Stationary distribution via matrix power (simple, robust for small k)
export function stationaryDistribution(P: Mat, iters = 200): Vec {
  if (!P.length) return [];
  const Pn = matPow(P, iters);
  return Pn[0].slice();
}

// Expected duration (in steps) for each state given P_ii
export function expectedDurations(P: Mat): Vec {
  return P.map((row, i) => 1 / Math.max(1e-9, 1 - row[i]));
}

// One-step-ahead forecast from a current state
export function nextStateForecast(P: Mat, current: number): Vec {
  return P[current] ? P[current].slice() : [];
}
