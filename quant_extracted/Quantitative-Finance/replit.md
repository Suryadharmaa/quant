# Workspace

## Overview

pnpm workspace monorepo using TypeScript. Each package manages its own dependencies.

## Stack

- **Monorepo tool**: pnpm workspaces
- **Node.js version**: 24
- **Package manager**: pnpm
- **TypeScript version**: 5.9
- **API framework**: Express 5
- **Database**: PostgreSQL + Drizzle ORM
- **Validation**: Zod (`zod/v4`), `drizzle-zod`
- **API codegen**: Orval (from OpenAPI spec)
- **Build**: esbuild (CJS bundle)

## Key Commands

- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from OpenAPI spec
- `pnpm --filter @workspace/db run push` — push DB schema changes (dev only)
- `pnpm --filter @workspace/api-server run dev` — run API server locally

See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details.

## Quant Regime Analyzer Artifact (artifacts/quant-regime)

Real-time market regime clustering for BTC/USD and XAU/USD with deep math:

- **Data**: real Yahoo Finance via `/api/market?symbol=btc|xauusd` proxy in api-server (10-min cache).
- **Models** (`src/lib/regime-models.ts`): K-Means++ (geometric), Gaussian Mixture (full-covariance EM), Gaussian HMM (Baum-Welch fitting + Viterbi decoding).
- **Markov math**: empirical or HMM-learned transition matrix, stationary distribution via matrix power, expected duration `1/(1-Pii)`, next-day forecast row.
- **Model selection**: log-likelihood, BIC, AIC computed for GMM and HMM so user can compare k=2..6.
- **AI Console**: Groq API key stored in localStorage; every prompt injected with full real-data context (price, model, regime stats, transition matrix, stationary, expected durations, next-day forecast).
