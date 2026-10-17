Sod, N = 400, t = 0.25. Runtime: median of 7 runs on one CPU core (first order: forward Euler, CFL 0.8; second order: SSP-RK2, CFL 0.5). Cells across a discontinuity: cells inside the 5-95% band of the jump (shock measured in p, contact in rho).

| Flux | Limiter | L1 error (rho) | Runtime (ms) | Steps | Cells across shock | Cells across contact |
|---|---|---|---|---|---|---|
| rusanov | none | 1.126e-02 | 14.7 | 272 | 4 | 49 |
| rusanov | minmod | 3.141e-03 | 44.1 | 437 | 4 | 17 |
| rusanov | vanleer | 2.089e-03 | 44.6 | 438 | 3 | 10 |
| rusanov | superbee | 1.838e-03 | 48.0 | 445 | 2 | 4 |
| roe | none | 6.609e-03 | 21.3 | 273 | 4 | 28 |
| roe | minmod | 2.457e-03 | 65.4 | 437 | 4 | 13 |
| roe | vanleer | 1.606e-03 | 66.4 | 438 | 3 | 9 |
| roe | superbee | **9.002e-04** | 69.8 | 442 | 2 | 4 |
| hllc | none | 6.758e-03 | 24.7 | 273 | 4 | 28 |
| hllc | minmod | 2.497e-03 | 75.4 | 437 | 4 | 13 |
| hllc | vanleer | 1.692e-03 | 75.4 | 438 | 3 | 9 |
| hllc | superbee | 9.686e-04 | 79.7 | 442 | 2 | 3 |
