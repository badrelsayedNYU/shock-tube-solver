Modified Sod (Case 4), first order, N = 100, t = 0.2. Density drop between the two cells adjacent to x0 = 0.3 (the sonic point).

| Scheme | Density drop at sonic point | L1 error (rho) |
|---|---|---|
| Exact solution (sampled at the two cell centers) | 0.0274 | - |
| Roe, no entropy fix | 0.1400 | 1.455e-02 |
| Roe + Harten fix, delta = 0.1 a~ (default) | 0.0775 | 1.347e-02 |
| Roe + Harten fix, delta = 0.2 a~ | 0.0386 | 1.286e-02 |
| Roe + Harten fix, delta = 0.3 a~ | 0.0301 | 1.315e-02 |
| HLLC | 0.0273 | 1.380e-02 |
| Rusanov | 0.0163 | 2.961e-02 |
