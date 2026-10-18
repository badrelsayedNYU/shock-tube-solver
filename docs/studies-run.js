const Studies = (() => {
  function measure(caseKey, options) {
    const cs = ST.CASES[caseKey], result = ST.solve(cs, options);
    const failure = result.failure;
    const status = failure ? `Positivity failure at t=${failure.t.toPrecision(5)}, cell ${failure.cell}` : "Complete";
    const errors = failure ? null : ["rho", "u", "p"].map(field => ST.l1(result, cs, field));
    return { N: result.N, t: result.t, steps: result.steps, status, errors,
      conservation: result.conservation, failure };
  }
  function run(p, progress = () => {}) {
    if (!ST.CASES[p.caseKey] || !["hllc", "roe", "rusanov"].includes(p.flux)
        || !Object.hasOwn(ST.LIMITERS, p.limiter) || ![100,200,400,800].includes(p.N)) {
      throw new Error("Choose a supported case, scheme and comparison grid.");
    }
    const base = { entropyFix: p.entropyFix, deltaFactor: 0.1, frames: 0 };
    const convergence = [], schemes = [];
    let count = 0;
    for (const N of [50,100,200,400,800]) {
      const row = measure(p.caseKey, { ...base, N, flux:p.flux, limiter:p.limiter });
      const previous = convergence.at(-1);
      row.order = row.errors && previous?.errors && row.errors[0] > 0 && previous.errors[0] > 0
        ? Math.log(previous.errors[0]/row.errors[0])/Math.log(N/previous.N) : null;
      convergence.push(row); progress(++count);
    }
    for (const flux of ["rusanov", "roe", "hllc"]) for (const limiter of ["none", "minmod", "vanleer", "superbee"]) {
      schemes.push({ flux, limiter, ...measure(p.caseKey, { ...base, N:p.N, flux, limiter }) });
      progress(++count);
    }
    return { p, convergence, schemes,
      selected: schemes.find(row => row.flux === p.flux && row.limiter === p.limiter) };
  }
  return { run, measure };
})();
