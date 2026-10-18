(() => {
  const $ = id => document.getElementById(id);
  const names = {none:"First order",minmod:"minmod",vanleer:"van Leer",superbee:"superbee",hllc:"HLLC",roe:"Roe",rusanov:"Rusanov"};
  const fmt = n => n == null ? "—" : n.toExponential(5);
  let current, url, worker;
  function addRow(id, values) {
    const row = document.createElement("tr");
    for (const value of values) { const cell = document.createElement("td"); cell.textContent = value; row.appendChild(cell); }
    $(id).appendChild(row);
  }
  $("run").addEventListener("click", () => {
    const p = {caseKey:$("case").value,flux:$("flux").value,limiter:$("limiter").value,N:+$("grid").value,entropyFix:$("fix").checked};
    $("run").disabled = true; $("status").textContent = "Computing 17 simulations…";
    $("results").hidden = true;
    try {
      worker = new Worker("studies-worker.js");
      const fail = message => { $("status").textContent = message; $("run").disabled = false; worker.terminate(); };
      worker.onerror = () => fail("Study worker failed. Reload the page and try again.");
      worker.onmessage = ({data}) => {
        if (data.error) { fail(data.error); return; }
        if (data.progress) { $("status").textContent = `Completed ${data.progress} of 17 simulations…`; return; }
        current = data.done; render(); worker.terminate(); $("run").disabled = false;
      };
      worker.postMessage(p);
    } catch (e) { $("status").textContent = "This page requires browser support for Web Workers. " + e.message; $("run").disabled = false; }
  });
  function render() {
    const {p,convergence,schemes,selected} = current;
    $("results").hidden = false;
    $("caption").textContent = `${ST.CASES[p.caseKey].label}, final time ${ST.CASES[p.caseKey].tEnd}; selected ${names[p.flux]} / ${names[p.limiter]}, comparison N=${p.N}; Roe fix ${p.entropyFix?"on":"off"}.`;
    for (const id of ["convergenceRows","schemeRows","budgetRows"]) $(id).replaceChildren();
    for (const row of convergence) addRow("convergenceRows", [row.N,...[0,1,2].map(i=>fmt(row.errors?.[i])),row.order?.toFixed(3)??"—",row.status]);
    for (const row of schemes) addRow("schemeRows", [`${names[row.flux]} / ${names[row.limiter]}`,...[0,1,2].map(i=>fmt(row.errors?.[i])),row.steps,row.status]);
    const b=selected.conservation;
    $("budgetStatus").textContent = selected.failure ? `${selected.status}. Budget covers only accepted steps through t=${selected.t.toPrecision(6)}; requested final time was not reached.` : `Final time reached: t=${selected.t}. All quantities are nondimensional.`;
    ["Mass","Momentum","Total energy"].forEach((name,i)=>addRow("budgetRows",[name,fmt(b.initial[i]),fmt(b.final[i]),fmt(b.final[i]-b.initial[i]),fmt(b.boundaryFlux[i]),fmt(b.residual[i])]));
    const header=["study","case","flux","limiter","N","time","density_L1","velocity_L1","pressure_L1","density_order","steps","status"];
    const rows=[...convergence.map(r=>["convergence",p.caseKey,p.flux,p.limiter,r.N,r.t,...(r.errors??[null,null,null]),r.order,r.steps,r.status]),...schemes.map(r=>["schemes",p.caseKey,r.flux,r.limiter,r.N,r.t,...(r.errors??[null,null,null]),null,r.steps,r.status])];
    rows.push([], ["budget_quantity","initial","final","change","boundary_integral","residual"]);
    ["mass","momentum","energy"].forEach((name,i)=>rows.push([name,b.initial[i],b.final[i],b.final[i]-b.initial[i],b.boundaryFlux[i],b.residual[i]]));
    const csv=[header,...rows].map(row=>row.map(v=>`"${String(v??"").replaceAll('"','""')}"`).join(",")).join("\n");
    if(url) URL.revokeObjectURL(url); url=URL.createObjectURL(new Blob([csv],{type:"text/csv"})); $("download").href=url;
    draw(); $("status").textContent = `Studies complete. ${schemes.filter(r=>r.failure).length} of 12 scheme combinations failed positivity; those errors are omitted.`;
  }
  function draw() {
    if(!current) return;
    const cv=$("convergence"), dpr=devicePixelRatio||1, w=cv.clientWidth,h=cv.clientHeight;
    cv.width=Math.round(w*dpr); cv.height=Math.round(h*dpr);
    const ctx=cv.getContext("2d"); ctx.scale(dpr,dpr);
    const rows=current.convergence.filter(r=>r.errors&&r.errors[0]>0);
    if(!rows.length) {ctx.fillStyle=getComputedStyle(document.body).color;ctx.fillText("No successful grids to plot.",30,40);return;}
    const lo=Math.floor(Math.log10(Math.min(...rows.map(r=>r.errors[0])))),hi=Math.max(lo+1,Math.ceil(Math.log10(Math.max(...rows.map(r=>r.errors[0])))));
    const X=n=>65+Math.log2(n/50)/4*(w-90),Y=e=>h-45-(Math.log10(e)-lo)/(hi-lo)*(h-70);
    ctx.font="12px system-ui";ctx.strokeStyle="#8090a080";ctx.fillStyle=getComputedStyle(document.body).color;
    for(const n of [50,100,200,400,800]) {const x=X(n);ctx.beginPath();ctx.moveTo(x,20);ctx.lineTo(x,h-45);ctx.stroke();ctx.fillText(n,x-10,h-25);}
    for(let e=lo;e<=hi;e++) {const y=Y(10**e);ctx.beginPath();ctx.moveTo(65,y);ctx.lineTo(w-25,y);ctx.stroke();ctx.fillText(`1e${e}`,10,y+4);}
    ctx.fillText("Grid cells N (log scale)",w/2-65,h-5);ctx.fillText("Density L1 error (log scale)",65,14);
    ctx.strokeStyle="#2d91bb";ctx.fillStyle="#2d91bb";ctx.lineWidth=2;let previous=false;ctx.beginPath();
    for(const r of current.convergence) {if(!r.errors||r.errors[0]<=0){previous=false;continue;}const x=X(r.N),y=Y(r.errors[0]);if(previous)ctx.lineTo(x,y);else ctx.moveTo(x,y);previous=true;}ctx.stroke();
    for(const r of rows) {ctx.beginPath();ctx.arc(X(r.N),Y(r.errors[0]),4,0,Math.PI*2);ctx.fill();}
  }
  addEventListener("resize",draw);
})();
