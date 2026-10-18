const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {test} = require('node:test');
const root = path.join(__dirname, '..');
const read = file => fs.readFileSync(path.join(root,file),'utf8');
const original = read('docs/index.html').match(/const ST = \(\(\) => \{[\s\S]*?\n\}\)\(\);/)[0];
const lab = vm.runInNewContext(original+'\nST;');
const {ST,Studies} = vm.runInNewContext(read('docs/studies-core.js')+'\n'+read('docs/studies-run.js')+'\n({ST,Studies});');

for(const flux of ['rusanov','roe','hllc']) for(const limiter of ['none','minmod','vanleer','superbee']) {
  test(`diagnostics preserve lab state and close the open budget: ${flux}/${limiter}`,()=>{
    const options={N:100,flux,limiter};
    const expected=lab.solve(lab.CASES.sod,options),actual=ST.solve(ST.CASES.sod,options);
    for(const field of ['rho','u','p']) assert.deepEqual(Array.from(actual[field]),Array.from(expected[field]));
    assert.equal(actual.steps,expected.steps);
    const b=actual.conservation;
    // Momentum changes physically through unequal boundary pressures.
    assert.ok(b.final[1]-b.initial[1]>0.2);
    for(let i=0;i<3;i++) assert.ok(Math.abs(b.residual[i])<1e-11*Math.max(1,Math.abs(b.initial[i])));
  });
}
test('failed Roe run excludes rejected stages from conservation',()=>{
  const r=Studies.measure('123',{N:100,flux:'roe',limiter:'none'});
  assert.ok(r.failure);
  assert.equal(r.errors,null);
  assert.ok(r.t<ST.CASES['123'].tEnd);
  assert.ok(r.conservation.residual.every(v=>Math.abs(v)<1e-11));
});
test('complete Sod study compares all schemes and converges toward exact density',()=>{
  const r=Studies.run({caseKey:'sod',flux:'hllc',limiter:'vanleer',N:100,entropyFix:true});
  assert.equal(r.schemes.length,12);
  assert.equal(r.convergence.length,5);
  assert.ok(r.schemes.every(row=>row.errors&&row.t===0.25));
  assert.ok(r.convergence.at(-1).errors[0]<r.convergence[0].errors[0]/3);
  assert.ok(r.convergence.slice(1).every(row=>row.order>0));
  assert.equal(r.selected.flux,'hllc');
  assert.equal(r.selected.limiter,'vanleer');
});
test('invalid study settings fail before solving',()=>{
  assert.throws(()=>Studies.run({caseKey:'unknown'}),/supported/);
});
