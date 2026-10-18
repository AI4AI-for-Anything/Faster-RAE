#!/usr/bin/env python3
"""Generate three file-change views from pinned Git diffs; Python stdlib only."""
from pathlib import Path
import hashlib
import html
import json
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REVS = ['a4d18c4db766419cbe7cb8c02cd9f7ceb0ec9041', '649d0e7', '86aee71']

def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()

REVS = [git('rev-parse', r) for r in REVS]
# Descriptions are reviewed against the pinned source, not inferred from filenames.
DETAILS = {
'.harness/config.json': ('Preset', 'Records the selected optimization switches as an environment mapping; this small preset is not the independent research harness.'),
'scripts/fd7800d0633a.env.sh': ('Preset', 'Shell version of the selected preset: TF32/cuBLAS, block compilation, LPIPS layout and local BF16 paths.'),
'src/stage1/decoders/decoder.py': ('Decoder', 'Routes attention through local BF16 SDPA and projection/MLP through scoped BF16 linear helpers; compiles decoder blocks; fixes the SDPA dropout attribute.'),
'src/utils/local_attention.py': ('Decoder', 'Casts Q/K/V operands to BF16 for a selected native SDPA backend, then restores the incoming output dtype.'),
'src/utils/local_linear.py': ('Decoder', 'Casts decoder linear operands locally while keeping original parameter ownership and gradients; restores the activation dtype.'),
'src/disc/dinodisc.py': ('Loss & discriminator', 'Adds local BF16 attention, projection and MLP paths; compiles discriminator transformer blocks and the normalization tensor region.'),
'src/disc/lpips.py': ('Loss & discriminator', 'Adds channels-last VGG/comparison layouts, a compiled LPIPS comparison region and optional convolution hooks.'),
'src/utils/fusion_backends.py': ('Compute controls', 'Centralizes opt-in torch.compile regions, ATen GEMM selection, repeated-backward buffer compatibility and call diagnostics.'),
'src/utils/compute_backends.py': ('Compute controls', 'Controls TF32, BLAS and SDPA selection; offers optional nondeterministic cuDNN autotuning.'),
'src/utils/local_auxiliary.py': ('Compute controls', 'Implements scoped BF16 discriminator linear operations plus optional encoder-linear and VGG-convolution hooks. The latter two are OFF in the selected preset.'),
'src/stage1/encoders/dinov2.py': ('Compute controls', 'Installs opt-in EncoderLinear hooks and an attention-backend context. Encoder-linear BF16 is OFF in the selected preset.'),
'src/stage1/rae.py': ('Compute controls', 'Applies the requested process-wide matrix multiplication settings when constructing RAE.'),
'src/utils/optim_utils.py': ('Compute controls', 'Adds an optional foreach AdamW route and resume-state handling. The selected preset retains upstream fused AdamW.'),
'src/utils/train_utils.py': ('Compute controls', 'Adds optional grouped foreach EMA updates and diagnostics. Foreach EMA is OFF in the selected preset.'),
'README.md': ('Documentation', 'Documents the optimized snapshot and usage; the later phase adds measured benchmarks, reconstruction quality and adaptation instructions.'),
'docs/source-snapshot.json': ('Documentation', 'Pins the imported snapshot with per-file SHA-256 values and the upstream source revision.'),
'.gitignore': ('Documentation', 'Excludes local adaptation environments and generated run artifacts.'),
'AGENTS.md': ('Environment & workflow', 'Defines the agent workflow: default 8 GPUs, fixed upstream global batch, bounded memory probing and release boundaries.'),
'docs/agent-environment.md': ('Environment & workflow', 'Documents isolated environment setup, dependency/toolchain checks, local assets and troubleshooting before GPU probing.'),
'docs/gpu-adaptation.md': ('Environment & workflow', 'Explains the fixed-global-batch plan, three probe phases, outputs, launch commands and limits of single-rank capacity evidence.'),
'docs/release-boundary.md': ('Documentation', 'Separates the repository adaptation entrypoint from the independently versioned research harness and its experiment ledger.'),
'requirements-adapt.txt': ('Environment & workflow', 'Lists adaptation dependencies alongside the repository environment requirements.'),
'skills/adapt-rae-gpu/SKILL.md': ('Environment & workflow', 'Gives agents a first-run GPU adaptation checklist while preserving model, loss, precision and global batch.'),
'src/disc/lpips_utils.py': ('Environment & workflow', 'Accepts an explicit local LPIPS checkpoint path and verifies its expected checksum before use.'),
'src/train_stage1.py': ('Trainer & accumulation', 'Adds bounded probes; splits an effective CPU batch into GPU microbatches; accumulates G/D gradients with DDP synchronization; advances G/EMA/schedules per effective update.'),
'src/utils/accumulation.py': ('Trainer & accumulation', 'Implements mean-gradient accumulation, final-microbatch DDP synchronization and RNG replay for rank-local GAN adaptive weighting.'),
'src/utils/batch_plan.py': ('Trainer & accumulation', 'Enforces global batch = world size × microbatch × accumulation and rejects invalid divisors or mismatched deployment sizes.'),
'src/utils/capacity_probe.py': ('Capacity validation', 'Bounds the real trainer run; applies allocator limits; records warmup-adjusted timing, memory and finite/update checks.'),
'tools/adapt_gpu.py': ('Capacity validation', 'Runs environment preflight and three real-trainer phases; lowers microbatch only on OOM; emits a fixed-global-batch training config and launch script.'),
 'tools/check_environment.py': ('Environment & workflow', 'Checks actual trainer imports, local assets, compiler forward/backward and available SDPA backends before probing.'),
'tests/test_accumulation.py': ('Capacity validation', 'Checks mean-gradient/adaptive-weight behavior on toy models and distributed synchronization; does not establish full-run convergence.'),
'tests/test_gpu_adaptation.py': ('Capacity validation', 'Checks batch arithmetic and configuration constraints for GPU adaptation.'),
}
PHASES = [
    {'id': 'optimization', 'label': 'Initial optimization snapshot', 'from': REVS[0], 'to': REVS[1]},
    {'id': 'adaptation', 'label': 'Later GPU adaptation', 'from': REVS[1], 'to': REVS[2]},
]
entries = []
for phase in PHASES:
    diff = git('diff', '--numstat', phase['from'], phase['to'])
    statuses = dict(line.split('\t', 1)[::-1] for line in git('diff', '--name-status', phase['from'], phase['to']).splitlines())
    for line in diff.splitlines():
        added, removed, path = line.split('\t')
        if not (ROOT / path).is_file() or path == 'docs/source-snapshot.json':
            continue
        assert path in DETAILS, f'Add a reviewed description for {path}'
        assert (ROOT / path).is_file(), path
        module, description = DETAILS[path]
        entries.append({'phase': phase['id'], 'module': module, 'path': path, 'status': statuses[path],
                        'added': int(added), 'removed': int(removed), 'description': description,
                        'from': phase['from'], 'to': phase['to'], 'source_link': '../../'+path,
                        'blob_sha': git('rev-parse', phase['to']+':'+path),
                        'file_speedup': None})
data = {'schema_version': 1, 'phases': PHASES, 'files': entries,
        'scope': 'Current code, presets, tests and documentation from two historical Git ranges. Retired reports, the source manifest and this presentation are excluded.',
        'speedup_attribution': 'No per-file causal speedups are claimed. An optimization setting can span multiple files; measurements apply to the benchmarked configuration as a whole.',
        'preset_note': 'Optional encoder-linear BF16, VGG-convolution BF16, foreach EMA and foreach AdamW hooks exist in source but are not selected by the imported preset.',
        'adaptation_note': 'Preserving effective global batch preserves sample/update cadence, not exact numerical equivalence. A single-rank capacity PASS does not validate full distributed convergence.'}
(HERE/'changes.json').write_text(json.dumps(data, indent=2)+'\n')

md = ['# What changed in Faster-RAE', '', 'Three views of the same pinned Git evidence: **[review table](review-table.md)** · [module map](module-map.svg) · [browsable cards](index.html). The cards are a standalone HTML file: download or open them locally; GitHub displays the source.', '',
      data['scope'], '', '**Attribution:** '+data['speedup_attribution'], '', '**Selected vs available:** '+data['preset_note'], '', '**Adaptation:** '+data['adaptation_note'], '']
for phase in PHASES:
    rows = [e for e in entries if e['phase']==phase['id']]
    md += ['## '+phase['label'], '',
           f"**{len(rows)} changed paths · +{sum(r['added'] for r in rows)} / −{sum(r['removed'] for r in rows)} lines.** Line counts describe the diff, not runtime benefit.", '',
           '| Module | File | Change | Purpose |', '|---|---|---:|---|']
    for e in rows:
        md.append(f"| {e['module']} | [{e['path']}]({e['source_link']}) | {e['status']} +{e['added']} / −{e['removed']} | {e['description']} |")
    md += ['']
md += ['## Reproduce and review', '', '```bash', 'python3 docs/change-views/generate.py', '```', '',
       'The generator requires both pinned commits in local Git history and uses Python’s standard library. `changes.json` stores per-phase paths, status, added/deleted lines, commit IDs and Git blob IDs. The curated descriptions live in `generate.py`; regenerating the views never runs GPU work.', '',
       'A = added; M = modified. A file can appear in both phases (for example README). Runtime comparisons belong to the [H20 results](../h20-results.md), not this line-count inventory.', '']
(HERE/'review-table.md').write_text('\n'.join(md))

# A compact map complements the full inventory; it intentionally is not a tiny table.
svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="690" viewBox="0 0 1200 690" role="img" aria-labelledby="title desc">',
'<title id="title">Faster-RAE: where the code changed</title>', '<desc id="desc">Initial optimized compute paths and later fixed-global-batch GPU adaptation, shown separately. Full file inventory is in review-table.md. This map does not attribute speedup to individual files.</desc>',
'<style>text{font-family:Arial,Helvetica,sans-serif;fill:#292122}.title{font-size:32px;font-weight:700}.label{font-size:18px;font-weight:700}.body{font-size:16px}.code{font-size:15px;font-family:monospace}.small{font-size:15px;fill:#79696B}</style>',
'<rect width="1200" height="690" rx="20" fill="#FAF5F5"/>',
'<text x="40" y="54" class="title">What changed in Faster-RAE</text>',
'<g transform="translate(0,-25)">']
def box(x,y,w,h,title,lines,accent,link):
    svg.extend([f'<a href="{html.escape(link)}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="white" stroke="#EADFE0"/>',f'<rect x="{x}" y="{y}" width="{w}" height="5" rx="2" fill="{accent}"/>',f'<text x="{x+20}" y="{y+34}" class="label">{html.escape(title)}</text>'])
    for i,line in enumerate(lines): svg.append(f'<text x="{x+20}" y="{y+65+i*25}" class="code">{html.escape(line)}</text>')
    svg.append('</a>')
svg += ['<text x="40" y="134" class="label">01 / Initial optimization</text>', '<text x="355" y="134" class="small">Compute paths and preset</text>']
box(40,153,360,186,'Decoder',['stage1/decoders/decoder.py','utils/local_attention.py','utils/local_linear.py','Local BF16 + compiled blocks'],'#8C0000','../../src/stage1/decoders/decoder.py')
box(420,153,360,186,'LPIPS + discriminator',['disc/lpips.py','disc/dinodisc.py','Layout + compiled regions','Local BF16 attention / MLP'],'#8C0000','../../src/disc/dinodisc.py')
box(800,153,360,186,'Shared controls + preset',['utils/{compute,fusion}_backends.py','utils/local_auxiliary.py','.harness/config.json','scripts/fd7800d0633a.env.sh'],'#8C0000','../../.harness/config.json')
svg += ['<text x="40" y="369" class="small">Also: encoder hooks, RAE setup, optimizer / EMA options, and README.</text>',
'<text x="40" y="394" class="small">Optional hooks are not all active in the selected preset; see the complete inventory.</text>',
'<text x="40" y="449" class="label">02 / Later GPU adaptation</text>', '<text x="440" y="449" class="small">Training and deployment tools</text>']
box(40,468,360,186,'Fixed-batch training',['train_stage1.py','utils/accumulation.py','utils/batch_plan.py','Global = GPUs × micro × accum'],'#8C0000','../../src/utils/accumulation.py')
box(420,468,360,186,'Capacity probing',['tools/adapt_gpu.py','utils/capacity_probe.py','tests/test_{accumulation,','  gpu_adaptation}.py'],'#8C0000','../../tools/adapt_gpu.py')
box(800,468,360,186,'Environment + agent entry',['tools/check_environment.py','disc/lpips_utils.py','AGENTS.md + adaptation skill','Setup docs + requirements'],'#8C0000','../../docs/agent-environment.md')
svg += ['<text x="40" y="684" class="small">Training quality and performance results are collected in the H20 results document.</text>',
'</g>', '</svg>']
(HERE/'module-map.svg').write_text('\n'.join(svg)+'\n')

page = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Faster-RAE · File change explorer</title>
<style>
:root{--ink:#292122;--muted:#79696B;--blue:#8C0000;--green:#8C0000}*{box-sizing:border-box}body{margin:0;background:#FFFFFF;color:var(--ink);font:16px/1.55 system-ui,-apple-system,sans-serif}main{max-width:1240px;margin:auto;padding:48px 30px}h1{font-size:clamp(30px,4vw,48px);line-height:1.1;margin:12px 0 20px}h2{font-size:20px}a{color:#8C0000}header p{max-width:900px;color:var(--muted)}.eyebrow{letter-spacing:.12em;font-size:13px;font-weight:700}.stats{display:flex;gap:16px;flex-wrap:wrap;margin:26px 0}.stat{background:white;padding:14px 22px;border-radius:12px;border:1px solid #EADFE0}.stat strong{display:block;font-size:28px}.note{padding:18px 22px;background:#FAF5F5;border-radius:12px;margin:22px 0}.controls{display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin:28px 0 16px}input,select,button{font:inherit;padding:10px 12px;border:1px solid #EADFE0;border-radius:8px;background:white;color:var(--ink)}input{flex:1;min-width:220px}button{cursor:pointer}input:focus,select:focus,button:focus,a:focus{outline:3px solid #D7BDBF;outline-offset:2px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,340px),1fr));gap:18px}.card{background:white;border:1px solid #EADFE0;border-top:4px solid var(--blue);border-radius:12px;padding:22px;display:flex;flex-direction:column;gap:12px}.card.adaptation{border-top-color:var(--green)}.badge{font-size:12px;font-weight:700;letter-spacing:.03em;color:var(--blue)}.adaptation .badge{color:var(--green)}.path{font:600 15px/1.5 ui-monospace,monospace;overflow-wrap:anywhere}.card p{margin:0;font-size:15px}.meta{font-size:12px;color:var(--muted);margin-top:auto;padding-top:8px}.meta code{font-size:12px}.count{color:var(--muted);font-size:14px;margin-bottom:18px}details{margin:28px 0}pre{overflow:auto;background:#FAF5F5;padding:18px;border-radius:10px}footer{margin-top:32px;color:var(--muted);font-size:14px}@media print{.controls{display:none}.grid{display:block}.card{break-inside:avoid;margin-bottom:12px}body{background:white}main{padding:10px}}
</style>
<main><header><div class="eyebrow">FASTER-RAE / SOURCE REVIEW</div><h1>See exactly what changed.</h1><p>Browse the initial performance snapshot and the later GPU adaptation work separately. Every card links to a real repository file and records its pinned Git diff.</p><p><a href="review-table.md">Review table</a> · <a href="module-map.svg">Module map</a> · <a href="changes.json">Machine-readable evidence</a></p></header>
<div class="stats"><div class="stat"><strong>__INITIAL_COUNT__</strong>initial changed paths</div><div class="stat"><strong>__LATER_COUNT__</strong>later changed paths</div><div class="stat"><strong>512</strong>upstream default global batch</div></div>
<div class="note"><strong>Files explain implementation; experiments establish speed.</strong><br>No individual-file speedup is inferred from this diff. Optional encoder/VGG BF16 and foreach optimizer/EMA hooks are not active in the imported preset. Fixed effective batch does not prove exact numerical equivalence.</div>
<div class="controls"><label for="phase">Phase</label><select id="phase"><option value="all">Both phases</option><option value="optimization">Initial optimization</option><option value="adaptation">Later GPU adaptation</option></select><label for="search">Find</label><input id="search" type="search" placeholder="File, module or purpose…"><button id="reset" type="button">Reset</button></div>
<div id="count" class="count" aria-live="polite"></div><div id="cards" class="grid"></div><noscript>This view requires JavaScript for filtering. The complete static inventory is in <a href="review-table.md">review-table.md</a>.</noscript>
<details><summary>Reproduce these views</summary><pre id="repro"></pre><p>This self-contained page has no CDN, analytics or network dependency. Open it from this checkout so relative source links resolve. GitHub does not render repository HTML as an application.</p></details>
<footer>The same file may appear in both phases. Counts and additions/deletions are source-diff statistics, not performance attribution. The module map is a summary; these cards and the review table cover the retained files from the historical ranges.</footer></main>
<script id="evidence" type="application/json">__DATA__</script>
<script>
'use strict';const data=JSON.parse(document.getElementById('evidence').textContent);const phase=document.getElementById('phase'),search=document.getElementById('search'),cards=document.getElementById('cards');
function el(tag,cls,text){const e=document.createElement(tag);if(cls)e.className=cls;if(text!==undefined)e.textContent=text;return e}
function render(){const q=search.value.toLowerCase().trim(),rows=data.files.filter(e=>(phase.value==='all'||e.phase===phase.value)&&[e.path,e.module,e.description].join(' ').toLowerCase().includes(q));cards.replaceChildren();for(const e of rows){const c=el('article','card '+e.phase);c.append(el('div','badge',(e.phase==='optimization'?'01 · INITIAL OPTIMIZATION':'02 · LATER ADAPTATION')+' / '+e.module));const a=el('a','path',e.path);a.href=e.source_link;c.append(a,el('p','',e.description));c.append(el('div','meta',`${e.status==='A'?'Added':'Modified'} · +${e.added} / −${e.removed} lines`));cards.append(c)}document.getElementById('count').textContent=rows.length+' of '+data.files.length+' file changes shown';}
phase.addEventListener('change',render);search.addEventListener('input',render);document.getElementById('reset').addEventListener('click',()=>{phase.value='all';search.value='';render()});document.getElementById('repro').textContent='python3 docs/change-views/generate.py';render();
</script></html>'''
(HERE/'index.html').write_text(page.replace('__INITIAL_COUNT__', str(sum(e['phase'] == 'optimization' for e in entries))).replace('__LATER_COUNT__', str(sum(e['phase'] == 'adaptation' for e in entries))).replace('__DATA__',json.dumps(data).replace('<','\\u003c')))
print(json.dumps({'phase_changes':len(entries),'unique_paths':len({e['path'] for e in entries}),'outputs':['changes.json','review-table.md','module-map.svg','index.html']}))
