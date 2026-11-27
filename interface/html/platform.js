/**
 * GajahMada AI Platform Control Plane Frontend (`/chat`, `/models/*`, `/model-factory`,
 * `/training`, `/benchmarks`, `/infrastructure`, `/nexus`, `/knowledge`, `/memory`,
 * `/settings`, `/documentation`).
 *
 * 100% live backend data, zero fake telemetry, browser localStorage + IndexedDB adapter.
 */
document.addEventListener('DOMContentLoaded', async () => {
  const path = window.location.pathname.replace(/\/$/, '') || '/';
  const root = document.getElementById('plt-root');
  const esc = (s) => String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  const capBadge = (st) => {
    const k = String(st || 'NOT CONFIGURED').split(/[\s_]/)[0];
    return `<span class="cap-badge cap-${esc(k)}">${esc(st || 'NOT CONFIGURED')}</span>`;
  };

  // Highlight active nav link
  document.querySelectorAll('#plt-nav .plt-link').forEach((a) => {
    if (a.getAttribute('href') === path) a.classList.add('active');
  });

  // Session API Key helper
  const getKey = () => { try { return (sessionStorage.getItem('gm_admin_api_key') || '').trim(); } catch { return ''; } };
  const authHeaders = (extra = {}) => { const h = { ...extra }; const k = getKey(); if (k) h['X-API-Key'] = k; return h; };
  document.getElementById('btn-plt-auth')?.addEventListener('click', () => {
    const v = window.prompt('Enter GajahMada Admin/Operator API Key (sessionStorage):', getKey() ? '********' : '');
    if (v === null || v === '********') return;
    try { if (v.trim()) sessionStorage.setItem('gm_admin_api_key', v.trim()); else sessionStorage.removeItem('gm_admin_api_key'); } catch {}
    window.location.reload();
  });

  // Lightweight browser IndexedDB helper for local structured notes / chat history
  const localDB = {
    open: () => new Promise((resolve, reject) => {
      if (!window.indexedDB) return resolve(null);
      const req = indexedDB.open('gajahmada_local_store', 1);
      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains('notes')) db.createObjectStore('notes', { keyPath: 'id' });
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    }),
    listNotes: async () => {
      try {
        const db = await localDB.open();
        if (!db) return [];
        return await new Promise((res) => {
          const tx = db.transaction('notes', 'readonly').objectStore('notes').getAll();
          tx.onsuccess = () => res(tx.result || []);
          tx.onerror = () => res([]);
        });
      } catch { return []; }
    },
    putNote: async (text) => {
      const db = await localDB.open();
      if (!db) return;
      return new Promise((res) => {
        const tx = db.transaction('notes', 'readwrite').objectStore('notes');
        tx.put({ id: `local-${Date.now()}`, text, created_at: new Date().toISOString() });
        tx.transaction.oncomplete = () => res();
      });
    },
  };

  // Load live platform status first
  let status = {};
  try {
    status = await (await fetch('/api/v2/platform/status')).json();
    document.getElementById('plt-subtitle').textContent =
      `${status.runtime_environment || 'RUNTIME'} · STORAGE: ${status.storage?.mode || 'UNKNOWN'}`;
    const infPill = document.getElementById('plt-inference-pill');
    if (infPill) infPill.textContent = status.native_model_inference?.status_banner || 'NATIVE MODEL INFERENCE: UNAVAILABLE IN CURRENT RUNTIME';
    if (status.storage?.warning) {
      const bn = document.getElementById('plt-storage-banner');
      bn.style.display = 'flex';
      bn.innerHTML = `<span>⚠ ${esc(status.storage.warning)}</span><span>Browser Storage: localStorage + IndexedDB Ready</span>`;
    }
  } catch (e) {
    document.getElementById('plt-subtitle').textContent = 'STATUS FETCH ERROR';
  }

  // ── Route Renderers ───────────────────────────────────────────────────────
  if (path === '/chat') {
    root.innerHTML = `
      <div class="plt-card">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px">
          <h3>GAJAHMADA SOVEREIGN CHAT</h3>
          <div>${capBadge('ACTIVE')} ${capBadge(status.native_model_inference?.active ? 'ACTIVE' : 'UNAVAILABLE ON VERCEL FREE')}</div>
        </div>
        <div style="font-size:11.5px;color:var(--txt-dim)">
          Runtime Inference: <b>${esc(status.native_model_inference?.status_banner)}</b> · Symbolic Reasoning + Hybrid Knowledge + World Facts active.
        </div>
        <div class="chat-box" id="plt-chat-box"></div>
        <div style="display:flex;gap:8px">
          <input class="obs-input" id="plt-chat-input" style="flex:1" placeholder="Ask GajahMada AI anything..." />
          <button class="obs-btn accent" id="plt-chat-send">SEND</button>
        </div>
      </div>`;
    const box = document.getElementById('plt-chat-box');
    const inp = document.getElementById('plt-chat-input');
    const send = async () => {
      const q = inp.value.trim();
      if (!q) return;
      inp.value = '';
      box.innerHTML += `<div class="chat-msg user">${esc(q)}</div>`;
      box.scrollTop = box.scrollHeight;
      try {
        const res = await fetch('/ai/chat', {
          method: 'POST',
          headers: authHeaders({ 'Content-Type': 'application/json' }),
          body: JSON.stringify({ prompt: q }),
        });
        const d = await res.json();
        box.innerHTML += `<div class="chat-msg ai">${esc(d.response || d.detail || 'No response')}</div>`;
      } catch (err) {
        box.innerHTML += `<div class="chat-msg ai">Error: ${esc(err.message)}</div>`;
      }
      box.scrollTop = box.scrollHeight;
    };
    document.getElementById('plt-chat-send').onclick = send;
    inp.addEventListener('keydown', (e) => { if (e.key === 'Enter') send(); });
    return;
  }

  if (path === '/knowledge') {
    const prov = await (await fetch('/api/v2/platform/providers')).json();
    const stats = await (await fetch('/knowledge/stats')).json();
    root.innerHTML = `
      <div class="plt-card">
        <div style="display:flex;justify-content:space-between;align-items:center">
          <h3>HYBRID KNOWLEDGE &amp; FAISS CPU PROVIDERS</h3>
          <a class="obs-btn accent" href="/brain/knowledge">Open Interactive Knowledge Universe →</a>
        </div>
        <div style="font-size:12px;color:var(--txt-dim)">
          Selected Provider: <b style="color:var(--cyan)">${esc(prov.selected_provider)}</b> (${esc(prov.selected_method)}) · Indexed Docs: <b>${esc(stats.total_docs)}</b>
        </div>
      </div>
      <div class="plt-grid">
        ${(prov.providers || []).map((p) => `
          <div class="plt-card">
            <div style="display:flex;justify-content:space-between;align-items:center">
              <h3>${esc(p.provider)}</h3>${capBadge(p.status)}
            </div>
            <div class="insp-kv"><span class="insp-k">Method</span><span class="insp-v">${esc(p.retrieval_method)}</span></div>
            <div style="font-size:11.5px;color:var(--txt-dim)">${esc(p.reason)}</div>
          </div>`).join('')}
      </div>`;
    return;
  }

  if (path === '/memory') {
    const mem = await (await fetch('/api/v2/observatory/memory?limit=25', { headers: authHeaders() })).json();
    const notes = await localDB.listNotes();
    root.innerHTML = `
      <div class="plt-card">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px">
          <h3>MEMORY ARCHITECTURE (SERVER vs BROWSER-LOCAL)</h3>
          <a class="obs-btn accent" href="/brain/memory">Open Memory Explorer →</a>
        </div>
        <div style="font-size:12px;color:var(--txt-dim)">
          Server Persistence: <b>${esc(status.storage?.mode)}</b> · Actor Role: <b>${esc(mem.actor_role)}</b> · Accessible Server Records: <b>${esc(mem.pagination?.total ?? 0)}</b>
        </div>
      </div>
      <div class="plt-grid">
        <div class="plt-card">
          <h3>Browser-Local IndexedDB Notes (${notes.length})</h3>
          <div style="display:flex;gap:6px">
            <input class="obs-input" id="idb-note-in" style="flex:1" placeholder="Save a browser-local note (stays in your browser)..." />
            <button class="obs-btn accent" id="idb-note-save">SAVE LOCAL</button>
          </div>
          ${notes.length ? notes.slice(-6).reverse().map((n) => `<div class="insp-kv"><span class="insp-k">${esc(n.created_at?.slice(0,19))}</span><span class="insp-v">${esc(n.text)}</span></div>`).join('') : '<div style="font-size:11px;color:var(--txt-dim)">NO DATA AVAILABLE in IndexedDB.</div>'}
        </div>
        <div class="plt-card">
          <h3>Accessible Server Memory Scopes</h3>
          ${Object.entries(mem.accessible_scope_counts || {}).map(([k, v]) => `<div class="insp-kv"><span class="insp-k">${esc(k)}</span><span class="insp-v">${esc(v)}</span></div>`).join('')}
        </div>
      </div>`;
    document.getElementById('idb-note-save')?.addEventListener('click', async () => {
      const t = document.getElementById('idb-note-in').value.trim();
      if (!t) return;
      await localDB.putNote(t);
      window.location.reload();
    });
    return;
  }

  if (path === '/models' || path.startsWith('/models/')) {
    const fam = path === '/models' ? null : path.split('/')[2];
    const data = fam
      ? await (await fetch(`/api/v2/platform/models/${encodeURIComponent(fam)}`)).json()
      : await (await fetch('/api/v2/platform/models')).json();
    const models = data.models || [];
    const famInfo = data.family_info;
    root.innerHTML = `
      <div class="plt-card">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px">
          <h3>${famInfo ? `${esc(famInfo.family)} — ${esc(famInfo.tagline)}` : 'GAJAHMADA UNIFIED GNNM MODEL REGISTRY'}</h3>
          ${capBadge(famInfo?.status || 'ACTIVE')}
        </div>
        <div style="font-size:12px;color:var(--txt-dim)">
          ${esc(famInfo ? famInfo.purpose : 'Preserves native GNNM families (BUMBLECORE, OPTICORE, TITANFORGE) and planned specializations (NEUROX, CODESTRIKE, KNOWTRON). Configurations are never misrepresented as trained weights.')}
        </div>
        ${famInfo?.note ? `<div style="font-size:11.5px;color:#fde68a">ℹ ${esc(famInfo.note)}</div>` : ''}
      </div>
      ${models.length ? `
      <div class="plt-card" style="overflow-x:auto">
        <table class="plt-table">
          <thead><tr><th>Model</th><th>Family</th><th>Type</th><th>Params</th><th>Lifecycle</th><th>Weights</th><th>BF16 / INT4 VRAM</th><th>Runtime</th></tr></thead>
          <tbody>
            ${models.map((m) => `
              <tr>
                <td><b style="color:var(--cyan)">${esc(m.model_name)}</b></td>
                <td>${esc(m.model_family)}</td>
                <td>${esc(m.architecture_type.toUpperCase())}</td>
                <td>${esc(m.parameter_billions)}B (${Number(m.parameter_count).toLocaleString()})</td>
                <td>${capBadge(m.lifecycle_state)}</td>
                <td>${m.weight_available ? `<span style="color:#86efac">YES (${esc(m.artifact_size_mb)} MB)</span>` : '<span style="color:#94a3b8">NONE</span>'}</td>
                <td>${esc(m.inference_vram_bf16_gb)} GB / ${esc(m.inference_vram_int4_gb)} GB</td>
                <td>${capBadge(m.runtime_compatibility)}</td>
              </tr>`).join('')}
          </tbody>
        </table>
      </div>` : `<div class="plt-card"><div style="font-size:12px;color:var(--txt-dim)">NO TRAINED ARTIFACTS — Planned specialization only.</div></div>`}`;
    return;
  }

  if (path === '/model-factory') {
    const estData = await (await fetch('/api/v2/platform/factory/estimator?preset=TITANFORGE-70B')).json();
    const est = estData.estimate;
    root.innerHTML = `
      <div class="plt-card">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px">
          <h3>GAJAHMADA MODEL FACTORY · ARCHITECTURE EDITOR &amp; RESOURCE ESTIMATOR</h3>
          ${capBadge('ACTIVE')} ${capBadge('UNAVAILABLE ON VERCEL FREE')}
        </div>
        <div style="font-size:11.5px;color:var(--txt-dim)">
          Analytical parameter calculator &amp; theoretical GPU/VRAM estimator (zero GPUs required). Training execution is disabled on Vercel Free.
        </div>
        <div class="form-grid" id="mf-form">
          <label class="form-field">Preset
            <select class="obs-select" id="mf-preset">${(estData.available_presets || []).map((p) => `<option value="${esc(p)}" ${p === 'TITANFORGE-70B' ? 'selected' : ''}>${esc(p)}</option>`).join('')}</select>
          </label>
          <label class="form-field">Target GPU
            <select class="obs-select" id="mf-gpu">${(estData.available_gpus || []).map((g) => `<option value="${esc(g)}">${esc(g)}</option>`).join('')}</select>
          </label>
          <label class="form-field">Architecture Type
            <select class="obs-select" id="mf-arch"><option value="dense">Dense</option><option value="moe">MoE (Experimental)</option></select>
          </label>
          <label class="form-field">d_model <input class="obs-input" type="number" id="mf-dmodel" value="8192" /></label>
          <label class="form-field">n_layers <input class="obs-input" type="number" id="mf-nlayers" value="80" /></label>
          <label class="form-field">n_heads <input class="obs-input" type="number" id="mf-nheads" value="64" /></label>
          <label class="form-field">n_kv_heads <input class="obs-input" type="number" id="mf-nkv" value="8" /></label>
          <label class="form-field">d_ff <input class="obs-input" type="number" id="mf-dff" value="28672" /></label>
          <label class="form-field">vocab_size <input class="obs-input" type="number" id="mf-vocab" value="128000" /></label>
        </div>
        <div style="display:flex;gap:8px;flex-wrap:wrap">
          <button class="obs-btn accent" id="mf-btn-preset">LOAD PRESET ESTIMATE</button>
          <button class="obs-btn" id="mf-btn-custom">VALIDATE CUSTOM ARCHITECTURE</button>
        </div>
      </div>
      <div class="plt-grid" id="mf-output"></div>`;

    const renderEst = (e) => {
      document.getElementById('mf-output').innerHTML = `
        <div class="plt-card">
          <h3>Analytical Parameters (${esc(e.model_name)})</h3>
          <div class="insp-kv"><span class="insp-k">Kind</span><span class="insp-v">${esc(e.estimate_kind)}</span></div>
          <div class="insp-kv"><span class="insp-k">Total Parameters</span><span class="insp-v">${esc(e.parameters.total_billions)}B (${Number(e.parameters.total).toLocaleString()})</span></div>
          <div class="insp-kv"><span class="insp-k">Active Parameters</span><span class="insp-v">${esc(e.parameters.active_billions)}B</span></div>
          <div class="insp-kv"><span class="insp-k">Checkpoint (BF16 / Full)</span><span class="insp-v">${esc(e.checkpoint_storage_gb.inference_weights_bf16_gb)} GB / ${esc(e.checkpoint_storage_gb.full_resumable_checkpoint_gb)} GB</span></div>
        </div>
        <div class="plt-card">
          <h3>Training Memory &amp; Cluster (${esc(e.compute_and_cluster.target_gpu)})</h3>
          <div class="insp-kv"><span class="insp-k">Weights + Grads (BF16)</span><span class="insp-v">${esc(e.training_memory_gb.weights_bf16_gb)} + ${esc(e.training_memory_gb.gradients_bf16_gb)} GB</span></div>
          <div class="insp-kv"><span class="insp-k">AdamW Optimizer (FP32)</span><span class="insp-v">${esc(e.training_memory_gb.optimizer_adamw_fp32_gb)} GB</span></div>
          <div class="insp-kv"><span class="insp-k">Total Sharded Mem</span><span class="insp-v">${esc(e.training_memory_gb.total_sharded_memory_gb)} GB</span></div>
          <div class="insp-kv"><span class="insp-k">Min GPUs (FSDP)</span><span class="insp-v">${esc(e.compute_and_cluster.recommended_min_gpus)} GPUs</span></div>
          <div class="insp-kv"><span class="insp-k">Chinchilla FLOPs</span><span class="insp-v">${esc(e.compute_and_cluster.estimated_training_zettaflops)} ZettaFLOPs</span></div>
          <div class="insp-kv"><span class="insp-k">Est. Days (64 / 256 GPUs)</span><span class="insp-v">${esc(e.compute_and_cluster.approx_duration_days_64_gpus)}d / ${esc(e.compute_and_cluster.approx_duration_days_256_gpus)}d</span></div>
        </div>
        <div class="plt-card">
          <h3>Inference VRAM by Precision</h3>
          ${Object.entries(e.inference_memory_by_precision || {}).map(([pr, v]) => `
            <div class="insp-kv"><span class="insp-k">${esc(pr.toUpperCase())}</span><span class="insp-v">${esc(v.total_vram_gb_est)} GB (weights ${esc(v.weights_gb)} GB)</span></div>`).join('')}
        </div>`;
    };
    renderEst(est);
    document.getElementById('mf-btn-preset').onclick = async () => {
      const p = document.getElementById('mf-preset').value;
      const g = document.getElementById('mf-gpu').value;
      const r = await (await fetch(`/api/v2/platform/factory/estimator?preset=${encodeURIComponent(p)}&target_gpu=${encodeURIComponent(g)}`)).json();
      renderEst(r.estimate);
    };
    document.getElementById('mf-btn-custom').onclick = async () => {
      const arch = document.getElementById('mf-arch').value;
      const res = await fetch('/api/v2/platform/factory/validate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: 'CUSTOM-GNNM',
          architecture_type: arch,
          num_experts: arch === 'moe' ? 8 : 1,
          num_experts_per_tok: arch === 'moe' ? 2 : 1,
          d_model: Number(document.getElementById('mf-dmodel').value),
          n_layers: Number(document.getElementById('mf-nlayers').value),
          n_heads: Number(document.getElementById('mf-nheads').value),
          n_kv_heads: Number(document.getElementById('mf-nkv').value),
          d_ff: Number(document.getElementById('mf-dff').value),
          vocab_size: Number(document.getElementById('mf-vocab').value),
          tie_embeddings: false,
          gradient_checkpointing: true,
          target_gpu: document.getElementById('mf-gpu').value,
        }),
      });
      const d = await res.json();
      if (!res.ok) return alert(`Validation Error: ${d.detail}`);
      renderEst(d.resource_estimate);
    };
    return;
  }

  if (path === '/training') {
    const plan = await (await fetch('/api/v2/platform/factory/distributed-plan?preset=TITANFORGE-70B')).json();
    root.innerHTML = `
      <div class="plt-card">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px">
          <h3>TITANFORGE 70B DISTRIBUTED TRAINING PREPARATION</h3>
          ${capBadge(plan.status)}
        </div>
        <div style="font-size:12px;color:#fca5a5">
          Distributed Tests Executed: <b>${esc(plan.distributed_tests_executed)}</b> · Execution Ready: <b>${esc(plan.execution_ready)}</b>
        </div>
        ${(plan.blockers || []).map((b) => `<div style="font-size:11.5px;color:#fde68a">• ${esc(b)}</div>`).join('')}
      </div>
      <div class="plt-grid">
        <div class="plt-card">
          <h3>Distributed Strategy Config</h3>
          ${Object.entries(plan.distributed_config || {}).map(([k, v]) => `<div class="insp-kv"><span class="insp-k">${esc(k)}</span><span class="insp-v">${esc(v)}</span></div>`).join('')}
        </div>
        <div class="plt-card">
          <h3>Modular Data Pipeline Stages</h3>
          ${(plan.data_pipeline?.stages || []).map((s, i) => `<div class="insp-kv"><span class="insp-k">Stage ${i + 1}</span><span class="insp-v">${esc(s)}</span></div>`).join('')}
        </div>
      </div>`;
    return;
  }

  if (path === '/benchmarks') {
    const bm = await (await fetch('/api/v2/platform/benchmarks')).json();
    root.innerHTML = `
      <div class="plt-card">
        <div style="display:flex;justify-content:space-between;align-items:center">
          <h3>VERIFIED BENCHMARKS &amp; EVALUATION METADATA</h3>
          ${capBadge(bm.status === 'AVAILABLE' ? 'ACTIVE' : 'NOT CONFIGURED')}
        </div>
        <pre style="font-family:var(--font-mono);font-size:11px;color:#cbd5e1;overflow-x:auto;white-space:pre-wrap">${esc(bm.report ? JSON.stringify(bm.report, null, 2) : (bm.note || 'NO DATA AVAILABLE'))}</pre>
      </div>`;
    return;
  }

  if (path === '/infrastructure') {
    root.innerHTML = `
      <div class="plt-grid">
        <div class="plt-card">
          <h3>Vercel Free Control Plane</h3>
          ${capBadge('ACTIVE')}
          <div class="insp-kv"><span class="insp-k">Runtime</span><span class="insp-v">${esc(status.runtime_environment)}</span></div>
          <div class="insp-kv"><span class="insp-k">Native Inference</span><span class="insp-v">${esc(status.native_model_inference?.status_banner)}</span></div>
          <div class="insp-kv"><span class="insp-k">Training Execution</span><span class="insp-v">${esc(status.training_execution?.status)}</span></div>
          <div class="insp-kv"><span class="insp-k">Storage Mode</span><span class="insp-v">${esc(status.storage?.mode)}</span></div>
        </div>
        <div class="plt-card">
          <h3>Future Large-Scale Infrastructure</h3>
          ${capBadge('PLANNED')}
          <div style="font-size:12px;color:var(--txt-dim);line-height:1.6">
            • TITANFORGE 70B &amp; 120B+ external GPU cluster training (FSDP / TP / PP)<br/>
            • Dedicated native GNNM inference serving cluster<br/>
            • Permanent multi-node GIP/2.0 federation &amp; federated learning<br/>
            • Optional object storage &amp; managed Postgres/pgvector adapters
          </div>
        </div>
      </div>`;
    return;
  }

  if (path === '/nexus') {
    const nx = await (await fetch('/api/v2/platform/nexus')).json();
    root.innerHTML = `
      <div class="plt-card">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px">
          <h3>NEXUS PRIME — ORCHESTRATION &amp; CAPABILITY ROUTING</h3>
          ${capBadge('ACTIVE')}
        </div>
        <div style="font-size:12px;color:var(--txt-dim)">Runtime: <b>${esc(nx.runtime_mode)}</b> · ${esc(nx.native_inference_banner)}</div>
      </div>
      <div class="plt-grid">
        <div class="plt-card">
          <h3>Active Capabilities</h3>
          ${(nx.active_capabilities || []).map((c) => `<div class="insp-kv"><span class="insp-k">${esc(c.name)}</span><span class="insp-v">${capBadge(c.status)}</span></div>`).join('')}
        </div>
        <div class="plt-card">
          <h3>Future Distributed Roadmap</h3>
          ${(nx.future_distributed_roadmap || []).map((c) => `<div class="insp-kv"><span class="insp-k">${esc(c.name)}</span><span class="insp-v">${capBadge(c.status)}</span></div>`).join('')}
        </div>
      </div>`;
    return;
  }

  if (path === '/settings') {
    const savedTheme = localStorage.getItem('gm_pref_density') || 'standard';
    root.innerHTML = `
      <div class="plt-card">
        <h3>SETTINGS &amp; BROWSER STORAGE</h3>
        <div class="insp-kv"><span class="insp-k">Browser localStorage</span><span class="insp-v">${capBadge('ACTIVE')}</span></div>
        <div class="insp-kv"><span class="insp-k">Browser IndexedDB</span><span class="insp-v">${capBadge(window.indexedDB ? 'ACTIVE' : 'UNAVAILABLE')}</span></div>
        <div class="insp-kv"><span class="insp-k">Server Persistent DB</span><span class="insp-v">${capBadge(status.storage?.persistent_database_configured ? 'ACTIVE' : 'NOT CONFIGURED')}</span></div>
        <label class="form-field" style="max-width:260px">UI Density Preference (saved in localStorage)
          <select class="obs-select" id="set-density">
            <option value="standard" ${savedTheme === 'standard' ? 'selected' : ''}>Standard</option>
            <option value="compact" ${savedTheme === 'compact' ? 'selected' : ''}>Compact</option>
          </select>
        </label>
      </div>`;
    document.getElementById('set-density').onchange = (e) => localStorage.setItem('gm_pref_density', e.target.value);
    return;
  }

  // Default: /documentation
  root.innerHTML = `
    <div class="plt-card">
      <h3>GAJAHMADA AI DOCUMENTATION &amp; VERCEL FREE GUIDE</h3>
      <div style="font-size:12.5px;color:#cbd5e1;line-height:1.65">
        <p><b>Independent Intelligence Principle:</b> GajahMada AI preserves its native PyTorch Decoder-Only Transformer (<b>GNNM</b>) and Hybrid Knowledge Architecture without mandatory commercial AI APIs.</p>
        <p style="margin-top:8px"><b>Separation of Concerns:</b> Vercel Free hosts the lightweight FastAPI control plane, Observatory, Model Factory metadata, and Hybrid Knowledge UI. Large-scale training (<b>TITANFORGE 70B</b>) and FAISS CPU native binaries remain cleanly separated for local/external GPU infrastructure.</p>
        <p style="margin-top:8px"><b>Vercel Hobby Note:</b> Vercel Free (Hobby) is intended for personal and non-commercial use. See <code>docs/VERCEL_DEPLOYMENT_GUIDE.md</code> and <code>README.md</code> in the repository for complete deployment instructions.</p>
      </div>
    </div>`;
});
