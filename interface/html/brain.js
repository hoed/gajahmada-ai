/**
 * GAJAHMADA BRAIN & KNOWLEDGE OBSERVATORY — Interactive Frontend Controller
 * Connected 100% to live `/api/v2/observatory/*` endpoints + SSE `/stream`.
 * Zero mock data, zero fake animations on inactive subsystems.
 */
document.addEventListener('DOMContentLoaded', () => {
  const API = window.location.origin;

  // ─── Session Auth Helper ──────────────────────────────────────────────────
  const getApiKey = () => {
    try { return (window.sessionStorage.getItem('gm_admin_api_key') || '').trim(); } catch { return ''; }
  };
  const authHeaders = (extra = {}) => {
    const h = { ...extra };
    const k = getApiKey();
    if (k) h['X-API-Key'] = k;
    return h;
  };
  const updateAuthBtn = () => {
    const btn = document.getElementById('btn-obs-auth');
    if (!btn) return;
    const ok = Boolean(getApiKey());
    btn.textContent = ok ? '🔑 AUTH ✓' : '🔑 AUTH';
    btn.classList.toggle('active', ok);
  };
  updateAuthBtn();
  document.getElementById('btn-obs-auth')?.addEventListener('click', () => {
    const entered = window.prompt(
      'Enter GajahMada Admin/Operator API Key (stored in browser sessionStorage):',
      getApiKey() ? '********' : ''
    );
    if (entered === null || entered.trim() === '********') return;
    try {
      if (entered.trim()) window.sessionStorage.setItem('gm_admin_api_key', entered.trim());
      else window.sessionStorage.removeItem('gm_admin_api_key');
    } catch { /* ignore */ }
    updateAuthBtn();
    refreshAllActive();
  });

  const esc = (s) => String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  const badge = (st) => `<span class="status-badge st-${esc(st || 'NO_DATA')}">${esc(st || 'N/A')}</span>`;

  // ─── Tab Navigation (supports /brain/{activity,knowledge,memory,reasoning,learning}) ───
  const PATH_TO_TAB = {
    '/brain/activity': 'cognitive-activity',
    '/brain/reasoning': 'cognitive-activity',
    '/brain/knowledge': 'knowledge-universe',
    '/brain/memory': 'memory-explorer',
    '/brain/learning': 'learning-observatory',
  };
  let activeTab = PATH_TO_TAB[window.location.pathname.replace(/\/$/, '')] || 'brain-overview';
  const activateTab = (tabId) => {
    activeTab = tabId;
    document.querySelectorAll('.obs-tab').forEach((b) => b.classList.toggle('active', b.dataset.tab === tabId));
    document.querySelectorAll('.obs-view').forEach((v) => v.classList.toggle('active', v.id === `view-${tabId}`));
    requestAnimationFrame(() => {
      if (activeTab === 'brain-overview') brainGraph.resize();
      else if (activeTab === 'knowledge-universe') kgGraph.resize();
      else if (activeTab === 'distributed-nodes') nodesGraph.resize();
    });
  };
  document.querySelectorAll('.obs-tab').forEach((btn) => {
    btn.addEventListener('click', () => {
      activateTab(btn.dataset.tab);
      refreshAllActive();
    });
  });
  if (activeTab !== 'brain-overview') {
    activateTab(activeTab);
  }

  // ─── Reusable Interactive 2D Neural / Knowledge Graph Engine ──────────────
  const GROUP_COLORS = {
    GNNM_CORE: '#ff5a1f',
    COGNITIVE_ENGINE: '#00e5ff',
    KNOWLEDGE_SYSTEM: '#3b82f6',
    MEMORY_SYSTEM: '#22c55e',
    LEARNING_SYSTEM: '#a855f7',
    DISTRIBUTED_NETWORK: '#f59e0b',
    MEMORY: '#22c55e',
    DOCUMENT: '#00e5ff',
    SOURCE: '#ff5a1f',
    ENTITY: '#a855f7',
    LOCAL: '#00e5ff',
    TRUSTED: '#22c55e',
    PENDING: '#f59e0b',
    REVOKED: '#ef4444',
  };

  class InteractiveGraphCanvas {
    constructor(canvasId, onSelect) {
      this.canvas = document.getElementById(canvasId);
      this.ctx = this.canvas ? this.canvas.getContext('2d') : null;
      this.onSelect = onSelect;
      this.nodes = [];
      this.edges = [];
      this.zoom = 1.0;
      this.panX = 0;
      this.panY = 0;
      this.dragging = false;
      this.lastX = 0;
      this.lastY = 0;
      this.hovered = null;
      this.selected = null;
      this.filterGroup = '';
      this.searchTerm = '';
      this.pulsePhase = 0;
      this.tooltip = document.getElementById('canvas-tooltip');
      if (this.canvas) this._bindEvents();
    }

    resize() {
      if (!this.canvas || !this.ctx || !this.canvas.parentElement) return false;
      const rect = this.canvas.parentElement.getBoundingClientRect();
      if (rect.width < 10 || rect.height < 10) return false;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const targetW = Math.round(rect.width * dpr);
      const targetH = Math.round(rect.height * dpr);
      if (this.canvas.width !== targetW || this.canvas.height !== targetH || this.width !== rect.width || this.height !== rect.height) {
        this.canvas.width = targetW;
        this.canvas.height = targetH;
        this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        this.width = rect.width;
        this.height = rect.height;
      }
      return true;
    }

    _bindEvents() {
      this.resize();
      window.addEventListener('resize', () => this.resize());
      if (typeof ResizeObserver !== 'undefined' && this.canvas.parentElement) {
        const ro = new ResizeObserver(() => this.resize());
        ro.observe(this.canvas.parentElement);
      }

      this.canvas.addEventListener('mousedown', (e) => {
        this.dragging = true; this.lastX = e.clientX; this.lastY = e.clientY;
      });
      window.addEventListener('mouseup', () => { this.dragging = false; });
      window.addEventListener('mousemove', (e) => {
        const rect = this.canvas.getBoundingClientRect();
        if (this.dragging) {
          this.panX += e.clientX - this.lastX;
          this.panY += e.clientY - this.lastY;
          this.lastX = e.clientX; this.lastY = e.clientY;
          return;
        }
        const mx = (e.clientX - rect.left - this.width / 2 - this.panX) / this.zoom;
        const my = (e.clientY - rect.top - this.height / 2 - this.panY) / this.zoom;
        const hit = this._hitTest(mx, my);
        this.hovered = hit;
        if (hit && this.tooltip) {
          this.tooltip.style.display = 'block';
          this.tooltip.style.left = `${Math.min(window.innerWidth - 270, e.clientX + 14)}px`;
          this.tooltip.style.top = `${Math.min(window.innerHeight - 90, e.clientY + 14)}px`;
          this.tooltip.innerHTML = `<b style="color:var(--cyan)">${esc(hit.label)}</b><br><span style="color:#94a3b8">${esc(hit.group || hit.category || '')} · ${esc(hit.status || hit.verification_status || '')}</span>`;
        } else if (this.tooltip) {
          this.tooltip.style.display = 'none';
        }
      });
      this.canvas.addEventListener('click', (e) => {
        const rect = this.canvas.getBoundingClientRect();
        const mx = (e.clientX - rect.left - this.width / 2 - this.panX) / this.zoom;
        const my = (e.clientY - rect.top - this.height / 2 - this.panY) / this.zoom;
        const hit = this._hitTest(mx, my);
        if (hit) {
          this.selected = hit;
          if (this.onSelect) this.onSelect(hit);
        }
      });
      this.canvas.addEventListener('wheel', (e) => {
        e.preventDefault();
        const factor = e.deltaY < 0 ? 1.12 : 0.89;
        this.zoom = Math.max(0.4, Math.min(2.8, this.zoom * factor));
      }, { passive: false });
    }

    _hitTest(mx, my) {
      for (const n of this.nodes) {
        if (!this._isVisible(n)) continue;
        const dx = mx - n.x, dy = my - n.y;
        if (dx * dx + dy * dy <= (n.r + 6) * (n.r + 6)) return n;
      }
      return null;
    }

    _isVisible(n) {
      if (this.filterGroup && (n.group || n.category) !== this.filterGroup) return false;
      if (this.searchTerm) {
        const hay = `${n.label} ${n.id} ${n.group || ''} ${n.category || ''} ${n.description || ''}`.toLowerCase();
        if (!hay.includes(this.searchTerm)) return false;
      }
      return true;
    }

    setData(nodes, edges, layoutMode = 'cortex') {
      const grouped = {};
      nodes.forEach((n) => {
        const g = n.group || n.category || 'DEFAULT';
        if (!grouped[g]) grouped[g] = [];
        grouped[g].push(n);
      });
      const groupKeys = Object.keys(grouped);
      const positioned = [];

      if (layoutMode === 'cortex') {
        // Brain-cortex cluster layout around 6 orbital lobes
        const lobeCenters = [
          { x: -180, y: -90 }, { x: 0, y: -135 }, { x: 185, y: -85 },
          { x: -175, y: 85 },  { x: 5, y: 120 },  { x: 185, y: 85 },
        ];
        groupKeys.forEach((gk, gi) => {
          const center = lobeCenters[gi % lobeCenters.length];
          const list = grouped[gk];
          list.forEach((n, ni) => {
            const ang = (ni / Math.max(1, list.length)) * Math.PI * 2 - Math.PI / 2;
            const rad = list.length === 1 ? 0 : 62;
            positioned.push({
              ...n,
              x: center.x + Math.cos(ang) * rad,
              y: center.y + Math.sin(ang) * rad,
              r: n.id === 'gnnm-core' || n.id === 'know-controller' ? 14 : 10,
              color: GROUP_COLORS[gk] || '#00e5ff',
            });
          });
        });
      } else {
        // Radial Knowledge / Peer network layout
        nodes.forEach((n, idx) => {
          const isCenter = idx === 0 && layoutMode === 'nodes';
          const ring = isCenter ? 0 : (1 + Math.floor(idx / 12)) * 105;
          const angle = ((idx % 12) / Math.min(12, Math.max(1, nodes.length))) * Math.PI * 2;
          const key = n.category || n.trust_status || 'DOCUMENT';
          positioned.push({
            ...n,
            x: isCenter ? 0 : Math.cos(angle) * ring,
            y: isCenter ? 0 : Math.sin(angle) * ring,
            r: isCenter ? 16 : (key === 'SOURCE' ? 11 : 8),
            color: GROUP_COLORS[key] || '#00e5ff',
          });
        });
      }

      const hadNodes = this.nodes.length > 0;
      this.nodes = positioned;
      this.nodeMap = Object.fromEntries(positioned.map((n) => [n.id, n]));
      this.edges = edges;
      if (!hadNodes && positioned.length > 0 && this.resize()) {
        this.fitView();
      }
    }

    resetView() { this.zoom = 1.0; this.panX = 0; this.panY = 0; }
    fitView() {
      if (!this.nodes.length || !this.width || !this.height) {
        this.zoom = 0.92; this.panX = 0; this.panY = 0;
        return;
      }
      let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
      for (const n of this.nodes) {
        if (!this._isVisible(n)) continue;
        if (n.x < minX) minX = n.x;
        if (n.x > maxX) maxX = n.x;
        if (n.y < minY) minY = n.y;
        if (n.y > maxY) maxY = n.y;
      }
      if (!isFinite(minX)) { this.resetView(); return; }
      const spanX = Math.max(120, maxX - minX + 120);
      const spanY = Math.max(120, maxY - minY + 120);
      const usableW = Math.max(200, this.width - 80);
      const usableH = Math.max(160, this.height - 140);
      this.zoom = Math.max(0.45, Math.min(1.35, Math.min(usableW / spanX, usableH / spanY)));
      this.panX = -((minX + maxX) / 2) * this.zoom;
      this.panY = -((minY + maxY) / 2) * this.zoom;
    }

    draw() {
      if (!this.ctx) return;
      if (!this.resize()) return;
      const ctx = this.ctx;
      ctx.clearRect(0, 0, this.width, this.height);

      // Subtle futuristic neural radar rings
      ctx.save();
      const cx = this.width / 2 + this.panX;
      const cy = this.height / 2 + this.panY;
      ctx.strokeStyle = 'rgba(56, 189, 248, 0.06)';
      ctx.lineWidth = 1;
      for (const rad of [95, 185, 280]) {
        ctx.beginPath();
        ctx.arc(cx, cy, rad * this.zoom, 0, Math.PI * 2);
        ctx.stroke();
      }
      ctx.restore();

      ctx.save();
      ctx.translate(cx, cy);
      ctx.scale(this.zoom, this.zoom);

      this.pulsePhase = (this.pulsePhase + 0.014) % 1;
      const focusNode = this.selected || this.hovered;

      // Draw edges
      for (const e of this.edges) {
        const a = this.nodeMap?.[e.source];
        const b = this.nodeMap?.[e.target];
        if (!a || !b || !this._isVisible(a) || !this._isVisible(b)) continue;
        const isConnected = focusNode && (focusNode.id === a.id || focusNode.id === b.id);
        const isContra = e.relation === 'CONTRADICTS';
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.strokeStyle = isContra
          ? 'rgba(239, 68, 68, 0.75)'
          : (isConnected ? 'rgba(0, 229, 255, 0.85)' : (e.active ? 'rgba(56, 189, 248, 0.36)' : 'rgba(148, 163, 184, 0.14)'));
        ctx.lineWidth = isConnected ? 1.8 : 1;
        ctx.stroke();

        // ONLY animate pulses when backend reports actual active flow!
        if (e.active) {
          const px = a.x + (b.x - a.x) * this.pulsePhase;
          const py = a.y + (b.y - a.y) * this.pulsePhase;
          ctx.beginPath();
          ctx.arc(px, py, 2.3, 0, Math.PI * 2);
          ctx.fillStyle = '#ff5a1f';
          ctx.fill();
        }
      }

      // Draw nodes
      for (const n of this.nodes) {
        if (!this._isVisible(n)) continue;
        const isSel = this.selected && this.selected.id === n.id;
        const isHov = this.hovered && this.hovered.id === n.id;
        const dim = n.status === 'UNAVAILABLE' || n.status === 'NOT_CONFIGURED';

        // Outer glow
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.r + (isSel || isHov ? 6 : 3), 0, Math.PI * 2);
        ctx.fillStyle = dim ? 'rgba(148,163,184,0.08)' : `${n.color}28`;
        ctx.fill();

        // Core circle
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
        ctx.fillStyle = dim ? '#334155' : n.color;
        ctx.fill();
        ctx.lineWidth = isSel ? 2.2 : 1;
        ctx.strokeStyle = isSel ? '#ffffff' : 'rgba(255,255,255,0.35)';
        ctx.stroke();

        // Label
        ctx.font = '600 10px "JetBrains Mono", monospace';
        ctx.fillStyle = dim ? '#64748b' : '#e2e8f0';
        ctx.textAlign = 'center';
        ctx.fillText(String(n.label || n.id || '').slice(0, 26), n.x, n.y + n.r + 13);
      }

      ctx.restore();
    }
  }

  // Instantiate 3 interactive graph canvases
  const brainGraph = new InteractiveGraphCanvas('canvas-brain', (node) => {
    const el = document.getElementById('brain-node-detail');
    if (!el) return;
    const metricsHtml = Object.entries(node.metrics || {}).map(([k, v]) =>
      `<div class="insp-kv"><span class="insp-k">${esc(k)}</span><span class="insp-v">${esc(typeof v === 'object' ? JSON.stringify(v) : v)}</span></div>`
    ).join('');
    el.innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
        <b style="color:var(--cyan);font-size:13px">${esc(node.label)}</b>
        ${badge(node.status)}
      </div>
      <div style="color:var(--txt-dim);margin-bottom:8px">${esc(node.description || '')}</div>
      <div class="insp-kv"><span class="insp-k">Group</span><span class="insp-v">${esc(node.group)}</span></div>
      <div class="insp-kv"><span class="insp-k">Active Flow</span><span class="insp-v">${node.active ? 'YES' : 'IDLE'}</span></div>
      ${metricsHtml}
    `;
  });

  const kgGraph = new InteractiveGraphCanvas('canvas-kg', (node) => {
    const el = document.getElementById('kg-node-detail');
    if (!el) return;
    el.innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
        <b style="color:var(--cyan);font-size:12.5px">${esc(node.label)}</b>
        ${badge(node.verification_status)}
      </div>
      <div style="color:#e2e8f0;margin-bottom:8px;white-space:pre-wrap">${esc(node.content_preview || '')}</div>
      <div class="insp-kv"><span class="insp-k">Node ID</span><span class="insp-v">${esc(node.id)}</span></div>
      <div class="insp-kv"><span class="insp-k">Category</span><span class="insp-v">${esc(node.category)}</span></div>
      <div class="insp-kv"><span class="insp-k">Domain</span><span class="insp-v">${esc(node.domain || 'N/A')}</span></div>
      <div class="insp-kv"><span class="insp-k">Source</span><span class="insp-v">${esc(node.source || 'N/A')}</span></div>
      <div class="insp-kv"><span class="insp-k">Access Scope</span><span class="insp-v">${esc(node.visibility_policy || 'N/A')}</span></div>
      <div class="insp-kv"><span class="insp-k">Created</span><span class="insp-v">${esc(node.created_at || 'N/A')}</span></div>
    `;
  });

  const nodesGraph = new InteractiveGraphCanvas('canvas-nodes', (node) => {
    const el = document.getElementById('dist-node-detail');
    if (!el) return;
    el.innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
        <b style="color:var(--cyan)">${esc(node.label)}</b>
        ${badge(node.trust_status || 'UNVERIFIED')}
      </div>
      <div class="insp-kv"><span class="insp-k">Key Fingerprint</span><span class="insp-v">${esc(node.fingerprint || 'N/A')}</span></div>
      <div class="insp-kv"><span class="insp-k">Connection</span><span class="insp-v">${esc(node.connection_status || 'N/A')}</span></div>
      <div class="insp-kv"><span class="insp-k">Organization</span><span class="insp-v">${esc(node.organization || 'UNVERIFIED_OPERATOR')}</span></div>
      <div class="insp-kv"><span class="insp-k">Endpoint</span><span class="insp-v">${esc(node.endpoint || 'local')}</span></div>
      <div class="insp-kv"><span class="insp-k">Last Heartbeat</span><span class="insp-v">${esc(node.last_heartbeat || 'N/A')}</span></div>
    `;
  });

  // Animation loop (lightweight)
  function animate() {
    if (activeTab === 'brain-overview') brainGraph.draw();
    else if (activeTab === 'knowledge-universe') kgGraph.draw();
    else if (activeTab === 'distributed-nodes') nodesGraph.draw();
    requestAnimationFrame(animate);
  }
  requestAnimationFrame(animate);

  // ─── Data Loaders (100% Real Backend Data) ────────────────────────────────
  async function fetchJson(path) {
    const res = await fetch(`${API}${path}`, { headers: authHeaders() });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  }

  function setOnlineState(online) {
    const pill = document.getElementById('pill-live');
    const txt = document.getElementById('txt-live-state');
    if (pill && txt) {
      pill.classList.toggle('live', online);
      pill.classList.toggle('offline', !online);
      txt.textContent = online ? 'LIVE' : 'OFFLINE';
    }
    const sync = document.getElementById('txt-last-sync');
    if (sync && online) sync.textContent = new Date().toLocaleTimeString();
  }

  async function loadOverviewAndBrain() {
    try {
      const [ov, br] = await Promise.all([
        fetchJson('/api/v2/observatory/overview'),
        fetchJson('/api/v2/observatory/brain'),
      ]);
      setOnlineState(true);
      document.getElementById('obs-node-subtitle').textContent = `NODE ${ov.node_id.slice(0, 14)}... · ROLE: ${ov.actor_role.toUpperCase()}`;
      const m = ov.active_runtime_model;
      document.getElementById('txt-active-model').textContent = m.neural_inference_active
        ? `${m.model_id} (${m.backend})`
        : `INACTIVE (${m.registry_active_version || 'NO WEIGHTS'})`;

      document.getElementById('kpi-mem').textContent = ov.counts.accessible_memory_records;
      document.getElementById('kpi-ext').textContent = ov.counts.accessible_external_chunks;
      document.getElementById('kpi-kg').textContent = ov.counts.knowledge_graph_relations;
      document.getElementById('kpi-agents').textContent = `${ov.counts.running_tasks} / ${ov.counts.registered_agents}`;
      document.getElementById('kpi-peers').textContent = `${ov.counts.trusted_peers} / ${ov.counts.registered_peers}`;
      document.getElementById('kpi-ledger').textContent = ov.counts.ledger_height;

      const healthEl = document.getElementById('brain-health-list');
      healthEl.innerHTML = Object.entries(ov.subsystems || {}).map(([k, v]) => `
        <div class="insp-kv"><span class="insp-k">${esc(k)}</span><span>${badge(v.status)}</span></div>
      `).join('');

      brainGraph.setData(br.nodes, br.edges, 'cortex');
    } catch {
      setOnlineState(false);
    }
  }

  async function loadCognition() {
    try {
      const data = await fetchJson('/api/v2/observatory/cognition?limit=15');
      setOnlineState(true);
      const latestStages = data.recent_traces?.[0]?.stages_executed || [];
      const stageStatusMap = Object.fromEntries(latestStages.map((s) => [s.stage, s.status]));

      document.getElementById('cog-stages-flow').innerHTML = data.pipeline_stages.map((st) => {
        const execSt = stageStatusMap[st.stage] || 'IDLE';
        return `
          <div class="stage-node ${execSt === 'COMPLETED' ? 'active-step' : ''}">
            <div style="display:flex;justify-content:space-between;align-items:center">
              <b style="font-size:10.5px;color:var(--cyan)">${st.index}. ${esc(st.stage)}</b>
              ${badge(execSt)}
            </div>
            <div style="font-size:10px;color:var(--txt-dim)">${esc(st.description)}</div>
          </div>
        `;
      }).join('');

      const listEl = document.getElementById('cog-traces-list');
      if (!data.recent_traces.length) {
        listEl.innerHTML = '<div class="insp-card">No reasoning traces recorded yet. Submit a query above to observe live pipeline execution.</div>';
        return;
      }
      listEl.innerHTML = data.recent_traces.map((t) => `
        <div class="insp-card">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px">
            <b style="color:var(--cyan);font-family:var(--font-mono);font-size:11px">${esc(t.request_id)}</b>
            ${badge(t.verification_status)}
          </div>
          <div style="margin-bottom:6px;color:#f8fafc">${esc(t.objective_preview)}</div>
          <div style="font-size:10px;font-family:var(--font-mono);color:var(--txt-dim)">
            ${esc(t.interpretation)} · conf=${t.confidence} · stages=${t.stages_executed.length}
          </div>
        </div>
      `).join('');
    } catch { setOnlineState(false); }
  }

  let kgOffset = 0;
  const kgLimit = 35;
  async function loadKnowledgeGraph() {
    try {
      const q = encodeURIComponent((document.getElementById('kg-search')?.value || '').trim());
      const vf = encodeURIComponent(document.getElementById('kg-ver-filter')?.value || '');
      const data = await fetchJson(`/api/v2/observatory/knowledge/graph?limit=${kgLimit}&offset=${kgOffset}&query=${q}&verification_status=${vf}`);
      setOnlineState(true);
      document.getElementById('kg-page-info').textContent = `${data.nodes.length} nodes · total ${data.pagination.total}`;
      kgGraph.setData(data.nodes, data.edges, 'radial');
    } catch { setOnlineState(false); }
  }

  let selectedMemScope = '';
  async function loadMemory() {
    try {
      const q = encodeURIComponent((document.getElementById('mem-search')?.value || '').trim());
      const sc = encodeURIComponent(selectedMemScope);
      const data = await fetchJson(`/api/v2/observatory/memory?limit=40&scope=${sc}&query=${q}`);
      setOnlineState(true);
      document.getElementById('mem-auth-banner').innerHTML =
        `Active Actor Role: <b style="color:var(--cyan)">${esc(data.actor_role.toUpperCase())}</b> · Showing ${data.records.length} of ${data.pagination.total} authorized memory records. PRIVATE/LOCAL/SESSION/AGENT/ORGANIZATION scopes require matching authenticated credentials.`;

      const pillsEl = document.getElementById('mem-scope-pills');
      pillsEl.innerHTML = `<button class="obs-btn ${selectedMemScope === '' ? 'accent' : ''}" data-sc="">ALL (${data.pagination.total})</button>` +
        data.available_scopes.map((s) =>
          `<button class="obs-btn ${selectedMemScope === s ? 'accent' : ''}" data-sc="${esc(s)}">${esc(s)} (${data.accessible_scope_counts[s] ?? 0})</button>`
        ).join('');
      pillsEl.querySelectorAll('button').forEach((b) => b.addEventListener('click', () => {
        selectedMemScope = b.dataset.sc;
        loadMemory();
      }));

      document.getElementById('mem-records-grid').innerHTML = data.records.length
        ? data.records.map((r) => `
            <div class="insp-card">
              <div style="display:flex;justify-content:space-between;gap:6px;margin-bottom:5px">
                <b style="color:var(--cyan);font-family:var(--font-mono);font-size:10.5px">${esc(r.record_id)}</b>
                <div>${badge(r.visibility_policy)} ${badge(r.verification_status)}</div>
              </div>
              <div style="margin-bottom:6px">${esc(r.content)}</div>
              <div style="font-size:10px;font-family:var(--font-mono);color:var(--txt-dim)">tier=${esc(r.memory_type)} · v${r.version} · ${esc(r.created_at || 'N/A')}</div>
            </div>
          `).join('')
        : '<div class="insp-card">No authorized memory records match the current filter.</div>';
    } catch { setOnlineState(false); }
  }

  async function loadLearning() {
    try {
      const data = await fetchJson('/api/v2/observatory/learning');
      setOnlineState(true);
      document.getElementById('learn-states-grid').innerHTML = Object.entries(data.knowledge_state_summary).map(([k, v]) => `
        <div class="insp-card">
          <div class="kpi-lbl">${esc(k)} KNOWLEDGE</div>
          <div style="margin-top:6px">${badge(v)}</div>
        </div>
      `).join('');

      document.getElementById('learn-lifecycle-flow').innerHTML = data.lifecycle_steps.map((s) => `
        <div class="stage-node">
          <b style="font-size:10.5px;color:var(--cyan)">${s.step}. ${esc(s.name)}</b>
          <div>${badge(s.status)}</div>
        </div>
      `).join('');

      document.getElementById('learn-versions-grid').innerHTML = data.model_versions.length
        ? data.model_versions.map((v) => `
            <div class="insp-card">
              <div style="display:flex;justify-content:space-between;margin-bottom:5px">
                <b style="color:var(--cyan)">${esc(v.version_id)}</b>
                ${badge(v.promotion_status)}
              </div>
              <div class="insp-kv"><span class="insp-k">Model</span><span class="insp-v">${esc(v.model_name)}</span></div>
              <div class="insp-kv"><span class="insp-k">SHA-256</span><span class="insp-v">${esc((v.checkpoint_sha256 || 'N/A').slice(0, 16))}</span></div>
              <div class="insp-kv"><span class="insp-k">Rollback Ref</span><span class="insp-v">${esc(v.rollback_reference || 'N/A')}</span></div>
            </div>
          `).join('')
        : '<div class="insp-card">No GNNM candidate versions registered yet (N/A). Run isolated training via GNNM Lab or /api/v2/learning/train.</div>';

      document.getElementById('learn-events-list').innerHTML = data.recent_learning_events.length
        ? data.recent_learning_events.map((e) => `
            <div class="insp-card">
              <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                <b style="font-family:var(--font-mono);font-size:10.5px;color:var(--orange)">${esc(e.event_type)}</b>
                ${badge(e.internalization_status)}
              </div>
              <div style="margin-bottom:4px">${esc(e.source_title || e.summary)}</div>
              <div style="font-size:10px;font-family:var(--font-mono);color:var(--txt-dim)">domain=${esc(e.knowledge_domain)} · ver=${esc(e.verification_status)}</div>
            </div>
          `).join('')
        : '<div class="insp-card">No temporal learning events recorded yet.</div>';
    } catch { setOnlineState(false); }
  }

  async function loadAgents() {
    try {
      const data = await fetchJson('/api/v2/observatory/agents');
      setOnlineState(true);
      document.getElementById('agents-grid').innerHTML = data.agents.map((a) => `
        <div class="insp-card">
          <div style="display:flex;justify-content:space-between;margin-bottom:6px">
            <b style="color:var(--cyan)">${esc(a.name)}</b>
            ${badge(a.status)}
          </div>
          <div class="insp-kv"><span class="insp-k">Role</span><span class="insp-v">${esc(a.role)}</span></div>
          <div class="insp-kv"><span class="insp-k">Memory Scope</span><span class="insp-v">${esc(a.memory_scope)}</span></div>
          <div class="insp-kv"><span class="insp-k">Node</span><span class="insp-v">${esc(a.assigned_node)}</span></div>
        </div>
      `).join('');

      document.getElementById('agent-tasks-grid').innerHTML = data.tasks.length
        ? data.tasks.map((t) => `
            <div class="insp-card">
              <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                <b style="font-family:var(--font-mono);font-size:10.5px">${esc(t.task_id)}</b>
                ${badge(t.state)}
              </div>
              <div>${esc(t.objective)}</div>
            </div>
          `).join('')
        : '<div class="insp-card">No autonomous agent tasks executed yet.</div>';
    } catch { setOnlineState(false); }
  }

  async function loadNodes() {
    try {
      const data = await fetchJson('/api/v2/observatory/nodes');
      setOnlineState(true);
      document.getElementById('dist-summary-pill').textContent =
        `LOCAL + ${data.registered_peers_count} PEERS (${data.trusted_peers_count} TRUSTED)`;

      const gNodes = [{
        id: data.local_node.node_id,
        label: 'LOCAL NODE',
        fingerprint: data.local_node.fingerprint,
        connection_status: 'CONNECTED',
        trust_status: 'LOCAL',
        organization: 'Sovereign Local Operator',
        endpoint: window.location.origin,
      }];
      const gEdges = [];
      data.peers.forEach((p) => {
        gNodes.push({
          id: p.node_id,
          label: p.node_name || p.node_id.slice(0, 12),
          fingerprint: p.fingerprint,
          connection_status: p.connection_status,
          trust_status: p.trust_status,
          organization: p.organization,
          endpoint: p.endpoint,
          last_heartbeat: p.last_heartbeat,
        });
        gEdges.push({
          source: data.local_node.node_id,
          target: p.node_id,
          active: p.trust_status === 'TRUSTED' && p.connection_status === 'CONNECTED',
        });
      });
      nodesGraph.setData(gNodes, gEdges, 'nodes');

      document.getElementById('dist-orgs-list').innerHTML = data.federation_organizations.length
        ? data.federation_organizations.map((o) => `
            <div class="insp-card">
              <b style="color:var(--cyan)">${esc(o.org_name)}</b> (${esc(o.org_type)})<br>
              <span style="font-family:var(--font-mono);font-size:10px;color:var(--txt-dim)">${esc(o.endpoint)} · ${esc(o.data_sovereignty_policy)}</span>
            </div>
          `).join('')
        : '<div class="insp-card">No federation organizations registered yet.</div>';
    } catch { setOnlineState(false); }
  }

  // ─── Live Event Stream (SSE + Filter + Pause/Resume) ──────────────────────
  let streamPaused = false;
  let cachedEvents = [];
  function renderStream() {
    const sub = (document.getElementById('stream-sub-filter')?.value || '').toLowerCase();
    const q = (document.getElementById('stream-search')?.value || '').trim().toLowerCase();
    const filtered = cachedEvents.filter((e) =>
      (!sub || (e.subsystem || '').toLowerCase() === sub) &&
      (!q || `${e.event_type} ${e.description} ${e.subsystem}`.toLowerCase().includes(q))
    );
    document.getElementById('stream-count').textContent = `${filtered.length} events`;
    document.getElementById('stream-feed').innerHTML = filtered.slice(0, 50).map((e) => `
      <div class="stream-row sub-${esc(e.subsystem)}">
        <span style="color:var(--txt-dim)">${esc((e.timestamp || '').slice(11, 19))}</span>
        <b style="color:var(--cyan)">[${esc(e.subsystem.toUpperCase())}]</b>
        <span style="color:var(--orange)">${esc(e.event_type)}</span>
        <span style="color:#e2e8f0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(e.description)}</span>
      </div>
    `).join('');
  }

  async function loadEventsSnapshot() {
    if (streamPaused) return;
    try {
      const data = await fetchJson('/api/v2/observatory/events?limit=40');
      cachedEvents = data.events || [];
      renderStream();
    } catch { /* ignore */ }
  }

  function connectSSE() {
    if (typeof EventSource === 'undefined') return;
    const es = new EventSource(`${API}/api/v2/observatory/stream`);
    es.addEventListener('observatory_update', (ev) => {
      if (streamPaused) return;
      try {
        const parsed = JSON.parse(ev.data);
        if (Array.isArray(parsed.events)) {
          cachedEvents = parsed.events;
          renderStream();
          setOnlineState(true);
        }
      } catch { /* ignore */ }
    });
    es.onerror = () => {
      es.close();
      setTimeout(connectSSE, 8000);
    };
  }

  // ─── Control Event Bindings ───────────────────────────────────────────────
  document.getElementById('brain-zoom-in')?.addEventListener('click', () => { brainGraph.zoom = Math.min(2.8, brainGraph.zoom * 1.2); });
  document.getElementById('brain-zoom-out')?.addEventListener('click', () => { brainGraph.zoom = Math.max(0.4, brainGraph.zoom * 0.82); });
  document.getElementById('brain-fit')?.addEventListener('click', () => brainGraph.fitView());
  document.getElementById('brain-reset')?.addEventListener('click', () => brainGraph.resetView());
  document.getElementById('brain-refresh')?.addEventListener('click', loadOverviewAndBrain);
  document.getElementById('brain-group-filter')?.addEventListener('change', (e) => { brainGraph.filterGroup = e.target.value; });
  document.getElementById('brain-search')?.addEventListener('input', (e) => { brainGraph.searchTerm = e.target.value.trim().toLowerCase(); });

  document.getElementById('kg-zoom-in')?.addEventListener('click', () => { kgGraph.zoom = Math.min(2.8, kgGraph.zoom * 1.2); });
  document.getElementById('kg-zoom-out')?.addEventListener('click', () => { kgGraph.zoom = Math.max(0.4, kgGraph.zoom * 0.82); });
  document.getElementById('kg-reset')?.addEventListener('click', () => kgGraph.resetView());
  document.getElementById('btn-kg-apply')?.addEventListener('click', () => { kgOffset = 0; loadKnowledgeGraph(); });
  document.getElementById('kg-prev')?.addEventListener('click', () => { kgOffset = Math.max(0, kgOffset - kgLimit); loadKnowledgeGraph(); });
  document.getElementById('kg-next')?.addEventListener('click', () => { kgOffset += kgLimit; loadKnowledgeGraph(); });

  document.getElementById('btn-mem-refresh')?.addEventListener('click', loadMemory);
  document.getElementById('mem-search')?.addEventListener('input', loadMemory);
  document.getElementById('btn-cog-refresh')?.addEventListener('click', loadCognition);
  document.getElementById('btn-dist-refresh')?.addEventListener('click', loadNodes);

  document.getElementById('btn-cog-run')?.addEventListener('click', async () => {
    const inp = document.getElementById('cog-live-query');
    const q = (inp?.value || '').trim();
    if (!q) return;
    await fetch(`${API}/api/v2/intelligence/ask`, {
      method: 'POST',
      headers: authHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({ objective: q }),
    });
    inp.value = '';
    await loadCognition();
    await loadEventsSnapshot();
  });

  document.getElementById('stream-sub-filter')?.addEventListener('change', renderStream);
  document.getElementById('stream-search')?.addEventListener('input', renderStream);
  document.getElementById('btn-stream-pause')?.addEventListener('click', (e) => {
    streamPaused = !streamPaused;
    e.target.textContent = streamPaused ? '▶ RESUME' : '⏸ PAUSE';
  });
  document.getElementById('btn-stream-toggle')?.addEventListener('click', (e) => {
    const bar = document.getElementById('obs-stream-bar');
    bar.classList.toggle('collapsed');
    e.target.textContent = bar.classList.contains('collapsed') ? '▲ EXPAND' : '▼ MINIMIZE';
  });

  function refreshAllActive() {
    loadOverviewAndBrain();
    if (activeTab === 'cognitive-activity') loadCognition();
    else if (activeTab === 'knowledge-universe') loadKnowledgeGraph();
    else if (activeTab === 'memory-explorer') loadMemory();
    else if (activeTab === 'learning-observatory') loadLearning();
    else if (activeTab === 'agent-network') loadAgents();
    else if (activeTab === 'distributed-nodes') loadNodes();
    loadEventsSnapshot();
  }

  refreshAllActive();
  connectSSE();
});
