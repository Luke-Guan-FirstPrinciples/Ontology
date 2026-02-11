/**
 * PhySH Network Graph Application
 */

const STATE = {
    concepts: [],
    stats: {},
    papers: [],
    conceptMap: new Map(),
    filters: {
        facet: 'all',
        discipline: 'all',
        maxDepth: 10,
        search: ''
    },
    selectedId: null
};

const FACET_COLORS = {
    "Research Areas": "#f472b6",
    "Physical Systems": "#60a5fa",
    "Properties": "#a78bfa",
    "Techniques": "#34d399",
    "Professional Topics": "#fbbf24"
};

let Graph;
let isRotating = true;

async function init() {
    try {
        await ensureDataLoaded();
        processData();
        populateFilters();
        initGraph();
        setupEventListeners();
        updateStats();

        // Hide loading
        const loadingEl = document.getElementById('loading');
        if (loadingEl) loadingEl.classList.add('hidden');
    } catch (e) {
        console.error(e);
        alert("Error: " + e.message);
    }
}

async function ensureDataLoaded() {
    if (!window.CONCEPTS || !window.STATS || !window.PAPERS) {
        throw new Error("Data scripts not loaded properly.");
    }
}

function processData() {
    STATE.concepts = window.CONCEPTS;
    STATE.papers = window.PAPERS;

    // Process Stats
    window.STATS.forEach(s => {
        STATE.stats[s.concept_id] = s.aps_frequency;
    });

    // Map hydration
    STATE.concepts.forEach(c => {
        c.freq = STATE.stats[c.id] || 0;
        STATE.conceptMap.set(c.id, c);
    });
}

function initGraph() {
    const elem = document.getElementById('graph');

    // Clear any existing content
    elem.innerHTML = '';

    Graph = ForceGraph3D()
        (elem)
        .nodeLabel('l')
        .nodeColor(node => FACET_COLORS[node.fl[0]] || '#94a3b8')
        .nodeRelSize(4)
        .linkWidth(1)
        .linkOpacity(0.3)
        .backgroundColor('#0f172a')
        .onNodeClick(node => {
            focusOnNode(node);
            showDetails(node);
        });

    updateGraphData();

    // Auto rotate
    Graph.controls().autoRotate = true;
    Graph.controls().autoRotateSpeed = 0.5;
}

function getFilteredData() {
    const { facet, discipline, maxDepth, search } = STATE.filters;
    const searchTerm = search.toLowerCase();

    // Filter plain list first
    const visibleNodes = STATE.concepts.filter(node => {
        if (node.d > maxDepth && maxDepth < 10) return false;
        if (facet !== 'all' && !node.fl.includes(facet)) return false;
        if (discipline !== 'all' && !node.dl.includes(discipline)) return false;
        if (search && !node.l.toLowerCase().includes(searchTerm)) return false;
        return true;
    });

    const visibleIds = new Set(visibleNodes.map(n => n.id));

    // Build Links (only if both ends are visible)
    const links = [];
    visibleNodes.forEach(node => {
        if (node.p && visibleIds.has(node.p)) {
            links.push({ source: node.p, target: node.id });
        }
    });

    return { nodes: visibleNodes, links };
}

function updateGraphData() {
    const data = getFilteredData();
    Graph.graphData(data);

    // Update stats count in sidebar
    const el = document.getElementById('totalConcepts');
    if (el) el.textContent = data.nodes.length.toLocaleString();
}

function focusOnNode(node) {
    const distance = 40;
    const distRatio = 1 + distance / Math.hypot(node.x, node.y, node.z);

    Graph.cameraPosition(
        { x: node.x * distRatio, y: node.y * distRatio, z: node.z * distRatio },
        node,
        3000
    );
}

function showDetails(node) {
    STATE.selectedId = node.id;

    const panel = document.getElementById('detailsContent');
    const emptyState = document.querySelector('.details-empty-state');

    if (emptyState) emptyState.classList.add('hidden');
    if (panel) panel.classList.remove('hidden');

    document.getElementById('dLabel').textContent = node.l;
    document.getElementById('dId').textContent = node.id;

    const defEl = document.getElementById('dDef');
    if (defEl) defEl.textContent = node.def || "No definition available.";

    document.getElementById('dApsFreq').textContent = node.freq.toLocaleString();
    document.getElementById('dDepth').textContent = node.d;

    const facetsContainer = document.getElementById('dFacets');
    if (facetsContainer) {
        facetsContainer.innerHTML = node.fl.map(f => `<span class="tag">${f}</span>`).join('');
    }

    const discContainer = document.getElementById('dDisciplines');
    if (discContainer) {
        discContainer.innerHTML = node.dl.map(d => `<span class="tag">${d}</span>`).join('');
    }

    // Papers
    const papersContainer = document.getElementById('dPapers');
    if (papersContainer) {
        papersContainer.innerHTML = '';
        const linkedPapers = STATE.papers.filter(p => p.c.includes(node.id));

        if (linkedPapers.length === 0) {
            papersContainer.innerHTML = '<p style="color:var(--text-muted)">No papers linked to this concept.</p>';
        } else {
            linkedPapers.slice(0, 50).forEach(p => {
                const pEl = document.createElement('div');
                pEl.className = 'paper-item';
                pEl.innerHTML = `
                    <a href="https://doi.org/${p.doi}" target="_blank" class="paper-title">${p.title || p.doi}</a>
                    <div class="paper-meta">${p.year || '?'} • ${p.doi}</div>
                `;
                papersContainer.appendChild(pEl);
            });
        }
    }
}

function populateFilters() {
    const facets = new Set();
    const disciplines = new Set();

    STATE.concepts.forEach(c => {
        c.fl.forEach(f => facets.add(f));
        c.dl.forEach(d => disciplines.add(d));
    });

    const fSelect = document.getElementById('facetFilter');
    if (fSelect) {
        // Clear existing options except first
        while (fSelect.options.length > 1) {
            fSelect.remove(1);
        }
        Array.from(facets).sort().forEach(f => {
            const opt = document.createElement('option');
            opt.value = f;
            opt.textContent = f;
            fSelect.appendChild(opt);
        });
    }

    const dSelect = document.getElementById('disciplineFilter');
    if (dSelect) {
        while (dSelect.options.length > 1) {
            dSelect.remove(1);
        }
        Array.from(disciplines).sort().forEach(d => {
            const opt = document.createElement('option');
            opt.value = d;
            opt.textContent = d;
            dSelect.appendChild(opt);
        });
    }
}

function setupEventListeners() {
    // Search
    let debounce;
    const searchIn = document.getElementById('searchInput');
    if (searchIn) {
        searchIn.addEventListener('input', (e) => {
            STATE.filters.search = e.target.value;
            if (debounce) clearTimeout(debounce);
            debounce = setTimeout(updateGraphData, 500);
        });
    }

    // Filters
    const facetFilter = document.getElementById('facetFilter');
    if (facetFilter) {
        facetFilter.addEventListener('change', (e) => {
            STATE.filters.facet = e.target.value;
            updateGraphData();
        });
    }

    const dipFilter = document.getElementById('disciplineFilter');
    if (dipFilter) {
        dipFilter.addEventListener('change', (e) => {
            STATE.filters.discipline = e.target.value;
            updateGraphData();
        });
    }

    // Depth
    const depthIn = document.getElementById('depthFilter');
    if (depthIn) {
        depthIn.addEventListener('input', (e) => {
            STATE.filters.maxDepth = parseInt(e.target.value);
            document.getElementById('depthValue').textContent = STATE.filters.maxDepth === 10 ? 'All' : STATE.filters.maxDepth;
            updateGraphData();
        });
    }
}

function updateStats() {
    const tc = document.getElementById('totalConcepts');
    if (tc) tc.textContent = STATE.concepts.length.toLocaleString();

    const tp = document.getElementById('totalPapers');
    if (tp) tp.textContent = STATE.papers.length.toLocaleString();
}

// Global functions for buttons
window.resetCamera = () => {
    if (Graph) Graph.cameraPosition({ x: 0, y: 0, z: 200 }, { x: 0, y: 0, z: 0 }, 2000);
};

window.toggleRotation = () => {
    isRotating = !isRotating;
    if (Graph) Graph.controls().autoRotate = isRotating;
};

// Start
window.addEventListener('load', init);
