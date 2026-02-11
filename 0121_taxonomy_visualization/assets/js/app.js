/**
 * PhySH Atlas - Tree View Application
 */

const STATE = {
    concepts: [],
    stats: {},
    papers: [],
    conceptMap: new Map(),
    roots: [],
    filters: {
        facet: 'all',
        discipline: 'all',
        maxDepth: 10,
        search: ''
    },
    expanded: new Set(),
    selectedId: null
};

// Facet Mapping for Colors/Labels
const FACET_MAP = {
    "Research Areas": "research",
    "Physical Systems": "systems",
    "Properties": "properties",
    "Techniques": "techniques",
    "Professional Topics": "topics"
};

// Helper to load script dynamically
const loadScript = (src) => {
    return new Promise((resolve, reject) => {
        const script = document.createElement('script');
        script.src = src;
        script.onload = resolve;
        script.onerror = reject;
        document.body.appendChild(script);
    });
};

async function init() {
    try {
        await loadData();
        buildTree();
        populateFilters();
        renderTree();
        setupEventListeners();
        updateStats();
    } catch (e) {
        console.error("Initialization Failed:", e);
        document.querySelector('.loading-state').innerHTML = `<p class="error">Failed to load data. <br> ${e.message}</p>`;
    }
}

async function loadData() {
    // Check if window.CONCEPTS exists (loaded via script tag in HTML)
    // If not, try to load dynamically here for robustness
    if (!window.CONCEPTS || !window.STATS) {
        console.log("Data not present globally, attempting dynamic load...");
        try {
            await Promise.all([
                loadScript('data/concepts.js'),
                loadScript('data/stats.js')
            ]);
        } catch (e) {
            throw new Error("Failed to load concepts.js or stats.js");
        }
    }

    // Double check
    if (!window.CONCEPTS || !window.STATS) {
        throw new Error("Taxonomy data not found. Ensure concepts.js and stats.js are valid.");
    }

    const concepts = window.CONCEPTS;
    const stats = window.STATS;

    STATE.concepts = concepts;

    // Process stats into a map
    stats.forEach(s => {
        STATE.stats[s.concept_id] = s.aps_frequency;
    });

    // Build Concept Map
    concepts.forEach(c => {
        // Hydrate with stats
        c.freq = STATE.stats[c.id] || 0;
        c.children = [];
        STATE.conceptMap.set(c.id, c);
    });

    console.log(`Loaded ${concepts.length} concepts.`);

    // Load Papers in Background
    loadPapers();
}

async function loadPapers() {
    try {
        await loadScript('data/papers.js');
        if (window.PAPERS) {
            STATE.papers = window.PAPERS;
            console.log(`Loaded ${STATE.papers.length} papers in background.`);

            // Only update this if the element exists
            const totalEl = document.getElementById('totalPapers');
            if (totalEl) totalEl.textContent = STATE.papers.length.toLocaleString();

            // If a node is already selected, refresh details
            if (STATE.selectedId) {
                const node = STATE.conceptMap.get(STATE.selectedId);
                if (node) selectNode(node);
            }
        }
    } catch (e) {
        console.warn("Background paper load failed", e);
    }
}

function buildTree() {
    STATE.roots = [];
    STATE.concepts.forEach(c => {
        if (c.p && STATE.conceptMap.has(c.p)) {
            STATE.conceptMap.get(c.p).children.push(c);
        } else {
            STATE.roots.push(c);
        }
    });

    // Sort by name
    const sorter = (a, b) => a.l.localeCompare(b.l);
    STATE.concepts.forEach(c => c.children.sort(sorter));
    STATE.roots.sort(sorter);
}

function populateFilters() {
    // Collect unique facets and disciplines
    const facets = new Set();
    const disciplines = new Set();

    STATE.concepts.forEach(c => {
        c.fl.forEach(f => facets.add(f));
        c.dl.forEach(d => disciplines.add(d));
    });

    const fSelect = document.getElementById('facetFilter');
    Array.from(facets).sort().forEach(f => {
        const opt = document.createElement('option');
        opt.value = f;
        opt.textContent = f;
        fSelect.appendChild(opt);
    });

    const dSelect = document.getElementById('disciplineFilter');
    Array.from(disciplines).sort().forEach(d => {
        const opt = document.createElement('option');
        opt.value = d;
        opt.textContent = d;
        dSelect.appendChild(opt);
    });
}

function matchesFilters(node) {
    const { facet, discipline, maxDepth, search } = STATE.filters;

    // Depth Check
    if (node.d > maxDepth && maxDepth < 10) return false;

    // Facet Check
    if (facet !== 'all' && !node.fl.includes(facet)) return false;

    // Discipline Check
    if (discipline !== 'all' && !node.dl.includes(discipline)) return false;

    // Search Check
    if (search) {
        const term = search.toLowerCase();
        const selfMatch = node.l.toLowerCase().includes(term);
        // If children match, this node matches (to show path)
        const childMatch = node.children.some(child => matchesRecursive(child));
        return selfMatch || childMatch;
    }

    return true;
}

// Optimization: Cache search results to avoid re-walking for every node
function matchesRecursive(node) {
    // Simplified for search logic path
    const term = STATE.filters.search.toLowerCase();
    if (node.l.toLowerCase().includes(term)) return true;
    return node.children.some(child => matchesRecursive(child));
}


function renderTree() {
    const container = document.getElementById('treeWrapper');
    container.innerHTML = '';

    // Sort roots by frequency if desired, or alphabetical
    // For now alphabetical
    STATE.roots.forEach(root => {
        if (shouldShow(root)) {
            container.appendChild(createNodeEl(root));
        }
    });

    if (container.children.length === 0) {
        container.innerHTML = '<div class="loading-state">No matching concepts found.</div>';
    }
}

function shouldShow(node) {
    // If searching, show only if part of the match path
    if (STATE.filters.search) return matchesRecursive(node);
    return matchesFilters(node);
}

function createNodeEl(node) {
    /* 
       Structure:
       <div class="tree-node">
         <div class="node-content">...</div>
         <div class="children-container">...</div>
       </div>
    */
    const wrapper = document.createElement('div');
    wrapper.className = 'tree-node';

    const content = document.createElement('div');
    content.className = `node-content ${STATE.selectedId === node.id ? 'selected' : ''}`;
    content.dataset.id = node.id;
    content.onclick = (e) => {
        e.stopPropagation();
        selectNode(node);
    };

    // Toggle Icon
    const hasChildren = node.children.length > 0;
    const isExpanded = STATE.expanded.has(node.id) || !!STATE.filters.search; // Always expand on search

    if (hasChildren) wrapper.classList.add('has-children');
    if (isExpanded) wrapper.classList.add('expanded');

    const toggle = document.createElement('div');
    toggle.className = 'toggle-icon';
    toggle.textContent = hasChildren ? '▶' : '•';
    toggle.onclick = (e) => {
        e.stopPropagation();
        toggleNode(node.id, wrapper);
    };
    content.appendChild(toggle);

    // Facet Dot
    const facet = node.fl[0]; // Primary facet
    const facetClass = FACET_MAP[facet] ? `facet-${FACET_MAP[facet]}` : 'facet-topics';
    const dot = document.createElement('span');
    dot.className = `facet-dot ${facetClass}`;
    content.appendChild(dot);

    // Label
    const label = document.createElement('span');
    label.className = 'node-label';
    label.textContent = node.l;
    content.appendChild(label);

    // Meta (Freq)
    if (node.freq > 0) {
        const meta = document.createElement('span');
        meta.className = 'node-meta';
        meta.textContent = node.freq.toLocaleString();
        content.appendChild(meta);
    }

    wrapper.appendChild(content);

    // Children
    if (hasChildren && isExpanded) {
        const childrenContainer = document.createElement('div');
        childrenContainer.className = 'children-container';
        node.children.forEach(child => {
            if (shouldShow(child)) {
                childrenContainer.appendChild(createNodeEl(child));
            }
        });
        wrapper.appendChild(childrenContainer);
    }

    return wrapper;
}

function toggleNode(id, el) {
    if (STATE.expanded.has(id)) {
        STATE.expanded.delete(id);
        el.classList.remove('expanded');
        // Remove children container
        const children = el.querySelector('.children-container');
        if (children) children.remove();
    } else {
        STATE.expanded.add(id);
        el.classList.add('expanded');
        const node = STATE.conceptMap.get(id);
        const childrenContainer = document.createElement('div');
        childrenContainer.className = 'children-container';
        node.children.forEach(child => {
            if (shouldShow(child)) {
                childrenContainer.appendChild(createNodeEl(child));
            }
        });
        el.appendChild(childrenContainer);
    }
}

function selectNode(node) {
    STATE.selectedId = node.id;

    // Update UI highlighs
    document.querySelectorAll('.node-content').forEach(el => el.classList.remove('selected'));
    const activeEl = document.querySelector(`.node-content[data-id="${node.id}"]`);
    if (activeEl) activeEl.classList.add('selected');

    // Update Details Panel
    const panel = document.getElementById('detailsContent');
    document.querySelector('.details-empty-state').classList.add('hidden');
    panel.classList.remove('hidden');

    document.getElementById('dLabel').textContent = node.l;
    document.getElementById('dId').textContent = node.id;
    document.getElementById('dDef').textContent = node.def || "No definition available.";

    document.getElementById('dApsFreq').textContent = node.freq.toLocaleString();
    document.getElementById('dDepth').textContent = node.d;

    // Classifications
    const facetsContainer = document.getElementById('dFacets');
    facetsContainer.innerHTML = node.fl.map(f => `<span class="tag">${f}</span>`).join('');

    const discContainer = document.getElementById('dDisciplines');
    discContainer.innerHTML = node.dl.map(d => `<span class="tag">${d}</span>`).join('');

    // Linked Papers
    const papersContainer = document.getElementById('dPapers');
    papersContainer.innerHTML = '';

    // Find papers with this concept
    const linkedPapers = STATE.papers.filter(p => p.c.includes(node.id));

    if (linkedPapers.length === 0) {
        papersContainer.innerHTML = '<p class="text-muted">No papers linked to this concept.</p>';
    } else {
        linkedPapers.slice(0, 50).forEach(p => { // Limit to 50
            const pEl = document.createElement('div');
            pEl.className = 'paper-item';
            pEl.innerHTML = `
                <a href="https://doi.org/${p.doi}" target="_blank" class="paper-title">${p.title || p.doi}</a>
                <div class="paper-meta">${p.year || 'Unknown Year'} • ${p.doi}</div>
            `;
            papersContainer.appendChild(pEl);
        });
    }
}

function setupEventListeners() {
    let debounce;
    document.getElementById('searchInput').addEventListener('input', (e) => {
        STATE.filters.search = e.target.value;
        if (debounce) clearTimeout(debounce);
        debounce = setTimeout(renderTree, 300);
    });

    document.getElementById('facetFilter').addEventListener('change', (e) => {
        STATE.filters.facet = e.target.value;
        renderTree();
    });

    document.getElementById('disciplineFilter').addEventListener('change', (e) => {
        STATE.filters.discipline = e.target.value;
        renderTree();
    });

    const depthIn = document.getElementById('depthFilter');
    depthIn.addEventListener('input', (e) => {
        STATE.filters.maxDepth = parseInt(e.target.value);
        document.getElementById('depthValue').textContent = STATE.filters.maxDepth === 10 ? 'All' : STATE.filters.maxDepth;
        renderTree();
    });

    document.getElementById('expandAllBtn').addEventListener('click', () => {
        // Only expand visible nodes to avoid lag
        STATE.expanded = new Set(STATE.concepts.map(c => c.id));
        renderTree();
    });

    document.getElementById('collapseAllBtn').addEventListener('click', () => {
        STATE.expanded.clear();
        renderTree();
    });
}

function updateStats() {
    document.getElementById('totalConcepts').textContent = STATE.concepts.length.toLocaleString();
    if (document.getElementById('totalPapers')) {
        document.getElementById('totalPapers').textContent = STATE.papers.length.toLocaleString();
    }
}

// Start
window.addEventListener('load', init);
