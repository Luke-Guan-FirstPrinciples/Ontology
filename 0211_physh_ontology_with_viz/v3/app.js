/**
 * PhySH Ontology Graph – v3 visualization
 * Uses 3d-force-graph with the polished dark theme.
 */

/* ─── Constants ─── */

const FACET_COLORS = {
    "Research Areas": "#f472b6",
    "Physical Systems": "#60a5fa",
    "Properties": "#a78bfa",
    "Techniques": "#34d399",
    "Unknown": "#94a3b8",
};

const RELATION_COLORS = {
    "STUDIES": "#f472b6",
    "EXHIBITS": "#60a5fa",
    "MEASURES": "#34d399",
    "PROBES": "#fbbf24",
};

const RELATION_LABELS = {
    "STUDIES": "Studies",
    "EXHIBITS": "Exhibits",
    "MEASURES": "Measures",
    "PROBES": "Probes",
};

/* ─── State ─── */

const STATE = {
    nodeMap: new Map(),
    edgeIndex: new Map(),       // nodeId -> [{edge, direction}]
    filters: {
        facet: "all",
        discipline: "all",
        minSupport: 10,
        search: "",
        relations: new Set(["STUDIES", "EXHIBITS", "MEASURES", "PROBES"]),
    },
    selectedId: null,
    showLabels: false,
};

let Graph;
let isRotating = true;

/* ─── Init ─── */

window.addEventListener("load", init);

function init() {
    try {
        if (!window.NODES || !window.EDGES || !window.GRAPH_STATS) {
            throw new Error("Data scripts not loaded.");
        }

        processData();
        populateFilters();
        initGraph();
        setupEventListeners();
        updateTotalStats();

        // Hide loading
        setTimeout(() => {
            const el = document.getElementById("loading");
            if (el) {
                el.style.opacity = "0";
                setTimeout(() => el.classList.add("hidden"), 500);
            }
        }, 300);

    } catch (e) {
        console.error("Init error:", e);
        const el = document.getElementById("loading");
        if (el) el.innerHTML = `<p style="color:#ef4444;">Error: ${e.message}</p>`;
    }
}

/* ─── Data Processing ─── */

function processData() {
    // Build node map
    window.NODES.forEach(n => {
        n.degree = 0;
        n.maxSupport = 0;
        STATE.nodeMap.set(n.id, n);
        STATE.edgeIndex.set(n.id, []);
    });

    // Build edge index & compute degrees
    window.EDGES.forEach(e => {
        const srcNode = STATE.nodeMap.get(e.s);
        const tgtNode = STATE.nodeMap.get(e.t);
        if (srcNode) {
            srcNode.degree++;
            srcNode.maxSupport = Math.max(srcNode.maxSupport, e.sup);
        }
        if (tgtNode) {
            tgtNode.degree++;
            tgtNode.maxSupport = Math.max(tgtNode.maxSupport, e.sup);
        }

        // Index for both directions
        const sIdx = STATE.edgeIndex.get(e.s);
        if (sIdx) sIdx.push({ edge: e, direction: "outgoing" });

        const tIdx = STATE.edgeIndex.get(e.t);
        if (tIdx) tIdx.push({ edge: e, direction: "incoming" });
    });
}

/* ─── Graph Initialization ─── */

function initGraph() {
    const elem = document.getElementById("graph");
    elem.innerHTML = "";

    Graph = ForceGraph3D()(elem)
        .nodeLabel(node => `<div style="background:rgba(15,23,42,0.92);padding:6px 10px;border-radius:6px;border:1px solid #334155;font-family:Inter,sans-serif;font-size:12px;color:#f1f5f9;max-width:200px"><b>${node.l}</b><br><span style="color:#94a3b8">${node.fl[0] || 'Unknown'}</span></div>`)
        .nodeColor(node => {
            if (STATE.selectedId === node.id) return "#ffffff";
            return FACET_COLORS[node.fl[0]] || FACET_COLORS["Unknown"];
        })
        .nodeVal(node => {
            // Scale node size by degree (log scale)
            return Math.max(1, Math.log2(node.degree + 1) * 1.5);
        })
        .nodeOpacity(0.92)
        .linkColor(link => {
            const color = RELATION_COLORS[link.r] || "#475569";
            return color + "88"; // Semi-transparent
        })
        .linkWidth(link => Math.max(0.3, Math.min(link.sup / 100, 3)))
        .linkOpacity(0.5)
        .linkDirectionalArrowLength(3)
        .linkDirectionalArrowRelPos(1)
        .linkDirectionalArrowColor(link => RELATION_COLORS[link.r] || "#475569")
        .backgroundColor("#0f172a")
        .onNodeClick(node => {
            focusOnNode(node);
            showDetails(node);
        })
        .onBackgroundClick(() => clearDetails())
        .d3AlphaDecay(0.02)
        .d3VelocityDecay(0.3)
        .warmupTicks(80)
        .cooldownTicks(100);

    // Node labels on top (only if toggled)
    Graph.nodeThreeObject(node => {
        if (!STATE.showLabels) return undefined;

        const sprite = new SpriteText(node.l);
        sprite.color = "#e2e8f0";
        sprite.textHeight = 3;
        sprite.fontFamily = "Inter, sans-serif";
        sprite.backgroundColor = "rgba(15, 23, 42, 0.6)";
        sprite.padding = [1, 2];
        sprite.borderRadius = 2;
        return sprite;
    });
    Graph.nodeThreeObjectExtend(true);

    updateGraphData();

    // Auto rotate
    Graph.controls().autoRotate = true;
    Graph.controls().autoRotateSpeed = 0.4;
}

/* ─── Filtering ─── */

function getFilteredData() {
    const { facet, discipline, minSupport, search, relations } = STATE.filters;
    const searchLower = search.toLowerCase();

    // Filter edges first
    const filteredEdges = window.EDGES.filter(e => {
        if (e.sup < minSupport) return false;
        if (!relations.has(e.r)) return false;
        return true;
    });

    // Collect node IDs from visible edges
    const edgeNodeIds = new Set();
    filteredEdges.forEach(e => {
        edgeNodeIds.add(e.s);
        edgeNodeIds.add(e.t);
    });

    // Filter nodes
    const filteredNodes = window.NODES.filter(n => {
        if (!edgeNodeIds.has(n.id)) return false;
        if (facet !== "all" && !n.fl.includes(facet)) return false;
        if (discipline !== "all" && !(n.dl || []).includes(discipline)) return false;
        if (search && !n.l.toLowerCase().includes(searchLower)) return false;
        return true;
    });

    const visibleIds = new Set(filteredNodes.map(n => n.id));

    // Only keep edges where both nodes are visible
    const visibleEdges = filteredEdges.filter(e =>
        visibleIds.has(e.s) && visibleIds.has(e.t)
    );

    // Convert to graph format
    const links = visibleEdges.map(e => ({
        source: e.s,
        target: e.t,
        r: e.r,
        sup: e.sup,
        w: e.w,
    }));

    return { nodes: filteredNodes, links };
}

function updateGraphData() {
    const data = getFilteredData();
    Graph.graphData(data);

    // Update visible counts
    setText("visibleNodes", data.nodes.length.toLocaleString());
    setText("visibleEdges", data.links.length.toLocaleString());
}

/* ─── Node Interaction ─── */

function focusOnNode(node) {
    STATE.selectedId = node.id;

    const dist = 80;
    const distRatio = 1 + dist / Math.hypot(node.x, node.y, node.z);

    Graph.cameraPosition(
        { x: node.x * distRatio, y: node.y * distRatio, z: node.z * distRatio },
        node,
        2000
    );

    // Refresh colors to show selection
    Graph.nodeColor(Graph.nodeColor());
}

function showDetails(node) {
    const panel = document.getElementById("detailsContent");
    const empty = document.getElementById("emptyState");

    if (empty) empty.classList.add("hidden");
    if (panel) panel.classList.remove("hidden");

    setText("dLabel", node.l);
    setText("dFacet", node.fl.join(", ") || "Unknown");
    setText("dDegree", node.degree);
    setText("dMaxSupport", node.maxSupport);

    // Facets
    const facetsEl = document.getElementById("dFacets");
    if (facetsEl) {
        facetsEl.innerHTML = (node.fl || []).map(f =>
            `<span class="tag" style="border-left:3px solid ${FACET_COLORS[f] || '#94a3b8'}">${f}</span>`
        ).join("");
    }

    // Disciplines
    const discEl = document.getElementById("dDisciplines");
    if (discEl) {
        const dl = node.dl || [];
        discEl.innerHTML = dl.length > 0
            ? dl.map(d => `<span class="tag">${d}</span>`).join("")
            : '<span style="color:var(--text-muted);font-size:0.85rem">No discipline data</span>';
    }

    // Connections
    const connEl = document.getElementById("dConnections");
    const connTitle = document.getElementById("connTitle");
    if (connEl) {
        const connections = STATE.edgeIndex.get(node.id) || [];

        // Filter by current filter settings
        const { minSupport, relations } = STATE.filters;
        const visible = connections
            .filter(c => c.edge.sup >= minSupport && relations.has(c.edge.r))
            .sort((a, b) => b.edge.sup - a.edge.sup);

        if (connTitle) connTitle.textContent = `Connections (${visible.length})`;

        const SHOW_LIMIT = 20;
        const toShow = visible.slice(0, SHOW_LIMIT);

        connEl.innerHTML = toShow.map(c => {
            const e = c.edge;
            const otherId = c.direction === "outgoing" ? e.t : e.s;
            const otherLabel = c.direction === "outgoing" ? e.tl : e.sl;
            const arrow = c.direction === "outgoing" ? "→" : "←";
            const relColor = RELATION_COLORS[e.r] || "#94a3b8";

            return `
                <div class="conn-item" onclick="navigateToNode('${otherId}')">
                    <div class="conn-rel" style="color:${relColor}">${arrow} ${RELATION_LABELS[e.r] || e.r}</div>
                    <div class="conn-label">${otherLabel}</div>
                    <div class="conn-support">Support: ${e.sup} · Weight: ${e.w.toFixed(2)}</div>
                </div>`;
        }).join("");

        if (visible.length > SHOW_LIMIT) {
            connEl.innerHTML += `<button class="show-more-btn" onclick="showAllConnections('${node.id}')">Show all ${visible.length} connections</button>`;
        }
    }
}

function clearDetails() {
    STATE.selectedId = null;
    const panel = document.getElementById("detailsContent");
    const empty = document.getElementById("emptyState");
    if (panel) panel.classList.add("hidden");
    if (empty) empty.classList.remove("hidden");
    Graph.nodeColor(Graph.nodeColor());
}

function navigateToNode(nodeId) {
    const nodeData = Graph.graphData().nodes.find(n => n.id === nodeId);
    if (nodeData) {
        focusOnNode(nodeData);
        showDetails(nodeData);
    }
}

function showAllConnections(nodeId) {
    const node = STATE.nodeMap.get(nodeId);
    if (!node) return;

    const connEl = document.getElementById("dConnections");
    const connections = STATE.edgeIndex.get(nodeId) || [];
    const { minSupport, relations } = STATE.filters;
    const visible = connections
        .filter(c => c.edge.sup >= minSupport && relations.has(c.edge.r))
        .sort((a, b) => b.edge.sup - a.edge.sup);

    connEl.innerHTML = visible.map(c => {
        const e = c.edge;
        const otherId = c.direction === "outgoing" ? e.t : e.s;
        const otherLabel = c.direction === "outgoing" ? e.tl : e.sl;
        const arrow = c.direction === "outgoing" ? "→" : "←";
        const relColor = RELATION_COLORS[e.r] || "#94a3b8";

        return `
            <div class="conn-item" onclick="navigateToNode('${otherId}')">
                <div class="conn-rel" style="color:${relColor}">${arrow} ${RELATION_LABELS[e.r] || e.r}</div>
                <div class="conn-label">${otherLabel}</div>
                <div class="conn-support">Support: ${e.sup} · Weight: ${e.w.toFixed(2)}</div>
            </div>`;
    }).join("");
}

/* ─── Filters UI ─── */

function populateFilters() {
    // Facets
    const facets = new Set();
    const disciplines = new Set();
    window.NODES.forEach(n => {
        (n.fl || []).forEach(f => facets.add(f));
        (n.dl || []).forEach(d => disciplines.add(d));
    });

    const fSelect = document.getElementById("facetFilter");
    if (fSelect) {
        Array.from(facets).sort().forEach(f => {
            const opt = document.createElement("option");
            opt.value = f;
            opt.textContent = f;
            fSelect.appendChild(opt);
        });
    }

    // Discipline dropdown
    const dSelect = document.getElementById("disciplineFilter");
    if (dSelect) {
        Array.from(disciplines).sort().forEach(d => {
            const opt = document.createElement("option");
            opt.value = d;
            opt.textContent = d;
            dSelect.appendChild(opt);
        });
    }

    // Relation checkboxes
    const relContainer = document.getElementById("relationFilters");
    if (relContainer) {
        const relCounts = window.GRAPH_STATS?.relations || {};
        const allRelations = Object.keys(RELATION_COLORS);

        allRelations.forEach(rel => {
            const count = relCounts[rel] || 0;
            const color = RELATION_COLORS[rel];
            const label = RELATION_LABELS[rel] || rel;

            const wrapper = document.createElement("label");
            wrapper.className = "relation-check";

            wrapper.innerHTML = `
                <input type="checkbox" value="${rel}" checked>
                <span class="rel-dot" style="background:${color}"></span>
                <span>${label}</span>
                <span class="rel-count">${count.toLocaleString()}</span>
            `;

            relContainer.appendChild(wrapper);
        });
    }
}

function setupEventListeners() {
    // Search (debounced)
    let debounce;
    const searchInput = document.getElementById("searchInput");
    if (searchInput) {
        searchInput.addEventListener("input", (e) => {
            STATE.filters.search = e.target.value;
            if (debounce) clearTimeout(debounce);
            debounce = setTimeout(updateGraphData, 400);
        });
    }

    // Facet filter
    const facetFilter = document.getElementById("facetFilter");
    if (facetFilter) {
        facetFilter.addEventListener("change", (e) => {
            STATE.filters.facet = e.target.value;
            updateGraphData();
        });
    }

    // Discipline filter
    const discFilter = document.getElementById("disciplineFilter");
    if (discFilter) {
        discFilter.addEventListener("change", (e) => {
            STATE.filters.discipline = e.target.value;
            updateGraphData();
        });
    }

    // Support slider
    const supportSlider = document.getElementById("supportFilter");
    if (supportSlider) {
        supportSlider.addEventListener("input", (e) => {
            STATE.filters.minSupport = parseInt(e.target.value);
            setText("supportValue", STATE.filters.minSupport);
        });
        // Update graph on release (for performance)
        supportSlider.addEventListener("change", () => updateGraphData());
    }

    // Relation checkboxes
    const relContainer = document.getElementById("relationFilters");
    if (relContainer) {
        relContainer.addEventListener("change", (e) => {
            if (e.target.type === "checkbox") {
                const rel = e.target.value;
                if (e.target.checked) {
                    STATE.filters.relations.add(rel);
                } else {
                    STATE.filters.relations.delete(rel);
                }
                updateGraphData();
            }
        });
    }
}

/* ─── Stats ─── */

function updateTotalStats() {
    setText("totalNodes", (window.GRAPH_STATS?.totalNodes || 0).toLocaleString());
    setText("totalEdges", (window.GRAPH_STATS?.totalEdges || 0).toLocaleString());
}

/* ─── Global Button Handlers ─── */

window.resetCamera = () => {
    if (Graph) Graph.cameraPosition({ x: 0, y: 0, z: 400 }, { x: 0, y: 0, z: 0 }, 2000);
};

window.toggleRotation = () => {
    isRotating = !isRotating;
    if (Graph) Graph.controls().autoRotate = isRotating;
};

window.toggleLabels = () => {
    STATE.showLabels = !STATE.showLabels;
    if (Graph) {
        // Re-initialize nodeThreeObject to toggle labels
        Graph.nodeThreeObject(node => {
            if (!STATE.showLabels) return undefined;

            // Check if SpriteText is available (from three-spritetext)
            if (typeof SpriteText !== "undefined") {
                const sprite = new SpriteText(node.l);
                sprite.color = "#e2e8f0";
                sprite.textHeight = 3;
                sprite.fontFamily = "Inter, sans-serif";
                sprite.backgroundColor = "rgba(15, 23, 42, 0.6)";
                sprite.padding = [1, 2];
                sprite.borderRadius = 2;
                return sprite;
            }
            return undefined;
        });
        Graph.nodeThreeObjectExtend(true);
    }
};

window.navigateToNode = navigateToNode;
window.showAllConnections = showAllConnections;

/* ─── Helpers ─── */

function setText(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
}
