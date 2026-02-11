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

    // Stop auto-rotation while focusing
    if (Graph.controls()) Graph.controls().autoRotate = false;
    isRotating = false;

    const dist = 120;
    // Camera looks at node from a position offset along the camera's current direction
    const camPos = Graph.cameraPosition();
    const dx = camPos.x - node.x;
    const dy = camPos.y - node.y;
    const dz = camPos.z - node.z;
    const currentDist = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1;

    // Position camera at fixed distance from node, along the current viewing direction
    const newPos = {
        x: node.x + (dx / currentDist) * dist,
        y: node.y + (dy / currentDist) * dist,
        z: node.z + (dz / currentDist) * dist,
    };

    Graph.cameraPosition(
        newPos,                         // camera position
        { x: node.x, y: node.y, z: node.z },  // lookAt: center exactly on the node
        1500
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

/* ─── Sidebar Toggle ─── */

window.toggleSidebar = () => {
    const sidebar = document.getElementById("sidebar");
    const showBtn = document.getElementById("showSidebar");
    if (!sidebar) return;
    sidebar.classList.toggle("collapsed");
    showBtn.classList.toggle("hidden", !sidebar.classList.contains("collapsed"));
    // Resize graph after transition
    setTimeout(() => { if (Graph) Graph.width(document.getElementById("graph")?.clientWidth); }, 350);
};

window.toggleDetails = () => {
    const panel = document.getElementById("detailsPanel");
    const showBtn = document.getElementById("showDetails");
    if (!panel) return;
    panel.classList.toggle("collapsed");
    showBtn.classList.toggle("hidden", !panel.classList.contains("collapsed"));
    setTimeout(() => { if (Graph) Graph.width(document.getElementById("graph")?.clientWidth); }, 350);
};

/* ─── View Switching ─── */

let currentView = "graph";

window.switchView = (view) => {
    currentView = view;
    const graphView = document.getElementById("graphView");
    const tableView = document.getElementById("tableView");
    const tabGraph = document.getElementById("tabGraph");
    const tabTable = document.getElementById("tabTable");

    if (view === "graph") {
        graphView.classList.remove("hidden");
        tableView.classList.add("hidden");
        tabGraph.classList.add("active");
        tabTable.classList.remove("active");
        // Refresh graph sizing
        setTimeout(() => { if (Graph) Graph.width(document.getElementById("graph")?.clientWidth); }, 100);
    } else {
        graphView.classList.add("hidden");
        tableView.classList.remove("hidden");
        tabGraph.classList.remove("active");
        tabTable.classList.add("active");
        renderTable();
    }
};

/* ─── Table View ─── */

const TABLE_STATE = {
    tab: "nodes",       // "nodes" | "edges"
    search: "",
    sortCol: null,
    sortDir: "asc",
    page: 0,
    pageSize: 100,
};

window.switchTableTab = (tab) => {
    TABLE_STATE.tab = tab;
    TABLE_STATE.page = 0;
    TABLE_STATE.sortCol = null;
    TABLE_STATE.sortDir = "asc";
    document.getElementById("tableTabNodes").classList.toggle("active", tab === "nodes");
    document.getElementById("tableTabEdges").classList.toggle("active", tab === "edges");
    renderTable();
};

function getTableData() {
    const { facet, discipline, minSupport, search: graphSearch, relations } = STATE.filters;
    const tableSearch = TABLE_STATE.search.toLowerCase();

    if (TABLE_STATE.tab === "nodes") {
        // Get filtered nodes (same logic as graph)
        const filteredEdges = window.EDGES.filter(e => e.sup >= minSupport && relations.has(e.r));
        const edgeNodeIds = new Set();
        filteredEdges.forEach(e => { edgeNodeIds.add(e.s); edgeNodeIds.add(e.t); });

        let nodes = window.NODES.filter(n => {
            if (!edgeNodeIds.has(n.id)) return false;
            if (facet !== "all" && !n.fl.includes(facet)) return false;
            if (discipline !== "all" && !(n.dl || []).includes(discipline)) return false;
            if (graphSearch && !n.l.toLowerCase().includes(graphSearch.toLowerCase())) return false;
            return true;
        });

        // Table-specific search
        if (tableSearch) {
            nodes = nodes.filter(n =>
                n.l.toLowerCase().includes(tableSearch) ||
                (n.fl || []).some(f => f.toLowerCase().includes(tableSearch)) ||
                (n.dl || []).some(d => d.toLowerCase().includes(tableSearch))
            );
        }

        // Sorting
        if (TABLE_STATE.sortCol !== null) {
            const col = TABLE_STATE.sortCol;
            const dir = TABLE_STATE.sortDir === "asc" ? 1 : -1;
            nodes.sort((a, b) => {
                let va, vb;
                switch (col) {
                    case "label": va = a.l; vb = b.l; return va.localeCompare(vb) * dir;
                    case "facet": va = (a.fl[0] || ""); vb = (b.fl[0] || ""); return va.localeCompare(vb) * dir;
                    case "degree": return (a.degree - b.degree) * dir;
                    case "maxSupport": return (a.maxSupport - b.maxSupport) * dir;
                    case "disciplines": va = (a.dl || []).length; vb = (b.dl || []).length; return (va - vb) * dir;
                    default: return 0;
                }
            });
        }

        return nodes;
    } else {
        // Edges tab
        let edges = window.EDGES.filter(e => {
            if (e.sup < minSupport) return false;
            if (!relations.has(e.r)) return false;
            return true;
        });

        // Facet/discipline filtering on source & target
        if (facet !== "all" || discipline !== "all" || graphSearch) {
            const visibleNodeIds = new Set(
                window.NODES.filter(n => {
                    if (facet !== "all" && !n.fl.includes(facet)) return false;
                    if (discipline !== "all" && !(n.dl || []).includes(discipline)) return false;
                    if (graphSearch && !n.l.toLowerCase().includes(graphSearch.toLowerCase())) return false;
                    return true;
                }).map(n => n.id)
            );
            edges = edges.filter(e => visibleNodeIds.has(e.s) && visibleNodeIds.has(e.t));
        }

        // Table-specific search
        if (tableSearch) {
            edges = edges.filter(e =>
                e.sl.toLowerCase().includes(tableSearch) ||
                e.tl.toLowerCase().includes(tableSearch) ||
                e.r.toLowerCase().includes(tableSearch)
            );
        }

        // Sorting
        if (TABLE_STATE.sortCol !== null) {
            const col = TABLE_STATE.sortCol;
            const dir = TABLE_STATE.sortDir === "asc" ? 1 : -1;
            edges.sort((a, b) => {
                switch (col) {
                    case "source": return a.sl.localeCompare(b.sl) * dir;
                    case "target": return a.tl.localeCompare(b.tl) * dir;
                    case "relation": return a.r.localeCompare(b.r) * dir;
                    case "support": return (a.sup - b.sup) * dir;
                    case "weight": return (a.w - b.w) * dir;
                    default: return 0;
                }
            });
        }

        return edges;
    }
}

function renderTable() {
    const data = getTableData();
    const wrapper = document.getElementById("tableWrapper");
    const countEl = document.getElementById("tableCount");
    const pagEl = document.getElementById("tablePagination");
    if (!wrapper) return;

    const total = data.length;
    const { page, pageSize } = TABLE_STATE;
    const totalPages = Math.max(1, Math.ceil(total / pageSize));
    TABLE_STATE.page = Math.min(page, totalPages - 1);
    const start = TABLE_STATE.page * pageSize;
    const pageData = data.slice(start, start + pageSize);

    countEl.textContent = `${total.toLocaleString()} rows`;

    function thClass(col) {
        if (TABLE_STATE.sortCol === col) return TABLE_STATE.sortDir === "asc" ? "sorted-asc" : "sorted-desc";
        return "";
    }

    if (TABLE_STATE.tab === "nodes") {
        wrapper.innerHTML = `<table>
            <thead><tr>
                <th class="${thClass("label")}" onclick="sortTable('label')">Label</th>
                <th class="${thClass("facet")}" onclick="sortTable('facet')">Facet</th>
                <th class="${thClass("degree")}" onclick="sortTable('degree')">Degree</th>
                <th class="${thClass("maxSupport")}" onclick="sortTable('maxSupport')">Max Support</th>
                <th class="${thClass("disciplines")}" onclick="sortTable('disciplines')">Disciplines</th>
            </tr></thead>
            <tbody>${pageData.map(n => {
                const facetColor = FACET_COLORS[n.fl[0]] || FACET_COLORS["Unknown"];
                return `<tr>
                    <td class="clickable-cell" onclick="tableNodeClick('${n.id}')">${escapeHtml(n.l)}</td>
                    <td><span class="facet-badge" style="background:${facetColor}55;color:${facetColor}">${n.fl[0] || 'Unknown'}</span></td>
                    <td>${n.degree}</td>
                    <td>${n.maxSupport}</td>
                    <td title="${(n.dl || []).join(', ')}">${(n.dl || []).length > 0 ? (n.dl || []).slice(0, 2).join(', ') + (n.dl.length > 2 ? '...' : '') : '—'}</td>
                </tr>`;
            }).join("")}</tbody></table>`;
    } else {
        wrapper.innerHTML = `<table>
            <thead><tr>
                <th class="${thClass("source")}" onclick="sortTable('source')">Source</th>
                <th class="${thClass("relation")}" onclick="sortTable('relation')">Relation</th>
                <th class="${thClass("target")}" onclick="sortTable('target')">Target</th>
                <th class="${thClass("support")}" onclick="sortTable('support')">Support</th>
                <th class="${thClass("weight")}" onclick="sortTable('weight')">Weight</th>
            </tr></thead>
            <tbody>${pageData.map(e => {
                const relColor = RELATION_COLORS[e.r] || "#94a3b8";
                return `<tr>
                    <td class="clickable-cell" onclick="tableNodeClick('${e.s}')">${escapeHtml(e.sl)}</td>
                    <td><span class="relation-badge" style="background:${relColor}55;color:${relColor}">${RELATION_LABELS[e.r] || e.r}</span></td>
                    <td class="clickable-cell" onclick="tableNodeClick('${e.t}')">${escapeHtml(e.tl)}</td>
                    <td>${e.sup}</td>
                    <td>${e.w.toFixed(3)}</td>
                </tr>`;
            }).join("")}</tbody></table>`;
    }

    // Pagination
    renderPagination(pagEl, totalPages);
}

function renderPagination(container, totalPages) {
    if (totalPages <= 1) { container.innerHTML = ""; return; }

    const cur = TABLE_STATE.page;
    let html = `<button class="page-btn" onclick="goToPage(0)" ${cur === 0 ? 'disabled' : ''}>&laquo;</button>`;
    html += `<button class="page-btn" onclick="goToPage(${cur - 1})" ${cur === 0 ? 'disabled' : ''}>&lsaquo;</button>`;

    // Show page window
    const windowSize = 5;
    let startPage = Math.max(0, cur - Math.floor(windowSize / 2));
    let endPage = Math.min(totalPages - 1, startPage + windowSize - 1);
    if (endPage - startPage < windowSize - 1) startPage = Math.max(0, endPage - windowSize + 1);

    if (startPage > 0) html += `<span style="color:var(--text-muted);padding:0 4px;">...</span>`;
    for (let i = startPage; i <= endPage; i++) {
        html += `<button class="page-btn ${i === cur ? 'active' : ''}" onclick="goToPage(${i})">${i + 1}</button>`;
    }
    if (endPage < totalPages - 1) html += `<span style="color:var(--text-muted);padding:0 4px;">...</span>`;

    html += `<button class="page-btn" onclick="goToPage(${cur + 1})" ${cur >= totalPages - 1 ? 'disabled' : ''}>&rsaquo;</button>`;
    html += `<button class="page-btn" onclick="goToPage(${totalPages - 1})" ${cur >= totalPages - 1 ? 'disabled' : ''}>&raquo;</button>`;
    container.innerHTML = html;
}

window.sortTable = (col) => {
    if (TABLE_STATE.sortCol === col) {
        TABLE_STATE.sortDir = TABLE_STATE.sortDir === "asc" ? "desc" : "asc";
    } else {
        TABLE_STATE.sortCol = col;
        TABLE_STATE.sortDir = "asc";
    }
    TABLE_STATE.page = 0;
    renderTable();
};

window.goToPage = (p) => {
    TABLE_STATE.page = Math.max(0, p);
    renderTable();
    // Scroll table to top
    const wrapper = document.getElementById("tableWrapper");
    if (wrapper) wrapper.scrollTop = 0;
};

window.tableNodeClick = (nodeId) => {
    // Switch to graph view, focus on node, and show details
    const node = STATE.nodeMap.get(nodeId);
    if (!node) return;

    switchView("graph");
    // Wait for graph to be visible, then navigate
    setTimeout(() => {
        const graphNode = Graph.graphData().nodes.find(n => n.id === nodeId);
        if (graphNode) {
            focusOnNode(graphNode);
            showDetails(graphNode);
        }
    }, 200);
};

function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
}

// Table search input listener
window.addEventListener("load", () => {
    const tSearchInput = document.getElementById("tableSearchInput");
    if (tSearchInput) {
        let tDebounce;
        tSearchInput.addEventListener("input", (e) => {
            TABLE_STATE.search = e.target.value;
            TABLE_STATE.page = 0;
            if (tDebounce) clearTimeout(tDebounce);
            tDebounce = setTimeout(renderTable, 300);
        });
    }
});

/* ─── Helpers ─── */

function setText(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
}
