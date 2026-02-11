import os
import json
import argparse
import psycopg2
from pathlib import Path
from dotenv import load_dotenv
from pyvis.network import Network
import networkx as nx

# Load environment variables
load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "database": os.getenv("DB_NAME", "postgres"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "postgres"),
}

# Color mapping for facets
FACET_COLORS = {
    "Area": "#FF6B6B",       # Red
    "System": "#4ECDC4",     # Teal
    "Property": "#45B7D1",   # Blue
    "Technique": "#FFA07A",  # Light Salmon
}

# Color mapping for disciplines (you can expand this)
DISCIPLINE_COLORS = {
    "Condensed Matter Physics": "#9B59B6",
    "Quantum Physics": "#3498DB",
    "Astrophysics": "#E74C3C",
    "High Energy Physics": "#F39C12",
    "Nuclear Physics": "#1ABC9C",
}

# Color mapping for relation types
RELATION_COLORS = {
    "STUDIES": "#E74C3C",      # Red
    "EXHIBITS": "#3498DB",     # Blue
    "MEASURES": "#2ECC71",     # Green
    "PROBES": "#F39C12",       # Orange
    "PARENT_OF": "#9B59B6",    # Purple
    "RELATED_TO": "#1ABC9C",   # Turquoise
    "HAS_FACET": "#E67E22",    # Dark Orange
    "IN_DISCIPLINE": "#95A5A6" # Gray
}

# Edge styles for relation types
RELATION_STYLES = {
    "PARENT_OF": {"dashes": False},
    "RELATED_TO": {"dashes": [5, 5]},
    "HAS_FACET": {"dashes": [2, 2]},
    "IN_DISCIPLINE": {"dashes": [10, 5]},
}

def parse_json_field(val):
    """Parse JSON field from database"""
    if not val:
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            loaded = json.loads(val)
            if isinstance(loaded, list):
                return loaded
            return [loaded]
        except:
            return [val]
    return []

def get_primary_value(json_list):
    """Extract primary value from JSON list"""
    if not json_list:
        return None
    values = parse_json_field(json_list)
    return values[0] if values else None

def get_node_color(facet_labels, discipline_labels):
    """Determine node color based on facet or discipline"""
    facets = parse_json_field(facet_labels)
    if facets:
        primary_facet = facets[0]
        return FACET_COLORS.get(primary_facet, "#95A5A6")  # Default gray

    disciplines = parse_json_field(discipline_labels)
    if disciplines:
        primary_discipline = disciplines[0]
        return DISCIPLINE_COLORS.get(primary_discipline, "#95A5A6")

    return "#95A5A6"  # Default gray

def fetch_graph_data(support_threshold=50, limit=1000000):
    """Fetch graph data from database"""
    print(f"Fetching graph data with support > {support_threshold}, limit {limit}...")

    conn = psycopg2.connect(**DB_CONFIG)

    query = """
    SELECT
        oe.source_id,
        source_pc.concept_label AS source_label,
        source_pc.discipline_labels AS source_discipline,
        source_pc.facet_labels AS source_facet_labels,
        oe.relation,
        oe.target_id,
        target_pc.concept_label AS target_label,
        target_pc.discipline_labels AS target_discipline,
        target_pc.facet_labels AS target_facet_labels,
        oe.support
    FROM third_party_aps.ontology_edges oe
    LEFT JOIN third_party_aps.physh_concepts source_pc
        ON oe.source_id = source_pc.concept_id
    LEFT JOIN third_party_aps.physh_concepts target_pc
        ON oe.target_id = target_pc.concept_id
    WHERE oe.support > %s
    ORDER BY oe.support DESC
    LIMIT %s;
    """

    try:
        with conn.cursor() as cur:
            cur.execute(query, (support_threshold, limit))
            rows = cur.fetchall()
            print(f"Fetched {len(rows)} edges")
            return rows
    finally:
        conn.close()

def compute_statistics(edges_data):
    """Compute statistics about the graph"""
    from collections import Counter

    facet_counts = Counter()
    discipline_counts = Counter()
    relation_counts = Counter()
    node_ids = set()

    for row in edges_data:
        (source_id, source_label, source_discipline, source_facet_labels,
         relation, target_id, target_label, target_discipline, target_facet_labels,
         support) = row

        node_ids.add(source_id)
        node_ids.add(target_id)

        # Count facets
        for facet in parse_json_field(source_facet_labels):
            facet_counts[facet] += 1
        for facet in parse_json_field(target_facet_labels):
            facet_counts[facet] += 1

        # Count disciplines
        for disc in parse_json_field(source_discipline):
            discipline_counts[disc] += 1
        for disc in parse_json_field(target_discipline):
            discipline_counts[disc] += 1

        # Count relations
        relation_counts[relation] += 1

    return {
        'facets': dict(facet_counts),
        'disciplines': dict(discipline_counts),
        'relations': dict(relation_counts),
        'total_nodes': len(node_ids),
        'total_edges': len(edges_data)
    }

def build_graph_visualization(edges_data, output_file="ontology_graph_v2.html", height="900px", width="100%"):
    """Build interactive graph visualization using pyvis"""
    print("Building graph visualization...")

    # Compute statistics
    stats = compute_statistics(edges_data)

    # Create pyvis network
    net = Network(height=height, width=width, directed=True, notebook=False)
    net.barnes_hut()  # Enable physics

    # Track unique nodes to avoid duplicates
    nodes_added = set()

    # Collect all data for filtering
    all_disciplines = set()
    all_relations = set()

    for row in edges_data:
        (source_id, source_label, source_discipline, source_facet_labels,
         relation, target_id, target_label, target_discipline, target_facet_labels,
         support) = row

        # Add source node
        if source_id not in nodes_added:
            color = get_node_color(source_facet_labels, source_discipline)
            facets = parse_json_field(source_facet_labels)
            disciplines = parse_json_field(source_discipline)
            all_disciplines.update(disciplines)

            title = f"""<b>{source_label or source_id}</b>
            <br>ID: {source_id}
            <br>Facets: {', '.join(facets) if facets else 'N/A'}
            <br>Disciplines: {', '.join(disciplines) if disciplines else 'N/A'}"""

            # Store metadata for filtering
            net.add_node(
                source_id,
                label=source_label or source_id,
                title=title,
                color=color,
                size=20,
                disciplines=disciplines,
                facets=facets
            )
            nodes_added.add(source_id)

        # Add target node
        if target_id not in nodes_added:
            color = get_node_color(target_facet_labels, target_discipline)
            facets = parse_json_field(target_facet_labels)
            disciplines = parse_json_field(target_discipline)
            all_disciplines.update(disciplines)

            title = f"""<b>{target_label or target_id}</b>
            <br>ID: {target_id}
            <br>Facets: {', '.join(facets) if facets else 'N/A'}
            <br>Disciplines: {', '.join(disciplines) if disciplines else 'N/A'}"""

            # Store metadata for filtering
            net.add_node(
                target_id,
                label=target_label or target_id,
                title=title,
                color=color,
                size=20,
                disciplines=disciplines,
                facets=facets
            )
            nodes_added.add(target_id)

        # Add edge with relation-specific styling
        all_relations.add(relation)
        edge_title = f"{relation} (support: {support})"
        edge_width = min(1 + support / 20, 10)  # Scale edge width by support
        edge_color = RELATION_COLORS.get(relation, "#95A5A6")
        edge_style = RELATION_STYLES.get(relation, {})

        net.add_edge(
            source_id,
            target_id,
            title=edge_title,
            label=relation,
            width=edge_width,
            color=edge_color,
            relation_type=relation,
            **edge_style
        )

    # Configure visualization options
    net.set_options("""
    {
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -30000,
          "centralGravity": 0.3,
          "springLength": 200,
          "springConstant": 0.04,
          "damping": 0.09,
          "avoidOverlap": 0.5
        },
        "minVelocity": 0.75,
        "stabilization": {
          "enabled": true,
          "iterations": 100
        }
      },
      "nodes": {
        "font": {
          "size": 14,
          "face": "arial"
        },
        "borderWidth": 2,
        "borderWidthSelected": 4
      },
      "edges": {
        "arrows": {
          "to": {
            "enabled": true,
            "scaleFactor": 0.5
          }
        },
        "smooth": {
          "enabled": true,
          "type": "continuous"
        },
        "font": {
          "size": 10,
          "align": "middle"
        }
      },
      "interaction": {
        "hover": true,
        "navigationButtons": true,
        "keyboard": true
      }
    }
    """)

    # Save initial HTML
    output_path = Path(__file__).parent / output_file
    net.save_graph(str(output_path))

    # Inject custom HTML with stats and controls
    inject_custom_ui(output_path, stats, sorted(all_disciplines), sorted(all_relations))

    print(f"Graph saved to: {output_path}")
    print(f"Total nodes: {len(nodes_added)}")
    print(f"Total edges: {len(edges_data)}")

    return output_path

def inject_custom_ui(html_path, stats, all_disciplines, all_relations):
    """Inject custom UI controls and statistics into the HTML"""
    print("Injecting custom UI controls and statistics...")

    with open(html_path, 'r') as f:
        html_content = f.read()

    # Build stats HTML
    facet_stats_html = ""
    for facet, count in sorted(stats['facets'].items()):
        color = FACET_COLORS.get(facet, "#95A5A6")
        facet_stats_html += f'<div style="margin: 5px 0;"><span style="display:inline-block;width:12px;height:12px;background-color:{color};margin-right:8px;"></span>{facet}: {count}</div>'

    discipline_stats_html = ""
    for disc, count in sorted(stats['disciplines'].items(), key=lambda x: x[1], reverse=True)[:10]:
        discipline_stats_html += f'<div style="margin: 5px 0;">{disc}: {count}</div>'

    relation_stats_html = ""
    for rel, count in sorted(stats['relations'].items(), key=lambda x: x[1], reverse=True):
        color = RELATION_COLORS.get(rel, "#95A5A6")
        relation_stats_html += f'<div style="margin: 5px 0;"><span style="display:inline-block;width:12px;height:12px;background-color:{color};margin-right:8px;"></span>{rel}: {count}</div>'

    # Build discipline filter options
    discipline_options = "".join([f'<option value="{d}">{d}</option>' for d in all_disciplines])

    # Build relation filter options
    relation_checkboxes = ""
    for rel in all_relations:
        color = RELATION_COLORS.get(rel, "#95A5A6")
        relation_checkboxes += f'''
        <div style="margin: 5px 0;">
            <label style="display: flex; align-items: center;">
                <input type="checkbox" value="{rel}" checked onchange="filterGraph()">
                <span style="display:inline-block;width:12px;height:12px;background-color:{color};margin:0 8px;"></span>
                {rel}
            </label>
        </div>
        '''

    # Custom HTML to inject
    custom_ui = f'''
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
        }}
        #sidebar {{
            position: fixed;
            left: 0;
            top: 0;
            width: 300px;
            height: 100vh;
            background: #f8f9fa;
            border-right: 2px solid #ddd;
            overflow-y: auto;
            padding: 15px;
            box-sizing: border-box;
            z-index: 1000;
        }}
        #mynetwork {{
            position: fixed;
            left: 300px;
            top: 0;
            width: calc(100% - 300px);
            height: 100vh;
        }}
        .section {{
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 1px solid #ddd;
        }}
        .section:last-child {{
            border-bottom: none;
        }}
        .section h3 {{
            margin: 0 0 10px 0;
            font-size: 14px;
            font-weight: bold;
            color: #333;
        }}
        .stats-item {{
            font-size: 12px;
            color: #666;
        }}
        input[type="text"], select {{
            width: 100%;
            padding: 8px;
            margin: 5px 0;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-sizing: border-box;
        }}
        button {{
            padding: 8px 15px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            margin-top: 5px;
        }}
        button:hover {{
            background: #0056b3;
        }}
        .checkbox-group {{
            max-height: 200px;
            overflow-y: auto;
            font-size: 12px;
        }}
    </style>

    <div id="sidebar">
        <div class="section">
            <h3>Graph Statistics</h3>
            <div class="stats-item">
                <strong>Total Nodes:</strong> {stats['total_nodes']}<br>
                <strong>Total Edges:</strong> {stats['total_edges']}
            </div>
        </div>

        <div class="section">
            <h3>Facet Distribution</h3>
            <div class="stats-item">
                {facet_stats_html}
            </div>
        </div>

        <div class="section">
            <h3>Top Disciplines</h3>
            <div class="stats-item">
                {discipline_stats_html}
            </div>
        </div>

        <div class="section">
            <h3>Relation Types</h3>
            <div class="stats-item">
                {relation_stats_html}
            </div>
        </div>

        <div class="section">
            <h3>Search</h3>
            <input type="text" id="searchInput" placeholder="Search nodes..." onkeyup="searchNodes()">
            <button onclick="resetSearch()">Reset</button>
        </div>

        <div class="section">
            <h3>Filter by Discipline</h3>
            <select id="disciplineFilter" onchange="filterGraph()">
                <option value="">All Disciplines</option>
                {discipline_options}
            </select>
        </div>

        <div class="section">
            <h3>Filter by Relation</h3>
            <div class="checkbox-group">
                {relation_checkboxes}
            </div>
        </div>
    </div>

    <script type="text/javascript">
        let allNodes = null;
        let allEdges = null;
        let highlightActive = false;

        // Store original data
        network.on("stabilizationIterationsDone", function () {{
            allNodes = nodes.get();
            allEdges = edges.get();
        }});

        function searchNodes() {{
            const searchTerm = document.getElementById('searchInput').value.toLowerCase();

            if (!allNodes || !allEdges) {{
                setTimeout(searchNodes, 100);
                return;
            }}

            if (searchTerm === '') {{
                // Reset to original
                nodes.update(allNodes.map(n => ({{
                    id: n.id,
                    hidden: false,
                    opacity: 1
                }})));
                edges.update(allEdges.map(e => ({{
                    id: e.id,
                    hidden: false
                }})));
                return;
            }}

            // Find matching nodes
            const matchingNodeIds = new Set();
            allNodes.forEach(node => {{
                const label = (node.label || '').toLowerCase();
                if (label.includes(searchTerm)) {{
                    matchingNodeIds.add(node.id);
                }}
            }});

            // Find connected edges
            const connectedEdges = new Set();
            allEdges.forEach(edge => {{
                if (matchingNodeIds.has(edge.from) || matchingNodeIds.has(edge.to)) {{
                    connectedEdges.add(edge.id);
                    matchingNodeIds.add(edge.from);
                    matchingNodeIds.add(edge.to);
                }}
            }});

            // Update visibility
            nodes.update(allNodes.map(n => ({{
                id: n.id,
                hidden: !matchingNodeIds.has(n.id),
                opacity: matchingNodeIds.has(n.id) ? 1 : 0.1
            }})));

            edges.update(allEdges.map(e => ({{
                id: e.id,
                hidden: !connectedEdges.has(e.id)
            }})));
        }}

        function resetSearch() {{
            document.getElementById('searchInput').value = '';
            searchNodes();
        }}

        function filterGraph() {{
            const disciplineFilter = document.getElementById('disciplineFilter').value;
            const relationCheckboxes = document.querySelectorAll('#sidebar input[type="checkbox"]');
            const selectedRelations = new Set();

            relationCheckboxes.forEach(cb => {{
                if (cb.checked) {{
                    selectedRelations.add(cb.value);
                }}
            }});

            if (!allNodes || !allEdges) {{
                setTimeout(filterGraph, 100);
                return;
            }}

            // Filter nodes by discipline
            const visibleNodeIds = new Set();
            allNodes.forEach(node => {{
                let visible = true;

                if (disciplineFilter && node.disciplines) {{
                    visible = node.disciplines.includes(disciplineFilter);
                }}

                if (visible) {{
                    visibleNodeIds.add(node.id);
                }}
            }});

            // Filter edges by relation type and connected nodes
            const visibleEdges = new Set();
            allEdges.forEach(edge => {{
                const relationVisible = selectedRelations.has(edge.relation_type);
                const nodesVisible = visibleNodeIds.has(edge.from) && visibleNodeIds.has(edge.to);

                if (relationVisible && nodesVisible) {{
                    visibleEdges.add(edge.id);
                }}
            }});

            // Update visibility
            nodes.update(allNodes.map(n => ({{
                id: n.id,
                hidden: !visibleNodeIds.has(n.id)
            }})));

            edges.update(allEdges.map(e => ({{
                id: e.id,
                hidden: !visibleEdges.has(e.id)
            }})));
        }}
    </script>
    '''

    # Inject before closing body tag
    html_content = html_content.replace('</body>', f'{custom_ui}</body>')

    # Write back
    with open(html_path, 'w') as f:
        f.write(html_content)

    print("Custom UI injected successfully")

def build_networkx_stats(edges_data):
    """Build NetworkX graph and compute statistics"""
    print("\nComputing graph statistics...")

    G = nx.DiGraph()

    for row in edges_data:
        source_id, source_label, _, source_facets, relation, target_id, target_label, _, target_facets, support = row

        # Add nodes with attributes
        if not G.has_node(source_id):
            G.add_node(
                source_id,
                label=source_label or source_id,
                facets=parse_json_field(source_facets)
            )

        if not G.has_node(target_id):
            G.add_node(
                target_id,
                label=target_label or target_id,
                facets=parse_json_field(target_facets)
            )

        # Add edge
        G.add_edge(source_id, target_id, relation=relation, support=support)

    # Compute statistics
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")

    # Analyze facets and relations
    from collections import Counter
    facet_counter = Counter()
    relation_counter = Counter()

    for node_id, node_data in G.nodes(data=True):
        facets = node_data.get('facets', [])
        for facet in facets:
            facet_counter[facet] += 1

    for _, _, edge_data in G.edges(data=True):
        relation = edge_data.get('relation', 'Unknown')
        relation_counter[relation] += 1

    print("\nFacet distribution:")
    for facet, count in facet_counter.most_common():
        print(f"  {facet}: {count}")

    print("\nRelation type distribution:")
    for relation, count in relation_counter.most_common():
        print(f"  {relation}: {count}")

    # Top nodes by degree
    degrees = dict(G.degree())
    top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\nTop 10 nodes by degree:")
    for node_id, degree in top_nodes:
        label = G.nodes[node_id].get('label', node_id)
        print(f"  {label}: {degree}")

    # Check if graph is connected
    if nx.is_weakly_connected(G):
        print("\nGraph is weakly connected")
    else:
        components = list(nx.weakly_connected_components(G))
        print(f"\nGraph has {len(components)} weakly connected components")
        print(f"Largest component size: {len(max(components, key=len))}")

def main():
    parser = argparse.ArgumentParser(description="Visualize ontology graph from database")
    parser.add_argument(
        "--support",
        type=int,
        default=50,
        help="Minimum support threshold for edges (default: 50)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100000,
        help="Maximum number of edges to fetch (default: 100)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ontology_graph_v2.html",
        help="Output filename for visualization (default: ontology_graph_v2.html)"
    )
    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="Skip computing graph statistics"
    )

    args = parser.parse_args()

    # Fetch data
    edges_data = fetch_graph_data(support_threshold=args.support, limit=args.limit)

    if not edges_data:
        print("No edges found with the specified criteria")
        return

    # Build visualization
    output_path = build_graph_visualization(edges_data, output_file=args.output)

    # Compute statistics
    if not args.no_stats:
        build_networkx_stats(edges_data)

    print(f"\n✓ Visualization complete! Open {output_path} in your browser to view the graph.")

if __name__ == "__main__":
    main()
