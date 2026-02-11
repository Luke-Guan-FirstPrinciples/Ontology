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

def build_graph_visualization(edges_data, output_file="ontology_graph.html", height="900px", width="100%"):
    """Build interactive graph visualization using pyvis"""
    print("Building graph visualization...")

    # Create pyvis network
    net = Network(height=height, width=width, directed=True, notebook=False)
    net.barnes_hut()  # Enable physics

    # Track unique nodes to avoid duplicates
    nodes_added = set()

    for row in edges_data:
        (source_id, source_label, source_discipline, source_facet_labels,
         relation, target_id, target_label, target_discipline, target_facet_labels,
         support) = row

        # Add source node
        if source_id not in nodes_added:
            color = get_node_color(source_facet_labels, source_discipline)
            facets = parse_json_field(source_facet_labels)
            disciplines = parse_json_field(source_discipline)

            title = f"""<b>{source_label or source_id}</b>
            <br>ID: {source_id}
            <br>Facets: {', '.join(facets) if facets else 'N/A'}
            <br>Disciplines: {', '.join(disciplines) if disciplines else 'N/A'}"""

            net.add_node(
                source_id,
                label=source_label or source_id,
                title=title,
                color=color,
                size=20
            )
            nodes_added.add(source_id)

        # Add target node
        if target_id not in nodes_added:
            color = get_node_color(target_facet_labels, target_discipline)
            facets = parse_json_field(target_facet_labels)
            disciplines = parse_json_field(target_discipline)

            title = f"""<b>{target_label or target_id}</b>
            <br>ID: {target_id}
            <br>Facets: {', '.join(facets) if facets else 'N/A'}
            <br>Disciplines: {', '.join(disciplines) if disciplines else 'N/A'}"""

            net.add_node(
                target_id,
                label=target_label or target_id,
                title=title,
                color=color,
                size=20
            )
            nodes_added.add(target_id)

        # Add edge
        edge_title = f"{relation} (support: {support})"
        edge_width = min(1 + support / 20, 10)  # Scale edge width by support

        net.add_edge(
            source_id,
            target_id,
            title=edge_title,
            label=relation,
            width=edge_width
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

    # Save
    output_path = Path(__file__).parent / output_file
    net.save_graph(str(output_path))
    print(f"Graph saved to: {output_path}")
    print(f"Total nodes: {len(nodes_added)}")
    print(f"Total edges: {len(edges_data)}")

    return output_path

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
        default="ontology_graph.html",
        help="Output filename for visualization (default: ontology_graph.html)"
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
