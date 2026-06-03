import re
import networkx as nx
from core.data_models import Book, ArticulationPoint, ArticulationAnalysis, TemporalSnapshot


def _build_graph(
    chapters: list,
    important_entities: list,
    min_cooccurrence: int = 1,
) -> nx.Graph:
    names = [e.name for e in important_entities if e.entity_type != "Place"]
    G: nx.Graph = nx.Graph()
    G.add_nodes_from(names)

    for chapter in chapters:
        present = [
            n for n in names
            if re.search(r"\b" + re.escape(n.lower()) + r"\b", chapter.content.lower())
        ]
        for i, n1 in enumerate(present):
            for n2 in present[i + 1:]:
                if G.has_edge(n1, n2):
                    G[n1][n2]["weight"] += 1
                else:
                    G.add_edge(n1, n2, weight=1)

    if min_cooccurrence > 1:
        weak = [(u, v) for u, v, d in G.edges(data=True) if d["weight"] < min_cooccurrence]
        G.remove_edges_from(weak)

    isolated = [n for n in G.nodes if G.degree(n) == 0]
    G.remove_nodes_from(isolated)
    return G


def _run_analysis(G: nx.Graph, min_cooccurrence: int) -> ArticulationAnalysis:
    if G.number_of_nodes() < 2:
        return ArticulationAnalysis(
            articulation_points=[],
            bridge_edges=[],
            biconnected_components=[],
            min_cooccurrence=min_cooccurrence,
        )
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

    raw_aps = set(nx.articulation_points(G))
    raw_bridges = list(nx.bridges(G))

    ap_details: list[ArticulationPoint] = []
    for name in raw_aps:
        G_r = G.copy()
        G_r.remove_node(name)
        clusters = [sorted(c) for c in nx.connected_components(G_r)]
        ap_details.append(ArticulationPoint(
            name=name,
            components_after_removal=len(clusters),
            clusters=clusters,
        ))
    ap_details.sort(key=lambda x: x.components_after_removal, reverse=True)

    biconnected = [sorted(c) for c in nx.biconnected_components(G) if len(c) > 2]

    return ArticulationAnalysis(
        articulation_points=ap_details,
        bridge_edges=[(u, v) for u, v in raw_bridges],
        biconnected_components=sorted(biconnected, key=len, reverse=True),
        min_cooccurrence=min_cooccurrence,
    )


def analyze_articulation_points(
    book: Book,
    min_cooccurrence: int = 1,
) -> ArticulationAnalysis:
    G = _build_graph(book.chapters, book.important_entities, min_cooccurrence)
    return _run_analysis(G, min_cooccurrence)


def analyze_temporal_articulation(
    book: Book,
    min_cooccurrence: int = 2,
) -> list[TemporalSnapshot]:
    snapshots: list[TemporalSnapshot] = []
    prev_aps: list[str] = []

    for n in range(3, len(book.chapters) + 1):
        G = _build_graph(book.chapters[:n], book.important_entities, min_cooccurrence)
        if G.number_of_nodes() < 3 or not nx.is_connected(G):
            continue

        aps = sorted(nx.articulation_points(G))
        bridges = list(nx.bridges(G))

        if aps != prev_aps:
            snapshots.append(TemporalSnapshot(
                up_to_scene=n,
                articulation_point_names=aps,
                bridge_edge_names=[(u, v) for u, v in bridges],
            ))
            prev_aps = aps

    return snapshots
