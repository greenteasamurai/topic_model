import networkx as nx
import matplotlib.pyplot as plt


def create_entity_network(entities: list[str]) -> nx.Graph:
    G: nx.Graph = nx.Graph()
    for entity in entities:
        if entity not in G:
            G.add_node(entity)
    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            if G.has_edge(entities[i], entities[j]):
                G[entities[i]][entities[j]]["weight"] += 1
            else:
                G.add_edge(entities[i], entities[j], weight=1)
    return G


def visualize_entity_network(G: nx.Graph, chapter_num: int) -> None:
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=1000, node_color="lightblue")
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.7)
    nx.draw_networkx_labels(G, pos, font_size=8, font_family="sans-serif")
    plt.title(f"Entity Network - Chapter {chapter_num}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"entity_network_chapter_{chapter_num}.png", dpi=300, bbox_inches="tight")
    plt.close()


def process_chapter_entities(chapter_num: int, entities: list[str]) -> nx.Graph:
    G = create_entity_network(entities)
    visualize_entity_network(G, chapter_num)
    return G
