from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from collections import Counter
from itertools import groupby
from core.data_models import Chapter, Entity

_OUT = Path(__file__).parent.parent.parent / "outputs"
_OUT.mkdir(exist_ok=True)


def visualize_mood_flow(chapters: list[Chapter]) -> None:
    scores = [ch.mood.vader_sentiment["compound"] for ch in chapters]
    scene_nums = [ch.number for ch in chapters]

    fig, ax = plt.subplots(figsize=(16, 7))

    seen_arcs: set[int] = set()
    arc_palette = plt.cm.Set3.colors
    for arc_label, group in groupby(chapters, key=lambda ch: ch.arc_label):
        if arc_label < 0:
            continue
        group_list = list(group)
        label = f"Arc {arc_label}" if arc_label not in seen_arcs else "_nolegend_"
        ax.axvspan(
            group_list[0].number - 0.5, group_list[-1].number + 0.5,
            alpha=0.15, color=arc_palette[arc_label % len(arc_palette)], label=label,
        )
        seen_arcs.add(arc_label)

    ax.plot(scene_nums, scores, marker="o", linewidth=2, color="steelblue", zorder=3)

    if len(scores) > 1:
        diffs = [scores[i + 1] - scores[i] for i in range(len(scores) - 1)]
        tp_idx = diffs.index(min(diffs))
        tp_entities = [e.name for e in chapters[tp_idx].entities[:2]]
        tp_label = f"Turning point\n({', '.join(tp_entities)})" if tp_entities else "Turning point"
        ax.axvline(x=scene_nums[tp_idx] + 0.5, color="crimson", linestyle="--",
                   linewidth=1.5, alpha=0.8, zorder=4)
        ax.text(
            scene_nums[tp_idx] + 0.6, 0.85, tp_label,
            color="crimson", fontsize=8, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="crimson"),
        )

    bottom_indices = sorted(range(len(scores)), key=lambda i: scores[i])[:3]
    for idx in bottom_indices:
        ch = chapters[idx]
        names = [e.name for e in ch.entities[:2]]
        if names:
            ax.annotate(
                ", ".join(names),
                xy=(ch.number, scores[idx]),
                xytext=(ch.number, scores[idx] + 0.18),
                fontsize=8, color="darkred", ha="center",
                arrowprops=dict(arrowstyle="->", color="darkred", lw=0.8),
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="darkred"),
            )

    ax.set_title("Mood Flow Across Scenes")
    ax.set_xlabel("Scene")
    ax.set_ylabel("VADER Sentiment Compound Score")
    ax.set_ylim(-1.15, 1.15)
    ax.grid(True, alpha=0.3)
    if seen_arcs:
        ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(_OUT / "mood_flow.png", dpi=300, bbox_inches="tight")
    plt.close()


def visualize_emotion_distribution(chapters: list[Chapter]) -> None:
    emotion_counts: Counter[str] = Counter()
    for ch in chapters:
        emotion_counts.update(ch.mood.emotions)
    emotions = list(emotion_counts.keys())
    counts = list(emotion_counts.values())
    plt.figure(figsize=(12, 8))
    sns.barplot(x=emotions, y=counts)
    plt.title("Overall Emotion Distribution")
    plt.xlabel("Emotion")
    plt.ylabel("Total Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(_OUT / "emotion_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()


def visualize_character_network(chapters: list[Chapter], important_entities: list[Entity]) -> None:
    top_names = {e.name for e in important_entities}

    G: nx.Graph = nx.Graph()
    for entity in important_entities:
        G.add_node(entity.name, weight=entity.count)

    for chapter in chapters:
        names_in_chapter = [e.name for e in chapter.entities if e.name in top_names]
        for i, n1 in enumerate(names_in_chapter):
            for n2 in names_in_chapter[i + 1:]:
                if G.has_edge(n1, n2):
                    G[n1][n2]["weight"] += 1
                else:
                    G.add_edge(n1, n2, weight=1)

    isolated = [n for n in G.nodes if G.degree(n) == 0]
    G.remove_nodes_from(isolated)

    if G.number_of_nodes() == 0:
        return

    communities = nx.community.louvain_communities(G, seed=42)
    community_map = {node: i for i, comm in enumerate(communities) for node in comm}
    node_colors = [community_map.get(n, 0) for n in G.nodes]

    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    normalized_widths = [w / max_weight * 5 for w in edge_weights]

    plt.figure(figsize=(15, 15))
    pos = nx.spring_layout(G, weight="weight", seed=42)
    nx.draw(
        G, pos, with_labels=True,
        node_color=node_colors, cmap=plt.cm.Set3,
        node_size=[G.nodes[n]["weight"] * 100 for n in G.nodes],
        width=normalized_widths,
        font_size=10, font_weight="bold",
    )
    plt.title("Character Relationship Network")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(_OUT / "character_network.png", dpi=300, bbox_inches="tight")
    plt.close()


def visualize_entity_flow(chapters: list[Chapter], important_entities: list[Entity]) -> None:
    top_entities = [e.name for e in important_entities[:10]]
    entity_matrix = [
        [1 if e in {ent.name for ent in ch.entities} else 0 for e in top_entities]
        for ch in chapters
    ]
    plt.figure(figsize=(15, 10))
    sns.heatmap(entity_matrix, cmap="YlOrRd", yticklabels=top_entities)
    plt.title("Entity Flow Across Chapters")
    plt.xlabel("Chapter")
    plt.ylabel("Entity")
    plt.tight_layout()
    plt.savefig(_OUT / "entity_flow.png", dpi=300, bbox_inches="tight")
    plt.close()


def visualize_articulation_structure(book, analysis) -> None:
    from core.data_models import ArticulationAnalysis, Book
    book_typed: Book = book
    analysis_typed: ArticulationAnalysis = analysis

    ap_names = {ap.name for ap in analysis_typed.articulation_points}
    bridge_set = {frozenset(e) for e in analysis_typed.bridge_edges}

    names = [e.name for e in book_typed.important_entities if e.entity_type != "Place"]
    G: nx.Graph = nx.Graph()
    G.add_nodes_from(names)
    for chapter in book_typed.chapters:
        present = [n for n in names if n.lower() in chapter.content.lower()]
        for i, n1 in enumerate(present):
            for n2 in present[i + 1:]:
                if G.has_edge(n1, n2):
                    G[n1][n2]["weight"] += 1
                else:
                    G.add_edge(n1, n2, weight=1)
    isolated = [n for n in G.nodes if G.degree(n) == 0]
    G.remove_nodes_from(isolated)

    communities = nx.community.louvain_communities(G, seed=42)
    community_map = {n: i for i, c in enumerate(communities) for n in c}

    node_colors = [
        "#e74c3c" if n in ap_names
        else plt.cm.Set3.colors[community_map.get(n, 0) % 12]
        for n in G.nodes
    ]
    node_sizes = [
        G.nodes[n].get("weight", 1) * 120 + 400 if n in ap_names
        else G.nodes[n].get("weight", 1) * 80 + 200
        for n in G.nodes
    ]

    bridge_edges = [(u, v) for u, v in G.edges if frozenset((u, v)) in bridge_set]
    regular_edges = [(u, v) for u, v in G.edges if frozenset((u, v)) not in bridge_set]
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w = max(edge_weights) if edge_weights else 1

    plt.figure(figsize=(16, 16))
    pos = nx.spring_layout(G, weight="weight", seed=42)
    nx.draw_networkx_edges(G, pos, edgelist=regular_edges,
                           width=[G[u][v]["weight"] / max_w * 4 for u, v in regular_edges],
                           edge_color="#bdc3c7", alpha=0.6)
    nx.draw_networkx_edges(G, pos, edgelist=bridge_edges,
                           width=3.0, edge_color="#e74c3c", style="dashed", alpha=0.9)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold")

    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c",
                   markersize=12, label="Articulation point"),
        plt.Line2D([0], [0], color="#e74c3c", linewidth=2, linestyle="dashed",
                   label="Bridge relationship"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#a8d8a8",
                   markersize=10, label="Regular character"),
    ]
    plt.legend(handles=legend_elements, loc="upper left", fontsize=9)
    plt.title("Character Network — Structural Bridges\n"
              "Red nodes: removal disconnects the network  ·  Red dashed: bridge relationships")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(_OUT / "articulation_network.png", dpi=300, bbox_inches="tight")
    plt.close()


def visualize_character_impact_scatter(impacts: list) -> None:
    from core.data_models import CharacterImpact
    typed: list[CharacterImpact] = impacts

    d_vals = [imp.cohens_d for imp in typed]
    lag_vals = [imp.lagged_delta for imp in typed]

    colors = []
    for d, l in zip(d_vals, lag_vals):
        if d > 0 and l < 0:
            colors.append("#e74c3c")
        elif d > 0 and l >= 0:
            colors.append("#e67e22")
        elif d <= 0 and l < 0:
            colors.append("#8e44ad")
        else:
            colors.append("#27ae60")

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.scatter(d_vals, lag_vals, c=colors, s=140, alpha=0.85, zorder=3)

    for imp, x, y in zip(typed, d_vals, lag_vals):
        ax.annotate(imp.name, (x, y), xytext=(6, 4),
                    textcoords="offset points", fontsize=8)

    ax.axvline(x=0, color="#95a5a6", linewidth=0.9, linestyle="--")
    ax.axhline(y=0, color="#95a5a6", linewidth=0.9, linestyle="--")

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    pad_x = (xmax - xmin) * 0.04
    pad_y = (ymax - ymin) * 0.04

    quadrant_labels = [
        (xmax - pad_x, ymin + pad_y, "Active catalyst\n(dark scenes + causes drops)",
         "#e74c3c", "right", "bottom"),
        (xmax - pad_x, ymax - pad_y, "Tragic presence\n(dark scenes, stabilises after)",
         "#e67e22", "right", "top"),
        (xmin + pad_x, ymin + pad_y, "Hidden driver\n(causes drops, not at the scene)",
         "#8e44ad", "left", "bottom"),
        (xmin + pad_x, ymax - pad_y, "Stabiliser\n(lighter scenes, mood improves after)",
         "#27ae60", "left", "top"),
    ]
    for x, y, label, color, ha, va in quadrant_labels:
        ax.text(x, y, label, color=color, fontsize=7, ha=ha, va=va, alpha=0.7)

    ax.set_xlabel("Cohen's d  (associative: how dark are scenes when present →)")
    ax.set_ylabel("← Lagged delta  (causal: sentiment change in scene after appearance)")
    ax.set_title("Character Impact — Associative vs. Causal")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(_OUT / "character_impact_scatter.png", dpi=300, bbox_inches="tight")
    plt.close()


def visualize_character_mood_impact(impacts: list) -> None:
    from core.data_models import CharacterImpact
    impacts_typed: list[CharacterImpact] = impacts

    ordered = sorted(impacts_typed, key=lambda x: x.cohens_d)
    names = [imp.name for imp in ordered]
    values = [imp.cohens_d for imp in ordered]
    colors = ["#c0392b" if v > 0.2 else "#2980b9" if v < -0.2 else "#7f8c8d" for v in values]

    fig, ax = plt.subplots(figsize=(12, max(6, len(ordered) * 0.55)))
    ax.barh(names, values, color=colors, alpha=0.85, height=0.6)

    for i, imp in enumerate(ordered):
        x = imp.cohens_d
        offset = 0.02 if x >= 0 else -0.02
        ha = "left" if x >= 0 else "right"
        ax.text(
            x + offset, i,
            f"δ={imp.mood_delta:+.2f}  avg={imp.presence_avg:+.2f}  n={imp.scene_count}  crisis={imp.crisis_scene_count}",
            va="center", ha=ha, fontsize=7.5, color="#2c3e50",
        )

    ax.axvline(x=0, color="#2c3e50", linewidth=0.9)
    ax.set_xlabel("Cohen's d  (standardized effect size; positive = presence correlates with darker scenes)")
    ax.set_title(
        "Character Mood Impact  (Cohen's d)\n"
        "Red → strong negative association  ·  Blue → strong positive association"
    )
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(_OUT / "character_mood_impact.png", dpi=300, bbox_inches="tight")
    plt.close()
