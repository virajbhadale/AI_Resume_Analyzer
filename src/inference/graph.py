"""Relational Episode Graph representation using NetworkX."""
from typing import List, Dict, Any
import networkx as nx
from src.document_models import ExperienceEpisode

class EpisodeGraphBuilder:
    def __init__(self):
        pass

    def build_graph_from_episodes(self, episodes: List[ExperienceEpisode]) -> Dict[str, Any]:
        """Builds a NetworkX directed graph connecting episodic entities (Role -> Company -> Dates -> Bullets)."""
        G = nx.DiGraph()

        nodes_list = []
        edges_list = []

        for ep in episodes:
            role_node = f"role_{ep.id}"
            comp_node = f"comp_{ep.id}"
            date_node = f"date_{ep.id}"

            # Add Role
            G.add_node(role_node, label=ep.role, type="ROLE")
            nodes_list.append({"id": role_node, "label": ep.role, "type": "ROLE"})

            # Add Company
            G.add_node(comp_node, label=ep.company, type="COMPANY")
            nodes_list.append({"id": comp_node, "label": ep.company, "type": "COMPANY"})
            G.add_edge(role_node, comp_node, relation="EMPLOYED_AT")
            edges_list.append({"source": role_node, "target": comp_node, "relation": "EMPLOYED_AT"})

            # Add Date
            if ep.start_date:
                date_str = f"{ep.start_date} - {ep.end_date or 'Present'}"
                G.add_node(date_node, label=date_str, type="DATE")
                nodes_list.append({"id": date_node, "label": date_str, "type": "DATE"})
                G.add_edge(comp_node, date_node, relation="TENURE")
                edges_list.append({"source": comp_node, "target": date_node, "relation": "TENURE"})

            # Connect sample bullet points
            for b_idx, bullet in enumerate(ep.bullet_points[:3]):
                bullet_node = f"bullet_{ep.id}_{b_idx}"
                short_text = bullet[:60] + "..." if len(bullet) > 60 else bullet
                G.add_node(bullet_node, label=short_text, type="DESCRIPTION")
                nodes_list.append({"id": bullet_node, "label": short_text, "type": "DESCRIPTION"})
                G.add_edge(role_node, bullet_node, relation="DELIVERED")
                edges_list.append({"source": role_node, "target": bullet_node, "relation": "DELIVERED"})

        return {
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "nodes": nodes_list,
            "edges": edges_list
        }
