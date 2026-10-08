"""Tests for the relational experience episode graph."""

from src.document_models import ExperienceEpisode
from src.inference.graph import EpisodeGraphBuilder


def test_empty_episode_graph():
    builder = EpisodeGraphBuilder()

    result = builder.build_graph_from_episodes([])

    assert result["num_nodes"] == 0
    assert result["num_edges"] == 0
    assert result["nodes"] == []
    assert result["edges"] == []


def test_single_episode_graph():
    builder = EpisodeGraphBuilder()

    episode = ExperienceEpisode(
        id="exp-1",
        role="Data Scientist",
        company="Nexus Analytics",
        start_date="Jan 2023",
        end_date="Present",
        is_current=True,
        description="Built machine learning systems.",
        bullet_points=[
            "Built ML models.",
            "Created data pipelines.",
        ],
    )

    result = builder.build_graph_from_episodes([episode])

    assert result["num_nodes"] == 5
    assert result["num_edges"] == 4

    node_types = [node["type"] for node in result["nodes"]]

    assert "ROLE" in node_types
    assert "COMPANY" in node_types
    assert "DATE" in node_types
    assert node_types.count("DESCRIPTION") == 2

    relations = [edge["relation"] for edge in result["edges"]]

    assert "EMPLOYED_AT" in relations
    assert "TENURE" in relations
    assert relations.count("DELIVERED") == 2


def test_multiple_episode_graphs_are_separate():
    builder = EpisodeGraphBuilder()

    episodes = [
        ExperienceEpisode(
            id="exp-1",
            role="Data Scientist",
            company="Company A",
            start_date="Jan 2023",
            end_date="Present",
            is_current=True,
            bullet_points=["Built ML models."],
        ),
        ExperienceEpisode(
            id="exp-2",
            role="Data Analyst",
            company="Company B",
            start_date="Jan 2021",
            end_date="Dec 2022",
            is_current=False,
            bullet_points=["Built dashboards."],
        ),
    ]

    result = builder.build_graph_from_episodes(episodes)

    assert result["num_nodes"] == 8
    assert result["num_edges"] == 6

    node_ids = [node["id"] for node in result["nodes"]]

    assert len(node_ids) == len(set(node_ids))

    assert any(node["label"] == "Company A" for node in result["nodes"])
    assert any(node["label"] == "Company B" for node in result["nodes"])