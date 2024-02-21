import pytest
from Graph import Node, Edge, GraphGenerator, Arena


def test_remove_negative_cycles(monkeypatch):
    """
    Test that the negative cycles are correctly removed from a given graph
    """
    monkeypatch.setattr('builtins.print', lambda x: None)

    nodes = [Node(1), Node(2), Node(3), Node(4)]
    edges = [Edge(Node(1), Node(2), 1.5),
             Edge(Node(2), Node(3), 12),
             Edge(Node(3), Node(4), 9),
             Edge(Node(4), Node(1), -20),
             Edge(Node(3), Node(1), -15)]
    arena = Arena(nodes, edges)
    print(f"Initial arena is {arena}")

    for edge in edges:
        edge.node1.add_edge(edge)

    for node in nodes:
        assert arena._check_negative_cycles(node)

    # for node in nodes:
    #     arena.remove_negative_cycles(node)

    print(f"Final arena is {arena}")
    for node in nodes:
        assert not arena._check_negative_cycles(node)

@pytest.mark.skip(reason="No way of currently testing this")
def test_graph_generation(monkeypatch):
    """
    Test that the negative cycles are correctly removed from a generated graph
    """
    def count_cycles():
        cycles = 0
        for node in nodes:
            if arena._check_negative_cycles(node):
                cycles += 1
        return cycles

    num_cycles = 0
    while num_cycles == 0:
        graph_generator = GraphGenerator(
            num_nodes=10, edge_probability=0.3)
        nodes, edges = graph_generator.generate_graph()
        arena = Arena(nodes, edges)
        num_cycles = count_cycles()

    for node in nodes:
        assert arena._check_negative_cycles(node)

    arena.generate_mean_payoff_arena()
    for node in nodes:
        arena.remove_negative_cycles(node)

    for node in nodes:
        assert not arena._check_negative_cycles(node)
