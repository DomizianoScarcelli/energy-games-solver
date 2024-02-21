import pytest
from Graph import Node, Edge, Player


def test_safe_update(monkeypatch):
    """
    Test that the safe update function works as expected
    """
    # monkeypatch.setattr('builtins.print', lambda x: None)
    player_1 = Player(1)
    player_2 = Player(2)
    node_1 = Node(1, player_1)
    node_2 = Node(2, player_2)
    node_3 = Node(3, player_2)
    node_4 = Node(4, player_1)
    edge_12 = Edge(node_1, node_2, 4)
    edge_23 = Edge(node_2, node_3, -2)
    edge_34 = Edge(node_3, node_4, -5)
    edge_41 = Edge(node_4, node_1, -1)

    node_1.safely_update(edge_12)
    node_2.safely_update(edge_23)
    node_3.safely_update(edge_34)
    node_4.safely_update(edge_41)

    # node_1.update(edge_12)
    # node_2.update(edge_23)
    # node_3.update(edge_34)
    # node_4.update(edge_41)

    print(
        f"""
        Final setting is:
        node_1.reaches = {node_1.reaches}
        node_1.parents = {node_1.parents}
        ---
        node_2.reaches = {node_2.reaches}
        node_2.parents = {node_2.parents}
        ---
        node_3.reaches = {node_3.reaches}
        node_3.parents = {node_3.parents}
        ---
        node_4.reaches = {node_4.reaches}
        node_4.parents = {node_4.parents}
        """
    )

    print(f"Node 1 has a cycle? {node_1._check_cycle()}, is it negative? {node_1.check_negative_cycle()}")
    print(f"Node 2 has a cycle? {node_2._check_cycle()}, is it negative? {node_2.check_negative_cycle()}")
    print(f"Node 3 has a cycle? {node_3._check_cycle()}, is it negative? {node_3.check_negative_cycle()}")
    print(f"Node 4 has a cycle? {node_4._check_cycle()}, is it negative? {node_4.check_negative_cycle()}")

   