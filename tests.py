# import pytest
# from Graph import Arena, Node, Edge, Player

# @pytest.mark.skip(reason="Not needed for now")
# def test_safe_update_1(monkeypatch):
#     """
#     Test that the safe update function works as expected
#     """
#     # monkeypatch.setattr('builtins.print', lambda x: None)
#     player_1 = Player(1)
#     player_2 = Player(2)
#     node_1 = Node(1, player_1)
#     node_2 = Node(2, player_2)
#     node_3 = Node(3, player_2)
#     node_4 = Node(4, player_1)
#     edge_12 = Edge(node_1, node_2, 4)
#     edge_23 = Edge(node_2, node_3, -2)
#     edge_34 = Edge(node_3, node_4, -5)
#     edge_41 = Edge(node_4, node_1, -1)

#     node_1.safely_update(edge_12)
#     node_2.safely_update(edge_23)
#     node_3.safely_update(edge_34)
#     node_4.safely_update(edge_41)

#     print(
#         f"""
#         Final setting is:
#         node_1.reaches = {node_1.reaches}
#         node_1.parents = {node_1.parents}
#         ---
#         node_2.reaches = {node_2.reaches}
#         node_2.parents = {node_2.parents}
#         ---
#         node_3.reaches = {node_3.reaches}
#         node_3.parents = {node_3.parents}
#         ---
#         node_4.reaches = {node_4.reaches}
#         node_4.parents = {node_4.parents}
#         """
#     )

#     print(f"Node 1 has a cycle? {node_1._check_cycle()}, is it negative? {node_1.check_negative_cycle()}")
#     print(f"Node 2 has a cycle? {node_2._check_cycle()}, is it negative? {node_2.check_negative_cycle()}")
#     print(f"Node 3 has a cycle? {node_3._check_cycle()}, is it negative? {node_3.check_negative_cycle()}")
#     print(f"Node 4 has a cycle? {node_4._check_cycle()}, is it negative? {node_4.check_negative_cycle()}")

#     # assert node_1.reaches == {edge_12, edge_23, edge_34} 
#     # assert len(node_1.parents) == 0

#     # assert node_2.reaches == {edge_23, edge_34}
#     # assert node_2.parents == edge_12

#     # assert node_3.reaches == {edge_34}
#     # assert node_3.parents == {edge_23}

#     # assert len(node_4.reaches) == 0
#     # assert node_4.parents == {edge_34}


# @pytest.mark.skip(reason="Not needed for now")
# def test_safe_update_2(monkeypatch):
#     """
#     Test that the safe update function works as expected
#     """
#     # monkeypatch.setattr('builtins.print', lambda x: None)
#     player_1 = Player(1)
#     player_2 = Player(2)
#     node_1 = Node(1, player_1)
#     node_2 = Node(2, player_2)
#     node_3 = Node(3, player_2)
#     node_4 = Node(4, player_1)
#     node_5 = Node(5, player_1)
#     node_6 = Node(6, player_2)
#     edge_12 = Edge(node_1, node_2, 4)
#     edge_23 = Edge(node_2, node_3, -2)
#     edge_34 = Edge(node_3, node_4, 3)
#     edge_53 = Edge(node_5, node_3, 4)
#     edge_46 = Edge(node_4, node_6, -1)
#     edge_65 = Edge(node_6, node_5, 30)

#     node_1.safely_update(edge_12)
#     node_2.safely_update(edge_23)
#     node_3.safely_update(edge_34)
#     node_5.safely_update(edge_53)
#     node_4.safely_update(edge_46)
#     node_6.safely_update(edge_65)

#     print(
#         f"""
#         Final setting is:
#         node_1.reaches = {node_1.reaches}
#         node_1.parents = {node_1.parents}
#         ---
#         node_2.reaches = {node_2.reaches}
#         node_2.parents = {node_2.parents}
#         ---
#         node_3.reaches = {node_3.reaches}
#         node_3.parents = {node_3.parents}
#         ---
#         node_4.reaches = {node_4.reaches}
#         node_4.parents = {node_4.parents}
#         ---
#         node_5.reaches = {node_5.reaches}
#         node_5.parents = {node_5.parents}
#         ---
#         node_6.reaches = {node_6.reaches}
#         node_6.parents = {node_6.parents}
#         """
#     )

# @pytest.mark.skip(reason="Not needed for now")
# def test_safe_update_3(monkeypatch):
#     """
#     Test that the safe update function works as expected.

#     Text maximum recursion error in the case of a non-negative cycle when saving the state
#     """
#     # monkeypatch.setattr('builtins.print', lambda x: None)
#     player_1 = Player(1)
#     player_2 = Player(2)
#     node_1 = Node(1, player_1)
#     node_2 = Node(2, player_2)
#     node_3 = Node(3, player_2)
#     node_4 = Node(4, player_1)
#     node_5 = Node(5, player_1)
#     node_6 = Node(6, player_2)
#     edge_12 = Edge(node_1, node_2, 4)
#     edge_23 = Edge(node_2, node_3, -2)
#     edge_34 = Edge(node_3, node_4, 3)
#     edge_53 = Edge(node_5, node_3, 4)
#     edge_46 = Edge(node_4, node_6, -100)
#     edge_65 = Edge(node_6, node_5, 30)
#     edge_56 = Edge(node_5, node_6, 1)
#     edge_64 = Edge(node_6, node_4, 10)
#     edge_44 = Edge(node_4, node_4, -1)
#     edge_45 = Edge(node_4, node_5, -10)

#     node_1.safely_add_edge(edge_12)
#     node_2.safely_add_edge(edge_23)
#     node_3.safely_add_edge(edge_34)
#     node_5.safely_add_edge(edge_53)
#     node_4.safely_add_edge(edge_46)
#     node_6.safely_add_edge(edge_65)
#     node_5.safely_add_edge(edge_56)
#     node_6.safely_add_edge(edge_64)
#     node_4.safely_add_edge(edge_44)
#     node_4.safely_add_edge(edge_45)
    
#     nodes = {node_1, node_2, node_3, node_4, node_5, node_6}
#     arena = Arena()
#     arena.load_from_nodes(nodes)

#     num_negative_cycles = len(arena.detect_negative_cycles())

#     assert num_negative_cycles == 0, f"Expected 0 negative cycles, got {num_negative_cycles}"

#     print(
#         f"""
#         Final setting is:
#         Arena: {arena}
#         ---
#         node_1.reaches = {node_1.reaches}
#         node_1.parents = {node_1.parents}
#         ---
#         node_2.reaches = {node_2.reaches}
#         node_2.parents = {node_2.parents}
#         ---
#         node_3.reaches = {node_3.reaches}
#         node_3.parents = {node_3.parents}
#         ---
#         node_4.reaches = {node_4.reaches}
#         node_4.parents = {node_4.parents}
#         ---
#         node_5.reaches = {node_5.reaches}
#         node_5.parents = {node_5.parents}
#         ---
#         node_6.reaches = {node_6.reaches}
#         node_6.parents = {node_6.parents}
#         """
#     )

# import json
# def test_serialization():
#     _dict = {0: {(0, 8, -8.63227481561914), (3, 0, 1.7886554660070182), (0, 3, 6.964664132393558), (0, 4, 1.0679336002545714)}, 1: {(0, 8, -8.63227481561914), (1, 8, -0.07505398815246878), (1, 0, -5.46025620618159), (1, 4, 2.066219437274448), (0, 3, 6.964664132393558), (1, 9, 2.084858131547069), (3, 0, 1.7886554660070182), (0, 4, 1.0679336002545714)}, 2: {(2, 0, -0.3513524109322592), (0, 8, -8.63227481561914), (1, 8, -0.07505398815246878), (1, 0, -5.46025620618159), (1, 4, 2.066219437274448), (0, 3, 6.964664132393558), (2, 3, -5.906936957454045), (2, 1, 5.623667152210125), (1, 9, 2.084858131547069), (2, 9, 4.47305245885196), (3, 0, 1.7886554660070182), (0, 4, 1.0679336002545714)}, 3: {(0, 8, -8.63227481561914), (0, 3, 6.964664132393558), (3, 0, 1.7886554660070182), (0, 4, 1.0679336002545714)}} 
#     result = json.loads(json.dumps(_dict, default=set_default))
#     print(result)
#     return result