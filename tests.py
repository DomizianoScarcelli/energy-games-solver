
from Graph import Node, Edge


def test_graph():
    node1 = Node("A")
    node2 = Node("A")
    node2alt = Node("A")
    node3 = Node("C")

    node_set = set()
    node_set.add(node1)
    node_set.add(node2)
    node_set.add(node2alt)
    node_set.add(node3)

    print(node_set)

    edge1 = Edge(node3, node2, 1)
    edge2 = Edge(node2, node3, 2)
    edge3 = Edge(node3, node1, 1)

    edge_set = set()
    edge_set.add(edge1)
    edge_set.add(edge2)
    edge_set.add(edge3)

    print(edge_set)

    edge1alt = Edge(node3, node2, 1)

    print(edge1 == edge1alt)
