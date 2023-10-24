from __future__ import annotations
from typing import List, Set
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.offline import plot

from plot_graph import plot_graph


class Node:
    _nodes = {}

    def __new__(cls, name: int, player: Player):
        if name in cls._nodes:
            return cls._nodes[name]
        else:
            cls._nodes[name] = super(Node, cls).__new__(cls)
            return cls._nodes[name]

    def __init__(self, name: int, player: Player):
        self.name = name
        self.edges: List[Edge] = []
        self.visited = False
        self.player = player

    def get_neighbours(self) -> Set[Node]:
        neigh = set()
        for edge in self.edges:
            neigh.add(edge.node1)
            neigh.add(edge.node2)
        neigh.remove(self)
        return neigh

    def add_edge(self, node):
        self.edges.append(node)

    def set_visited(self):
        self.visited = True

    def __eq__(self, __value: Node) -> bool:
        return self.name == __value.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __str__(self):
        # return f"({self.name})_p{self.player.name}"
        return f"{self.name}"

    def __repr__(self):
        # return f"({self.name})_p{self.player.name}"
        return f"{self.name}"


class Edge:
    _edges = {}

    def __new__(cls, node1: Node, node2: Node, weight: float):
        if (node1, node2) in cls._edges:
            return cls._edges[(node1, node2)]
        else:
            cls._edges[(node1, node2)] = super(Edge, cls).__new__(cls)
            return cls._edges[(node1, node2)]

    def __init__(self, node1: Node, node2: Node, weight: float):
        self.node1 = node1
        self.node2 = node2
        self.weight = weight

    def __eq__(self, __value: Edge) -> bool:
        return self.node1 == __value.node1 and self.node2 == __value.node2

    def __hash__(self) -> int:
        return abs(hash(self.node1) - hash(self.node2))

    def __str__(self) -> str:
        return f"{self.node1} -> {self.node2} ({self.weight})"

    def __repr__(self) -> str:
        return f"{self.node1} -> {self.node2} ({self.weight})"


class Player:
    def __init__(self, name: int):
        self.name = name
        self.energy = 0


class Arena:
    def __init__(self, nodes, edges):
        self.nodes: List[Node] = nodes
        self.edges: List[Edge] = edges
        self.players: List[Player] = [Player(1), Player(2)]

    def __repr__(self):
        return f"{self.nodes}\n{self.edges}"

    def __str__(self):
        return f"{self.nodes}\n{self.edges}"


if __name__ == "__main__":
    pass
