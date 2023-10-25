from __future__ import annotations
from typing import Dict, List, Set
from tqdm import tqdm
import random
import numpy as np

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
        self.visited: bool = False
        self.player = player
        self.value: int = 0

    def get_neighbours(self) -> Set[Node]:
        neigh = set()
        for edge in self.edges:
            neigh.add(edge.node1)
            neigh.add(edge.node2)
        if self in neigh:
            neigh.remove(self)
        return neigh

    def get_neighbours_with_edges(self) -> Dict[Node, Edge]:
        neigh: Dict[Node, Edge] = {}
        neighbours = self.get_neighbours()
        for edge in self.edges:
            if edge.node1 == self and edge.node2 in neighbours:
                neigh[edge.node2] = edge
        return neigh

    def add_edge(self, node: Node):
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
    _players = {}

    def __new__(self, name: int, energy: float = 0):
        if name in self._players:
            return self._players[name]
        else:
            self._players[name] = super(Player, self).__new__(self)
            return self._players[name]

    def __init__(self, name: int, energy: float = 0):
        self.name = name
        self.energy = 0

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return f"Player {self.name}, energy: {self.energy}"

    def __repr__(self):
        return f"Player {self.name}, energy: {self.energy}"


class GraphGenerator:
    def __init__(self, num_nodes, edge_probability=0.01):
        self.num_nodes = num_nodes
        self.edge_probability = edge_probability
        self.max_weight = 10
        self.weight_range = (-self.max_weight, self.max_weight)

    def generate_graph(self):
        nodes: Set[Node] = set()
        while len(nodes) < self.num_nodes:
            nodes.add(Node(random.randint(1, self.num_nodes),
                      Player(random.randint(1, 2))))
        nodes: List[Node] = list(nodes)
        edges = []

        for i in tqdm(range(self.num_nodes), desc="Generating edges"):
            for j in range(i + 1, self.num_nodes):
                # Generate a random number from a uniform distribution and check if it's less than the edge probability
                if random.random() < self.edge_probability and len(edges) < self.num_nodes - 1:
                    # Adjust the weight range as needed
                    weight = random.uniform(*self.weight_range)
                    edge = Edge(nodes[i], nodes[j], weight)
                    nodes[i].add_edge(edge)
                    nodes[j].add_edge(edge)
                    edges.append(edge)

        return nodes, edges


class Arena:
    def __init__(self, nodes, edges):
        self.nodes: List[Node] = nodes
        self.edges: List[Edge] = edges
        self.players: List[Player] = [Player(1), Player(2)]

    def __repr__(self):
        return f"{self.nodes}\n{self.edges}"

    def __str__(self):
        return f"{self.nodes}\n{self.edges}"

    def find_min_energy(self):
        """
        For each player, finds the minimum amount of energy needed in order to keep it positive for an infinite amount of steps.
        (Energy is lost when a player moves to a node whose edge has negative weight, and gained when it moves to a node whose edge has a positive weight.)
        """
        pass

    def value_iteration(self):

        def delta(l, w): return max(l-w, 0)

        def Q(node: Node):
            outgoing_edges = node.get_neighbours_with_edges()
            print(f"Node {node} has outgoing edges {outgoing_edges}")
            values = [delta(node.value, edge.weight)
                      for node, edge in outgoing_edges.items()]
            if values == []:
                return 0
            print(f"Q({node}) = {values}")

            if node.player.name == 1:  # max
                return max(values)
            else:  # min
                return min(values)

        converged = {node: False for node in self.nodes}
        print(f"Converged dict is: {converged}")
        threshold = 0.0001
        steps = 0
        while not all(converged.values()):
            steps += 1
            for node in self.nodes:
                old_value = node.value
                node.value = Q(node)
                print(
                    f"At step {steps}, node {node} has old value {old_value} and new value {node.value}")
                if abs(node.value - old_value) < threshold:
                    converged[node] = True
        print(f"Converged after {steps} steps")
        return self.nodes


if __name__ == "__main__":
    graph_generator = GraphGenerator(10, 1)
    nodes, edges = graph_generator.generate_graph()
    arena = Arena(nodes, edges)
    plot_graph(arena)
    arena.value_iteration()
    value_dict = {node: node.value for node in arena.nodes}
    print(f"Final state: {value_dict}")
