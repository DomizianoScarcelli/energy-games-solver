from __future__ import annotations
from typing import Dict, List, Set
from tqdm import tqdm
import random
import numpy as np

from plot_graph import plot_graph, plot_3D_graph, plot_2D_graph


class Node:
    _nodes = {}

    def __new__(cls, name: int, player: Player = None):
        if name in cls._nodes:
            return cls._nodes[name]
        else:
            cls._nodes[name] = super(Node, cls).__new__(cls)
            return cls._nodes[name]

    def __init__(self, name: int, player: Player = None):
        self.name = name
        self.edges: List[Edge] = []
        self.visited: bool = False
        if player is None:
            self.player = Player(1)
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
        return f"{self.name}"

    def __repr__(self):
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
    def __init__(self, num_nodes: int,
                 edge_probability: float = 0.01,
                 seed: int or None = None):
        self.num_nodes = num_nodes
        self.edge_probability = edge_probability
        self.max_weight = 10
        self.weight_range = (-self.max_weight, self.max_weight)
        if seed is not None:
            random.seed(seed)

    def generate_graph(self):
        nodes: Set[Node] = set()
        while len(nodes) < self.num_nodes:
            nodes.add(Node(random.randint(1, self.num_nodes),
                      Player(random.randint(1, 2))))
        nodes: List[Node] = list(nodes)

        edges = []

        for origin in tqdm(nodes, desc="Creating graph"):
            for dest in nodes:
                if random.random() < self.edge_probability:
                    weight = random.uniform(*self.weight_range)
                    edge = Edge(origin, dest, weight)
                    origin.add_edge(edge)
                    dest.add_edge(edge)
                    edges.append(edge)
                if random.random() < self.edge_probability:
                    weight = random.uniform(*self.weight_range)
                    edge = Edge(dest, origin, weight)
                    origin.add_edge(edge)
                    dest.add_edge(edge)
                    edges.append(edge)

        return nodes, edges


class Arena:
    def __init__(self, nodes: List[Node], edges: List[Edge]):
        self.nodes: List[Node] = nodes
        self.edges: List[Edge] = edges
        self.players: List[Player] = [Player(1), Player(2)]
        self.max_weight = 10

    def __repr__(self):
        return f"{self.nodes}\n{self.edges}"

    def __str__(self):
        return f"{self.nodes}\n{self.edges}"

    def get_edge(self, node1: Node, node2: Node) -> Edge:
        for edge in self.edges:
            if edge.node1 == node1 and edge.node2 == node2:
                return edge
        return None

    def check_arena_conditions(self):
        """
        Let us say that a cycle in G is non-negative if the sum of its weights is nonnegative. We consider the following properties:
        (i) G satisfies MeanPayoff− ≥ 0.
        (ii) All cycles in G are non-negative.
        (iii) G satisfies Energy < ∞
        """
        # Checkign if sum of weights is non negative and if there are both negative and positive weights

        weight_sum = sum([edge.weight for edge in self.edges])
        num_positive_weights = len(
            [edge for edge in self.edges if edge.weight > 0])
        num_negative_weights = len(
            [edge for edge in self.edges if edge.weight < 0])
        print(f"Sum is non negative: {weight_sum}")
        print(
            f"There are {num_positive_weights} positive weights and {num_negative_weights} negative weights")

        for node in self.nodes:
            assert not self._check_negative_cycles(node)

    def generate_mean_payoff_arena(self):
        # Remove negative self loops:
        for edge in tqdm(self.edges, desc="Removing negative self loops"):
            if edge.node1 == edge.node2:
                edge.weight = random.uniform(0, self.max_weight)
        # while any(self._check_negative_cycles(node) for node in self.nodes):

        def count_negative_cycles():
            number_of_negative_cycles = 0
            for node in tqdm(self.nodes, desc="Checking negative cycles"):
                negative_cycles = self._check_negative_cycles(node)
                if negative_cycles:
                    number_of_negative_cycles += 1
            return number_of_negative_cycles

        number_of_negative_cycles = count_negative_cycles()

        print(f"There are {number_of_negative_cycles} negative cycles")
        for node in tqdm(self.nodes, desc="Removing negative cycles"):
            self.remove_negative_cycles(source_node=node,
                                        visited=set([node]),
                                        curr_sum=0)
            number_of_negative_cycles = count_negative_cycles()
            print(f"There are {number_of_negative_cycles} negative cycles")
            if number_of_negative_cycles == 0:
                break

        return self.nodes, self.edges

    def remove_negative_cycles(self,
                               source_node: Node,
                               visited: Set[Node] = set(),
                               curr_sum: int = 0):
        """
        Remove all the negative cycles from the graph from the source node
        """
        successors = source_node.get_neighbours_with_edges()
        for succ, edge in successors.items():
            updated_sum = curr_sum + edge.weight
            if updated_sum < 0:
                edge.weight += abs(updated_sum)

            if succ in visited:
                return self.edges

            self.remove_negative_cycles(source_node=succ,
                                        visited=visited.union(
                                            set([source_node])),
                                        curr_sum=updated_sum)

        return self

    def _check_negative_cycles(self, node: Node):
        """
        Check if the graph has any negative cycle and print the negative cycle path if it exists.
        """
        distances = {node: 0}
        predecessors = {node: None}

        # Relax edges repeatedly
        for i in range(len(self.nodes) - 1):
            for edge in self.edges:
                if edge.node1 in distances:
                    if edge.node2 not in distances or distances[edge.node2] > distances[edge.node1] + edge.weight:
                        distances[edge.node2] = distances[edge.node1] + \
                            edge.weight
                        predecessors[edge.node2] = edge.node1

        # Check for negative cycles
        for edge in self.edges:
            if edge.node1 in distances and distances[edge.node1] + edge.weight < distances.get(edge.node2, float('inf')):
                # Negative cycle found
                cycle = [edge.node2]
                node = edge.node1
                while node not in cycle:
                    cycle.append(node)
                    node = predecessors[node]
                cycle.append(node)
                cycle.reverse()
                # print(
                # f"Negative cycle found: {cycle} with edges {[self.get_edge(cycle[i], cycle[i+1]) for i in range(len(cycle)-1)]}")
                return True

        return False

    def value_iteration(self):

        def delta(l, w): return max(l-w, 0)

        def Q(node: Node):
            outgoing_edges = node.get_neighbours_with_edges()
            # print(f"Node {node} has outgoing edges {outgoing_edges}")
            values = [delta(node.value, edge.weight)
                      for node, edge in outgoing_edges.items()]
            if values == []:
                return 0
            # print(f"Q({node}) = {values}")

            if node.player.name == 1:  # max
                return max(values)
            else:  # min
                return min(values)

        converged = {node: False for node in self.nodes}
        threshold = 0.0001
        steps = 0
        max_steps = 50_000
        pbar = tqdm(total=max_steps, desc="Value iteration")
        while not all(converged.values()):
            steps += 1
            if steps > max_steps:
                break
            pbar.update(1)
            for node in self.nodes:
                old_value = node.value
                node.value = Q(node)
                if abs(node.value - old_value) < threshold:
                    converged[node] = True
        pbar.close()
        if steps > max_steps:
            print(f"Did not converge after {steps} steps")
        else:
            print(f"Converged after {steps} steps")
        return self.nodes

    def get_min_energy(self, round_to: int = 2):
        """
        For each player, finds the minimum amount of energy needed in order to keep it positive for an infinite amount of steps.
        (Energy is lost when a player moves to a node whose edge has negative weight, and gained when it moves to a node whose edge has a positive weight.)
        """
        min_energy_dict = {}
        for player in self.players:
            # Get the nodes owned by the player
            player_nodes = [
                node for node in self.nodes if node.player == player]

            # Find the maximum value among the player's nodes
            max_value = max(node.value for node in player_nodes)

            min_energy_dict[player.name] = round(max_value, round_to)
        return min_energy_dict


if __name__ == "__main__":
    graph_generator = GraphGenerator(
        num_nodes=100, edge_probability=0.01)
    nodes, edges = graph_generator.generate_graph()
    arena = Arena(nodes, edges)
    arena.generate_mean_payoff_arena()
    plot_2D_graph(arena)
    arena.check_arena_conditions()
    arena.value_iteration()
    value_dict = {node: round(node.value, 2) for node in arena.nodes}
    print(f"Final state: {value_dict}")
    min_energy_dict = arena.get_min_energy()
    print(f"Min energy: {min_energy_dict}")
