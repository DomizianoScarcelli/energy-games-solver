from __future__ import annotations
from typing import Dict, List, Set
from tqdm import tqdm
import random
import numpy as np
from collections import deque

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
        if self is None or __value is None:
            return False
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
                 seed: int | None = None):
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
                if random.random() < random.uniform(0, self.edge_probability):
                    weight = random.uniform(*self.weight_range)
                    edge = Edge(origin, dest, weight)
                    origin.add_edge(edge)
                    dest.add_edge(edge)
                    edges.append(edge)
                if random.random() < random.uniform(0, self.edge_probability):
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

    def get_edge(self, node1, node2):
        """
        Retrieve an edge between two nodes.
        """
        for edge in self.edges:
            if (edge.node1 == node1 and edge.node2 == node2) or (edge.node1 == node2 and edge.node2 == node1):
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

        weight_sum_player_1 = sum(
            [edge.weight for edge in self.edges if edge.node1.player.name == 1])
        weight_sum_player_2 = sum(
            [edge.weight for edge in self.edges if edge.node1.player.name == 2])
        print(f"Sum of weights for player 1: {weight_sum_player_1}")
        print(f"Sum of weights for player 2: {weight_sum_player_2}")
        print(f"Sum is non negative: {weight_sum}")
        print(
            f"There are {num_positive_weights} positive weights and {num_negative_weights} negative weights")

        num_negative_cycles = len(self._detect_negative_cycles())
        assert weight_sum >= 0, "Graph does not satisfy MeanPayoff− ≥ 0"
        assert num_negative_cycles == 0, f"Graph has {num_negative_cycles} negative cycles"
        # for node in self.nodes:
        #     assert not self._check_and_remove_negative_cycles(node), f"Graph has negative cycles"

    def generate_mean_payoff_arena(self):
        # Remove negative self loops:
        for edge in tqdm(self.edges, desc="Removing negative self loops"):
            if edge.node1 == edge.node2:
                edge.weight = random.uniform(0, self.max_weight)

        def count_negative_cycles():
            number_of_negative_cycles = self._detect_negative_cycles()
            return number_of_negative_cycles

        number_of_negative_cycles = len(count_negative_cycles())

        print(f"There are {number_of_negative_cycles} negative cycles")
        # for node in tqdm(self.nodes, desc="Removing negative cycles"):
        #     # self.remove_negative_cycles(source_node=node,
        #     #                             visited=set([node]),
        #     #                             curr_sum=0)
        #     self.remove_negative_cycles(source_node=node)
        #     number_of_negative_cycles = count_negative_cycles()
        #     print(f"There are {number_of_negative_cycles} negative cycles")
        #     if number_of_negative_cycles == 0:
        #         break

        return self.nodes, self.edges

    # def remove_negative_cycles(self,
    #                            source_node: Node,
    #                            visited: Set[Node] = set(),
    #                            curr_sum: int = 0):
    #     """
    #     Remove all the negative cycles from the graph that starts from the source node
    #     """
    #     successors = source_node.get_neighbours_with_edges()
    #     for succ, edge in successors.items():
    #         updated_sum = curr_sum + edge.weight
    #         if updated_sum < 0:
    #             #TODO: here I can change the logic in order to maintain the negative weight but remove the cycle, for example by just removing the edge that creates the cycle.
    #             edge.weight += abs(updated_sum)

    #             # Remove edge (TODO:)
    #             # if edge in self.edges:
    #             #     self.edges.remove(edge)


    #         if succ in visited:
    #             return self.edges

    #         self.remove_negative_cycles(source_node=succ,
    #                                     visited=visited.union(
    #                                         set([source_node])),
    #                                     curr_sum=updated_sum)

    #     return self
    
    # def remove_negative_cycles(self, source_node: Node):
    #     """
    #     Remove all the negative cycles from the graph that starts from the source node
    #     """
    #     # Step 1: Initialize distances
    #     distance = {node: float('inf') for node in self.nodes}
    #     distance[source_node] = 0
    #     predecessor = {node: None for node in self.nodes}
    #     pbar = tqdm(desc=f"Removing negative cycles from node {source_node}")
    #     # Step 2: Relax edges |V| - 1 times
    #     for _ in range(len(self.nodes) - 1):
    #         for edge in self.edges:
    #             if distance[edge.node1] + edge.weight < distance[edge.node2]:
    #                 distance[edge.node2] = distance[edge.node1] + edge.weight
    #                 predecessor[edge.node2] = edge.node1

    #             pbar.update(1)
    #     # Step 3: Check for negative-weight cycles
    #     for edge in self.edges:
    #         if distance[edge.node1] + edge.weight < distance[edge.node2]:
    #             # We found a negative cycle, now let's find its nodes
    #             cycle = [edge.node2]
    #             while True:
    #                 pbar.update(1)
    #                 cycle.append(predecessor[cycle[-1]])
    #                 if cycle[-1] == edge.node2:
    #                     break

    #             # Step 4: Remove an edge from the cycle
    #             for i in range(len(cycle) - 1):
    #                 pbar.update(1)
    #                 self.edges.remove(Edge(cycle[i], cycle[i + 1]))

    #             break

    #     return self

    def _is_reachable(self, start):
        """
        Check if all nodes are reachable from the start node using BFS.
        """
        visited = {node: False for node in self.nodes}
        queue = deque([start])
        visited[start] = True

        while queue:
            current_node = queue.popleft()
            for edge in self.edges:
                if edge.node1 == current_node and not visited[edge.node2]:
                    visited[edge.node2] = True
                    queue.append(edge.node2)

        return all(visited.values())

    def _detect_negative_cycles(self):
        """
        Detect negative cycles using Bellman-Ford algorithm.
        """
        distances = {node: 0 for node in self.nodes}
        predecessors = {node: None for node in self.nodes}

        # Relax edges repeatedly
        for _ in range(len(self.nodes) - 1):
            for edge in self.edges:
                if distances[edge.node1] + edge.weight < distances.get(edge.node2, float('inf')):
                    distances[edge.node2] = distances[edge.node1] + edge.weight
                    predecessors[edge.node2] = edge.node1

        # Check for negative cycles
        negative_cycles = []
        for edge in self.edges:
            if distances[edge.node1] + edge.weight < distances.get(edge.node2, float('inf')):
                # Negative cycle found
                cycle = [edge.node2]
                node = edge.node1
                while node not in cycle:
                    cycle.append(node)
                    node = predecessors[node]
                cycle.append(node)
                cycle.reverse()
                negative_cycles.append(cycle)

        return negative_cycles

    def _remove_edges_from_cycle(self, cycle):
        """
        Remove edges from the negative cycle to disconnect it.
        """
        visited = set()
        stack = []
        cycles = []

        def dfs(node: Node):
            visited.add(node)
            stack.append(node)
            for neighbor in node.get_neighbours():
                if neighbor not in visited:
                    dfs(neighbor)
                elif neighbor in stack:
                    cycles.append(stack[stack.index(neighbor):])
            stack.pop()

        dfs(cycle[0])

        for cycle in cycles:
            min_weight_edge = None
            min_weight = float('inf')
            for i in range(len(cycle) - 1):
                edge = self.get_edge(cycle[i], cycle[i+1])
                # Handle the case where the edge is not in the graph
                if edge is None:
                    continue
                if edge.weight < min_weight:
                    min_weight = edge.weight
                    min_weight_edge = edge
            if min_weight_edge in self.edges:
                self.edges.remove(min_weight_edge)

    def remove_negative_cycles(self):
        """
        Remove edges from negative cycles to ensure graph connectivity.
        """
        negative_cycles = self._detect_negative_cycles()

        for cycle in negative_cycles:
            start_node = cycle[0]
            if self._is_reachable(start_node):
                self._remove_edges_from_cycle(cycle)

    def get_edge(self, node1, node2):
        """
        Retrieve an edge between two nodes.
        """
        for edge in self.edges:
            if (edge.node1 == node1 and edge.node2 == node2) or (edge.node1 == node2 and edge.node2 == node1):
                return edge
        return None

    # def _check_and_remove_negative_cycles(self, node: Node):
    #     """
    #     Check if the graph has any negative cycle and print the negative cycle path if it exists.
    #     Explaination of Bellman-Ford algorithm: https://www.geeksforgeeks.org/detect-negative-cycle-graph-bellman-ford/
    #     """
    #     distances = {node: 0}
    #     predecessors = {node: None}

    #     # Relax edges repeatedly
    #     for i in range(len(self.nodes) - 1):
    #         for edge in self.edges:
    #             if edge.node1 in distances:
    #                 if edge.node2 not in distances or distances[edge.node2] > distances[edge.node1] + edge.weight:
    #                     distances[edge.node2] = distances[edge.node1] + \
    #                         edge.weight
    #                     predecessors[edge.node2] = edge.node1

    #     # Check for negative cycles
    #     for edge in self.edges:
    #         if edge.node1 in distances and distances[edge.node1] + edge.weight < distances.get(edge.node2, float('inf')):
    #             # Negative cycle found
    #             cycle = [edge.node2]
    #             node = edge.node1
    #             while node not in cycle:
    #                 cycle.append(node)
    #                 node = predecessors[node]
    #             cycle.append(node)
    #             cycle.reverse()
                
    #             self.edges.remove(self.get_edge(cycle[-2], cycle[-1]))

    #             print(
    #             f"Negative cycle found: {cycle} with edges {[self.get_edge(cycle[i], cycle[i+1]) for i in range(len(cycle)-1)]}")
    #             # return True
    #             return False

    #     return False

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


def run_solver(num_nodes: int = 30, edge_probability: float = 0.1):
    graph_generator = GraphGenerator(
        num_nodes=num_nodes, edge_probability=edge_probability)
    nodes, edges = graph_generator.generate_graph()
    arena = Arena(nodes, edges)
    arena.generate_mean_payoff_arena()
    arena.remove_negative_cycles()
    # plot_2D_graph(arena)
    arena.check_arena_conditions()
    arena.value_iteration()
    value_dict = {node: round(node.value, 2) for node in arena.nodes}
    # print(f"Final state: {value_dict}")
    min_energy_dict = arena.get_min_energy()
    return min_energy_dict
    # print(f"Min energy: {min_energy_dict}")

if __name__ == "__main__":
    # solutions = []
    # for i in range(10):
    #     solution = run_solver()
    #     solutions.append(solution)
    # print(solutions)
    solution = run_solver(num_nodes=50)
    print(solution)
