from __future__ import annotations
from typing import Dict, List, Set, Tuple
from tqdm import tqdm
import random
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
        else:
            self.player = player
        # The list of edges regarding the same player that this node reaches.
        self.reaches: Set[Edge] = set()
        # The list of edges regarding the same player that reach this node.
        self.parents: Set[Edge]  = set()

        # A copy of the reaches and parents list before the last update.
        self.backtracking_reaches: Set[Edge] = set()
        self.backtracking_parents: Set[Edge] = set()

    def update(self, edge: Edge):
        """
        Update the `reaches` list of the node with the new edge.
        Also update the reaches list of the self's parent nodes of the same player with the new edge.
        """
        print(f"Updating edge {edge} for node {self}")
        old_reaches = self.reaches.copy()
        self.reaches.add(edge)
        self.reaches = self.reaches.union(edge.node2.reaches)

        old_parents = self.parents.copy()
        dest = edge.node2
        dest.parents.add(edge)

        if self.reaches == old_reaches and self.parents == old_parents:
            # Avoid max recursion depth
            return

        for p_edge in self.parents:
            p_origin = p_edge.node1
            p_origin.update(edge)
        
        print(f"""
        Node: {self}
        Edge: {edge}
        Reaches: {self.reaches}
        Backtracking Reaches: {self.backtracking_reaches}
        Parent: {self.parents}
        Backtracking Parents: {self.backtracking_parents}
              """)

    def backtrack(self, edge: Edge):
        """
        Remove the edge from the `reaches` list of the node.
        Also remove the edge from the reaches list of the self's parent nodes of the same player.
        """
        print(f"[Node {self}]: Backtracking edge {edge}. Reaches is {self.reaches} and parents is {self.parents}")
        if edge not in self.reaches:
            print(f"[Node {self}]: Edge {edge} not in reaches {self.reaches}")
            return
        self.reaches.remove(edge)
        print(f"[Node {self}]: Removed edge {edge} from reaches {self.reaches}")

        #TODO: Remove also all the other edges that are not reachable anymore
        for p_edge in self.parents:
            p_origin = p_edge.node1
            p_origin.backtrack(edge) 
    
    def new_backtrack(self):
        """
        Remove the edge from the `reaches` list of the node.
        Also remove the edge from the reaches list of the self's parent nodes of the same player.
        """
        old_reaches = self.reaches.copy()
        old_parents = self.parents.copy()
        self.reaches = self.backtracking_reaches.copy()
        self.parents = self.backtracking_parents.copy()

        if self.reaches == old_reaches and old_parents == self.parents:
            # Nothing changed, avoid max recursion depth
            print(f"[Node {self}], nothing changed")
            return

        # for edge in self.reaches:
        #     edge.node2.new_backtrack()
        for edge in self.parents:
            edge.node1.new_backtrack()

        

    def _check_cycle(self):
        """
        Returns true if the node is part of a negative cycle.
        A cycle exists if, for each (i, j) in self.reaches, it exists at least one j == self.
        """
        for edge in self.reaches:
            if edge.node2 == self:
                return True
        return False

    def check_negative_cycle(self):
        """
        Returns true if the node is part of a negative cycle.
        """
        if self._check_cycle():
            weight_sum_player_1 = sum(
                [edge.weight for edge in self.reaches if edge.node1.player.name == 1])
            weight_sum_player_2 = sum(
                [edge.weight for edge in self.reaches if edge.node1.player.name == 2])
            
            return weight_sum_player_1 < 0 or weight_sum_player_2 < 0
        return False

    def save_state(self):
        print(f"[Node {self}] saving state")
        self.backtracking_reaches = self.reaches.copy()
        self.backtracking_parents = self.parents.copy()
        for p_edge in self.parents:
            p_origin = p_edge.node1
            p_origin.save_state()

    def safely_update(self, next_edge: Edge):
        """
        Update the node with the next edge and backtrack if a negative cycle is detected.
        """
        print(f"[Node {self}]: Safely update on edge {next_edge}")
        self.save_state()

        self.update(next_edge)
        if self.check_negative_cycle():
            print(f"Negative cycle detected for node {self} with edge {next_edge}")
            self.new_backtrack()
            
        # for d_edge in self.reaches.copy():
        #     d_origin, d_dest = d_edge.node1, d_edge.node2
        #     if d_dest.check_negative_cycle():
        #         d_dest.backtrack(d_edge)

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
        return abs(hash(self.node1) + hash(self.node2))

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

class Arena:
    def __init__(self, num_nodes: float, edge_probability: float = 0.01, seed: int | None = None):
        self.players: List[Player] = [Player(1), Player(2)]
        self.num_nodes = num_nodes
        self.edge_probability = edge_probability
        self.max_weight = 10
        self.weight_range = (-self.max_weight, self.max_weight)
        if seed is not None:
            random.seed(seed)
        self.nodes, self.edges = self.generate()

    def __repr__(self):
        return f"{self.nodes}\n{self.edges}"

    def __str__(self):
        return f"{self.nodes}\n{self.edges}"

    def generate(self):
        nodes: Set[Node] = set()
        for i in range(self.num_nodes):
            nodes.add(Node(name=i, player=Player(random.randint(1, 2))))

        edges: Set[Edge] = set()

        for origin in tqdm(nodes, desc="Creating graph"):
            for dest in nodes:
                if random.random() < random.uniform(0, self.edge_probability):
                    weight = random.uniform(*self.weight_range)
                    edge = Edge(origin, dest, weight)
                    origin.safely_update(edge)

                    origin.add_edge(edge)
                    dest.add_edge(edge)
                    edges.add(edge)
                if random.random() < random.uniform(0, self.edge_probability):
                    weight = random.uniform(*self.weight_range)
                    edge = Edge(dest, origin, weight)
                    dest.safely_update(edge) 

                    dest.add_edge(edge)
                    origin.add_edge(edge)
                    edges.add(edge)
        return nodes, edges

    def get_edge(self, node1, node2):
        """
        Retrieve an edge between two nodes.
        """
        for edge in self.edges:
            if (edge.node1 == node1 and edge.node2 == node2): 
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

        num_negative_cycles = len(self.detect_negative_cycles())
        #TODO: commented by now
        # assert weight_sum >= 0, "Graph does not satisfy MeanPayoff− ≥ 0"
        assert num_negative_cycles == 0, f"Graph has {num_negative_cycles} negative cycles"


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

    def detect_negative_cycles(self):
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
    arena = Arena(num_nodes=num_nodes,
                  edge_probability=edge_probability)
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
    solution = run_solver(num_nodes=10)
    print(solution)
