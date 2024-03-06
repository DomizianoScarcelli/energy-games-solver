from __future__ import annotations
import cProfile
from itertools import product
import json
import math
import time
from pprint import pprint
from typing import Dict, List, Set
from tqdm import tqdm
import random
from collections import deque
from plot_graph import plot_graph
import logging
import pickle
import sys
from copy import deepcopy
import time

#Set this to False to disable debug prints
DEBUG = False
INFO = True
# Set up logging
if DEBUG:
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
if INFO:
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def timeit(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = (end_time - start_time) * 1000
            print(f"Execution time of {func.__name__}: {execution_time} ms")
            return result
        return wrapper
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
        self.value = 0 #TODO: don't remember if this should be initialized to 0 


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

    def __deepcopy__(self, memo):
        # Create a new instance of Node with the same properties
        # return Node(self.name, self.player)
        return self


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
    
    def __deepcopy__(self, memo):
        # Create a new instance of Edge with the same properties
        # return Edge(self.node1, self.node2, self.weight)
        return self


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
    def __init__(self, num_nodes: float = 10, edge_probability: float = 0.01, seed: int | None = None):
        self.players: List[Player] = [Player(1), Player(2)]
        self.num_nodes = num_nodes
        self.edge_probability = edge_probability
        self.max_weight = 10
        self.weight_range = (-self.max_weight, self.max_weight)
        self.backtracking_reaches: Dict[Node, Set[Edge]] = {}
        self.backtracking_parents: Dict[Node, Set[Edge]] = {}
        self.backtracking_edges: Set[Edge] = set()

        self.reaches: Dict[Node, Set[Edge]] = {}
        self.parents: Dict[Node, Set[Edge]] = {}

        if seed is not None:
            random.seed(seed)

    def __repr__(self):
        return f"Nodes: {self.nodes}\nEdges: {self.edges}"

    def __str__(self):
        return f"Nodes: {self.nodes}\nEdges: {self.edges}"

    def load_from_nodes(self, nodes: Set[Node]):
        self.nodes = nodes
        self.edges = set()
        for node in nodes:
            self.edges = self.edges.union(node.reaches)
            self.edges = self.edges.union(node.parents)
        for edge in self.edges.copy():
            if edge.node1 not in self.nodes:
                self.nodes.add(edge.node1)
            if edge.node2 not in self.nodes:
                self.nodes.add(edge.node2)
                
        # logging.debug("Final edges are: ", self.edges)

    def load_from_pickle(self, filename: str):
        with open(filename, 'rb') as f:
            arena = pickle.load(f)
        self = arena

    def save(self):
        with open('arena.pkl', 'wb') as f:
            pickle.dump(self, f)

    def _deepcopy(self, _dict, strategy: str = "deepcopy"):
        
        if strategy == "json":
            def set_default(obj):
                if isinstance(obj, set):
                    return list(obj)
                raise TypeError

            result = json.dumps(_dict, default=set_default)
            return result
        elif strategy == "deepcopy":
            return deepcopy(_dict)
        else:
            return NotImplementedError("Strategy not implemented")
        
    def save_state(self):
        self.backtracking_reaches = self._deepcopy(self.reaches)
        self.backtracking_parents = self._deepcopy(self.parents)
        self.backtracking_edges = self._deepcopy(self.edges)

    def backtrack(self, message: str = ""):
        self.reaches = self._deepcopy(self.backtracking_reaches)
        self.parents = self._deepcopy(self.backtracking_parents)
        self.edges = self._deepcopy(self.backtracking_edges)
    
    def update_reaches(self, node: Node, edge: Edge):
        if self.reaches.get(node) is None:
            self.reaches[node] = {edge}
        else:
            self.reaches[node].add(edge)

    def update_parents(self, node: Node, edge: Edge):
        if self.parents.get(node) is None:
            self.parents[node] = {edge}
        else:
            self.parents[node].add(edge)

    def update(self, node: Node, edge: Edge):
        """
        Update the `reaches` list of the node with the new edge.
        Also update the reaches list of the self's parent nodes of the same player with the new edge.
        """
        logging.debug(f"Updating edge {edge} for node {self}")
        old_reaches = self._deepcopy(self.reaches)
        self.update_reaches(node, edge)
        self.reaches[node] = self.reaches[node].union(self.reaches.get(edge.node2, set()))

        old_parents = self._deepcopy(self.parents)
        dest = edge.node2
        self.update_parents(dest, edge)

        if self.reaches == old_reaches and self.parents == old_parents:
            # Avoid max recursion depth
            return

        self.edges.add(edge)

        for p_edge in self.parents.get(node, set()):
            p_origin = p_edge.node1
            self.update(p_origin, edge)

    def print_state(self, message):
        negative_cycles = self.detect_negative_cycles()
        logging.debug("-"*25 + message + "-"*25)
        logging.debug(f"Negative cycles: {len(negative_cycles)}")
        logging.debug(f"Arena: {self}")
        logging.debug(f"Reaches:")
        logging.debug(self.reaches)
        logging.debug(f"Parents:")
        logging.debug(self.parents)
        for node in self.nodes:
            logging.debug(f"   Node {node}: edges {node.edges}")

    # @timeit 
    def safely_update(self, node: Node, edge: Edge):
        """
        Update the arena with the new edge and backtrack if a negative cycle is detected.
        """
        self.update(node, edge)
        negative_cycles = self.detect_negative_cycles()
        if len(negative_cycles) > 0:
            self.backtrack(message=f"removing {edge}")
            return False
        node.add_edge(edge)
        return True

    def safely_add_edge(self, node: Node, edge: Edge):
        if edge in self.edges:
            return
        if self.safely_update(node, edge):
            # assert len(self.detect_negative_cycles()) == 0, f"Negative cycles after adding edge {edge}. Expected 0"
            return
        return

    def generate(self):
        self.nodes: Set[Node] = {Node(name=i, player=Player(random.randint(1, 2))) for i in range(self.num_nodes)}
        self.edges: Set[Edge] = set()

        pbar = tqdm(total=self.num_nodes ** 2, desc="Creating graph")
        update_delta = round(math.sqrt(pbar.total))
        for i, (origin, dest) in enumerate(product(self.nodes, repeat=2)):
            if i % update_delta == 0:
                pbar.update(update_delta)
            if random.random() < self.edge_probability:
                weight = random.uniform(*self.weight_range)
                if origin != dest or weight >= 0:
                    edge = Edge(origin, dest, weight)
                    self.save_state()
                    self.safely_add_edge(origin, edge)
                    self.edges.add(edge)
        pbar.close()

    # def generate(self):
    #     self.nodes: Set[Node] = set()
    #     self.edges: Set[Edge] = set()
    #     for i in range(self.num_nodes):
    #         self.nodes.add(Node(name=i, player=Player(random.randint(1, 2))))

    #     pbar = tqdm(total=self.num_nodes ** 2, desc="Creating graph")
    #     for origin in self.nodes:
    #         for dest in self.nodes:
    #             pbar.update(1)
    #             if random.random() < self.edge_probability:
    #                 # weight = round(random.uniform(*self.weight_range))
    #                 weight = random.uniform(*self.weight_range)
    #                 if origin == dest and weight < 0:
    #                     continue
    #                 edge = Edge(origin, dest, weight)
    #                 self.save_state()
    #                 self.safely_add_edge(origin, edge)

    #                 for node in self.nodes:
    #                     self.edges = self.edges.union(self.reaches.get(node, set()))
    #                     self.edges = self.edges.union(self.parents.get(node, set()))
                
    #             logging.debug(f"[generate] Edges are {self.edges}")

     

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
        logging.debug(f"Sum of weights for player 1: {weight_sum_player_1}")
        logging.debug(f"Sum of weights for player 2: {weight_sum_player_2}")
        logging.debug(f"Sum is non negative: {weight_sum}")
        logging.debug(
            f"There are {num_positive_weights} positive weights and {num_negative_weights} negative weights")

        num_negative_cycles = len(self.detect_negative_cycles())
        #TODO: commented by now
        # assert weight_sum >= 0, "Graph does not satisfy MeanPayoff− ≥ 0"
        assert num_negative_cycles == 0, f"Graph has {num_negative_cycles} negative cycles"


    # def _is_reachable(self, start):
    #     """
    #     Check if all nodes are reachable from the start node using BFS.
    #     """
    #     visited = {node: False for node in self.nodes}
    #     queue = deque([start])
    #     visited[start] = True
    #     while queue:
    #         current_node = queue.popleft()
    #         for edge in self.edges:
    #             if edge.node1 == current_node and not visited[edge.node2]:
    #                 visited[edge.node2] = True
    #                 queue.append(edge.node2)

    #     return all(visited.values())

    # def detect_negative_cycles(self):
    #     """
    #     Detect negative cycles using Bellman-Ford algorithm.
    #     """
    #     distances = {node: 0 for node in self.nodes}
    #     predecessors = {node: None for node in self.nodes}

    #     # Relax edges repeatedly
    #     for _ in range(len(self.nodes) - 1):
    #         for edge in self.edges:
    #             if distances[edge.node1] + edge.weight < distances.get(edge.node2, float('inf')):
    #                 distances[edge.node2] = distances[edge.node1] + edge.weight
    #                 predecessors[edge.node2] = edge.node1

    #     # Check for negative cycles
    #     negative_cycles = []
    #     for edge in self.edges:
    #         if distances[edge.node1] + edge.weight < distances.get(edge.node2, float('inf')):
    #             # Negative cycle found
    #             cycle = [edge.node2]
    #             node = edge.node1
    #             while node not in cycle:
    #                 cycle.append(node)
    #                 node = predecessors[node]
    #             cycle.append(node)
    #             cycle.reverse()
    #             negative_cycles.append(cycle)

    #     return negative_cycles

    def detect_negative_cycles(self):
        for key, value in self.reaches.items():
            for edge in value:
                if edge.node2 == key:
                    return [0]
        return []

   
    def value_iteration(self):
        def delta(l, w): return max(l-w, 0)

        def Q(node: Node):
            outgoing_edges = node.get_neighbours_with_edges()
            print(f"Outgoing edges for node {node} are {outgoing_edges}")
            # for n, e in outgoing_edges.items():
            #     print(f"Value is {n.value}, weight is {e.weight}, delta is {delta(n.value, e.weight)}")
            values = [delta(node.value, edge.weight)
                      for node, edge in outgoing_edges.items()]
            if values == []:
                return 0
            # logging.debug(f"Q({node}) = {values}")

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
                print(f"Analyzing node {node} with value {node.value} and player {node.player.name}")
                old_value = node.value
                node.value = Q(node)
                if abs(node.value - old_value) < threshold:
                    converged[node] = True
        pbar.close()
        if steps > max_steps:
            logging.info(f"Did not converge after {steps} steps")
        else:
            logging.info(f"Converged after {steps} steps")
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
            max_value = max((node.value for node in player_nodes))

            min_energy_dict[player.name] = round(max_value, round_to)
        return min_energy_dict


def run_solver(num_nodes: int = 30, edge_probability: float = 0.1, seed: int | None = None):
    arena = Arena(num_nodes=num_nodes,
                  edge_probability=edge_probability, 
                  seed=seed) 
    arena.generate()
    # arena.check_arena_conditions()
    plot_graph(arena)
    arena.value_iteration()
    # value_dict = {node: round(node.value, 2) for node in arena.nodes}
    min_energy_dict = arena.get_min_energy()
    return min_energy_dict

def run_multiple():
    times = []
    for seed in range(0, 1):
        try:
            start = time.time()
            solution = run_solver(num_nodes=10, edge_probability=0.3, seed=seed)
            end = time.time()
            times.append(end - start)
            logging.info(f"Solution: {solution}")
        except AssertionError as e:
            logging.error(f"Seed {seed} failed with error: {e}")
            break
    avg_time = (sum(times) / len(times)) * 1000
    logging.info(f"Average time: {avg_time:f} ms")

def profile():
    import cProfile
    import pstats

    profiler = cProfile.Profile()
    profiler.enable()

    # Run your function
    run_solver(num_nodes=200, edge_probability=0.01)

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()

if __name__ == "__main__":
    run_multiple()
