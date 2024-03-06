from __future__ import annotations
import cProfile
from copy import deepcopy
import pstats
from itertools import product
import json
import math
import time
from pprint import pprint
from typing import Dict, List, Set, Tuple
from tqdm import tqdm
import random
from collections import deque
from plot_graph import plot_graph
import logging
import pickle
import sys
import time

def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError(f"Error in set_default: Object of type {obj.__class__.__name__} is not JSON serializable")

def deserialize_dict(_dict):
    return {int(k): {tuple(e) for e in v} for k, v in _dict.items()}

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

    def __deepcopy__(self, memo):
        return self

class Arena:
    def __init__(self, num_nodes: float = 10, edge_probability: float = 0.01, seed: int | None = None):
        self.nodes = set(range(num_nodes))
        self.players: List[Player] = [Player(1), Player(2)]
        self.player_mapping: Dict[int, Player] = {i: Player(random.randint(1, 2)) for i in range(num_nodes)} 
        self.value_mapping: Dict[int, float] = {i: 0 for i in range(num_nodes)}
        self.edges_mapping: Dict[int, Set[Tuple[int, int, float]]] = {i: set() for i in range(num_nodes)}
        self.edges = set()
        self.num_nodes = num_nodes
        self.edge_probability = edge_probability
        self.max_weight = 10
        self.weight_range = (-self.max_weight, self.max_weight)
        self.backtracking_reaches: Dict[int, Set[Tuple[int, int]]] = {}
        self.backtracking_parents: Dict[int, Set[Tuple[int, int]]] = {}
        self.backtracking_edges: Set[Tuple[int, int]] = set()
        self.backtracking_edges_mapping: Dict[int, Set[Tuple[int, int, float]]] = {}

        self.reaches: Dict[int, Set[Tuple[int, int]]] = {}
        self.parents: Dict[int, Set[Tuple[int, int]]] = {}


        self.considered: Set[Tuple[int, Tuple[int, int, float]]]= set()

        if seed is not None:
            random.seed(seed)

    def __repr__(self):
        return f"Nodes: {self.nodes}\nEdges: {self.edges}"

    def __str__(self):
        return f"Nodes: {self.nodes}\nEdges: {self.edges}"

    # def load_from_nodes(self, nodes: Set[Node]):
    #     self.nodes = nodes
    #     self.edges = set()
    #     for node in nodes:
    #         self.edges = self.edges.union(node.reaches)
    #         self.edges = self.edges.union(node.parents)
    #     for edge in self.edges.copy():
    #         if edge.node1 not in self.nodes:
    #             self.nodes.add(edge.node1)
    #         if edge.node2 not in self.nodes:
    #             self.nodes.add(edge.node2)
                
        # logging.debug("Final edges are: ", self.edges)

    # def load_from_pickle(self, filename: str):
    #     with open(filename, 'rb') as f:
    #         arena = pickle.load(f)
    #     self = arena

    # def save(self):
    #     with open('arena.pkl', 'wb') as f:
    #         pickle.dump(self, f)

    def _deepcopy(self, _dict, strategy: str = "json", loading: bool = False):
        if strategy == "json":
            if loading:
                result = json.loads(_dict)
            else:
                result = json.dumps(_dict, default=set_default)
            return result
        elif strategy == "deepcopy":
            if not loading:
                return deepcopy(_dict)
            return _dict
        else:
            raise NotImplementedError("Strategy not implemented")
        
    def save_state(self):
        self.backtracking_reaches = self._deepcopy(self.reaches, loading=False)
        self.backtracking_parents = self._deepcopy(self.parents, loading=False)
        self.backtracking_edges = self.edges.copy()

    def backtrack(self, message: str = ""):
        self.reaches = deserialize_dict(self._deepcopy(self.backtracking_reaches, loading=True))
        self.parents = deserialize_dict(self._deepcopy(self.backtracking_parents, loading=True))
        self.edges = self.backtracking_edges.copy()
    
    def update_reaches(self, node: int, edge: Tuple[int, int, float]):
        # key = str(node) if isinstance(node, int) else node
        key = node
        if self.reaches.get(key) is None:
            self.reaches[key] = {edge}
        else:
            self.reaches[key].add(edge)

    def update_parents(self, node: int, edge: Tuple[int, int, float]):
        # key = str(node) if isinstance(node, int) else node
        key = node
        if self.parents.get(key) is None:
            self.parents[key] = {edge}
        else:
            self.parents[key].add(edge)

    def update(self, node: int, edge: Tuple[int, int, float]):
        """
        Update the `reaches` list of the node with the new edge.
        Also update the reaches list of the self's parent nodes of the same player with the new edge.
        """
        logging.debug(f"Updating edge {edge} for node {node}")
        if (node, edge) in self.considered:
            # Avoid maximum recursion depth
            return

        self.considered.add((node, edge))
        self.update_reaches(node, edge)
        self.reaches[node] = self.reaches[node].union(self.reaches.get(edge[1], set()))

        dest = edge[1]
        self.update_parents(dest, edge)

        self.edges.add(edge)

        for p_edge in self.parents.get(node, set()):
            p_origin = p_edge[0]
            self.update(p_origin, edge)

    # @timeit 
    def safely_update(self, node: int, edge: Tuple[int, int, float]):
        """
        Update the arena with the new edge and backtrack if a negative cycle is detected.
        """
        self.update(node, edge)
        negative_cycles = self.detect_negative_cycles()
        if negative_cycles > 0:
            self.backtrack(message=f"removing {edge}")
            return False
        self.edges_mapping[node].add(edge)
        return True

    # def safely_add_edge(self, node: int, edge: Tuple[int, int, float]):
    #     # if (edge[0], edge[1]) in {(e[0], e[1]) for e in self.edges}:
    #     #     print(f"HERE")
    #     #     return False
    #     self.safely_update(node, edge)
        
    def generate(self):
        pbar = tqdm(total=self.num_nodes ** 2, desc="Creating graph")
        update_delta = round(math.sqrt(pbar.total))
        for i, (origin, dest) in enumerate(product(self.nodes, repeat=2)):
            if i % update_delta == 0:
                pbar.update(update_delta)
            if random.random() < self.edge_probability:
                weight = random.uniform(*self.weight_range)
                if origin != dest or weight >= 0:
                    edge = (origin, dest, weight)
                    self.save_state()
                    self.safely_update(origin, edge)
        pbar.close()

    # def check_arena_conditions(self):
    #     """
    #     Let us say that a cycle in G is non-negative if the sum of its weights is nonnegative. We consider the following properties:
    #     (i) G satisfies MeanPayoff− ≥ 0.
    #     (ii) All cycles in G are non-negative.
    #     (iii) G satisfies Energy < ∞
    #     """
    #     # Checkign if sum of weights is non negative and if there are both negative and positive weights

    #     weight_sum = sum([edge.weight for edge in self.edges])
    #     num_positive_weights = len(
    #         [edge for edge in self.edges if edge.weight > 0])
    #     num_negative_weights = len(
    #         [edge for edge in self.edges if edge.weight < 0])

    #     weight_sum_player_1 = sum(
    #         [edge.weight for edge in self.edges if edge.node1.player.name == 1])
    #     weight_sum_player_2 = sum(
    #         [edge.weight for edge in self.edges if edge.node1.player.name == 2])
    #     logging.debug(f"Sum of weights for player 1: {weight_sum_player_1}")
    #     logging.debug(f"Sum of weights for player 2: {weight_sum_player_2}")
    #     logging.debug(f"Sum is non negative: {weight_sum}")
    #     logging.debug(
    #         f"There are {num_positive_weights} positive weights and {num_negative_weights} negative weights")

    #     num_negative_cycles = len(self.detect_negative_cycles())
    #     #TODO: commented by now
    #     # assert weight_sum >= 0, "Graph does not satisfy MeanPayoff− ≥ 0"
    #     assert num_negative_cycles == 0, f"Graph has {num_negative_cycles} negative cycles"


    def bellman_ford(self):
        """
        Detect negative cycles using Bellman-Ford algorithm.
        """
        distances = {node: 0 for node in self.nodes}
        predecessors = {node: None for node in self.nodes}

        # Relax edges repeatedly
        for _ in range(len(self.nodes) - 1):
            for edge in self.edges:
                if distances[edge[0]] + edge[2] < distances.get(edge[1], float('inf')):
                    distances[edge[1]] = distances[edge[0]] + edge[2]
                    predecessors[edge[1]] = edge[0]

        # Check for negative cycles
        negative_cycles = []
        for edge in self.edges:
            if distances[edge[0]] + edge[2] < distances.get(edge[1], float('inf')):
                # Negative cycle found
                cycle = [edge[1]]
                node = edge[0]
                while node not in cycle:
                    cycle.append(node)
                    node = predecessors[node]
                cycle.append(node)
                # cycle.reverse() #NOTE: removed for efficiency
                negative_cycles.append(cycle)

        return negative_cycles


    def bool_bellman_ford(self):
        """
        Detect negative cycles using Bellman-Ford algorithm.
        """
        distances = {node: 0 for node in self.nodes}

        # Relax edges repeatedly
        for _ in range(len(self.nodes) - 1):
            for edge in self.edges:
                if distances[edge[0]] + edge[2] < distances.get(edge[1], float('inf')):
                    distances[edge[1]] = distances[edge[0]] + edge[2]

        # Check for negative cycles
        for edge in self.edges:
            if distances[edge[0]] + edge[2] < distances.get(edge[1], float('inf')):
                return True  # Negative cycle found

        return False  # No negative cycle found
    
   

    def detect_negative_cycles(self):
        # Note: this piece of code doesn't allow for ANY cycle, and not only negative cycles, hence I use bellman_ford. 
        # for key, value in self.reaches.items():
        #     for edge in value:
        #         if edge[1] == key:
        #             return [0] #there is a cycle
        # return [] #no negative cycle

        return self.bool_bellman_ford()

    def get_node_neighbours_with_edges(self, node: int) -> Dict[int, Tuple[int, int, float]]:
        """
        Get the neighbours of a node, along with the edge that connects them.
        """
        # if isinstance(node, int):
        #     node = str(node)
        outgoing_edges = self.edges_mapping[node]
        neighbours = {edge[1]: edge for edge in outgoing_edges if edge[1] != node}
        return neighbours

    def value_iteration(self):
        def delta(l, w): return max(l-w, 0)
        def Q(node: int):
            outgoing_edges = self.get_node_neighbours_with_edges(node)
            # print(f"Outgoing edges for node {node} are {outgoing_edges}")
            # for n, e in outgoing_edges.items():
            #     print(f"Value is {self.value_mapping[n]}, weight is {e[2]}, delta is {delta(self.value_mapping[n], e[2])}")
            values = [delta(self.value_mapping[node], edge[2])
                      for node, edge in outgoing_edges.items()]
            if values == []:
                return 0
            if self.player_mapping[node].name == 1:  # max
                return max(values)
            else:  # min
                return min(values)
        converged = {node: False for node in self.nodes}
        threshold = 0.0001
        steps = 0
        max_steps = 10_000
        pbar = tqdm(total=max_steps, desc="Value iteration")
        while not all(converged.values()):
            steps += 1
            if steps > max_steps:
                break
            pbar.update(1)
            for node in self.nodes:
                # print(f"Analyizing node {node} with value {self.value_mapping[node]} and player {self.player_mapping[node].name}")
                old_value = self.value_mapping[node]
                self.value_mapping[node] = Q(node)
                if abs(self.value_mapping[node] - old_value) < threshold:
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
                node for node in self.nodes if self.player_mapping[node] == player]
            # Find the maximum value among the player's nodes
            max_value = max(self.value_mapping[node] for node in player_nodes)

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
    for seed in range(0, 100):
        try:
            start = time.time()
            solution = run_solver(num_nodes=100, edge_probability=0.2, seed=seed)
            end = time.time()
            times.append(end - start)
            logging.info(f"Solution: {solution}")
        except AssertionError as e:
            logging.error(f"Seed {seed} failed with error: {e}")
            break
    avg_time = (sum(times) / len(times)) * 1000
    logging.info(f"Average time: {avg_time:f} ms")

def profile():
    profiler = cProfile.Profile()
    profiler.enable()

    # Run your function
    # Best time: 25sec (num_nodes=100, edge_probability=0.1, seed=3, json deepcopy)
    run_solver(num_nodes=100, edge_probability=0.1, seed=3)

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()

if __name__ == "__main__":
    profile()    
    # run_multiple()
