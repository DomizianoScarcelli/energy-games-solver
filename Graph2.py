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
    # if isinstance(obj, Node):
    #     return (obj.name, obj.player.name)
    raise TypeError(f"Error in set_default: Object of type {obj.__class__.__name__} is not JSON serializable")


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

    def __new__(cls, name: int, player: Player = None, value: float = 0.0):
        if name in cls._nodes:
            return cls._nodes[name]
        else:
            cls._nodes[name] = super(Node, cls).__new__(cls)
            return cls._nodes[name]

    def __init__(self, name: int, player: Player = None, value: float = 0.0):
        self.name = name
        self.edges: List[Tuple[int, int, float]] = []
        self.visited: bool = False
        self.value = value
        if player is None:
            self.player = Player(1)
        else:
            self.player = player


    # def get_neighbours(self) -> Set[int]:
    #     neigh = set()
    #     #TODO: edges is empty
    #     for edge in self.edges:
    #         neigh.add(edge[0])
    #         neigh.add(edge[1])
    #     if self in neigh:
    #         neigh.remove(self)
    #     return neigh

    # def get_neighbours_with_edges(self, player_mapping, value_mapping) -> Dict[Node, Tuple[int, int, float]]:
    #     neigh: Dict[Node, Tuple[int, int, float]] = {}
    #     neighbours = self.get_neighbours()
    #     for edge in self.edges:
    #         if edge[0] == self.name and edge[1] in neighbours:
    #             neigh[edge[0]] = edge
    #     return {Node(name=name, player=player_mapping[name], value=value_mapping[name]): edge for name, edge in neigh.items()} 

    # def add_edge(self, edge: Tuple[int, int, float]):
    #     self.edges.append(edge)

    # def set_visited(self):
    #     self.visited = True

    def __eq__(self, __value: Node) -> bool:
        left, right = None, None
        if isinstance(self, str):
            left = self.split("-")[0]
        if isinstance(__value, str):
            right = __value.split("-")[0]
        if isinstance(self, Node):
            left = self.name
        if isinstance(__value, Node):
            right = __value.name
        if isinstance(self, int):
            left = self
        if isinstance(__value, int):
            right = __value
        if left is not None and right is not None:
            return left == right

    def __hash__(self) -> int:
        return hash(self.name)

    def __str__(self):
        return f"N({self.name, self.player.name, self.value})"

    def __repr__(self):
        return f"N({self.name, self.player.name, self.value})"

    def __deepcopy__(self, memo):
        # Create a new instance of Node with the same properties
        # return Node(self.name, self.player)
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
        self.player_mapping: Dict[int, Player] = dict()
        self.value_mapping: Dict[int, float] = dict()
        self.edges_mapping: Dict[int, Set[Tuple[int, int, float]]] = dict()
        self.num_nodes = num_nodes
        self.edge_probability = edge_probability
        self.max_weight = 10
        self.weight_range = (-self.max_weight, self.max_weight)
        self.backtracking_reaches: Dict[int, Set[Tuple[int, int]]] = {}
        self.backtracking_parents: Dict[int, Set[Tuple[int, int]]] = {}
        self.backtracking_edges: Set[Tuple[int, int]] = set()

        self.reaches: Dict[int, Set[Tuple[int, int]]] = {}
        self.parents: Dict[int, Set[Tuple[int, int]]] = {}

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

    def _deepcopy(self, _dict, strategy: str = "json"):
        if strategy == "json":
            result = json.loads(json.dumps(_dict, default=set_default))
            return result
        elif strategy == "deepcopy":
            return deepcopy(_dict)
        else:
            raise NotImplementedError("Strategy not implemented")
        
    def save_state(self):
        self.backtracking_reaches = self._deepcopy(self.reaches)
        self.backtracking_parents = self._deepcopy(self.parents)
        self.backtracking_edges = self.edges.copy()

    def backtrack(self, message: str = ""):
        self.reaches = self._deepcopy(self.backtracking_reaches)
        self.parents = self._deepcopy(self.backtracking_parents)
        self.edges = self.backtracking_edges.copy()
    
    def update_reaches(self, node: Node, edge: Tuple[int, int, float]):
        name = node.name if isinstance(node, Node) else node
        if self.reaches.get(name) is None:
            self.reaches[name] = {edge}
        else:
            self.reaches[name].add(edge)

    def update_parents(self, node: Node, edge: Tuple[int, int, float]):
        name = node.name if isinstance(node, Node) else node
        if self.parents.get(name) is None:
            self.parents[name] = {edge}
        else:
            self.parents[name].add(edge)

    def update(self, node: Node, edge: Tuple[int, int, float]):
        """
        Update the `reaches` list of the node with the new edge.
        Also update the reaches list of the self's parent nodes of the same player with the new edge.
        """
        logging.debug(f"Updating edge {edge} for node {self}")
        old_reaches = self._deepcopy(self.reaches)
        self.update_reaches(node, edge)
        name = node.name if isinstance(node, Node) else node
        self.reaches[name] = self.reaches[name].union(self.reaches.get(Node(edge[1]), set()))

        old_parents = self._deepcopy(self.parents)
        dest = edge[1]
        self.update_parents(dest, edge)

        if self._deepcopy(self.reaches) == old_reaches and self._deepcopy(self.parents) == old_parents:
            # Avoid max recursion depth
            return

        self.edges.add(edge)

        for p_edge in self.parents.get(name, set()):
            p_origin = p_edge[0]
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
    def safely_update(self, node: Node, edge: Tuple[int, int, float]):
        """
        Update the arena with the new edge and backtrack if a negative cycle is detected.
        """
        self.update(node, edge)
        negative_cycles = self.detect_negative_cycles()
        if len(negative_cycles) > 0:
            self.backtrack(message=f"removing {edge}")
            return False
        self.edges_mapping[node.name].add(edge)
        return True

    def safely_add_edge(self, node: Node, edge: Tuple[int, int, float]):
        if edge in self.edges:
            return False
        return self.safely_update(node, edge)
        

    def generate(self):
        self.nodes: Set[Node] = {Node(name=i, player=Player(random.randint(1, 2))) for i in range(self.num_nodes)}
        self.player_mapping = {node.name: node.player for node in self.nodes}
        self.value_mapping = {node.name: 0 for node in self.nodes}
        self.edges_mapping = {node.name: set() for node in self.nodes}
        self.edges: Set[Tuple[int, int,float]] = set()

        pbar = tqdm(total=self.num_nodes ** 2, desc="Creating graph")
        update_delta = round(math.sqrt(pbar.total))
        for i, (origin, dest) in enumerate(product(self.nodes, repeat=2)):
            if i % update_delta == 0:
                pbar.update(update_delta)
            if random.random() < self.edge_probability:
                weight = random.uniform(*self.weight_range)
                if origin != dest or weight >= 0:
                    edge = (origin.name, dest.name, weight)
                    self.save_state()
                    self.safely_add_edge(origin, edge)
                    self.edges.add(edge)
        pbar.close()

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

    # def bellman_ford(self):
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
                if edge[1] == key:
                    return [0] #there is a cycle
        return [] #no negative cycle

    def get_node_neighbours_with_edges(self, node: Node) -> Dict[Node, Tuple[int, int, float]]:
        """
        Get the neighbours of a node, along with the edge that connects them.
        """
        outgoing_edges = self.edges_mapping[node.name]
        neighbours = {edge[1]: edge for edge in outgoing_edges if edge[1] != node.name}
        return neighbours

    def value_iteration(self):
        def delta(l, w): return max(l-w, 0)
        def Q(node: Node):
            outgoing_edges = self.get_node_neighbours_with_edges(node)
            for node, edge in outgoing_edges.items():
                print(f"Value is {self.value_mapping[node]}, weight is {edge[2]}, delta is {delta(self.value_mapping[node], edge[2])}")
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
        max_steps = 50_000
        pbar = tqdm(total=max_steps, desc="Value iteration")
        while not all(converged.values()):
            steps += 1
            if steps > max_steps:
                break
            pbar.update(1)
            for node in self.nodes:
                old_value = self.value_mapping[node.name]
                self.value_mapping[node.name] = Q(node)
                if abs(self.value_mapping[node.name] - old_value) < threshold:
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
        self.nodes = [Node(node.name, self.player_mapping[node.name]) for node in self.nodes]
        for player in self.players:
            # Get the nodes owned by the player
            player_nodes = [
                node for node in self.nodes if node.player == player]
            if player_nodes == []:
                raise ValueError(f"No nodes found for player {player.name}. The nodes are {dict({node: node.player for node in self.nodes})}")
            # Find the maximum value among the player's nodes
            max_value = max(self.value_mapping[node.name] for node in player_nodes)

            min_energy_dict[player.name] = round(max_value, round_to)
        return min_energy_dict


def run_solver(num_nodes: int = 30, edge_probability: float = 0.1, seed: int | None = None):
    arena = Arena(num_nodes=num_nodes,
                  edge_probability=edge_probability, 
                  seed=seed) 
    arena.generate()
    # arena.check_arena_conditions()
    # plot_graph(arena)
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
    profiler = cProfile.Profile()
    profiler.enable()

    # Run your function
    run_solver(num_nodes=100, edge_probability=0.2)

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()

if __name__ == "__main__":
    # profile()    
    run_multiple()
