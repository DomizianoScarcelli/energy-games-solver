from __future__ import annotations
from enum import Enum
from itertools import product
import math
from typing import Dict, Set, Tuple, Optional
from tqdm import tqdm
import random
import logging
import pickle
import sys

#Set this to False to disable debug prints
DEBUG = False
INFO = True
# Set up logging
if DEBUG:
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
if INFO:
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

class Player(Enum):
    MIN = 1
    MAX = 2

class GenerationStrategy(Enum):
    BELLMAN_FORD = "bellman_ford"
    INCREMENTAL_BELLMAN_FORD = "incremental_bellman_ford"
    NONE = 'none'

class Arena:
    def __init__(self, 
                 num_nodes: int = 10, 
                 edge_probability: float = 0.01, 
                 seed: int | None = None):
        self.nodes = set(range(num_nodes))
        self.random = random.Random(seed)
        self.value_mapping: Dict[int, float] = {i: 0 for i in range(num_nodes)}
        self.edges_mapping: Dict[int, Set[Tuple[int, int, float]]] = {i: set() for i in range(num_nodes)}
        self.edges = set()
        self.num_nodes = num_nodes
        self.edge_probability = edge_probability
        self.max_weight = 10
        self.weight_range = (-self.max_weight, self.max_weight)
        self.distances: Dict[int, Optional[float]] = {node: 0 for node in self.nodes}
        self.considered: Set[Tuple[int, Tuple[int, int, float]]]= set()
        self.considered_edges_bf: Set[Tuple[int, int, float]] = set()
        self.fast_edges: Dict[int, Dict[int, float]] = {i: {} for i in range(num_nodes)}
        self.player_mapping: Dict[int, Player] = self._assign_players(equal=True) 
        self.ingoing_edges: Dict[int, Set[Tuple[int, int, float]]] = {i: set() for i in range(num_nodes)}

    def __repr__(self):
        return f"Nodes: {self.nodes}\nEdges: {self.edges}"

    def __str__(self):
        return f"Nodes: {self.nodes}\nEdges: {self.edges}"


    def save(self, save_path: str = "arena.pkl") -> None:
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, pickle_file: str) -> Arena:
        with open(pickle_file, 'rb') as f:
            arena = pickle.load(f)
        return arena

    def _assign_players(self, equal: bool = False) -> Dict[int, Player]:
        """
        Assign players to the nodes.
        """
        if equal:
            players = [Player.MAX] * (self.num_nodes // 2) + [Player.MIN] * (self.num_nodes // 2)
            self.random.shuffle(players)
            player_mapping = {i: player for i, player in enumerate(players)}
        else:
            player_mapping = {i: self.random.choice([Player.MIN, Player.MAX]) for i in range(self.num_nodes)}
        return player_mapping

    
    def safely_update(self, 
                      node: int, 
                      edge: Tuple[int, int, float], 
                      strategy: GenerationStrategy):
        """
        Update the arena with the new edge and backtrack if a negative cycle is detected.
        """
        if strategy == GenerationStrategy.NONE.value:
            negative_cycles = False
        elif strategy == GenerationStrategy.BELLMAN_FORD.value:
            negative_cycles = self.bellman_ford(nodes=[*self.nodes, node], edges=[*self.edges, edge])
        elif strategy == GenerationStrategy.INCREMENTAL_BELLMAN_FORD.value:
            negative_cycles = self.bellman_ford_incremental(edge)
        else:
            raise ValueError(f"Invalid strategy: {strategy}")

        if negative_cycles:
            return 

        # assert node == edge[0]
        self.edges_mapping[node].add(edge)
        self.ingoing_edges[edge[1]].add(edge)
        

    def generate(self, 
                 strategy: GenerationStrategy = GenerationStrategy.INCREMENTAL_BELLMAN_FORD):
        pbar = tqdm(total=self.num_nodes ** 2, desc=f"Creating graph (n = {self.num_nodes}, p = {self.edge_probability})")
        update_delta = round(math.sqrt(pbar.total))
        for i, (origin, dest) in enumerate(product(self.nodes, repeat=2)):
            if i % update_delta == 0:
                pbar.update(update_delta)
            if self.random.random() < self.edge_probability:
                weight = self.random.uniform(*self.weight_range)
                # Avoid self loops with negative weight
                if not (origin == dest and weight < 0): 
                    edge = (origin, dest, weight)
                    self.safely_update(node=origin, 
                                       edge=edge, 
                                       strategy=strategy)
        pbar.close()

    def bellman_ford_incremental(self, new_edge: Tuple[int, int, float]) -> bool:
        """
        A very efficient implementation of the Bellman-Ford algorithm that only checks for negative cycles related to the new edge.
        It uses the fast_edges dictionary to keep track of the edges and their weights, in order to avoid iterating over all the edges.
        """
        #TODO: this creates a valid graph, but sometimes it has some false positives


        # Add the new edge
        self.edges.add(new_edge)
        self.fast_edges[new_edge[0]][new_edge[1]] = new_edge[2]

        new_distance_0 = self.distances.get(new_edge[0], 0) + new_edge[2]
        new_distance_1 = self.distances.get(new_edge[1], float('inf'))

        # previous_distance_0 = self.distances.get(new_edge[0], None)
        previous_distance_1 = self.distances.get(new_edge[1], None)

        # Relax edges related to the new edge
        if new_distance_0 < new_distance_1:
            new_distance_1 = new_distance_0

        self.distances[new_edge[1]] = new_distance_1

        origin_to = self.fast_edges[new_edge[0]] #this is a dict that maps the destination to the weight
        dest_to = self.fast_edges[new_edge[1]] #this is a dict that maps the origin to the weight


        edges = {(new_edge[0], dest, weight) for dest, weight in origin_to.items()} | {(new_edge[1], origin, weight) for origin, weight in dest_to.items()}
        for edge in edges:
            if self.distances[edge[0]] + edge[2] < self.distances.get(edge[1], float('inf')):
                self.edges.remove(new_edge)
                self.fast_edges[new_edge[0]].pop(new_edge[1])
                self.distances[new_edge[1]] = previous_distance_1
                return True

        return False  # No negative cycle found

    def bellman_ford(self, 
                    nodes: Optional[Set[int]] = None, 
                    edges: Optional[Set[Tuple[int, int, float]]] = None) -> bool:
        """
        Detect negative cycles using Bellman-Ford algorithm.
        """
        if nodes is None:
            nodes = self.nodes
        if edges is None:
            edges = self.edges

        distances: Dict[int, float] = {node: 0 for node in nodes}

        # Relax edges repeatedly
        for _ in range(len(nodes) - 1):
            for edge in edges:
                if distances[edge[0]] + edge[2] < distances.get(edge[1], float('inf')):
                    distances[edge[1]] = distances[edge[0]] + edge[2]

        # Check for negative cycles
        for edge in edges:
            if distances[edge[0]] + edge[2] < distances.get(edge[1], float('inf')):
                return True  # Negative cycle found

        return False  # No negative cycle found

    def get_outgoing_edges(self, node: int) -> Set[Tuple[int, int, float]]:
        """
        Get the outgoing edges of a node.
        """
        return self.edges_mapping[node]


    def get_node_degree(self, node: int) -> int:
        """
        Get the degree of a node.
        """
        return len(self.get_outgoing_edges(node)) 
    


