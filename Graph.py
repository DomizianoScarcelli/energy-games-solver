from __future__ import annotations
import cProfile
import pstats
from itertools import product
import math
import time
from typing import Dict, List, Set, Tuple
from tqdm import tqdm
import random
from plot_graph import plot_graph
import logging
import pickle
import sys
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

class Arena:
    def __init__(self, num_nodes: float = 10, edge_probability: float = 0.01, seed: int | None = None):
        self.nodes = set(range(num_nodes))
        self.random = random.Random(seed)
        self.player_mapping: Dict[int, int] = {i: self.random.randint(1, 2) for i in range(num_nodes)} 
        self.value_mapping: Dict[int, float] = {i: 0 for i in range(num_nodes)}
        self.edges_mapping: Dict[int, Set[Tuple[int, int, float]]] = {i: set() for i in range(num_nodes)}
        self.edges = set()
        self.num_nodes = num_nodes
        self.edge_probability = edge_probability
        self.max_weight = 10
        self.weight_range = (-self.max_weight, self.max_weight)
        self.distances = {node: 0 for node in self.nodes}
        self.considered: Set[Tuple[int, Tuple[int, int, float]]]= set()
        self.considered_edges_bf: Set[Tuple[int, int, float]] = set()
        self.fast_edges: Dict[int, Dict[int, float]] = {i: {} for i in range(num_nodes)}

        if seed is not None:
            random.seed(seed)

    def __repr__(self):
        return f"Nodes: {self.nodes}\nEdges: {self.edges}"

    def __str__(self):
        return f"Nodes: {self.nodes}\nEdges: {self.edges}"


    def save(self, save_path: str = "arena.pkl"):
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, pickle_file: str):
        with open(pickle_file, 'rb') as f:
            arena = pickle.load(f)
        return arena

    
    def safely_update(self, node: int, edge: Tuple[int, int, float]):
        """
        Update the arena with the new edge and backtrack if a negative cycle is detected.
        """
        negative_cycles = self.bool_bellman_ford_incremental(edge)
        # alt_negative_cycles = self.bool_bellman_ford(nodes=[*self.nodes, node], edges=[*self.edges, edge])
        # assert negative_cycles == alt_negative_cycles, f"Negative cycles do not match: {negative_cycles} vs {alt_negative_cycles}"
        if negative_cycles:
            return 
        # self.distances = new_distances
        self.edges_mapping[node].add(edge)

        
    def generate(self):
        pbar = tqdm(total=self.num_nodes ** 2, desc="Creating graph")
        update_delta = round(math.sqrt(pbar.total))
        for i, (origin, dest) in enumerate(product(self.nodes, repeat=2)):
            if i % update_delta == 0:
                pbar.update(update_delta)
            if random.random() < self.edge_probability:
                weight = random.uniform(*self.weight_range)
                if not (origin == dest and weight < 0): # Avoid self loops with negative weight
                    edge = (origin, dest, weight)
                    self.safely_update(origin, edge)
        pbar.close()

    def bool_bellman_ford_incremental(self, new_edge: Tuple[int, int, float]):
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
        for _ in range(len(self.nodes) - 1):
            if new_distance_0 < new_distance_1:
                new_distance_1 = new_distance_0

        self.distances[new_edge[1]] = new_distance_1

        # Check for negative cycles related to the new edge
        # for edge in self.edges:
        #     if edge[0] == new_edge[0] or edge[1] == new_edge[0] or edge[0] == new_edge[1] or edge[1] == new_edge[1]:
        #         if self.distances[edge[0]] + edge[2] < self.distances.get(edge[1], float('inf')):
        #             self.edges.remove(new_edge)
        #             self.distances[new_edge[1]] = previous_distance_1
        #             return True # Negative cycle found

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

    def bool_bellman_ford(self, nodes: List[int] = None, edges: List[Tuple[int, int, float]] = None):
        """
        Detect negative cycles using Bellman-Ford algorithm.
        """
        if nodes is None:
            nodes = self.nodes
        if edges is None:
            edges = self.edges

        distances = {node: 0 for node in nodes}

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

    def get_node_neighbours_with_edges(self, node: int) -> Dict[int, Tuple[int, int, float]]:
        """
        Get the neighbours of a node, along with the edge that connects them.
        """
        outgoing_edges = self.edges_mapping[node]
        neighbours = {edge[1]: edge for edge in outgoing_edges if edge[1] != node}
        return neighbours

    def value_iteration(self):
        def delta(l, w): return max(l-w, 0)
        def Q(node: int):
            outgoing_edges = self.get_node_neighbours_with_edges(node)
            values = [delta(self.value_mapping[node], edge[2])
                      for node, edge in outgoing_edges.items()]
            if values == []:
                return 0
            if self.player_mapping[node] == 1:  # max
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
        for player in range(1, 3):
            # Get the nodes owned by the player
            player_nodes = [
                node for node in self.nodes if self.player_mapping[node] == player]
            # Find the maximum value among the player's nodes
            max_value = max(self.value_mapping[node] for node in player_nodes)

            min_energy_dict[player] = round(max_value, round_to)
        return min_energy_dict


def run_solver(num_nodes: int = 30, edge_probability: float = 0.1, seed: int | None = None, plot: bool = False, save: bool = False):
    arena = Arena(num_nodes=num_nodes,
                  edge_probability=edge_probability, 
                  seed=seed) 
    arena.generate()
    if plot:
        plot_graph(arena)
    if save:
        arena.save(f"arena_{num_nodes}_{edge_probability}.pkl")
    arena.value_iteration()
    min_energy_dict = arena.get_min_energy()
    return min_energy_dict

def run_multiple(n_runs: int = 100, plot: bool = False, save: bool = False):
    times = []
    for seed in range(0, n_runs):
        try:
            start = time.time()
            solution = run_solver(num_nodes=5000, edge_probability=0.1, seed=seed, plot=plot, save=save)
            end = time.time()
            times.append(end - start)
            logging.info(f"Solution: {solution}")
        except AssertionError as e:
            logging.error(f"Seed {seed} failed with error: {e}")
            break
    avg_time = (sum(times) / len(times)) * 1000
    logging.info(f"Average time: {avg_time:f} ms")

def profile(seed: int = 0, plot: bool = False, save: bool = False):
    profiler = cProfile.Profile()
    profiler.enable()

    # Run your function
    solution = run_solver(num_nodes=5000, edge_probability=0.2, seed=seed, plot=plot, save=save)    
    logging.info(f"Solution: {solution}")
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()

if __name__ == "__main__":
    profile(plot=False, save=True, seed=0)    
    # run_multiple(n_runs=1, plot=False, save=True)
