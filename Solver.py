from functools import lru_cache
import logging
from typing import Dict, Set
from tqdm import tqdm
from Graph import Arena, Player


class Solver:
    def __init__(self, arena: Arena):
        self.arena = arena

    def optimized_value_iteration(self):
        incorrect: Set[int] = set() 
        incorrect_prime: Set[int] = set()
        count: Dict[int, int] = {node: 0 for node in self.arena.nodes}

        def init():
            self.arena.value_mapping = {node: 0 for node in self.arena.nodes}
            pbar = tqdm(total=len(self.arena.nodes), desc="Opt Value Iteration - Init")
            # For each MIN node
            min_nodes = (n for n in self.arena.nodes 
                         if self.arena.player_mapping[n] == Player.MIN)
            for node in min_nodes:
                pbar.update(1)
                for (u, v, w) in self.arena.get_outgoing_edges(node):
                    # Node u is incorrect
                    assert u == node
                    if self.arena.value_mapping[u] < self._delta(self.arena.value_mapping[v], w):
                        count[u] += 1
                # If count == degree of node
                if count[node] == self.arena.get_node_degree(node):
                    incorrect.add(node)

            # For each MAX node
            max_nodes = (n for n in self.arena.nodes
                            if self.arena.player_mapping[n] == Player.MAX)
            for node in max_nodes:
                pbar.update(1)
                for (u, v, w) in self.arena.get_outgoing_edges(node):
                    # Node u is incorrect
                    assert u == node

                    if self.arena.value_mapping[u] < self._delta(self.arena.value_mapping[v], w):
                        incorrect.add(u)
        
        def treat(u: int):
            self.arena.value_mapping[u] = self._O(u)

        def update(u: int):
            if self.arena.player_mapping[u] == Player.MIN:
                count[u] = 0

            for (v, _, w) in self.arena.ingoing_edges.get(u, set()):
                # Only consider nodes that are still incorrect
                if not (self.arena.value_mapping[v] < self._delta(self.arena.value_mapping[u], w)):
                    continue

                if self.arena.player_mapping[v] == Player.MIN:
                    count[v] += 1
                    if count[v] == self.arena.get_node_degree(v):
                        incorrect_prime.add(v)
                if self.arena.player_mapping[v] == Player.MAX:
                    incorrect_prime.add(v)

        # def stop(old_values: Dict[int, int], new_values: Dict[int, int]):
        #     threshold = 0.000001
        #     return all(abs(old_values[node] - new_values[node]) < threshold for node in self.arena.nodes)

        init()
        n = len(self.arena.nodes)
        W = self.arena.max_weight
        max_steps = n * W
        steps = 0
        # Maximum steps = n, so complexity is O(kn^2) in the case of edge_probability = 1. 
        for i in tqdm(range(max_steps)):
            steps += 1
            incorrect_prime = set()

            # O(k * (|Out(u)| + |In(u)|) = O(k * 2n) = O(kn) where k is the number of incorrect nodes
            for u in incorrect:
                # Complexity of treat is O(|Out(u)|)
                treat(u)
                # Complexity of update is O(|In(u)|)
                update(u)

            # print(f"Step {i} - Incorrect: {len(incorrect)} | Incorrect prime: {len(incorrect_prime)}")
            if incorrect_prime == set():
                print(f"Converged after {i} steps")
                return steps
            incorrect = incorrect_prime 
        return steps

    def _delta(self,l, w): 
        return max(l-w, 0)

    def _O(self, node: int):
        """
        The O^G function which returns the max value between all the outgoing edges from the node (if player is Max), or the min value (if player is Min).
        """
        values = (self._delta(self.arena.value_mapping[v], w) for (u, v, w) in self.arena.get_outgoing_edges(node))
        if self.arena.player_mapping[node] == Player.MAX:  
            return max(values, default=0)
        else:  # player is MIN
            return min(values, default=0)

    def value_iteration(self):
        """
        The naive value iteration algorithm to compute the value function.
        """
        threshold = 0.000001
        steps = 0
        max_steps = 50_000
        pbar = tqdm(total=max_steps, desc="Value iteration")

        # Maximum n iterationr, so complexity is O(n^3) in the case of edge_probability = 1
        while True:
            pbar.update(1)
            steps += 1
            if steps > max_steps:
                break

            old_value = self.arena.value_mapping.copy()
            # O(n^2) complexity
            for node in self.arena.nodes:
                # O(n) complexity
                self.arena.value_mapping[node] = self._O(node)

            if all((abs(self.arena.value_mapping[node] - old_value[node]) < threshold for node in self.arena.nodes)):
               break

        pbar.close()

        if steps > max_steps:
            print(f"Naive Value Iteration - Did not converge after {steps} steps")
        else:
            print(f"Naive Value Iteration - Converged after {steps} steps")
        return steps

    def get_min_energy(self, round_to: int = 2):
        """
        For each player, finds the minimum amount of energy needed in order to keep it positive for an infinite amount of steps.
        (Energy is lost when a player moves to a node whose edge has negative weight, and gained when it moves to a node whose edge has a positive weight.)
        """
        min_energy_dict = {}
        for player in [Player.MIN, Player.MAX]: 
            # Get the nodes owned by the player
            player_nodes = [
                node for node in self.arena.nodes if self.arena.player_mapping[node] == player]
            # Find the maximum value among the playeor's nodes
            max_value = max(self.arena.value_mapping[node] for node in player_nodes)

            min_energy_dict[player] = round(max_value, round_to)

        return min_energy_dict