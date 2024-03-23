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
            for node in [n for n in self.arena.nodes if self.arena.player_mapping[n] == Player.MIN]:
                pbar.update(1)
                outgoing_edges = self.arena.get_node_neighbours_with_edges(node)
                for edge in outgoing_edges.values():
                    u, v, w = edge
                    # Node u is incorrect
                    if self.arena.value_mapping[u] < self._delta(self.arena.value_mapping[v], w):
                        count[u] += 1
                # If count == degree of node
                if count[u] == self.arena.get_node_degree(u):
                    incorrect.add(u)

            # For each MAX node
            for node in [n for n in self.arena.nodes if self.arena.player_mapping[n] == Player.MAX]:
                pbar.update(1)
                outgoing_edges = self.arena.get_node_neighbours_with_edges(node)
                for edge in outgoing_edges.values():
                    u, v, w = edge
                    # Node u is incorrect
                    if self.arena.value_mapping[u] < self._delta(self.arena.value_mapping[v], w):
                        incorrect.add(u)
        
        def treat(u: int):
            self.arena.value_mapping[u] = self._O(u)

        def update(u: int):
            if self.arena.player_mapping[u] == Player.MIN:
                count[u] = 0

            ingoing_edges = self.arena.ingoing_edges.get(u, set())
            for (v, u, _) in ingoing_edges:
                # Only consider nodes that are incorrect

                #TODO: if I put this, the results is different from the other algorithm, otherwise it's the same
                # if v not in incorrect:
                #     return

                if self.arena.player_mapping[v] == Player.MIN:
                    count[v] += 1
                    if count[v] == self.arena.get_node_degree(v):
                        incorrect_prime.add(v)
                if self.arena.player_mapping[v] == Player.MAX:
                    incorrect_prime.add(v)

        def stop(old_values: Dict[int, int]):
            """
            (Added by me)
            Stopping criterion: if the difference between the old and new values is less than a threshold for all nodes
            """
            threshold = 0.0001
            for node in self.arena.nodes:
                if abs(self.arena.value_mapping[node] - old_values[node]) > threshold:
                    return False
            return True

        init()
        m = self.arena.num_nodes
        n = len(self.arena.edges)
        W = self.arena.max_weight
        max_steps = m * n * W 
        for i in tqdm(range(max_steps)):
            incorrect_prime = set()
            old_values = self.arena.value_mapping.copy()
            for u in incorrect:
                treat(u)
                update(u)
            if stop(old_values) or len(incorrect_prime) == 0:
                print(f"Converged after {i} steps")
                return {node: self.arena.value_mapping[node] for node in self.arena.nodes}
            incorrect = incorrect_prime
            print(self.get_min_energy())

    def _delta(self,l, w): return max(l-w, 0) 

    def _O(self, node: int):
            outgoing_edges = self.arena.get_node_neighbours_with_edges(node)
            # print(f"Outgoing edges for node {node}: {outgoing_edges}")

            values = [self._delta(self.arena.value_mapping[out_node], weight) # delta between the value of the connected node and the weight of the edge for each outgoing edge 
                      for (_, out_node, weight) in outgoing_edges.values()]

            if values == []:
                return 0

            if self.arena.player_mapping[node] == Player.MAX:  
                return max(values)
            else:  # player is MIN
                return min(values)

    def value_iteration(self):
        threshold = 0.0001
        steps = 0
        max_steps = 10_000
        pbar = tqdm(total=max_steps, desc="Value iteration")

        converged = {node: False for node in self.arena.nodes}
        while not all(converged.values()):
            converged = {node: False for node in self.arena.nodes}
            steps += 1
            if steps > max_steps:
                break
            pbar.update(1)
            for node in self.arena.nodes:
                old_value = self.arena.value_mapping[node]
                self.arena.value_mapping[node] = self._O(node)
                if abs(self.arena.value_mapping[node] - old_value) < threshold:
                    converged[node] = True
            print(self.get_min_energy())
        pbar.close()

        if steps > max_steps:
            logging.info(f"Did not converge after {steps} steps")
        else:
            logging.info(f"Converged after {steps} steps")
        return self.arena.nodes, steps

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
            # Find the maximum value among the player's nodes
            max_value = max(self.arena.value_mapping[node] for node in player_nodes)

            min_energy_dict[player] = round(max_value, round_to)

        return min_energy_dict