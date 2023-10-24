import random
from typing import List, Set

from tqdm import tqdm

from Graph import Node, Player, Edge, Arena
from plot_graph import plot_graph


class GraphGenerator:
    def __init__(self, num_nodes, edge_probability=0.01):
        self.num_nodes = num_nodes
        self.edge_probability = edge_probability
        self.max_weight = 10
        self.weight_range = (-self.max_weight, self.max_weight)

    def generate_graph(self):
        nodes: Set[Node] = set()
        while len(nodes) < self.num_nodes:
            nodes.add(Node(random.randint(1, self.num_nodes),
                      Player(random.randint(1, 2))))
        nodes: List[Node] = list(nodes)
        edges = []

        for i in tqdm(range(self.num_nodes), desc="Generating edges"):
            for j in range(i + 1, self.num_nodes):
                # Generate a random number from a uniform distribution and check if it's less than the edge probability
                if random.random() < self.edge_probability and len(edges) < self.num_nodes - 1:
                    # Adjust the weight range as needed
                    weight = random.uniform(*self.weight_range)
                    edge = Edge(nodes[i], nodes[j], weight)
                    nodes[i].add_edge(edge)
                    nodes[j].add_edge(edge)
                    edges.append(edge)

        return nodes, edges


if __name__ == "__main__":
    num_nodes = 1_000  # Adjust the number of nodes as needed

    graph_generator = GraphGenerator(num_nodes, 0.5)
    nodes, edges = graph_generator.generate_graph()

    # Now you have a random graph with nodes and edges for two players.
    arena = Arena(nodes, edges)

    plot_graph(arena)
