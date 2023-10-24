import random

from Graph import Node, Player, Edge, Arena


class GraphGenerator:
    def __init__(self, num_nodes, max_edges):
        self.num_nodes = num_nodes
        self.max_edges = max_edges
        self.max_weight = 10
        self.weight_range = (-self.max_weight, self.max_weight)

    def generate_graph(self):
        nodes = [Node(name, Player(random.randint(1, 2)))
                 for name in range(1, self.num_nodes + 1)]
        edges = []

        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if random.random() < 0.5 and len(edges) < self.max_edges:
                    # Adjust the weight range as needed
                    weight = random.uniform(*self.weight_range)
                    edge = Edge(nodes[i], nodes[j], weight)
                    nodes[i].add_edge(edge)
                    nodes[j].add_edge(edge)
                    edges.append(edge)

        return nodes, edges


if __name__ == "__main__":
    num_nodes = 10  # Adjust the number of nodes as needed
    max_edges = 15  # Adjust the maximum number of edges as needed

    graph_generator = GraphGenerator(num_nodes, max_edges)
    nodes, edges = graph_generator.generate_graph()

    # Now you have a random graph with nodes and edges for two players.
    arena = Arena(nodes, edges)

    print(edges)

    print(arena)
