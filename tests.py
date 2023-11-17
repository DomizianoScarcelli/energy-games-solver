from unittest import TestCase

from Graph import Node, Edge, GraphGenerator, Arena


class GraphGeneratorTest(TestCase):
    def test_remove_negative_cycles(self):
        nodes = [Node(1), Node(2), Node(3), Node(4)]
        edges = [Edge(Node(1), Node(2), 1.5),
                 Edge(Node(2), Node(3), 12),
                 Edge(Node(3), Node(4), 9),
                 Edge(Node(4), Node(1), -20),
                 Edge(Node(3), Node(1), -15)]
        arena = Arena(nodes, edges)
        print(f"Initial arena is {arena}")

        for edge in edges:
            edge.node1.add_edge(edge)
        self.assertTrue(arena._check_negative_cycles(nodes[0]))

        for node in nodes:
            arena.remove_negative_cycles(node)

        print(f"Final arena is {arena}")
        for node in nodes:
            self.assertFalse(arena._check_negative_cycles(node))

    def test_graph_generation(self):
        graph_generator = GraphGenerator(
            num_nodes=10, edge_probability=0.3)
        nodes, edges = graph_generator.generate_graph()
        arena = Arena(nodes, edges)

        self.assertTrue(arena._check_negative_cycles(nodes[0]))

        arena.generate_mean_payoff_arena()
        for node in nodes:
            arena.remove_negative_cycles(node)

        print(f"Final arena is {arena}")
        for node in nodes:
            self.assertFalse(arena._check_negative_cycles(node))
