from __future__ import annotations
import pprint
import gravis as gv
import networkx as nx


class EnergyGameArena:
    def __init__(self, nodes, edges, player_mapping=None):
        self.nodes = nodes
        self.edges = edges
        self.player_mapping = player_mapping
        self._init_config()

    def _nodes_dict(self):
        if self.player_mapping:
            return {node: {'metadata': {'color': 'red' if self.player_mapping[node].name == 1 else 'blue', 'label_color':'red' if self.player_mapping[node] == 1 else 'blue'  }} for node in self.nodes}
        return {node.name: {'metadata': {'color': 'red' if node.player.name == 1 else 'blue', 'label_color':'red' if node.player.name == 1 else 'blue'  }} for node in self.nodes}

    def _edges_dict_list(self):
        default_color = 'black'
        highlight_size = 1.5
        if hasattr(list(self.edges)[0], "node1"):
            return [{'source': edge.node1.name, 'target': edge.node2.name, 'label': round(edge.weight, 2), 'metadata': {
                        'color': default_color,
                        'opacity': 1.0,
                        'size': highlight_size,
                        'label_color': "black",
                        'label_size': 12,
                    },} for edge in self.edges]
        return [{'source': edge[0], 'target': edge[1], 'label': round(edge[2], 2), 'metadata': {
                    'color': default_color,
                    'opacity': 1.0,
                    'size': highlight_size,
                    'label_color': "black",
                    'label_size': 12,
                },} for edge in self.edges]

    def _init_config(self):
        self.config = {
        'graph': {
            'label': 'Directed graph',
            'directed': True,
        'nodes': 
            self._nodes_dict(),
        'edges': 
            self._edges_dict_list()
    }}

    def create_plot(self):
        self.fig = gv.d3(self.config, 
                         edge_label_data_source='label', 
                         show_edge_label=True, 
                         use_edge_size_normalization=True,
                            node_drag_fix=True, 
                            node_hover_neighborhood=True, show_edge_label_border=True, use_many_body_force=True, many_body_force_strength=-1000)
        return self.fig
    
# def save_graph(graph, filename):
#     EnergyGameArena(graph.nodes, graph.edges).create_plot().export_png(filename)

def plot_graph(graph):
    EnergyGameArena(graph.nodes, graph.edges, graph.player_mapping if hasattr(graph, "player_mapping") else None).create_plot().display()
