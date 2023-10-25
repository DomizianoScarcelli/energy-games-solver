from __future__ import annotations
import random
import plotly.offline as pyo
import plotly.graph_objs as go
from plotly.offline import plot
import plotly.graph_objects as go
import networkx as nx
import matplotlib.pyplot as plt


def plot_graph(arena):
    # Assuming you have generated nodes and edges as in the previous code
    G = nx.Graph()

    for node in arena.nodes:
        G.add_node(node)  # Add all nodes to the graph

    for edge in arena.edges:
        G.add_edge(edge.node1, edge.node2, weight=edge.weight)

    # Set the random seed for layout consistency
    layout = nx.spring_layout(G, seed=10)

    # Create a list of node colors based on the player
    node_colors = ['blue' if node.player.name ==
                   1 else 'red' for node in G.nodes()]

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = layout[edge[0]]
        x1, y1 = layout[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = layout[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='RdBu',
            size=10,
            colorbar=dict(
                thickness=15,
                title='Player',
                xanchor='left',
                titleside='right'
            ),
            color=node_colors,
            line=dict(width=2)))

    node_trace.marker.color = node_colors

    node_adjacencies = []
    node_text = []
    for node in G.nodes():
        node_adjacencies.append(len(list(G.neighbors(node))))
        node_text.append(
            f"Node: {node.name}<br>Player: {node.player.name}<br>Energy: {node.player.energy}<br>Adjacent Nodes: {len(list(G.neighbors(node)))}<br>Outgoing Edges: {len(node.get_neighbours_with_edges())}"
        )

    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=0, l=0, r=0, t=0),
                    xaxis=dict(showgrid=False, zeroline=False,
                               showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

    plot(fig)


def plot_3D_graph(arena):
    # Create a directed graph
    directed_graph = go.Figure()

    # Create lists to store the node positions and colors
    node_x = []
    node_y = []
    node_z = []
    node_colors = []

    # Create an edge trace
    edge_trace = go.Scatter3d(x=(), y=(), z=(), mode='lines', line=dict(
        width=2), hoverinfo='text', text=[])

    # Iterate through nodes to set positions and colors
    for node in arena.nodes:
        x, y, z = random.uniform(-5, 5), random.uniform(-5,
                                                        5), random.uniform(-5, 5)
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        if node.player.name == 1:
            node_colors.append('red')
        else:
            node_colors.append('blue')

    # Update the edge trace with the positions and edge weights
    edge_text = []  # To store edge weight labels
    for edge in arena.edges:
        x0, y0, z0 = node_x[arena.nodes.index(edge.node1)], node_y[arena.nodes.index(
            edge.node1)], node_z[arena.nodes.index(edge.node1)]
        x1, y1, z1 = node_x[arena.nodes.index(edge.node2)], node_y[arena.nodes.index(
            edge.node2)], node_z[arena.nodes.index(edge.node2)]
        edge_trace['x'] += (x0, x1, None)
        edge_trace['y'] += (y0, y1, None)
        edge_trace['z'] += (z0, z1, None)
        # Create a label with edge weight
        weight_label = f"Weight: {edge.weight:.2f}"
        edge_text.extend(["", weight_label, ""])  # Add labels to text

    # Add nodes to the graph
    node_trace = go.Scatter3d(x=node_x, y=node_y, z=node_z, mode='markers', marker=dict(
        size=7, color=node_colors, opacity=1))

    # Add nodes and edges to the graph
    directed_graph.add_trace(edge_trace)
    directed_graph.add_trace(node_trace)

    # Set layout properties
    directed_graph.update_layout(
        scene=dict(
            xaxis=dict(nticks=4, range=[-10, 10]),
            yaxis=dict(nticks=4, range=[-10, 10]),
            zaxis=dict(nticks=4, range=[-10, 10]),
            aspectmode='cube'
        )
    )

    # Assign edge weight labels to text
    directed_graph.update_traces(text=edge_text)

    # Display the graph
    pyo.iplot(directed_graph)


def plot_2D_graph(arena):
    # Create a directed graph
    G = nx.DiGraph()

    # Create a dictionary to store node positions
    node_positions = {}

    # Iterate through nodes to set positions and colors
    for node in arena.nodes:
        x, y = random.uniform(-5, 5), random.uniform(-5, 5)
        node_positions[node] = (x, y)
        if node.player.name == 1:
            G.add_node(node, color='red')
        else:
            G.add_node(node, color='blue')

    # Create a list to store separate edges with different weights
    separate_edges = []

    # Iterate through edges to create separate edges
    for edge in arena.edges:
        G.add_edge(edge.node1, edge.node2, weight=round(edge.weight, 2),
                   label=f"{edge.weight:.2f}")
        separate_edges.append((edge.node1, edge.node2, edge.weight))

    # Get node colors from the node attributes
    node_colors = [node[1]['color'] for node in G.nodes(data=True)]

    # Create a layout for the graph
    pos = nx.spring_layout(G, seed=42)

    # Draw the nodes
    nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, node_size=200, alpha=1)

    # Draw the edges
    nx.draw_networkx_edges(G, pos, edgelist=separate_edges,
                           edge_color='k', arrows=True, connectionstyle="arc3,rad=0.1")

    # Draw the labels
    nx.draw_networkx_labels(G, pos)

    # Draw edge labels (weights)
    edge_labels = {(edge[0], edge[1]): edge[2] for edge in separate_edges}
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, font_color='black', font_size=8)

    plt.axis('off')
    plt.show()
