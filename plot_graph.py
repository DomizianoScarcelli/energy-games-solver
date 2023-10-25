from __future__ import annotations
from plotly.offline import plot
import plotly.graph_objects as go
import networkx as nx


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
