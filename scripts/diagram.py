import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# Define node positions for the flowchart
nodes = {
    'start': {'pos': (0, 8), 'text': 'Neurona en reposo<br>V = V_rest', 'color': '#F5F5F5'},
    'impulse': {'pos': (0, 6), 'text': '¿Llega un<br>impulso?', 'color': '#F5F5F5'},
    'accumulate': {'pos': (3, 4), 'text': 'Acumular carga:<br>V aumenta', 'color': '#A5D6A7'},
    'leak': {'pos': (-3, 4), 'text': 'Fuga:<br>V decae hacia V_rest', 'color': '#B3E5EC'},
    'threshold': {'pos': (0, 2), 'text': '¿V ≥ Umbral?', 'color': '#F5F5F5'},
    'fire': {'pos': (0, 0), 'text': '¡DISPARAR!<br>Emitir spike', 'color': '#FFCDD2'},
    'reset': {'pos': (0, -2), 'text': 'Reiniciar:<br>V = V_rest', 'color': '#E0E0E0'}
}

# Define edges (arrows) between nodes
edges = [
    ('start', 'impulse'),
    ('impulse', 'accumulate', 'Sí'),
    ('impulse', 'leak', 'No'),
    ('accumulate', 'threshold'),
    ('leak', 'threshold'),
    ('threshold', 'fire', 'Sí'),
    ('fire', 'reset'),
    ('reset', 'impulse'),
    ('threshold', 'impulse', 'No')
]

# Create figure
fig = go.Figure()

# Add nodes
for node_id, node_data in nodes.items():
    x, y = node_data['pos']
    fig.add_trace(go.Scatter(
        x=[x], y=[y],
        mode='markers+text',
        marker=dict(
            size=80,
            color=node_data['color'],
            line=dict(width=2, color='#333333')
        ),
        text=node_data['text'],
        textposition='middle center',
        textfont=dict(size=10, color='black'),
        showlegend=False,
        hoverinfo='none'
    ))

# Add edges (arrows)
for edge in edges:
    start_node = edge[0]
    end_node = edge[1]
    label = edge[2] if len(edge) > 2 else ""
    
    x0, y0 = nodes[start_node]['pos']
    x1, y1 = nodes[end_node]['pos']
    
    # Add arrow line
    fig.add_trace(go.Scatter(
        x=[x0, x1], y=[y0, y1],
        mode='lines',
        line=dict(width=2, color='#333333'),
        showlegend=False,
        hoverinfo='none'
    ))
    
    # Add arrowhead
    dx = x1 - x0
    dy = y1 - y0
    length = np.sqrt(dx**2 + dy**2)
    if length > 0:
        # Normalize direction
        dx_norm = dx / length
        dy_norm = dy / length
        
        # Calculate arrowhead position (closer to target node)
        arrow_x = x1 - 0.3 * dx_norm
        arrow_y = y1 - 0.3 * dy_norm
        
        # Add arrowhead
        fig.add_annotation(
            x=arrow_x, y=arrow_y,
            ax=x0, ay=y0,
            xref='x', yref='y',
            axref='x', ayref='y',
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor='#333333',
            showarrow=True,
            text=""
        )
    
    # Add edge labels
    if label:
        mid_x = (x0 + x1) / 2
        mid_y = (y0 + y1) / 2
        
        # Offset label position slightly
        if label == "Sí" and start_node == 'impulse':
            if end_node == 'accumulate':
                mid_x += 0.3
            else:  # fire
                mid_x += 0.3
        elif label == "No":
            if start_node == 'impulse':
                mid_x -= 0.3
            else:  # threshold back to impulse
                mid_x -= 0.5
        
        fig.add_trace(go.Scatter(
            x=[mid_x], y=[mid_y],
            mode='text',
            text=label,
            textfont=dict(size=12, color='#333333'),
            showlegend=False,
            hoverinfo='none'
        ))

# Update layout
fig.update_layout(
    title="Ciclo de decisión neurona LIF",
    showlegend=False,
    xaxis=dict(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        range=[-5, 5]
    ),
    yaxis=dict(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        range=[-3, 9]
    ),
    plot_bgcolor='white',
    paper_bgcolor='white'
)

# Save the chart
fig.write_image("lif_neuron_flowchart.png")
fig.write_image("lif_neuron_flowchart.svg", format="svg")

print("LIF neuron decision cycle flowchart created successfully!")