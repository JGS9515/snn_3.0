"""
Network architecture visualization for publication
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

def save_publication_snn_diagram(network, filepath, rankdir="LR", show_time_axis=True):
    """
    Save a publication-ready diagram of the SNN architecture
    
    Args:
        network: The bindsnet network object
        filepath: Path where to save the diagram
        rankdir: Direction of the graph layout ("LR" for left-to-right)
        show_time_axis: Whether to show time axis
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Create a simple visualization
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Define layer positions and sizes
        layers = []
        if hasattr(network, 'layers'):
            layer_names = list(network.layers.keys())
            layer_count = len(layer_names)
        else:
            layer_names = ['Input', 'Hidden', 'Output']  # Default layers
            layer_count = 3
        
        # Calculate positions
        x_positions = np.linspace(0.1, 0.9, layer_count)
        y_center = 0.5
        box_width = 0.15
        box_height = 0.3
        
        # Draw layers
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
        for i, (x_pos, layer_name) in enumerate(zip(x_positions, layer_names)):
            color = colors[i % len(colors)]
            
            # Draw layer box
            box = FancyBboxPatch(
                (x_pos - box_width/2, y_center - box_height/2),
                box_width, box_height,
                boxstyle="round,pad=0.02",
                facecolor=color,
                edgecolor='black',
                linewidth=2
            )
            ax.add_patch(box)
            
            # Add layer label
            ax.text(x_pos, y_center, layer_name, 
                   ha='center', va='center', fontsize=12, fontweight='bold')
            
            # Get layer size if available
            if hasattr(network, 'layers') and layer_name in network.layers:
                layer_obj = network.layers[layer_name]
                if hasattr(layer_obj, 'n'):
                    size_text = f"n={layer_obj.n}"
                    ax.text(x_pos, y_center - 0.1, size_text, 
                           ha='center', va='center', fontsize=10)
        
        # Draw connections between layers
        for i in range(len(x_positions) - 1):
            start_x = x_positions[i] + box_width/2
            end_x = x_positions[i + 1] - box_width/2
            
            # Draw arrow
            ax.annotate('', xy=(end_x, y_center), xytext=(start_x, y_center),
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        
        # Add time axis if requested
        if show_time_axis:
            ax.text(0.5, 0.1, 'Time →', ha='center', va='center', 
                   fontsize=12, style='italic')
        
        # Set title
        ax.set_title('Spiking Neural Network Architecture', fontsize=16, fontweight='bold')
        
        # Remove axes
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Architecture diagram saved to: {filepath}")
        
    except Exception as e:
        print(f"Warning: Failed to create architecture diagram: {e}")
        # Create a simple placeholder file to avoid import errors
        with open(filepath, 'w') as f:
            f.write("# Architecture diagram could not be generated\n")
