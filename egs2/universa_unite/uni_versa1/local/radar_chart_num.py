import matplotlib.pyplot as plt
import numpy as np

def create_multi_radar_chart(labels, system_values, system_names, title='Radar Chart Comparison', save_path='radar_chart.pdf'):
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for values, name in zip(system_values, system_names):
        values += values[:1]
        ax.plot(angles, values, label=name)
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title(title, y=1.08)
    ax.set_rlabel_position(30)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    plt.savefig(save_path, format='pdf')
    print(f"Radar chart saved to {save_path}")

# Example usage
labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
system_values = [
  [0.69, 0.45, 0.49, 0.45],
  [0.71, 0.51, 0.52, 0.49],
  [0.74, 0.58, 0.59, 0.57],
]
system_names = ['UniVERSA', 'UniVERSA-T', 'ECHO']

create_multi_radar_chart(labels, system_values, system_names, title='Model Comparison', save_path='model_comparison_class.pdf')

