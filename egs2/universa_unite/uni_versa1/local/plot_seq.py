import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_ordered_metric_sequence(seq, figsize=(20, 1.5)):
    fig, ax = plt.subplots(figsize=figsize)
    curr_token = seq[0]
    start_idx = 0

    for i in range(1, len(seq)):
        if seq[i] != curr_token:
            ax.add_patch(patches.Rectangle(
                (start_idx, 0), i - start_idx, 1,
                color=plt.cm.tab20(hash(curr_token) % 20)))
            ax.text(start_idx + (i - start_idx)/2, 0.5, curr_token,
                    va='center', ha='center', fontsize=6, rotation=90)
            curr_token = seq[i]
            start_idx = i

    # Add final segment
    ax.add_patch(patches.Rectangle(
        (start_idx, 0), len(seq) - start_idx, 1,
        color=plt.cm.tab20(hash(curr_token) % 20)))
    ax.text(start_idx + (len(seq) - start_idx)/2, 0.5, curr_token,
            va='center', ha='center', fontsize=6, rotation=90)

    ax.set_xlim(0, len(seq))
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.tight_layout()
    plt.show()

# Example
metrics_seq = ['Q-Gender', 'Q-Gender', 'Q-Emotion', 'Q-Emotion', 'Q-Pitch', 'Q-Volume'] * 50
plot_ordered_metric_sequence(metrics_seq)

