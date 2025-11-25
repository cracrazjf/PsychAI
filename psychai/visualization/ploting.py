import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import math

def plot_lines(lines, x_labels, y_labels, title, output_path, y_lim=None, x=None, sems=None, line_props=None, max_ticks=10, figsize=(8,6)):
    plt.figure(figsize=figsize)
    default_linestyles = ['-', '--', '-.', ':']
    default_colors = ['black']
    order = line_props.keys() if line_props else None
    x = np.arange(len(next(iter(lines.values())))) if x is None else np.array(x)
    for i, line_name in enumerate(lines):
        props = line_props.get(line_name, {}) if line_props else {}
        linestyle = props.get("linestyle", default_linestyles[i % len(default_linestyles)])
        color = props.get("color", default_colors[0])
        linewidth = props.get("linewidth", 2)
        y = np.array(lines[line_name])
        plt.plot(x, y, label=line_name, linestyle=linestyle, color=color, linewidth=linewidth)
        if sems:
            y_err = np.array(sems[line_name])
            plt.fill_between(x, y - y_err, y + y_err, alpha=0.2, color=color)
    
    if len(x) > max_ticks:
        step = len(x) // max_ticks
        plt.xticks(x[::step], x[::step])

    if order:
        handles, labels = plt.gca().get_legend_handles_labels()  
        ordered_handles = [handles[labels.index(o)] for o in order]
        plt.legend(ordered_handles, order, fontsize=15)
    else:
        plt.legend(fontsize=10)

    plt.xlabel(x_labels)
    plt.ylabel(y_labels)
    if y_lim:
        plt.ylim(y_lim)
    plt.title(title)
    plt.savefig(output_path)
    plt.close()

def plot_heatmap(data, output_path, x_labels=None, y_labels=None, title=None, figsize=(12, 4)):
    # Convert dict to DataFrame
    if isinstance(data, dict):
        data = pd.DataFrame.from_dict(data, orient='index')
        data.columns = x_labels if x_labels else data.columns.tolist()
        if y_labels:
            data.index = y_labels

    # Convert numpy array to DataFrame (so we get nice labeling)
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data, 
                            index=y_labels if y_labels else range(data.shape[0]),
                            columns=x_labels if x_labels else range(data.shape[1]))
    
    sns.heatmap(data,
                cmap='viridis',
                square=True,
                annot=True,
                fmt=".2f",
                cbar=False,
                annot_kws={"size": 5, "color": "black"}
    )

    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close() 

def plot_heatmaps(df_list, output_path, titles=None, figsize=(12, 4), cmap="coolwarm", vmin=-1, vmax=1, ncols=3):
    n = len(df_list)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
    
    if n == 1:
        axes = [axes] 

    axes = axes.flatten()
    
    for i, df in enumerate(df_list):
        ax = axes[i]
        sns.heatmap(df, cmap=cmap, vmin=vmin, vmax=vmax, ax=ax, square=True, cbar=False)
        # ax.set_title(titles[i] if titles else f"DF {i+1}")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_scatter(x, y, output_path, x_label=None, y_label=None, x_lim=None, y_lim=None, title=None, figsize=(8,6)):
    plt.figure(figsize=figsize)
    plt.scatter(x, y, color="blue", alpha=0.6)
    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)
    if title:
        plt.title(title)
    if x_lim:
        plt.xlim(x_lim)
    if y_lim:
        plt.ylim(y_lim)
    plt.savefig(output_path)
    plt.close()

def plot_bars(data: dict, output_path, title="Model Comparison", ylabel="Accuracy", figsize=(8,5)):
    bar_names = list(data.keys())
    values = [np.array(data[m]) for m in bar_names]
    
    means = [v.mean() for v in values]
    stds = [v.std() for v in values]

    plt.figure(figsize=figsize)
    x = np.arange(len(bar_names))

    plt.bar(x, means, yerr=stds, capsize=6)
    plt.xticks(x, bar_names)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()