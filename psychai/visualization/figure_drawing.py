import matplotlib.pyplot as plt
import numpy as np

def draw_scatter_xy(summary_df, scatter_params):
    x_col, y_col = scatter_params.xy_cols
    group_col = scatter_params.group_col
    label_col = scatter_params.label_col
    jitter = scatter_params.jitter
    size_scale = scatter_params.size_scale
    title = scatter_params.title

    counts = summary_df.groupby([x_col, y_col]).size().reset_index(name="_dup_n")
    df = summary_df.merge(counts, on=[x_col, y_col], how="left")

    plt.figure()

    if group_col is None:
        plt.scatter(df[x_col], df[y_col], s=size_scale * df["_dup_n"])

        if label_col is not None:
            for _, r in df.iterrows():
                plt.text(r[x_col], r[y_col], str(r[label_col]))

    else:
        groups = list(df[group_col].dropna().unique())
        for i, grp in enumerate(groups):
            sub = df[df[group_col] == grp]
            dx = jitter * i
            dy = jitter * i

            plt.scatter(sub[x_col] + dx, sub[y_col] + dy,
                        s=size_scale * sub["_dup_n"], label=str(grp))

            if label_col is not None:
                for _, r in sub.iterrows():
                    plt.text(r[x_col] + dx, r[y_col] + dy, str(r[label_col]))

        plt.legend()

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title if title is not None else f"{y_col} vs {x_col}")
    plt.show()
    

def draw_barplot(summary, current_figure_params):
    group_cols = current_figure_params.groupby_cols
    mean_col = current_figure_params.mean_col
    ci_col = current_figure_params.ci_col

    width = current_figure_params.width
    gap_group = current_figure_params.gap_group
    gap_block = current_figure_params.gap_block

    capsize = getattr(current_figure_params, "capsize", 4)
    xtick_rotation = getattr(current_figure_params, "xtick_rotation", 45)
    xtick_ha = getattr(current_figure_params, "xtick_ha", "right")
    legend_title = getattr(current_figure_params, "legend_title", None)

    # group_cols length 1-3; last is hue
    hue_col = group_cols[-1]
    hues = summary[hue_col].unique()

    x1_col = group_cols[0] if len(group_cols) >= 2 else None
    x2_col = group_cols[1] if len(group_cols) == 3 else None

    # lookup for fast access
    if len(group_cols) == 1:
        lookup = {r[hue_col]: (r[mean_col], r[ci_col]) for _, r in summary.iterrows()}
    elif len(group_cols) == 2:
        lookup = {(r[x1_col], r[hue_col]): (r[mean_col], r[ci_col]) for _, r in summary.iterrows()}
    else:  # len == 3
        lookup = {(r[x1_col], r[x2_col], r[hue_col]): (r[mean_col], r[ci_col]) for _, r in summary.iterrows()}

    xs_by_hue = [[] for _ in hues]
    ys_by_hue = [[] for _ in hues]
    es_by_hue = [[] for _ in hues]

    xticks, xticklabels = [], []
    centers = {}  # for block labels when len==3

    x = 0.0

    if len(group_cols) == 1:
        group_center = x + (len(hues) - 1) * width / 2
        xticks.append(group_center)
        xticklabels.append("all")

        for hi, h in enumerate(hues):
            m, c = lookup.get(h, (np.nan, np.nan))
            xs_by_hue[hi].append(x + hi * width)
            ys_by_hue[hi].append(m)
            es_by_hue[hi].append(c)

    elif len(group_cols) == 2:
        x1_vals = summary[x1_col].unique()
        for v1 in x1_vals:
            group_center = x + (len(hues) - 1) * width / 2
            xticks.append(group_center)
            xticklabels.append(str(v1))

            for hi, h in enumerate(hues):
                m, c = lookup.get((v1, h), (np.nan, np.nan))
                xs_by_hue[hi].append(x + hi * width)
                ys_by_hue[hi].append(m)
                es_by_hue[hi].append(c)

            x += len(hues) * width + gap_group

    else:  # len == 3
        x1_vals = summary[x1_col].unique()
        for v1 in x1_vals:
            start_x = x

            sub = summary[summary[x1_col] == v1]
            x2_vals = sub[x2_col].unique()

            for v2 in x2_vals:
                group_center = x + (len(hues) - 1) * width / 2
                xticks.append(group_center)
                xticklabels.append(str(v2))

                for hi, h in enumerate(hues):
                    m, c = lookup.get((v1, v2, h), (np.nan, np.nan))
                    xs_by_hue[hi].append(x + hi * width)
                    ys_by_hue[hi].append(m)
                    es_by_hue[hi].append(c)

                x += len(hues) * width + gap_group

            end_x = x - gap_group
            centers[v1] = (start_x + end_x) / 2
            x += gap_block

    plt.figure()
    for hi, h in enumerate(hues):
        plt.bar(xs_by_hue[hi], ys_by_hue[hi], yerr=es_by_hue[hi],
                width=width, capsize=capsize, label=str(h))

    plt.xticks(xticks, xticklabels, rotation=xtick_rotation, ha=xtick_ha)
    plt.legend(title=(legend_title if legend_title is not None else hue_col))

    if len(group_cols) == 3:
        ymin, ymax = plt.ylim()
        y_text = ymin - (ymax - ymin) * 0.12
        for v1, cx in centers.items():
            plt.text(cx, 1.01, str(v1), ha="center", va="bottom", transform=plt.gca().get_xaxis_transform())

    plt.tight_layout()
    plt.show()