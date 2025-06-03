from typing import Union
import re
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
import math
import warnings
import os
import argparse
import textwrap

def join_sorted(chars):
    return '-'.join(sorted(chars))  

def clean_name(name):
    return re.sub(r'^\[\d+\]', '', name)
    
def scale_pvals(
    pvals: Union[list, np.array],
) -> list:
    """
    Function to scale p-values that are already negative log10 transformed.
    In this context, scaling refers to assigning the p-values to a specific
    significance bin. The resulting significance bins are formatted as string
    for plotting purposes.

    Parameters
    ----------
    pvals : list or np.array of integers
        List (or any other iterable) of p-values that are already negative log10 transformed.

    Returns
    -------
    : list
        The lists of significance bins as strings. This will be added to the 'neg_log10_adj_p_round' column
    """
    steps = [100, 50, 10, 5, 2]
    r = []
    for xi in pvals:
        s_max = 0
        for s in steps:
            if xi >= s:
                if s > s_max:
                    s_max = s
        r.append('> '+str(s_max))
    return(r)

def custom_sort_key(x):
        custom_order = ['3₁₀-helix', 'α-helix', 'π-helix', 'PPII-helix', 'ß-strand', 'ß-bridge', 'turn', 'bend', 'unassigned', 'loop', 'IDR']
        order_map = {desc: i for i, desc in enumerate(custom_order)}
        # If it starts with '[', sort by the integer inside the brackets, with priority 0
        if x.startswith('['):
            num = int(x.split(']')[0][1:])
            return (0, num, x)
        # Otherwise, sort by custom order (priority 1)
        else:
            # Items not in custom_order get len(custom_order) as their order index
            order_idx = order_map.get(x, len(custom_order))
            return (1, order_idx, x)

def prepare_plot(data: pd.DataFrame,
                x: set,
                y: set,
                BH: float):
    """
    Custom functionality to color based on significance level and to show -+ inf values.
    """
    df = pd.read_csv(data)
    
    if len(y) == 0:
        if len(x & set(df.x.unique())) == 0:
            y = set(df.x.unique())
        else:
            y = set(df.y.unique())

    # convert sec names
    user_sec = ['310HELX', 'AHELX', 'PIHELX', 'PPIIHELX', 'STRAND', 'BRIDGE', 'TURN', 'BEND', 'unassigned', 'LOOP', 'IDR']
    true_sec = ['3₁₀-helix', 'α-helix', 'π-helix', 'PPII-helix', 'ß-strand', 'ß-bridge', 'turn', 'bend', 'unassigned', 'loop', 'IDR']
    mapping = dict(zip(user_sec, true_sec))
    def rename_elements(input_list):
        return set(mapping.get(item, item) for item in input_list)

    x = rename_elements(x)
    y = rename_elements(y)

    df = df[((df.x.isin(x)) | (df.y.isin(x))) & ((df.x.isin(y)) | (df.y.isin(y))) & (df.p_adj_bh<=BH)]

    # redefine x and y post-filtering
    if len(x & set(df.x.unique())) == 0:
        x = set(df.y.unique())
        y = set(df.x.unique())
    else:
        x = set(df.x.unique())
        y = set(df.y.unique())
    
    # sort alphabetically
    x_sorted = sorted(x, key=custom_sort_key)

    # Calculate limits based on actual data
    log2_odds_values = []
    for _, row in df.iterrows():
        if row['oddsr'] > 0 and not np.isinf(row['oddsr']):
            log2_odds_values.append(np.log2(row['oddsr']))
    
    # Define default values
    min_value = 0
    max_value = 0
    
    # Find the maximum and minimum log2 odds values
    if log2_odds_values:
        min_log2_odds = min(log2_odds_values)
        max_log2_odds = max(log2_odds_values)
        
        if min_log2_odds < 0:
            min_value = math.ceil(abs(min_log2_odds))
        
        if max_log2_odds > 0:
            max_value = math.ceil(max_log2_odds)
            
    # Determine whether to include +/- Inf based on the presence of infinite values in the data
    show_negative_inf = any(df['oddsr'] == 0)
    show_positive_inf = any(np.isinf(df['oddsr']))
    
    # Define edge positions for plotting infinite values
    negative_inf_pos = -(min_value + 1) if show_negative_inf else None # Position for -∞
    positive_inf_pos = max_value + 1 if show_positive_inf else None  # Position for +∞

    # Handle special cases for log2 odds ratio
    df['actual_log2_odds_ratio'] = np.nan
    df['display_log2_odds_ratio'] = np.nan
    df['is_inf'] = False
    df['inf_type'] = ""

    for idx, row in df.iterrows():
        if row['oddsr'] == 0:
            df.at[idx, 'actual_log2_odds_ratio'] = float('-inf')
            df.at[idx, 'display_log2_odds_ratio'] = negative_inf_pos
            df.at[idx, 'is_inf'] = True
            df.at[idx, 'inf_type'] = "-∞"
        elif np.isinf(row['oddsr']):
            df.at[idx, 'actual_log2_odds_ratio'] = float('inf')
            df.at[idx, 'display_log2_odds_ratio'] = positive_inf_pos
            df.at[idx, 'is_inf'] = True
            df.at[idx, 'inf_type'] = "+∞"
        else:
            df.at[idx, 'actual_log2_odds_ratio'] = np.log2(row['oddsr'])
            df.at[idx, 'display_log2_odds_ratio'] = np.log2(row['oddsr'])
    
    warnings.filterwarnings("ignore", message="divide by zero encountered in log10")
    df['neg_log10_adj_p'] = -np.log10(df.p_adj_bh)
    df['neg_log10_adj_p_round'] = scale_pvals(df.neg_log10_adj_p)
    
    # Color scale based on -log10 adj. p-value ranges
    color_ranges = [
        (100, 'rgb(120,0,0)'),
        (50, 'rgb(177, 63, 100)'),
        (10, 'rgb(221, 104, 108)'),
        (5, 'rgb(241, 156, 124)'),
        (2, 'rgb(246, 210, 169)'),
        (0, 'rgb(128,128,128)')
    ]

    # Generate custom ticks based on the requirements
    tick_vals = []
    tick_text = []

    # Add -inf position
    if show_negative_inf:
        tick_vals.append(negative_inf_pos)
        tick_text.append("∞")

    # Add negative ticks in descending order (e.g., -3, -2, -1)
    for i in range(min_value, 0, -1):
        tick_vals.append(-i)
        tick_text.append(str(-i)) # show negative ticks as positive numbers (but depleted)

    # Add zero
    tick_vals.append(0)
    tick_text.append("0")

    # Add positive ticks in ascending order (e.g., 1, 2)
    for i in range(1, max_value + 1):
        tick_vals.append(i)
        tick_text.append(str(i))

    # Add +inf position
    if show_positive_inf:
        tick_vals.append(positive_inf_pos)
        tick_text.append("∞")
    
    # Dynamically define x-axis range based on whether infinite values are present
    x_axis_range = []
    if show_negative_inf:
        x_axis_range.append(negative_inf_pos - 0.5)
    else:
        x_axis_range.append(-min_value - 0.5 if min_value > 0 else -1) # set a default to -1 if there is no minimum

    if show_positive_inf:
        x_axis_range.append(positive_inf_pos + 0.5)
    else:
        x_axis_range.append(max_value + 0.5 if max_value > 0 else 1) # Set a default to 1 if there is no maximum
        
    return df, x, y, x_sorted, color_ranges, tick_vals, tick_text, x_axis_range
    
def single_vis(
    data: pd.DataFrame,
    x: set,
    y: set,
    BH: float
):
    """
    Plot the enrichment of ROIs in different protein regions. Only plot enrichments that with BH multiple-testing corrected <= BH.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe with enrichment results.
    x : set
        Set of dependent data to show.
    y : set
        Set of independent data to show.
        Default is empty set, which shows all data.
    Returns
    -------
    : plot
        Figure showing enrichment of dependent data in independent data.
    """
    df, x, y, x_sorted, color_ranges, tick_vals, tick_text, x_axis_range = prepare_plot(data, x, y, BH)
    # Create the figure
    fig = go.Figure()

    significant_visualisations = False
    
    # First, check if values in y are in the x column or y column
    values_in_x = any(val in df['x'].values for val in y)
    values_in_y = any(val in df['y'].values for val in y)
    
    # Determine which column to use for filtering
    filter_column = 'x' if values_in_x else ('y' if values_in_y else None)
    if filter_column:
        # Sort the y_set alphabetically for ordered x-axis
        sorted_y = sorted(y, key=custom_sort_key)
        
        # Add bars for each value in sorted y_set
        for value in sorted_y:
            if value in df[filter_column].values:
                # Filter the dataframe to only include rows where the filter column equals value
                df_filtered = df[df[filter_column] == value]                
                if not df_filtered.empty:
                    for threshold, color in color_ranges:
                        category = '> ' + str(threshold)
                        # Filter further by the threshold category
                        df_category = df_filtered[df_filtered['neg_log10_adj_p_round'] == category]
                        
                        if not df_category.empty:
                            significant_visualisations = True
                            # Add a bar for this category
                            fig.add_trace(go.Bar(
                                x=[value],  # Use the value as the x-position in the plot
                                y=df_category['display_log2_odds_ratio'],
                                name=f"{value} ({category})",
                                marker_color=color,
                                legendgroup=value,
                                showlegend=False,
                                hoverinfo='none'
                            ))

    if significant_visualisations:
        # Keep track of used colors
        used_colors = set()
        
        # Add colors to legend with corresponding adj. p-value thresholds
        for threshold, color in color_ranges:
            category = '> ' + str(threshold)
            if category in df['neg_log10_adj_p_round'].unique():
                used_colors.add(color)
                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode='markers',
                    marker=dict(size=10, color=color, symbol='square'),
                    showlegend=True,
                    legendgroup='colors',
                    legendgrouptitle_text='-log₁₀(BH adj. p-value)',
                    name=f'> {threshold}'
                ))
        fig.update_layout(
            yaxis_title='Depleted ← log₂(odds ratio) → Enriched',
            yaxis_title_standoff=0,
            barmode='relative',
            width=600 + len(x) * 20,
            height=500,
            xaxis=dict(
                categoryarray=x_sorted,
                tickangle=0 if any(val in {'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'} for val in y) else -45, # -90 = vertical labels <> -45 = diagonal labels <> 0 = horizontal label rotation
                tickfont=dict(size=12),
                automargin=True,
                showgrid=True,
                gridcolor='rgb(229, 236, 246)',
                gridwidth=0.5),
            yaxis=dict(
                range=[min(tick_vals) - 0.5, max(tick_vals) + 0.5],
                tickvals=tick_vals,
                ticktext=tick_text,
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='black',
                showgrid=True,
                gridcolor='rgb(229, 236, 246)',
                gridwidth=1,
                title_font=dict(size=16)
            ),
            plot_bgcolor='white',
            showlegend=True,
            margin=dict(
                l=20,
                r=20,
                b=20,  # Further reduce bottom margin
                t=20,  # Further reduce top margin
                pad=1
            )    
        )
        
        return fig 
    
def multiple_vis(
    data: pd.DataFrame,
    x: set,
    y: set,
    BH: float
):

    df, x, y, sorted_x, color_ranges, tick_vals, tick_text, x_axis_range = prepare_plot(data, x, y, BH)
    
    # Load the domain descriptions
    with open('data/domain_descriptions.json', 'r') as f:
        domain_descriptions = json.load(f)
       
    # Define modification symbols for endpoints
    x_symbols = {
        x: symbol for x, symbol in zip(sorted_x, ['circle', 'diamond', 'star', 'square', 'pentagon'])
    }
        
    # Create combined i and description, with line break
    if len(x & set(df.x.unique())) != 0:
        df['description'] = df['y'].apply(lambda x: f"{x}<br>{domain_descriptions[x]}" if x in domain_descriptions else x)
    else:
        df['description'] = df['x'].apply(lambda x: f"{x}<br>{domain_descriptions[x]}" if x in domain_descriptions else x)
    # Change sorting to alphabetical order
    unique_descriptions = sorted(df['description'].unique(), key=lambda x: int(x.split(']')[0][1:]) if x.startswith('[') else x)

    custom_order = ['3₁₀-helix', 'α-helix', 'π-helix', 'PPII-helix', 'ß-strand', 'ß-bridge', 'turn', 'bend', 'unassigned', 'loop', 'IDR']
    # Create a mapping from description to its order index
    order_map = {desc: i for i, desc in enumerate(custom_order)}
    # Only sort if there is overlap
    if len(set(unique_descriptions) & set(custom_order)) != 0:
        unique_descriptions = sorted(
            unique_descriptions,
            key=lambda x: order_map.get(x, len(custom_order))  # items not in custom_order go last
        )

    # Calculate vertical positions for grouped lines
    description_to_y = {description: i for i, description in enumerate(unique_descriptions)}
    line_spacing = 0.2  # Spacing between lines in the same group
    
    # First, count how many x are present for each description
    x_per_description = {}
    for desc in unique_descriptions:
        x_per_description[desc] = df[df['description'] == desc].shape[0]

    # Keep track of the current x count for each description
    current_ptm_count = {desc: 0 for desc in unique_descriptions}

    # Create a custom figure
    fig = go.Figure()
    
    # Keep track of used colors
    used_colors = set()
    
    # First, check if values in y are in the x column or y column
    values_in_x = any(val in df['x'].values for val in sorted_x)
    values_in_y = any(val in df['y'].values for val in sorted_x)
    
    # Determine which column to use for filtering
    filter_column = 'x' if values_in_x else ('y' if values_in_y else None)

    # Add lines and endpoints for each x
    for _, x in enumerate(sorted_x):
        ptm_data = df[df[filter_column] == x]
        for _, row in ptm_data.iterrows():
            # Determine color based on adj. p-value
            color = 'grey'
            for threshold, col in color_ranges:
                if -np.log10(row['p_adj_bh']) > threshold:
                    color = col
                    break
            used_colors.add(color)
            
            # Calculate y-position for this line
            base_y = description_to_y[row['description']]
            
            # Only apply offset if there's more than one x for this description
            total_ptms_for_desc = x_per_description[row['description']]
            if total_ptms_for_desc > 1:
                offset = (current_ptm_count[row['description']] - (total_ptms_for_desc - 1) / 2) * line_spacing
            else:
                offset = 0
                
            y_pos = base_y + offset
            
            # Increment the x counter for this description
            current_ptm_count[row['description']] += 1
            
            # Add line from 0 to log2_odds_ratio (or to our edge position for inf values)
            fig.add_trace(go.Scatter(
                name=x,
                y=[y_pos, y_pos],
                x=[0, row['display_log2_odds_ratio']],
                mode='lines',
                line=dict(color=color, width=2),
                showlegend=False,
                hoverinfo='none'
            ))
            
            # Add symbol at the end of the line
            fig.add_trace(go.Scatter(
                name=x,
                y=[y_pos],
                x=[row['display_log2_odds_ratio']],
                mode='markers',
                marker=dict(
                    symbol=x_symbols[x],
                    size=10,
                    color=color,
                    line=dict(color='white', width=1)
                ),
                showlegend=False,
                hoverinfo='none'
            ))

    # Add modification symbols to legend with corresponding names
    for x in sorted_x:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, symbol=x_symbols[x], color='black'),
            name=x,
            showlegend=True,
            legendgroup='symbols',
            legendgrouptitle_text='Annotation',
            legendgrouptitle_font=dict(size=12)
        ))
    
    # Add colors to legend with corresponding adj. p-value thresholds
    for threshold, color in color_ranges:
        if color in used_colors:
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color=color, symbol='square'),
                showlegend=True,
                legendgroup='colors',
                legendgrouptitle_text='-log₁₀(BH adj. p-value)',
                name=f'> {threshold}'
            ))

    # Add shapes for all grid lines FIRST (so they're underneath the data)
    for i in range(len(unique_descriptions)):       
        # Add intermediate gridline if not the last position
        if i < len(unique_descriptions) - 1:
            fig.add_shape(
                type="line",
                x0=x_axis_range[0],
                y0=i + 0.5,  # Positioned halfway between labels
                x1=x_axis_range[1],
                y1=i + 0.5,
                line=dict(color="rgb(229, 236, 246)", width=2, dash="solid"),
                layer="below"
            )
    
    # Update layout
    fig.update_layout(
        title=None,
        width=800,
        height= max(300, 100 + (len(unique_descriptions) * 50)),  # Increased spacing for better readability
        margin=dict(
            l=20,  # Increased left margin for longer IPR and descriptions
            r=20,
            t=60,
            b=20,
            pad=1
        ),
        yaxis=dict(
            showgrid=False,
            side='right',
            ticktext=unique_descriptions,
            tickvals=list(range(len(unique_descriptions))),
            showline=False,
            mirror=False,
            range=[len(unique_descriptions) - 0.5, -0.5],  # Reverse the range to put first element at top
            tickfont=dict(size=12),
            ticks='',
            zeroline=False

        ),
        xaxis=dict(
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=1,
            side="top",
            showgrid=True,
            gridwidth=1,
            gridcolor='rgb(229, 236, 246)',
            range=x_axis_range,  # Use the dynamically determined range
            tickvals=tick_vals,
            ticktext=tick_text,
            tickangle=0
        ),
        legend=dict(
            xanchor="right",
            x=0.02
        ),
        plot_bgcolor='white',
        showlegend=True
    )

    fig.add_annotation(
    text="Depleted ← log₂(odds ratio) → Enriched",
    font=dict(size=16),
    xref='x',         # Position in x-axis data coordinates
    yref='paper',     # Position in normalized plot coordinates (0=bottom, 1=top)
    x=0,              # Centered at x=0
    y=1,              # Exactly at the x-axis
    yshift=40,        # Shift up by 24 pixels (adjust as needed)
    showarrow=False,
    xanchor='center',
    xshift=16        # Optional: fine-tune horizontal alignment
)
    
    return fig 

import argparse
import textwrap

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='''Visualise enrichment/depletion results from protmodcon function.''',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''\
            Example:
                python my_plotting.py --data results_protmodcon/ptm_name_IN_AA_IPR000182_proteome_wide.csv --x "[34]Methyl" --BH 0.05 --output my_plot.png''')
    )
    parser.add_argument(
        '--data', required=True,
        help='Path to data to be visualised. Should be output of the protmodcon function.'
    )
    parser.add_argument('--x', type=str, required=True,
        help='''Comma-separated list for x. A maximum of five annotations may be specified within any single annotation category (ptm_name, AA, sec, domain). For the ptm_name category, annotations must be formatted using double quotes, like "[Unimod accession]name". The name should correspond to the Unimod PSI-MS Name or, if that is unavailable, the Unimod Interim name. For example: "[1]Acetyl" or "[21]Phospho". Valid ptm_names can be found as keys in data/ptm_name_AA.json. For the AA category, specify amino acids using their single-letter codes, e.g., C S. For the sec category, indicate secondary structure elements such as 310HELX, AHELX, PIHELX, PPIIHELX, STRAND, BRIDGE, TURN, BEND, unassigned, LOOP, IDR. If only a single annotation (x) is provided, a different plot type will be generated compared to when multiple annotations (up to five) are specified. If you want the multiple-like visualisation for a single annotation, you can add an invalid argument to --x (next to your argument of interest).
        '''
    )
    parser.add_argument(
        '--y', type=str, required=False,
        help='Optional: Comma-separated list for y. Regions of interest to be visualised. This pre-filters the provided data to only show enrichment/depletions in the regions of interest.'
    )
    parser.add_argument(
        '--BH', default=0.01,
        help='Threshold for Benjamini-Hochberg multiple testing corrected p-values. Default 0.01.'
    )
    parser.add_argument(
        '--output', required=False, default='plot.png',
        help='Filename for the output plot. Default: plot.png'
    )
    
    args = parser.parse_args()
    def parse_list(arg):
    return [item.strip() for item in arg.split(',') if item.strip()]

    x_data = parse_list(args.x)
    y_data = parse_list(args.y)
    
    if len(set(args.x)) == 1:
        fig = single_vis(
                data=args.data,
                x=set(args.x),
                y=set(args.y),        
                BH=float(args.BH)
        )

    elif 1 < len(set(args.x)) <= 5:
        fig = multiple_vis(
                data=args.data,
                x=set(args.x),
                y=set(args.y),        
                BH=float(args.BH)
        )

    else:
        print("--x can contain at most 5 elements, otherwise the plot becomes too messy. No visualisation created.")

    if fig:   
        output_dir = 'figures_protmodcon'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, args.output)

        fig.write_image(
            output_path,
            scale=2,
            width=None,
            height=None,
            engine="kaleido"
         )
        print(f"{output_path} created.")

    else:
        print("No signficant results, so no file created.")
        