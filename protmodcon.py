#!/usr/bin/env python3
import importlib
import sys
import subprocess
import json
import os
import re
from multiprocessing import Pool
from functools import lru_cache
import time
import pandas as pd
import numpy as np
import scipy.stats
from statsmodels.stats.multitest import multipletests
import numba as nb
from tqdm import tqdm
import plotly.graph_objects as go
import argparse
import textwrap

def install_if_missing(package):
    try:
        importlib.import_module(package)
    except ImportError:
        print(f"Installing '{package}'...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        
# Use numba to accelerate calculations
@nb.jit(nopython=True)
def calculate_expected_values(row1_sum, row2_sum, col1_sum, col2_sum, total_sum):
    """Calculate expected values for contingency tables"""
    expected_a = (row1_sum * col1_sum) / total_sum
    expected_b = (row1_sum * col2_sum) / total_sum
    expected_c = (row2_sum * col1_sum) / total_sum
    expected_d = (row2_sum * col2_sum) / total_sum
    return expected_a, expected_b, expected_c, expected_d

@nb.jit(nopython=True)
def calculate_odds_ratios(table_a, table_b, table_c, table_d):
    """Calculate odds ratios from contingency table values"""
    return (table_a * table_d) / (table_b * table_c)

# Function to compute chi-square for parallel processing
def compute_chisq(argms):
    idx, a, b, c, d = argms
    """Compute chi-square test for a single contingency table"""
    table = np.array([[a, b], [c, d]])
    _, p, _, _ = scipy.stats.chi2_contingency(table)
    return p

def process_i(x_types, y_types, n, nm_per_d, nc_per_i, k_per_d_i):
    """Process the analysis with optimized computation - parallel version"""    
    start_time = time.time()
    
    # Vectorize operations for improved performance
    x_array = np.array([d for d in x_types for _ in y_types])
    y_array = np.array([i for _ in x_types for i in y_types])
    n_array = np.array([n for _ in x_array], dtype=np.int32)
    nm_array = np.array([nm_per_d[d] for d in x_array], dtype=np.int32)
    nc_array = np.array([nc_per_i[i] for i in y_array], dtype=np.int32)
    k_array = np.array([k_per_d_i.get((d, i), 0) for d, i in zip(x_array, y_array)], dtype=np.int32)
    # Efficient computation of contingency tables
    table_a = k_array  # nx_in_y
    table_b = nc_array - k_array  # not_nx_in_y
    table_c = nm_array - k_array  # nx_not_in_y
    table_d = n_array - nm_array - nc_array + k_array  # not_nx_not_in_y
    
    # Expected values calculation
    row1_sum = table_a + table_b
    row2_sum = table_c + table_d
    col1_sum = table_a + table_c
    col2_sum = table_b + table_d
    total_sum = row1_sum + row2_sum
    # Add a check to avoid division by zero
    mask = total_sum > 0
    
    # Only calculate where total_sum > 0
    expected_a = np.zeros_like(total_sum, dtype=float)
    expected_b = np.zeros_like(total_sum, dtype=float)
    expected_c = np.zeros_like(total_sum, dtype=float)
    expected_d = np.zeros_like(total_sum, dtype=float)
    
    # Extract valid indices for processing
    expected_a[mask], expected_b[mask], expected_c[mask], expected_d[mask] = calculate_expected_values(
        row1_sum[mask], row2_sum[mask], col1_sum[mask], col2_sum[mask], total_sum[mask])
    
    valid_indices = np.where(
        (expected_a >= 5) & (expected_b >= 5) & (expected_c >= 5) & (expected_d >= 5) & (total_sum > 0))[0]
    
    # Handle case with no valid tables
    if len(valid_indices) == 0:
        print("No valid contingency tables found.")
        return pd.DataFrame(columns=[
            'd', 'i', 'nx_in_y', 'nx_not_in_y', 
            'not_nx_in_y', 'not_nx_not_in_y', 'oddsr', 'p'
        ])
    
    print(f"Found {len(valid_indices)} valid tables out of {len(x_array)} combinations.")
    
    # Vectorized odds ratio calculation for valid indices
    odds_ratios = calculate_odds_ratios(
        table_a[valid_indices], 
        table_b[valid_indices], 
        table_c[valid_indices], 
        table_d[valid_indices]
    )
    
    # Prepare arguments for parallel chi-square computation
    chisq_argms = [
        (idx, table_a[idx], table_b[idx], table_c[idx], table_d[idx]) 
        for idx in valid_indices
    ]

    # Parallel computation of p-values
    num_processes = min(os.cpu_count(), len(valid_indices))
    if num_processes > 1 and len(valid_indices) > 10:  # Only use multiprocessing for larger datasets
        print(f"Using {num_processes} processes for chi-square calculations")
        with Pool(processes=num_processes) as pool:
            p_values = list(tqdm(
                pool.imap(compute_chisq, chisq_argms, chunksize=max(1, len(chisq_argms) // (num_processes * 4))),
                total=len(chisq_argms),
                desc="Computing p-values"
            ))
    else:
        # Fall back to sequential for small datasets
        print("Computing p-values sequentially")
        p_values = [compute_chisq(argms) for argms in tqdm(chisq_argms, desc="Computing p-values")]
    
    # Create DataFrame for valid results
    result_df = pd.DataFrame({
        'x': x_array[valid_indices],
        'y': y_array[valid_indices],
        'nx_in_y': table_a[valid_indices],
        'nx_not_in_y': table_c[valid_indices],
        'not_nx_in_y': table_b[valid_indices],
        'not_nx_not_in_y': table_d[valid_indices],
        'oddsr': odds_ratios,
        'p': p_values
    })
    
    print(f"Analysis processing completed in {time.time() - start_time:.2f} seconds")
    return result_df
    
def perform_enrichment_analysis(
    x_types: set,
    y_types: set,
    filters: list,
    modifiability: dict,
    x_mode: str,
    y_mode: str
) -> pd.DataFrame:
    """
    Perform enrichment analysis to evaluate which modifications or annotations (x_types) are enriched / depleted 
    within specific protein features (y_types).

    Example usage:
        "Which modifications (x_types) are enriched in secondary structures (y_types)?"

    Parameters
    ----------
    x_types : set
        The "dependent data type" for enrichment analysis.
    
    y_types : set
        The "independent data type" for enrichment analysis.
    
    filters: list, optional
        Lists specifying which amino acids, secondary structures, domains, or proteins to include in the analysis.
        By default, all lists are empty, resulting in a proteome-wide analysis.
        
    modifiability : dict, optional
        Dictionary specifying eligible amino acids for each modification type. 
        Example: {'[21]Phospho': ['S', 'T', 'Y']}.
        If empty, all amino acids evaluated by ionbot are considered modifiable for each ptm_name.

    x_mode / y_mode : str, required
        Whether to calculate individual enrichment or for a group (e.g. proteins) of interest

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the results of the enrichment analysis.
    """
    """
    ptm_name_AA = {}
    if x_types[0].startswith('['):
        with open('data/ptm_name_AA.json') as file:
            ptm_name_AA = json.load(file)
        if len(modifiability) != 0:
            for d in modifiability:
                ptm_name_AA[d] = modifiability[d]
    # also for y?
    # Process modifiability argument if provided
    if modifiability:
        for mod_str in modifiability:
            ptm, aas = mod_str.split(':')
            modifiability[ptm] = aas.split(',')
    """   
                
    # Mapping for secondary structure elements (to overcome HTML issues)
    sec_mapping = {
        '310HELX': '3₁₀-helix',
         'AHELX': 'α-helix',
         'PIHELX': 'π-helix',
         'PPIIHELX': 'PPII-helix',
         'STRAND': 'ß-strand',
         'BRIDGE': 'ß-bridge',
         'TURN': 'turn',
         'BEND': 'bend',
         'unassigned': 'unassigned',
         'LOOP': 'loop',
         'IDR': 'IDR'
    }
        
    def map_types(types, mapping):
        mapped = [mapping[sec] for sec in types if sec in mapping]
        return mapped if mapped else types
    
    x_types = map_types(x_types, sec_mapping)
    y_types = map_types(y_types, sec_mapping)
    if filters:
        filters = map_types(y_types, sec_mapping)
    
    # Check if there exists a file in which user's request has already been calculated
    file_path = 'data/requests.json'
    request = [x_types, y_types, filters, modifiability, [x_mode], [y_mode]]

    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            requests = json.load(f)
        # get index of the request and the corresponding filename (its index)
        try:
            idx = requests.index(request)
            filename = f'results_protmodcon/{idx}.csv'
            print(f"Results saved to {filename}")
            return # stop function
        except ValueError:
            ''

    # Update types - convert to tuples for caching
    x_types = tuple(sorted(x_types, key=lambda x: int(x.split(']')[0][1:]) if x.startswith('[') else x))
    y_types = tuple(sorted(y_types, key=lambda x: int(x.split(']')[0][1:]) if x.startswith('[') else x))
    
    total_start_time = time.time()

    protmodcon = pd.read_csv('protmodcon.csv')
    
    if filters:
        mask = protmodcon.isin(filters).any(axis=1)
        protmodcon = protmodcon[mask]

    n = protmodcon[['protein_id', 'position']].drop_duplicates().shape[0]
        
    # compute pairwise overlaps across all columns
    nm_per_d, nc_per_i, k_per_d_i = {}, {}, {}
    
    def get_mask_set(types):
        mask = protmodcon.isin([types]).any(axis=1)
        return set(protmodcon[mask]['protein_id'].astype(str) + '_' + protmodcon[mask]['position'].astype(str))

    if x_mode == "x_bulk":
        x = x_types
        x_set = get_mask_set(x)
        nm_per_d[x] = len(x_set)
        x_iter = [x]
    else:
        x_iter = x_types
    
    if y_mode == "y_bulk":
        y = y_types
        y_set = get_mask_set(y)
        nc_per_i[y] = len(y_set)
        y_iter = [y]
    else:
        y_iter = y_types
    
    for x in x_iter:
        if x_mode == "x_individual":
            x_set = get_mask_set(x)
            nm_per_d[x] = len(x_set)
        for y in y_iter:
            if y_mode == "y_individual":
                y_set = get_mask_set(y)
                nc_per_i[y] = len(y_set)
            k_per_d_i[(x, y)] = len(get_mask_set(x) & get_mask_set(y))

    print("Processing data...")
    start_time = time.time()

    print(x_types, y_types, n, nm_per_d, nc_per_i, k_per_d_i)
          
    # Process the data
    results_df = process_i(x_types, y_types, n, nm_per_d, nc_per_i, k_per_d_i)
    
    print(f"Data processed in {time.time() - start_time:.2f} seconds")

    # Apply multiple testing correction if needed and if there are results
    if not results_df.empty:
        print("Applying multiple testing correction...")
        results_df['p_adj_bh'] = multipletests(
            pvals=results_df.p, method='fdr_bh')[1]
   
    # Custom sorting function for ptm_names (contain brackets)
    def custom_sort(value):
        if isinstance(value, str) and value.startswith('['):
            try:
                return int(value.split(']')[0][1:])  # Extract integer from bracketed values
            except ValueError:
                return value
        return value  # Use original value for non-bracketed values
    
    print(f"Analysis complete. Found {len(results_df)} results.")
    print(f"Total analysis time: {time.time() - total_start_time:.2f} seconds")
    
    # Apply custom sorting
    if not results_df.empty:
        results_df = results_df.sort_values(
            by=['x', 'y'], 
            key=lambda x: x.map(custom_sort)
        )

        if not os.path.exists(file_path):
            # Initialize file with the first request
            requests = [request]
            idx = 0
        else:
            with open(file_path, 'r') as f:
                requests = json.load(f)
                requests.append(request)
                idx = len(requests) - 1
        
        # Save the updated requests back to the file
        with open(file_path, 'w') as f:
            json.dump(requests, f)
        
        # idx now holds the index of your request in the file        
        filename = f'results_protmodcon/{idx}.csv'
        results_df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='protmodcon: Comprehensive Analysis of Protein Modifications from a Conformational Perspective',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''\
            Example:
                python protmodcon.py --x-types [21]Phospho --y-types AHELX
                sec options = 310HELX, AHELX, PIHELX, PPIIHELX, STRAND, BRIDGE, TURN, BEND, unassigned, LOOP, IDR
            ''')
    )
    parser.add_argument(
        '--x-types', required=True, nargs='+',
        help='First data type for enrichment analysis, arbitrarily indicated by x.'
    )
    parser.add_argument(
        '--y-types', required=True, nargs='+',
        help='Second data type for enrichment analysis, arbitrarily indicated by y.'
    )
    parser.add_argument(
        '--filters', nargs='*', default=[],
        help='Optional: list of filters'
    )
    parser.add_argument(
        '--modifiability', nargs='*',
        help='''\
        Optional: Modifiability of amino acids per modification type. Format: "[Unimod accession]name:AA1,AA2", where the name corresponds to the Unimod
        PSI-MS Name or, if unavailable, the Unimod Interim name, e.g. --modifiability "[21]Phospho:S,T,Y" (double quotes mandatory). Valid ptm_name-AA 
        combinations can be found as keys in data/ptm_name_AA.json.'''
    )
    parser.add_argument(
        '--x-mode',
        required=True,
        choices=['x_individual', 'x_bulk'],
        help='Required: Specify the mode for x. Must be "x_individual" or "x_bulk".'
    )
    parser.add_argument(
        '--y-mode',
        required=True,
        choices=['y_individual', 'y_bulk'],
        help='Required: Specify the mode for y. Must be "y_individual" or "y_bulk".'
    )

    args = parser.parse_args()

    # Only install third-party packages (not standard library modules)
    third_party_packages = [
        "pandas",
        "numpy",
        "scipy",
        "statsmodels",
        "numba",
        "tqdm",
        "plotly",
        "kaleido"
    ]
    
    for package in third_party_packages:
        install_if_missing(package)

    # Run enrichment analysis
    perform_enrichment_analysis(
        x_types=args.x_types,
        y_types=args.y_types,
        filters=args.filters,
        modifiability=args.modifiability,
        x_mode=args.x_mode,
        y_mode=args.y_mode,
    )