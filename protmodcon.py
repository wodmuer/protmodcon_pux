#!/usr/bin/env python3
import importlib
import sys
import subprocess
import json
import os
import re
from multiprocessing import Pool, cpu_count
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
from pathlib import Path
import pickle
import hashlib

@nb.njit(parallel=True, fastmath=True)
def calculate_expected_values_vectorized(row1_sum, row2_sum, col1_sum, col2_sum, total_sum):
    n = len(row1_sum)
    expected_a = np.empty(n, dtype=np.float64)
    expected_b = np.empty(n, dtype=np.float64)
    expected_c = np.empty(n, dtype=np.float64)
    expected_d = np.empty(n, dtype=np.float64)
    for i in nb.prange(n):
        if total_sum[i] > 0:
            expected_a[i] = (row1_sum[i] * col1_sum[i]) / total_sum[i]
            expected_b[i] = (row1_sum[i] * col2_sum[i]) / total_sum[i]
            expected_c[i] = (row2_sum[i] * col1_sum[i]) / total_sum[i]
            expected_d[i] = (row2_sum[i] * col2_sum[i]) / total_sum[i]
        else:
            expected_a[i] = expected_b[i] = expected_c[i] = expected_d[i] = 0.0
    return expected_a, expected_b, expected_c, expected_d

@nb.njit(parallel=True, fastmath=True)
def calculate_odds_ratios_vectorized(table_a, table_b, table_c, table_d):
    n = len(table_a)
    odds_ratios = np.empty(n, dtype=np.float64)
    for i in nb.prange(n):
        denominator = table_b[i] * table_c[i]
        if denominator != 0:
            odds_ratios[i] = (table_a[i] * table_d[i]) / denominator
        else:
            odds_ratios[i] = np.inf
    return odds_ratios

@nb.njit(parallel=True, fastmath=True)
def create_contingency_tables_vectorized(k_array, nc_array, nm_array, n_array):
    n = len(k_array)
    table_a = np.empty(n, dtype=np.int32)
    table_b = np.empty(n, dtype=np.int32)
    table_c = np.empty(n, dtype=np.int32)
    table_d = np.empty(n, dtype=np.int32)
    for i in nb.prange(n):
        table_a[i] = k_array[i]
        table_b[i] = nc_array[i] - k_array[i]
        table_c[i] = nm_array[i] - k_array[i]
        table_d[i] = n_array[i] - nm_array[i] - nc_array[i] + k_array[i]
    return table_a, table_b, table_c, table_d

def compute_chisq_batch(tables_batch):
    p_values = []
    for table_a, table_b, table_c, table_d in tables_batch:
        try:
            table = np.array([[table_a, table_b], [table_c, table_d]], dtype=np.float64)
            _, p, _, _ = scipy.stats.chi2_contingency(table, correction=False)
            p_values.append(p)
        except (ValueError, ZeroDivisionError):
            p_values.append(1.0)
    return p_values

class DataCache:
    def __init__(self, cache_dir='cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    def get_cache_key(self, *args):
        key_str = repr(args)
        return hashlib.md5(key_str.encode()).hexdigest()
    def get(self, key):
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None
    def set(self, key, value):
        cache_file = self.cache_dir / f"{key}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(value, f)

cache = DataCache()

def create_type_mapping_optimized(protmodcon, all_types):
    type_to_posids = {}
    type_columns = [col for col in protmodcon.columns if col not in ['position', 'pos_id', 'protein_position']]
    for t in all_types:
        mask = (protmodcon[type_columns] == t).any(axis=1)
        type_to_posids[t] = set(protmodcon.loc[mask, 'pos_id'].to_numpy())
    return type_to_posids

def create_indicator_matrix_vectorized(types_list, type_to_posids, n, posid_to_index):
    # Build a 2D indicator matrix (rows: types_list, cols: positions)
    matrix = np.zeros((len(types_list), n), dtype=np.uint8)
    for i, types in enumerate(types_list):
        pos_ids = set()
        for t in types:
            pos_ids |= type_to_posids.get(t, set())
        if pos_ids:
            indices = [posid_to_index[pid] for pid in pos_ids if pid in posid_to_index]
            matrix[i, indices] = 1
    return matrix

def process_enrichment_optimized(x_types, y_types, n, nm_per_d, nc_per_i, k_per_d_i):
    start_time = time.time()
    combinations = [(d, i) for d in x_types for i in y_types]
    x_array = np.array([d for d, _ in combinations])
    y_array = np.array([i for _, i in combinations])
    n_array = np.full(len(combinations), n, dtype=np.int32)
    nm_array = np.array([nm_per_d[d] for d, _ in combinations], dtype=np.int32)
    nc_array = np.array([nc_per_i[i] for _, i in combinations], dtype=np.int32)
    k_array = np.array([k_per_d_i.get((d, i), 0) for d, i in combinations], dtype=np.int32)
    table_a, table_b, table_c, table_d = create_contingency_tables_vectorized(
        k_array, nc_array, nm_array, n_array
    )
    row1_sum = table_a + table_b
    row2_sum = table_c + table_d
    col1_sum = table_a + table_c
    col2_sum = table_b + table_d
    total_sum = row1_sum + row2_sum
    expected_a, expected_b, expected_c, expected_d = calculate_expected_values_vectorized(
        row1_sum, row2_sum, col1_sum, col2_sum, total_sum
    )
    valid_mask = (
        (expected_a >= 5) & (expected_b >= 5) &
        (expected_c >= 5) & (expected_d >= 5) & (total_sum > 0)
    )
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) == 0:
        print("No valid contingency tables found.")
        return pd.DataFrame(columns=[
            'x', 'y', 'nx_in_y', 'nx_not_in_y',
            'not_nx_in_y', 'not_nx_not_in_y', 'oddsr', 'p'
        ])
    print(f"Found {len(valid_indices)} valid tables out of {len(combinations)} combinations.")
    odds_ratios = calculate_odds_ratios_vectorized(
        table_a[valid_indices], table_b[valid_indices],
        table_c[valid_indices], table_d[valid_indices]
    )
    valid_tables = list(zip(
        table_a[valid_indices], table_b[valid_indices],
        table_c[valid_indices], table_d[valid_indices]
    ))
    batch_size = min(4000, len(valid_tables))
    batches = [valid_tables[i:i + batch_size] for i in range(0, len(valid_tables), batch_size)]
    num_processes = min(cpu_count() or 1, len(batches))
    if num_processes > 1 and len(valid_tables) > 100:
        print(f"Computing p-values in parallel with {num_processes} processes")
        with Pool(processes=num_processes) as pool:
            batch_results = list(tqdm(
                pool.imap(compute_chisq_batch, batches),
                total=len(batches),
                desc="Computing p-values"
            ))
        p_values = [p for batch in batch_results for p in batch]
    else:
        print("Computing p-values sequentially")
        p_values = []
        for batch in tqdm(batches, desc="Computing p-values"):
            p_values.extend(compute_chisq_batch(batch))
    
    def smart_join(subarr):
        # If it's a list or array with more than one element, join
        if isinstance(subarr, (list, np.ndarray)):
            if len(subarr) > 1:
                return '_'.join(subarr)
            else:
                return subarr[0]
        # If it's already a string, just return it
        return subarr

    result_df = pd.DataFrame({
        'x': [smart_join(subarr) for subarr in x_array[valid_indices]],
        'y': [smart_join(subarr) for subarr in y_array[valid_indices]],
        'nx_in_y': table_a[valid_indices],
        'nx_not_in_y': table_c[valid_indices],
        'not_nx_in_y': table_b[valid_indices],
        'not_nx_not_in_y': table_d[valid_indices],
        'oddsr': odds_ratios,
        'p': p_values
    })
    print(f"Enrichment processing completed in {time.time() - start_time:.2f} seconds")
    return result_df

def perform_enrichment_analysis(
    x_types: set,
    y_types: set,
    filters: list,
    modifiability: dict,
    x_mode: str,
    y_mode: str
) -> pd.DataFrame:
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
        filters = map_types(filters, sec_mapping)
    request_key = cache.get_cache_key(x_types, y_types, filters, modifiability, x_mode, y_mode)
    print('protmodcon', x_types, y_types, filters, modifiability, x_mode, y_mode)
    cached_result = cache.get(f"analysis_{request_key}")
    if cached_result is not None:
        return f"Using cached analysis results: {request_key}"
    x_types = tuple(sorted(x_types, key=lambda x: int(x.split(']')[0][1:]) if x.startswith('[') else x))
    y_types = tuple(sorted(y_types, key=lambda x: int(x.split(']')[0][1:]) if x.startswith('[') else x))
    total_start_time = time.time()
    dtype_dict = {'protein_id': 'category', 'position': 'int32'}
    protmodcon = pd.read_csv('protmodcon.csv', dtype=dtype_dict, low_memory=False)
    
    # Step 1: Get a set of pos_ids matching the filter
    if filters:
        mask = protmodcon['protein_id'].isin(filters) | protmodcon['annotation'].isin(filters)
        pos_id_set = set(protmodcon.loc[mask, 'pos_id'])
        protmodcon = protmodcon[protmodcon['pos_id'].isin(pos_id_set)]

    n = protmodcon['pos_id'].nunique()
    all_types = set(x_types) | set(y_types)
    type_to_posids = create_type_mapping_optimized(protmodcon, all_types)

    # Determine iteration modes
    if x_mode == "x_bulk":
        x_iter = [tuple(x_types)]
        x_types_for_matrix = [x_types]
    else:
        x_iter = x_types
        x_types_for_matrix = [[x] for x in x_types]
    if y_mode == "y_bulk":
        y_iter = [tuple(y_types)]
        y_types_for_matrix = [y_types]
    else:
        y_iter = y_types
        y_types_for_matrix = [[y] for y in y_types]

    # Build a mapping from pos_id to matrix index
    unique_pos_ids = sorted(protmodcon['pos_id'].unique())
    posid_to_index = {pid: idx for idx, pid in enumerate(unique_pos_ids)}
    n = len(unique_pos_ids)  # Ensure n matches the number of unique pos_ids
    
    x_matrix = create_indicator_matrix_vectorized(x_types_for_matrix, type_to_posids, n, posid_to_index)
    y_matrix = create_indicator_matrix_vectorized(y_types_for_matrix, type_to_posids, n, posid_to_index)
    
    # x_matrix: shape (n_x, n_positions)
    # y_matrix: shape (n_y, n_positions)

    # Expand x_matrix and y_matrix to compare every row pair
    batch_size = 100  # adjust as needed for memory
    k_matrix = np.zeros((x_matrix.shape[0], y_matrix.shape[0]), dtype=np.int32)
    for i in range(x_matrix.shape[0]):
        for j_start in range(0, y_matrix.shape[0], batch_size):
            j_end = min(j_start + batch_size, y_matrix.shape[0])
            k_matrix[i, j_start:j_end] = np.sum(
                np.logical_and(x_matrix[i, :], y_matrix[j_start:j_end, :]), axis=1
            )

    nm_per_d = {x: int(x_matrix[i].sum()) for i, x in enumerate(x_iter)}
    nc_per_i = {y: int(y_matrix[j].sum()) for j, y in enumerate(y_iter)}
    k_per_d_i = {(x, y): int(k_matrix[i, j]) for i, x in enumerate(x_iter) for j, y in enumerate(y_iter)}

    print("Processing enrichment analysis...")
    results_df = process_enrichment_optimized(x_iter, y_iter, n, nm_per_d, nc_per_i, k_per_d_i)
    if not results_df.empty:
        print("Applying multiple testing correction...")
        results_df['p_adj_bh'] = multipletests(results_df['p'], method='fdr_bh')[1]
        def custom_sort_key(series):
            def extract_key(value):
                if isinstance(value, str) and value.startswith('['):
                    try:
                        return int(value.split(']')[0][1:])
                    except ValueError:
                        return value
                return value
            return series.map(extract_key)
        results_df = results_df.sort_values(
            by=['x', 'y'],
            key=custom_sort_key
        )
    cache.set(f"analysis_{request_key}", results_df)
    print(f"Analysis complete. Found {len(results_df)} results.")
    print(f"Total analysis time: {time.time() - total_start_time:.2f} seconds")
    os.makedirs('results_protmodcon', exist_ok=True)
    timestamp = int(time.time())
    filename = f'results_protmodcon/analysis_{timestamp}.csv'
    results_df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")
    return results_df

def parse_modifiability(modifiability_args):
    # Accepts a list of strings like "[21]Phospho:S,T,Y"
    result = {}
    for item in modifiability_args:
        if ':' in item:
            k, v = item.split(':', 1)
            result[k.strip()] = [aa.strip() for aa in v.split(',')]
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='protmodcon: Comprehensive Analysis of Protein Modifications from a Conformational Perspective',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''\
            Example:
                python protmodcon.py --x-types [21]Phospho --y-types AHELX --x-mode x_individual --y-mode y_individual
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
        '--modifiability', nargs='*', default=[],
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
    third_party_packages = [
        "pandas", "numpy", "scipy", "statsmodels",
        "numba", "tqdm", "plotly", "kaleido"
    ]
    for package in third_party_packages:
        install_if_missing(package)
    modifiability = parse_modifiability(args.modifiability)
    perform_enrichment_analysis(
        x_types=args.x_types,
        y_types=args.y_types,
        filters=args.filters,
        modifiability=modifiability,
        x_mode=args.x_mode,
        y_mode=args.y_mode,
    )
