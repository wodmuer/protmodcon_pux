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
def compute_chisq(args):
    """Compute chi-square test for a single contingency table"""
    idx, a, b, c, d = args
    table = np.array([[a, b], [c, d]])
    _, p, _, _ = scipy.stats.chi2_contingency(table)
    return p

def get_n(protein_id_annotation_position, protein_id_position_AA, t, AAs=[]):
    """Highly optimized get_n function with advanced filtering"""
        
    # Fast path: check if annotation exists at all
    has_annotation = False
    for annotations in protein_id_annotation_position.values():
        if t in annotations:
            has_annotation = True
            break
            
    if not has_annotation:
        return set()
    # Use set comprehension for better performance
    result = set()
    for protein_id, annotations in protein_id_annotation_position.items():
        if t in annotations:
            positions = annotations.get(t, [])
            
            # If we need to filter by AAs, do it efficiently
            if len(AAs) != 0:
                for position in positions:
                    key = (protein_id, position)
                    AA = protein_id_position_AA.get(key)
                    if AA in AAs:
                        result.add(key)
            else:
                # No AA filtering needed, add all positions
                result.update((protein_id, position) for position in positions)
    
    return result

def precompute_nm_nc_k(x_types, y_types, ptm_name_AA, protein_id_annotation_position, protein_id_position_AA):
    """Precompute nm_per_d, nc_per_i and k_per_d_i with optimized caching strategy."""
    start_time = time.time()
    
    # First precompute all positions for each annotation - do the work once
    print("Caching positions for all annotation types...")
    position_cache = {}
    
    # Process all unique annotation types at once
    types = list(set(x_types) | set(y_types))
    
    for t in tqdm(types, desc="Building position cache"):
        position_cache[t] = get_n(protein_id_annotation_position, protein_id_position_AA, t, ptm_name_AA.get(t, []))
    
    print(f"Position cache built for {len(types)} annotation types in {time.time() - start_time:.2f} seconds")
    
    # Now compute nm_per_d, nc_per_i and k_per_d_i using the cached positions
    nm_per_d = {}
    nc_per_i = {}
    k_per_d_i = {}
    
    # Use batching to balance between parallelism overhead and utilization
    task_count = len(x_types) * len(y_types)
    # For large tasks, use multiprocessing
    if task_count > 5000 and len(x_types) > 50:  
        print(f"Using parallel processing for {task_count} combinations...")
        batch_size = max(1, len(x_types) // (os.cpu_count() * 2))       
        chunks = [(x_types[i:i+batch_size], y_types, position_cache) for i in range(0, len(x_types), batch_size)]       
        
        with Pool(processes=os.cpu_count()) as pool:
            all_results = list(tqdm(
                pool.imap(process_position_batch, chunks),
                total=len(chunks),
                desc="Processing position batches"
            ))  
           
        # Merge results from all batches
        for batch_nm, batch_nc, batch_k in all_results:
            nm_per_d.update(batch_nm)
            nc_per_i.update(batch_nc)
            k_per_d_i.update(batch_k)
    
    else:
        # For smaller tasks, process directly without multiprocessing
        print(f"Processing {task_count} position calculations directly...")
        total = len(x_types) * len(y_types)
        count = 0
        
        for d in x_types:
            x_positions = position_cache[d]
            nm_per_d[d] = len(x_positions)
            for i in y_types:
                y_positions = position_cache[i]
                nc_per_i[i] = len(y_positions)
                k_per_d_i[(d, i)] = len(x_positions & y_positions)
                
                # Update progress every 1000 iterations
                count += 1
                if count % 1000 == 0 or count == total:
                    print(f"Progress: {count}/{total} combinations processed ({count/total*100:.1f}%)")
    
    print(f"Precomputation completed in {time.time() - start_time:.2f} seconds")
    return nm_per_d, nc_per_i, k_per_d_i

def process_position_batch(args):
    """Process a batch of x_types against all y_types."""
    d_batch, y_types, position_cache = args
    batch_nm = {}  
    batch_nc = {}
    batch_k = {}
    
    for d in d_batch:
        x_positions = position_cache[d]
        # Calculate nm for this d_type
        batch_nm[d] = len(x_positions)
        
        for i in y_types:
            y_positions = position_cache[i]
            
            # Fast set operations
            batch_nc[i] = len(y_positions)  
            batch_k[(d, i)] = len(x_positions & y_positions)
    
    return batch_nm, batch_nc, batch_k

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
    chisq_args = [
        (idx, table_a[idx], table_b[idx], table_c[idx], table_d[idx]) 
        for idx in valid_indices
    ]
    # Parallel computation of p-values
    num_processes = min(os.cpu_count(), len(valid_indices))
    if num_processes > 1 and len(valid_indices) > 10:  # Only use multiprocessing for larger datasets
        print(f"Using {num_processes} processes for chi-square calculations")
        with Pool(processes=num_processes) as pool:
            p_values = list(tqdm(
                pool.imap(compute_chisq, chisq_args, chunksize=max(1, len(chisq_args) // (num_processes * 4))),
                total=len(chisq_args),
                desc="Computing p-values"
            ))
    else:
        # Fall back to sequential for small datasets
        print("Computing p-values sequentially")
        p_values = [compute_chisq(args) for args in tqdm(chisq_args, desc="Computing p-values")]
    
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
    
def filter_positions_with_all_categories(protein_id_annotation_position, sec_ids=[], domain_ids=[], protein_ids=[]):
    """
    Filter positions in protein_id_annotation_position to only include positions 
    that have annotations from specified categories (AA, sec, domain).
    Only filters for categories where IDs lists are non-empty.
    """
    if not any([sec_ids, domain_ids, protein_ids]):
        return protein_id_annotation_position
    
    result = {}
    
    # First filter proteins by protein_ids if provided
    proteins_to_process = protein_id_annotation_position
    if protein_ids:
        proteins_to_process = {pid: annotations for pid, annotations in protein_id_annotation_position.items() 
                              if pid in protein_ids}
    
    # If we're only filtering by protein_ids and no other filters are active, just return the filtered proteins
    if not any([sec_ids, domain_ids]):
        return proteins_to_process
    
    # Process each selected protein
    for protein_id, annotations in proteins_to_process.items():
        # Create position sets for each annotation type we need to filter for
        positions_by_type = {}
        if sec_ids:
            positions_by_type['sec'] = set()
        if domain_ids:
            positions_by_type['domain'] = set()
        
        # Track which positions have which annotations
        position_annotations = {}
        
        # Process all annotations in this protein
        for annotation, positions in annotations.items():           
            # Check if this is a secondary structure annotation we're interested in
            if sec_ids and annotation in sec_ids:
                for pos in positions:
                    positions_by_type['sec'].add(pos)
                    if pos not in position_annotations:
                        position_annotations[pos] = {'aa': [], 'sec': [], 'domain': []}
                    position_annotations[pos]['sec'].append(annotation)
            
            # Check if this is a domain annotation we're interested in
            elif domain_ids and annotation in domain_ids:
                for pos in positions:
                    positions_by_type['domain'].add(pos)
                    if pos not in position_annotations:
                        position_annotations[pos] = {'aa': [], 'sec': [], 'domain': []}
                    position_annotations[pos]['domain'].append(annotation)
        
        # Find positions that exist in all requested annotation types
        shared_positions = None
        for req_type, positions in positions_by_type.items():
            if shared_positions is None:
                shared_positions = positions
            else:
                shared_positions = shared_positions.intersection(positions)
        
        # If no shared positions found, skip this protein
        if shared_positions is not None and not shared_positions:
            continue
            
        # Create result dictionary with only the annotations that apply to shared positions
        protein_result = {}
        for annotation, positions in annotations.items():
            # If we have no filtering criteria other than protein_id, include all positions
            if shared_positions is None:
                protein_result[annotation] = positions
                continue
                
            # Keep only positions that are in the shared set
            filtered_positions = [pos for pos in positions if pos in shared_positions]
            if filtered_positions:
                protein_result[annotation] = filtered_positions
        
        result[protein_id] = protein_result
    
    return result

def convert_to_cross_reference_format(original_dict, aa_dict):
    """
    Create cross-reference dictionary mapping (protein_id, position) tuples to amino acids
    """
    result = {}
    
    # Process each entry in the amino acid dictionary
    for key, amino_acid in aa_dict.items():
        # Parse the protein ID and position from the key
        parts = key.split('_')
        if len(parts) != 2:
            continue
            
        protein_id = parts[0]
        try:
            position = int(parts[1])
        except ValueError:
            continue
        
        # Check if this protein exists in the original dictionary
        if protein_id in original_dict:
            # Check if this amino acid and position exist in the original annotations
            if amino_acid in original_dict[protein_id]:
                if position in original_dict[protein_id][amino_acid]:
                    # Add to result dictionary
                    result[(protein_id, position)] = amino_acid
    
    return result
    
def perform_enrichment_analysis(
    x_types: str,
    y_types: str,
    modifiability: dict = {},
    sec_ids: list = [],
    domain_ids: list = [],
    protein_ids: list = []
) -> pd.DataFrame:
    """
    Perform enrichment analysis to evaluate which modifications or annotations (x_types) are enriched / depleted 
    within specific protein features (y_types).

    Example usage:
        "Which modifications (x_types) are enriched in secondary structures (y_types)?"

    Parameters
    ----------
    x_types : str
        The "dependent data type" for enrichment analysis. Must be one of: 'ptm_name', 'AA', 'sec', or 'domain'.
        Do not specify individual or selected annotations; always test all annotations within a conformational level 
        to allow for multiple testing correction.
    
    y_types : str
        The "independent data type" for enrichment analysis. Must be one of: 'ptm_name', 'AA', 'sec', or 'domain'.
    
    modifiability : dict, optional
        Dictionary specifying eligible amino acids for each modification type. 
        Example: {'[21]Phospho': ['S', 'T', 'Y']}.
        If empty, all amino acids evaluated by ionbot are considered modifiable for each ptm_name.
    
    sec_ids, domain_ids, protein_ids : list, optional
        Lists specifying which amino acids, secondary structures, domains, or proteins to include in the analysis.
        By default, all lists are empty, resulting in a proteome-wide analysis.
    
    Notes
    -----
    - The input must represent a "hierarchical biological question".
    - For example, if 'domains' are specified as x_types, only 'protein_ids' can be provided; 
      do not specify 'lower conformational annotations' like sec_ids or domain_ids in this case.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the results of the enrichment analysis.
    """

    total_start_time = time.time()

    # When building the string, clean the modification name and join with hyphens
    def clean_ptm(ptm):
        return re.sub(r'^\[\d+\]', '', ptm)
    
    modifiability_str = "_".join(
        "-".join([clean_ptm(key)] + values)
        for key, values in modifiability.items()
    )
    
    d_part = x_types + (f'_{modifiability_str}' if modifiability else '')
    IN_part = f"{d_part}_IN_{y_types}"
    
    # Handle ID lists and proteome-wide designation
    id_list = sec_ids + domain_ids + protein_ids
    if not id_list:
        id_list.append("proteome_wide")
    
    # Limit filename elements to 5 and add "_and_more" if needed
    max_elements = 5
    if len(id_list) > max_elements:
        truncated_ids = id_list[:max_elements]
        suffix = "_and_more"
    else:
        truncated_ids = id_list
        suffix = ""
    
    # Create final filename
    output_filename = f"{IN_part}_{'_'.join(truncated_ids)}{suffix}.csv"
    
    # Load data
    with open('data/protein_id_annotation_position.json') as file:
        protein_id_annotation_position = json.load(file)
        
    protein_id_annotation_position = filter_positions_with_all_categories(
        protein_id_annotation_position,
        sec_ids,
        domain_ids,
        protein_ids
    )
    with open('data/protein_id_position_AA.json') as position_file:
        protein_id_position_AA = json.load(position_file) 
    protein_id_position_AA = convert_to_cross_reference_format(protein_id_annotation_position, protein_id_position_AA)
    print(f"Loaded annotation data for {len(protein_id_annotation_position)} proteins")
    ptm_name_AA = {}
    if x_types == 'ptm_name':
        with open('data/ptm_name_AA.json') as file:
            ptm_name_AA = json.load(file)
        if len(modifiability) != 0:
            for d in modifiability:
                ptm_name_AA[d] = modifiability[d]
    
    protein_ids = tuple(protein_id_annotation_position.keys()) # required for cache
    @lru_cache(maxsize=256)
    def update_types(types, protein_ids):
        """Updates the given types list"""
        if types == 'domain':
            return tuple(sorted({annotation for annotations in protein_id_annotation_position.values() 
                          for annotation in annotations if 'IPR' in annotation}))
        elif types == 'sec':
            return ('3₁₀-helix', 'α-helix', 'π-helix', 'PPII-helix', 'ß-strand', 'ß-bridge', 'turn', 'bend', 'unassigned', 'loop', 'IDR')
        elif types == 'AA':
            return ('A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y') 
        elif types == 'ptm_name':
            with open('data/ptm_name_AA.json') as file:
                ptm_name_AA = json.load(file)
            return tuple(sorted(ptm_name_AA))
                
        # Custom sort function for original types
        return tuple(sorted(types, key=lambda x: int(x.split(']')[0][1:]) if x.startswith('[') else x))
    
    # Update types - convert to tuples for caching
    x_types = update_types(x_types, protein_ids)
    y_types = update_types(y_types, protein_ids)
    n = len(protein_id_position_AA)
    
    nm_per_d, nc_per_i, k_per_d_i = precompute_nm_nc_k(
        x_types, y_types, ptm_name_AA, 
        protein_id_annotation_position, protein_id_position_AA
    )
    
    print("Processing data...")
    start_time = time.time()
          
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
        # Define the directory and output file path
        output_dir = 'results_protmodcon'
        output_path = os.path.join(output_dir, output_filename)
        
        # Create the directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save the results
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='protmodcon: Comprehensive Analysis of Protein Modifications from a Conformational Perspective',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''\
            Example:
                python protmodcon.py --x-types ptm_name --y-types sec --protein-ids P05067 --modifiability "[21]Phospho:S,T,Y"
            ''')
    )
    parser.add_argument(
        '--x-types', required=True, choices=['ptm_name', 'AA', 'sec', 'domain'],
        help='First data type for enrichment analysis, arbitrarily indicated by x. Choose exactly one of: ptm_name, AA, sec.'
    )
    parser.add_argument(
        '--y-types', required=True, choices=['ptm_name', 'AA', 'sec', 'domain'],
        help='Second data type for enrichment analysis, arbitrarily indicated by y. Choose exactly one of: AA, sec, domain.'
    )
    parser.add_argument(
        '--sec-ids', nargs='*', default=[],
        help='Optional: List of secondary structure IDs to include, e.g. --sec-ids 310HELX unassigned. Choose one (or more) of 310HELX, AHELX, PIHELX, PPIIHELX, STRAND, BRIDGE, TURN, BEND, unassigned, LOOP, IDR.'
    )
    parser.add_argument(
        '--domain-ids', nargs='*', default=[],
        help='Optional: List of domain IDs (from InterPro) to include, e.g. --domain-ids IPR002223 IPR003165. 7,406 domain-ids can be chosen, which can be found as keys in data/domain_descriptions.json.'
    )
    parser.add_argument(
        '--protein-ids', nargs='*', default=[],
        help='Optional: List of protein IDs (from UniProt) to include, e.g. --protein-ids P05067 Q9Y6K9. 20,059 protein-ids can be chosen, which can be found as keys in data/protein_id_annotation_position.json.'
    )
    parser.add_argument(
        '--modifiability', nargs='*', default=[],
        help='Optional: Modifiability of amino acids per modification type. Format: "[Unimod accession]name:AA1,AA2", where the name corresponds to the Unimod PSI-MS Name or, if unavailable, the Unimod Interim name, e.g. --modifiability "[21]Phospho:S,T,Y" (double quotes mandatory). Valid ptm_name-AA combinations can be found as keys in data/ptm_name_AA.json.'
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

    # Process modifiability argument if provided
    modifiability = {}
    if args.modifiability:
        # Parse modifiability strings like "[21]Phospho:S,T,Y" (double quotes inclusive)
        for mod_str in args.modifiability:
            if ':' in mod_str:
                ptm, aas = mod_str.split(':')
                modifiability[ptm] = aas.split(',')
    
    # Run enrichment analysis
    perform_enrichment_analysis(
        x_types=args.x_types,
        y_types=args.y_types,
        modifiability=modifiability,
        sec_ids=args.sec_ids,
        domain_ids=args.domain_ids,
        protein_ids=args.protein_ids
    )