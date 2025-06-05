from flask import Flask, request, render_template_string, render_template, send_from_directory
import requests
import numpy as np
import scipy
import subprocess
import os
import json

HASURA_URL = "http://localhost:8080/v1/graphql"
def fetch_annotations_from_hasura(x_types, second_x_list):
    query = '''
    query ($x: [String!], $second_x: [String!]) {
      first: protmodcon(where: { annotation: { _in: $x } }) {
        protein_id
        position
      }
      second: protmodcon(where: { annotation: { _in: $second_x } }) {
        protein_id
        position
      }
    }
    '''
    variables = {"x": x_types, "second_x": second_x_list}
    headers = {"Content-Type": "application/json"}
    response = requests.post(
        HASURA_URL,
        json={"query": query, "variables": variables},
        headers=headers
    )
    data = response.json()
    first_x_rows = set((item["protein_id"], item["position"]) for item in data["data"]["first"])
    second_x_rows = set((item["protein_id"], item["position"]) for item in data["data"]["second"])
    return first_x_rows, second_x_rows

def reorder_by_hierarchy(x, y):
    """make a fictive hierarchy so that protmodcon.py is not only run 'in one direction' to save computational time"""
    hierarchy = ['ptm_name', 'AA', 'sec', 'domain']
    x_index = hierarchy.index(x)
    y_index = hierarchy.index(y)
    if x_index < y_index:
        return (x, y, False)
    else:
        return (y, x, True)

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')
    
@app.route('/results', methods=['POST'])
def results():  
    z_data = request.form.getlist('z_data[]')
    z_data = [item for sublist in z_data for item in sublist.split() if item]  

    # first split z_data in AA-ids, sec-ids, domain-ids, protein-ids
    sec_ids, domain_ids, protein_ids = [], [], []
    for item in z_data:
        if item in ('3₁₀-helix', 'α-helix', 'π-helix', 'PPII-helix', 'ß-strand', 'ß-bridge', 'turn', 'bend', 'unassigned', 'loop', 'IDR'):
            sec_ids.append(item)
        elif item.startswith('IPR'):
            domain_ids.append(item)
        else:
            protein_ids.append(item)

    # to save time, check if enrichment was already calculated 
    """
    # When building the string, clean the modification name and join with hyphens
    def clean_ptm(ptm):
        return re.sub(r'^\[\d+\]', '', ptm)
    
    modifiability_str = "_".join(
        "-".join([clean_ptm(key)] + values)
        for key, values in modifiability.items()
    )
    """
    x = request.form.get('x')          
    y = request.form.get('y')
    z = request.form.get('z')
    x, y, reorder = reorder_by_hierarchy(x, y)

    # Print all variables for debugging
    print("x:", x)
    print("y:", y)
    print("z:", z)    
    x_types = x
    y_types = y
    modifiability = ''
    
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
    data = f"results_protmodcon/{IN_part}_{'_'.join(truncated_ids)}{suffix}.csv"
    print(data)

    # create enrichment data, if not existing
    if not os.path.isfile(data):
        subprocess.run([
            'conda', 'run', '-n', 'protmodcon', 'python', 'protmodcon.py',
            "--x-types", x,
            "--y-types", y,
            "--sec-ids"] + sec_ids + [
                "--domain-ids" ] + domain_ids + [
                    "--protein-ids" ] + protein_ids
        )

    x_data = request.form.getlist('x_data[]')
    x_data = [item for sublist in x_data for item in sublist.split() if item]  # Flatten and filter out empty strings
    y_data = request.form.getlist('y_data[]')
    y_data = [item for sublist in y_data for item in sublist.split() if item] 

    if reorder:
        x_data, y_data = y_data, x_data

    # in visualize: --x is required, --y NOT
    
    if not x_data: 
        if x == 'ptm_name':
            with open('static/valid_PTMS.json', 'r') as f:
                x_data = json.load(f)
        elif x == 'domain':
             with open('static/valid_domains.json', 'r') as f:
                x_data = json.load(f)
        elif x == 'protein':
            with open('static/valid_proteins.json', 'r') as f:
                x_data = json.load(f)   

    # Mapping for secondary structure elements
    sec_mapping = {
        '3₁₀-helix': '310HELX',
        'α-helix': 'AHELX',
        'π-helix': 'PIHELX',
        'PPII-helix': 'PPIIHELX',
        'ß-strand': 'STRAND',
        'ß-bridge': 'BRIDGE',
        'turn': 'TURN',
        'bend': 'BEND',
        'unassigned': 'unassigned',
        'loop': 'LOOP',
        'IDR': 'IDR'
    }

    if x == "sec":
        x_data = [sec_mapping[sec] for sec in x_data if sec in sec_mapping]

    if y == "sec":
        if y_data:
            y_data = [sec_mapping[sec] for sec in y_data if sec in sec_mapping]

    y_data = x_data
    x_data = ['[1]Acetyl']
    
    cmd = [
        'conda', 'run', '-n', 'protmodcon', 'python', 'visualise_protmodcon.py',
        '--data', data,
        '--x'
    ] + x_data + [
        '--y'
    ] + y_data

    subprocess.run(cmd)
    print(cmd)
    
    return render_template('results.html')
    
    """ 
    Hasura extension
    first_x_rows, second_x_rows = fetch_annotations_from_hasura(first_x_list, second_x_list)
    intersection = first_x_rows & second_x_rows
    
    a = len(intersection)
    b = len(first_x_rows) - a
    c = len(second_x_rows) - a
    d = 10489156 - len(first_x_rows) - len(second_x_rows) + a
    
    odds = (a * d) / (b * c) if b and c else 0
    p_value = scipy.stats.chi2_contingency([[a, b], [c, d]])[1] if b and c else 1.0
    
    results = {
        "intersection": a,
        "odds": odds,
        "p_value": p_value,
        "x": first_x_list,
        "second_x": second_x_list,
    }
    """

@app.route('/results/plot.png')
def serve_plot():
    return send_from_directory('figures_protmodcon', 'plot.png')
    
if __name__ == '__main__':
    app.run(debug=True)