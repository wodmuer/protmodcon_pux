from flask import Flask, request, render_template_string, render_template, send_from_directory
import requests
import numpy as np
import scipy
import subprocess
import os

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

def build_command(x, x_data, y, y_data, data_path):
    command = [
        'conda', 'run', '-n', 'protmodcon', 'python', 'visualise_protmodcon.py',
        '--data', data_path,
        '--x'
    ]

    # For PTM and domain: pass as multiple arguments
    # For AA and sec: pass as a single comma-separated string
    if x in ['ptm_name', 'domain']:
        command.extend(x_data)
    elif x in ['AA', 'sec']:
        command.append(','.join(x_data))
    else:
        raise ValueError(f"Unknown x: {x}")

    command.append('--y')
    if y in ['ptm_name', 'domain']:
        command.extend(y_data)
    elif y in ['AA', 'sec']:
        command.append(','.join(y_data))
    else:
        raise ValueError(f"Unknown y: {y}")

    return command

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')
    
@app.route('/results', methods=['POST'])
def results():
    # Extracting values from the form
    x = request.form.get('x')
    # Get x_data and split by spaces if it's not empty
    x_data = request.form.getlist('x_data[]')
    
    y = request.form.get('y')
    y_data = request.form.getlist('y_data[]')

    # Print all variables for debugging
    print("x:", x)
    print("x_data:", x_data)
    print("y:", y)
    print("y_data:", y_data)
   
    # first_x_rows, second_x_rows = fetch_annotations_from_hasura(first_x_list, second_x_list)
    # intersection = first_x_rows & second_x_rows
    # 
    # a = len(intersection)
    # b = len(first_x_rows) - a
    # c = len(second_x_rows) - a
    # d = 10489156 - len(first_x_rows) - len(second_x_rows) + a
    # 
    # odds = (a * d) / (b * c) if b and c else 0
    # p_value = scipy.stats.chi2_contingency([[a, b], [c, d]])[1] if b and c else 1.0
    # 
    # results = {
    #     "intersection": a,
    #     "odds": odds,
    #     "p_value": p_value,
    #     "x": first_x_list,
    #     "second_x": second_x_list,
    # }

    data = f"results_protmodcon/{x}_IN_{y}_proteome_wide.csv"
    print(data)
    
    # create enrichment data, if not existing
    if not os.path.isfile(data):
        subprocess.run([
            "conda", "run", "-n", "protmodcon",
            "python", "protmodcon.py",
            "--x-types", x,
            "--y-types", y
        ])
    
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
    
    if x == "ptm_name":
        x_data = x_data.split()
    elif x == "AA":
        "" # x_data already correctly formatted
    elif x == "sec":
        x_data = [sec_mapping[sec] for sec in x_data if sec in sec_mapping]
    elif x == "domain":
        x_data = x_data.split()

    if y == "ptm_name":
        y_data = y_data.split()
    elif y == "AA":
        "" # y_data already correctly formatted
    elif y == "sec":
        y_data = [sec_mapping[sec] for sec in y_data if sec in sec_mapping]
    elif y == "domain":
        y_data = y_data.split()
        
    # make shortest arg the x_arg -> only up to 5 allowed. So, if both x_arg and y_arg have length > 5 -> NOTHING returned
    if len(x_data) > 5 and len(y_data) > 5:
        return f"One of two selections must NOT contain more than five elements."
    elif len(x_data) > len(y_data):
        x, y = y, x
        x_data, y_data = y_data, x_data

    command = build_command(x, x_data, y, y_data, data)
    print(command)
    subprocess.run(command)

    return render_template('results.html')

@app.route('/results/plot.png')
def serve_plot():
    return send_from_directory('figures_protmodcon', 'plot.png')
    
if __name__ == '__main__':
    app.run(debug=True)