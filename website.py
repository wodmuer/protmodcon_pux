from flask import Flask, request, render_template_string, render_template, send_from_directory
import requests
import numpy as np
import scipy
import subprocess
import os
import json

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')
    
@app.route('/results', methods=['POST'])
def results():
    x = request.form.get('x')    
    x_types = request.form.getlist('x_types[]')
    x_types = [item for sublist in x_types for item in sublist.split() if item]  # Flatten and filter out empty strings
    if not x_types and (request.form.get('x_ptm_all') or request.form.get('x_domain_all') or request.form.get('x_protein_all')):
        if x == 'ptm':
            with open('static/valid_PTMS.json', 'r') as f:
                x_types = json.load(f)
        elif x == 'domain':
             with open('static/valid_domains.json', 'r') as f:
                x_types = json.load(f)
        elif x == 'protein':
            with open('static/valid_proteins.json', 'r') as f:
                x_types = json.load(f)

    y = request.form.get('y')
    y_types = request.form.getlist('y_types[]')
    y_types = [item for sublist in y_types for item in sublist.split() if item] 
    if not y_types:
        if y == 'ptm':
            with open('static/valid_PTMS.json', 'r') as f:
                y_types = json.load(f)
        elif y == 'domain':
             with open('static/valid_domains.json', 'r') as f:
                y_types = json.load(f)
        elif y == 'protein':
            with open('static/valid_proteins.json', 'r') as f:
                y_types = json.load(f)

    filters = request.form.getlist('filters[]')
    filters = [item for sublist in filters for item in sublist.split() if item]  
    
    modifiability = request.form.getlist('modifiability[]')
    modifiability = [item for sublist in modifiability for item in sublist.split() if item] 

    # analyze bulk (e.g. list of proteins) or each item individually (each protein in your list)
    if request.form.get('bulk'):
        # Switch is ON (Bulk selected)
        mode = 'bulk'
    else:
        # Switch is OFF (Individual selected)
        mode = 'individual'

    print(x_types, y_types, mode)
    return f"hello friend"
    # create enrichment data
    subprocess.run([
        'conda', 'run', '-n', 'protmodcon', 'python', 'protmodcon.py'] +
                   ["--x-types"] + x_types +
                   ["--y-types"] + y_types +
                   ["--filters"] + filters +
                   ["--modifiability"] + modifiability
        )
    return f"hello friend"
    # Mapping for secondary structure elements should be done inside visualize_protmodcon.py 
    cmd = [
        'conda', 'run', '-n', 'protmodcon', 'python', 'visualise_protmodcon.py',
        '--data', data,
        '--x'
    ] + x_types + [
        '--y'
    ] + y_types

    subprocess.run(cmd)
    print(cmd)
    
    return render_template('results.html')

@app.route('/results/plot.png')
def serve_plot():
    return send_from_directory('figures_protmodcon', 'plot.png')
    
if __name__ == '__main__':
    app.run(debug=True)