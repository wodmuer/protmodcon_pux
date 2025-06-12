from flask import Flask, request, render_template_string, render_template, send_from_directory
import requests
import numpy as np
import scipy
import subprocess
import os
import json
from pathlib import Path
import pickle
import hashlib

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')
    
@app.route('/results', methods=['POST'])
def results():
    x = request.form.get('x')    
    x_types = []
    if x == 'AA':
        x_types = request.form.getlist('x_aa_types[]')
    elif x == 'sec':
        x_types = request.form.getlist('x_sec_types[]')
    else:
        x_types = request.form.getlist('x_types[]')
    if request.form.get('x_all'):
        if x == 'ptm':
            with open('static/valid_PTMS.json', 'r') as f:
                x_types = json.load(f)
        elif x == 'domain':
             with open('static/valid_domains.json', 'r') as f:
                x_types = json.load(f)
        elif x == 'protein':
            with open('static/valid_proteins.json', 'r') as f:
                x_types = json.load(f)
    x_types = [item for sublist in x_types for item in sublist.split() if item]  # Flatten and filter out empty strings

    y = request.form.get('y')
    y_types = []
    if y == 'AA':
        y_types = request.form.getlist('y_aa_types[]')
    elif y == 'sec':
        y_types = request.form.getlist('y_sec_types[]')
    else:
        y_types = request.form.getlist('y_types[]')
    if request.form.get('y_all'):
        if y == 'ptm':
            with open('static/valid_PTMS.json', 'r') as f:
                y_types = json.load(f)
        elif y == 'domain':
             with open('static/valid_domains.json', 'r') as f:
                y_types = json.load(f)
        elif y == 'protein':
            with open('static/valid_proteins.json', 'r') as f:
                y_types = json.load(f)
    y_types = [item for sublist in y_types for item in sublist.split() if item] 

    filters = request.form.get('filters')
    filter_types = []
    if filters:
        if filters == 'AA':
            filter_types = request.form.getlist('filters_aa_types[]')
        elif filters == 'sec':
            filter_types = request.form.getlist('filters_sec_types[]')
        else:
            filter_types = request.form.getlist('filters_types[]')
            if not filter_types:
                if filters == 'ptm':
                    with open('static/valid_PTMS.json', 'r') as f:
                        filter_types = json.load(f)
                elif filters == 'domain':
                     with open('static/valid_domains.json', 'r') as f:
                        filter_types = json.load(f)
                elif filters == 'protein':
                    with open('static/valid_proteins.json', 'r') as f:
                        filter_types = json.load(f)
        filter_types = [item for sublist in filter_types for item in sublist.split() if item] 
    
    modifiability = request.form.getlist('modifiability[]')
    modifiability = [item for sublist in modifiability for item in sublist.split() if item]

    # analyze bulk (e.g. list of proteins) or each item individually (each protein in your list)
    if request.form.get('x_bulk'):
        # Switch is ON (Bulk selected)
        x_mode = ['x_bulk']
    else:
        # Switch is OFF (Individual selected)
        x_mode = ['x_individual']

    if request.form.get('y_bulk'):
        # Switch is ON (Bulk selected)
        y_mode = ['y_bulk']
    else:
        # Switch is OFF (Individual selected)
        y_mode = ['y_individual']

    subprocess.run([
            'conda', 'run', '-n', 'protmodcon', 'python', 'protmodcon.py'] +
                       ["--x-types"] + x_types +
                       ["--y-types"] + y_types +
                       ["--filters"] + filter_types +
                       ["--modifiability"] + modifiability +
                       ["--x-mode"] + x_mode +
                       ["--y-mode"] + y_mode
            )

    # visualize results
    if cached_result is not None:
        # Mapping for secondary structure elements should be done inside visualize_protmodcon.py 
        subprocess.run([
            'conda', 'run', '-n', 'protmodcon', 'python', 'visualise_protmodcon.py',
            '--data', cached_result,
            '--x'
        ] + x_types + [
            '--y'
        ] + y_types)
        return render_template('results.html')
    else:
        return f"No enrichment/depletion found."

@app.route('/results/plot.png')
def serve_plot():
    return send_from_directory('figures_protmodcon', 'plot.png')
    
if __name__ == '__main__':
    app.run(debug=True)