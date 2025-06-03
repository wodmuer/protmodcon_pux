from flask import Flask, request, render_template_string
import requests

HASURA_URL = "http://localhost:8080/v1/graphql"

app = Flask(__name__)

# HTML template with a form and a section to display results
HTML_TEMPLATE = '''
<!doctype html>
<title>Protein Annotations</title>
<h2>Enter Protein ID</h2>
<form method="post">
  <input type="text" name="protein_id" required>
  <input type="submit" value="Get Annotations">
</form>
{% if annotations is not none %}
  <h3>Annotations for {{ protein_id }}:</h3>
  <ul>
    {% for ann in annotations %}
      <li>{{ ann }}</li>
    {% endfor %}
  </ul>
{% endif %}
'''

def fetch_annotations_from_hasura(protein_id):
    query = """
    query GetAnnotations($protein_id: String!) {
      protmodcon(where: {protein_id: {_eq: $protein_id}}) {
        annotation
      }
    }
    """
    variables = {"protein_id": protein_id}
    headers = {
        "Content-Type": "application/json"
    }
    response = requests.post(
        HASURA_URL,
        json={"query": query, "variables": variables},
        headers=headers
    )
    data = response.json()
    # Extract just the annotation values
    return set([item["annotation"] for item in data["data"]["protmodcon"]])

@app.route('/', methods=['GET', 'POST'])
def home():
    annotations = None
    protein_id = None
    if request.method == 'POST':
        protein_id = request.form['protein_id']
        annotations = fetch_annotations_from_hasura(protein_id)
    return render_template_string(HTML_TEMPLATE, annotations=annotations, protein_id=protein_id)

if __name__ == '__main__':
    app.run(debug=True)
