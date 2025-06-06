// Wait for the DOM to be ready
document.addEventListener('DOMContentLoaded', function() {
    function renderFormSection(text, my_id) {
    return `
        <div class="form-section">
            <div class="row align-items-center">
            <div class="col-auto">
                <label class="form-label">${text}</label>
            </div>
                <div class="col-auto">
                <select name="${my_id}" id="${my_id}" class="form-select" required>
                <option value="" disabled selected>Select annotation type</option>
                <option value="ptm">PTM (Post-translational modification)</option>
                <option value="AA">Amino Acid</option>
                <option value="sec">Secondary Structure</option>
                <option value="domain">Protein Domain</option>
                <option value="protein">Protein ID (UniProt)</option>
                </select>
                </div>
            </div>
                
            <div class="form-check form-switch">
                <input class="form-check-input" type="checkbox" id="${my_id}_bulk" name="${my_id}_bulk">
                <label class="form-check-label" for="${my_id}_bulk">
                    <span id="${my_id}_switch-label">Individual</span>
                </label>
            </div>
            <div id="${my_id}_checkboxes" class="option-checkboxes">
                <label class="form-check-label">
                <input type="checkbox" id="${my_id}_all" name="${my_id}_all" class="form-check-input">
                Select All
                </label>
            </div>
            
            <div id="${my_id}_ptm_input" class="ptm-input input-with-overlay">
                <div style="position:relative;">
                    <div id="${my_id}_ptm-overlay" class="input-token-overlay"></div>
                    <input type="text" id="${my_id}_ptm_text" name="${my_id}_types[]" class="form-control" placeholder="[1]Acetyl [21]Phospho [35]Oxidation" autocomplete="off">
                    </div>
                    <span class="ptm-hint">Enter PTM modifications separated by spaces, e.g. [1]Acetyl [21]Phospho [35]Oxidation<br>Modifications are annotated as [Unimod accession]name, where the name corresponds to the Unimod PSI-MS Name or, if unavailable, the Unimod Interim name.</span>
                </div>
            </div>
            
            <div id="${my_id}_aa_checkboxes" class="option-checkboxes">
                <div class="row">
                    <div class="col-md-2 col-4">
                        <label class="form-check-label"><input type="checkbox" name="${my_id}_aa_types[]" value="A" class="form-check-input"> A (Ala)</label>
                    </div>
                    <div class="col-md-2 col-4">
                        <label class="form-check-label"><input type="checkbox" name="${my_id}_aa_types[]" value="C" class="form-check-input"> C (Cys)</label>
                    </div>
                    <div class="col-md-2 col-4">
                        <label class="form-check-label"><input type="checkbox" name="${my_id}_aa_types[]" value="D" class="form-check-input"> D (Asp)</label>
                    </div>
                    <div class="col-md-2 col-4">
                        <label class="form-check-label"><input type="checkbox" name="${my_id}_aa_types[]" value="E" class="form-check-input"> E (Glu)</label>
                    </div>
                    <div class="col-md-2 col-4">
                        <label class="form-check-label"><input type="checkbox" name="${my_id}_aa_types[]" value="F" class="form-check-input"> F (Phe)</label>
                    </div>
                    <div class="col-md-2 col-4">
                        <label class="form-check-label"><input type="checkbox" name="${my_id}_aa_types[]" value="G" class="form-check-input"> G (Gly)</label>
                    </div>
                    <div class="col-md-2 col-4">
                        <label class="form-check-label"><input type="checkbox" name="${my_id}_aa_types[]" value="H" class="form-check-input"> H (His)</label>
                    </div>
                    <div class="col-md-2 col-4">
                        <label class="form-check-label"><input type="checkbox" name="${my_id}_aa_types[]" value="I" class="form-check-input"> I (Ile)</label>
                    </div>
                    <div class="col-md-2 col-4">
                        <label class="form-check-label"><input type="checkbox" name="${my_id}_aa_types[]" value="K" class="form-check-input"> K (Lys)</label>
                    </div>
                    <div class="col-md-2 col-4">
                        <label class="form-check-label"><input type="checkbox" name="${my_id}_aa_types[]" value="L" class="form-check-input"> L (Leu)</label>
                    </div>
                    <div class="col-md-2 col-4">
                        <label class="form-check-label"><input type="checkbox" name="${my_id}_aa_types[]" value="M" class="form-check-input"> M (Met)</label>
                    </div>
                    <div class="col-md-2 col-4">
                        <label class="form-check-label"><input type="checkbox" name="${my_id}_aa_types[]" value="N" class="form-check-input"> N (Asn)</label>
                    </div>
                    <div class="col-md-2 col-4">
                        <label class="form-check-label"><input type="checkbox" name="${my_id}_aa_types[]" value="P" class="form-check-input"> P (Pro)</label>
                    </div>
                    <div class="col-md-2 col-4">
                        <label class="form-check-label"><input type="checkbox" name="${my_id}_aa_types[]" value="Q" class="form-check-input"> Q (Gln)</label>
                    </div>
                    <div class="col-md-2 col-4">
                        <label class="form-check-label"><input type="checkbox" name="${my_id}_aa_types[]" value="R" class="form-check-input"> R (Arg)</label>
                    </div>
                    <div class="col-md-2 col-4">
                        <label class="form-check-label"><input type="checkbox" name="${my_id}_types[]" value="S" class="form-check-input"> S (Ser)</label>
                    </div>
                    <div class="col-md-2 col-4">
                        <label class="form-check-label"><input type="checkbox" name="${my_id}_aa_types[]" value="T" class="form-check-input"> T (Thr)</label>
                    </div>
                    <div class="col-md-2 col-4">
                        <label class="form-check-label"><input type="checkbox" name="${my_id}_aa_types[]" value="V" class="form-check-input"> V (Val)</label>
                    </div>
                    <div class="col-md-2 col-4">
                        <label class="form-check-label"><input type="checkbox" name="${my_id}_aa_types[]" value="W" class="form-check-input"> W (Trp)</label>
                    </div>
                    <div class="col-md-2 col-4">
                        <label class="form-check-label"><input type="checkbox" name="${my_id}_aa_types[]" value="Y" class="form-check-input"> Y (Tyr)</label>
                    </div>
                </div>
            </div>

            <div id="${my_id}_sec_checkboxes" class="option-checkboxes">
                <div class="row">
                    <div class="col-md-3 col-6">
                        <label class="form-check-label"><input type="checkbox" name="${my_id}_sec_types[]" value="3₁₀-helix" class="form-check-input"> 3₁₀-helix</label>
                    </div>
                    <div class="col-md-3 col-6">
                        <label class="form-check-label"><input type="checkbox" name="${my_id}_sec_types[]" value="α-helix" class="form-check-input"> α-helix</label>
                    </div>
                    <div class="col-md-3 col-6">
                        <label class="form-check-label"><input type="checkbox" name="${my_id}_sec_types[]" value="π-helix" class="form-check-input"> π-helix</label>
                    </div>
                    <div class="col-md-3 col-6">
                        <label class="form-check-label"><input type="checkbox" name="${my_id}_sec_types[]" value="PPII-helix" class="form-check-input"> PPII-helix</label>
                    </div>
                    <div class="col-md-3 col-6">
                        <label class="form-check-label"><input type="checkbox" name="${my_id}_sec_types[]" value="ß-bridge" class="form-check-input"> ß-bridge</label>
                    </div>
                    <div class="col-md-3 col-6">
                        <label class="form-check-label"><input type="checkbox" name="${my_id}_sec_types[]" value="ß-strand" class="form-check-input"> ß-strand</label>
                    </div>
                    <div class="col-md-3 col-6">
                        <label class="form-check-label"><input type="checkbox" name="${my_id}_sec_types[]" value="turn" class="form-check-input"> turn</label>
                    </div>
                    <div class="col-md-3 col-6">
                        <label class="form-check-label"><input type="checkbox" name="${my_id}_sec_types[]" value="bend" class="form-check-input"> bend</label>
                    </div>
                    <div class="col-md-3 col-6">
                        <label class="form-check-label"><input type="checkbox" name="${my_id}_sec_types[]" value="loop" class="form-check-input"> loop</label>
                    </div>
                    <div class="col-md-3 col-6">
                        <label class="form-check-label"><input type="checkbox" name="${my_id}_sec_types[]" value="unassigned" class="form-check-input"> unassigned</label>
                    </div>
                    <div class="col-md-3 col-6">
                        <label class="form-check-label"><input type="checkbox" name="${my_id}_sec_types[]" value="IDR" class="form-check-input"> IDR</label>
                    </div>
                </div>
            </div>
                    
            <div id="${my_id}_domain_input" class="domain-input input-with-overlay">
                <div style="position:relative;">
                    <div id="${my_id}_domain-overlay" class="input-token-overlay"></div>
                    <input type="text" id="${my_id}_domain_text" name="${my_id}_types[]" class="form-control" placeholder="e.g. IPR000001 IPR000007" autocomplete="off">
                    </div>
                    <span class="domain-hint">Enter protein domain(s) separated by spaces, e.g. IPR000001 IPR000007</span>
                </div>
                        
            <div id="${my_id}_protein_input" class="protein-input input-with-overlay">
                <div style="position:relative;">
                    <div id="${my_id}_protein-overlay" class="input-token-overlay"></div>
                    <input type="text" id="${my_id}_protein_text" name="${my_id}_types[]" class="form-control" placeholder="e.g. P05067 P41227" autocomplete="off">
                    </div>
                    <span class="protein-hint">Enter protein IDs (UniProt) separated by spaces, e.g. P05067 P41227</span>
                </div>
            </div>
    `;
}

  document.getElementById('form-sections').innerHTML =
    renderFormSection('What is the enrichment/depletion of', 'x') +
    renderFormSection('in', 'y');
    
    let validPTMs = [];
    let validDomains = [];
    let validIDs = [];  // valid protein ids
    const hierarchy = ['ptm', 'AA', 'sec', 'domain', 'protein'];
    
    // Mapping for corresponding options between X and Y dropdowns
    const xToYMap = {
        'ptm': 'ptm',
        'AA': 'AA',
        'sec': 'sec',
        'domain': 'domain',
        'protein':'protein'
    };
    const yToXMap = {
        'ptm': 'ptm',
        'AA': 'AA',
        'sec': 'sec',
        'domain': 'domain',
        'protein':'protein'
    };

    // Function to update X options based on Y selection
    function updateXOptions(selectedYValue) {
        const xSelect = document.getElementById('x');
        const currentXValue = xSelect.value;
        
        // Reset all X options to enabled first
        Array.from(xSelect.options).forEach(option => {
            option.disabled = false;
            option.style.color = '';
            // Remove any previously added text
            if (option.textContent.includes(' (already selected')) {
                option.textContent = option.textContent.split(' (already selected')[0];
            }
        });
        
        // Disable the option that corresponds to Y selection
        if (selectedYValue && selectedYValue !== '') {
            const correspondingXValue = yToXMap[selectedYValue];
            if (correspondingXValue) {
                Array.from(xSelect.options).forEach(option => {
                    if (option.value === correspondingXValue) {
                        option.disabled = true;
                        option.style.color = '#666';
                        option.textContent += ' (already selected in second dropdown)';
                    }
                });
            }
            
            // If X currently has the same corresponding value as Y, reset X selection
            if (currentXValue === correspondingXValue) {
                xSelect.value = '';
                hideAllXInputs();
            }
        }
    }

    // Function to update Y options based on X selection
    function updateYOptions(selectedXValue) {
        const ySelect = document.getElementById('y');
        const currentYValue = ySelect.value;
        
        // Reset all Y options to enabled first
        Array.from(ySelect.options).forEach(option => {
            option.disabled = false;
            option.style.color = '';
            // Remove any previously added text
            if (option.textContent.includes(' (already selected')) {
                option.textContent = option.textContent.split(' (already selected')[0];
            }
        });
        
        // Disable the option that corresponds to X selection
        if (selectedXValue && selectedXValue !== '') {
            const correspondingYValue = xToYMap[selectedXValue];
            if (correspondingYValue) {
                Array.from(ySelect.options).forEach(option => {
                    if (option.value === correspondingYValue) {
                        option.disabled = true;
                        option.style.color = '#666';
                        option.textContent += ' (already selected in first dropdown)';
                    }
                });
            }
            
            // If Y currently has the same corresponding value as X, reset Y selection
            if (currentYValue === correspondingYValue) {
                ySelect.value = '';
                hideAllYInputs();
            }
        }
    }

    // Function to hide all X input fields
    function hideAllXInputs() {
        document.getElementById('x_ptm_input').style.display = 'none';
        document.getElementById('x_aa_checkboxes').style.display = 'none';
        document.getElementById('x_sec_checkboxes').style.display = 'none';
        document.getElementById('x_domain_input').style.display = 'none';
        document.getElementById('x_protein_input').style.display = 'none';
    }

    // Function to hide all Y input fields
    function hideAllYInputs() {
        document.getElementById('y_ptm_input').style.display = 'none';
        document.getElementById('y_aa_checkboxes').style.display = 'none';
        document.getElementById('y_sec_checkboxes').style.display = 'none';
        document.getElementById('y_domain_input').style.display = 'none';
        document.getElementById('y_protein_input').style.display = 'none';
    }    

    async function fetchValidLists() {
        try {
            const [ptms, domains, proteins] = await Promise.all([
                fetch('/static/valid_PTMs.json').then(r => r.json()),
                fetch('/static/valid_domains.json').then(r => r.json()),
                fetch('/static/valid_proteins.json').then(r => r.json())
            ]);
            validPTMs = ptms;
            validDomains = domains;
            validIDs = proteins;
        } catch (error) {
            console.error('Error loading validation lists:', error);
        }
    }
    fetchValidLists();

    function renderOverlay(inputId, overlayId, delimiter, validList) {
        const input = document.getElementById(inputId);
        const overlay = document.getElementById(overlayId);
        if (!input || !overlay) return;
        let value = input.value;
        let html = '';
        let parts, sep;
        if (delimiter === ' ') {
            parts = value.split(/\s+/);
            sep = ' ';
        } else {
            parts = value.split(',');
            sep = ', ';
        }
        for (let i = 0; i < parts.length; ++i) {
            let token = parts[i].trim();
            if (token === '') continue;
            let isValid = validList.includes(token);
            // Use proper template literals with backticks and ${}:
            html += `<span class="${isValid ? 'token-valid' : 'token-invalid'}">${token}</span>`;
            if (i !== parts.length - 1) html += sep;
        }
        overlay.innerHTML = html;
        overlay.scrollLeft = input.scrollLeft;
    }

    // x PTM overlay
    document.getElementById('x_ptm_text').addEventListener('input', function() {
        renderOverlay('x_ptm_text', 'x_ptm-overlay', ' ', validPTMs);
    });
    document.getElementById('x_ptm_text').addEventListener('scroll', function() {
        document.getElementById('x_ptm-overlay').scrollLeft = this.scrollLeft;
    });

    // y PTM overlay
    document.getElementById('y_ptm_text').addEventListener('input', function() {
        renderOverlay('y_ptm_text', 'y_ptm-overlay', ' ', validPTMs);
    });
    document.getElementById('y_ptm_text').addEventListener('scroll', function() {
        document.getElementById('y_ptm-overlay').scrollLeft = this.scrollLeft;
    });

    // x Domain overlay
    document.getElementById('x_domain_text').addEventListener('input', function() {
        renderOverlay('x_domain_text', 'x_domain-overlay', ' ', validDomains);
    });
    document.getElementById('x_domain_text').addEventListener('scroll', function() {
        document.getElementById('x_domain-overlay').scrollLeft = this.scrollLeft;
    });

    // y Domain overlay
    document.getElementById('y_domain_text').addEventListener('input', function() {
        renderOverlay('y_domain_text', 'y_domain-overlay', ' ', validDomains);
    });
    document.getElementById('y_domain_text').addEventListener('scroll', function() {
        document.getElementById('y_domain-overlay').scrollLeft = this.scrollLeft;
    });

    // x protein overlay
    document.getElementById('x_protein_text').addEventListener('input', function() {
        renderOverlay('x_protein_text', 'x_protein-overlay', ' ', validIDs);
    });
    document.getElementById('x_protein_text').addEventListener('scroll', function() {
        document.getElementById('x_protein-overlay').scrollLeft = this.scrollLeft;
    });

    // y protein overlay
    document.getElementById('y_protein_text').addEventListener('input', function() {
        renderOverlay('y_protein_text', 'y_protein-overlay', ' ', validIDs);
    });
    document.getElementById('y_protein_text').addEventListener('scroll', function() {
        document.getElementById('y_protein-overlay').scrollLeft = this.scrollLeft;
    });

    document.getElementById('x').addEventListener('change', function() {
        hideAllXInputs();
        if (this.value === 'ptm') {
            document.getElementById('x_ptm_input').style.display = 'block';
        } else if (this.value === 'AA') {
            document.getElementById('x_aa_checkboxes').style.display = 'block';
        } else if (this.value === 'sec') {
            document.getElementById('x_sec_checkboxes').style.display = 'block';
        } else if (this.value === 'domain') {
            document.getElementById('x_domain_input').style.display = 'block';
        } else if (this.value === 'protein') {
            document.getElementById('x_protein_input').style.display = 'block';
        }
        updateYOptions(this.value);
});

    document.getElementById('y').addEventListener('change', function() {
        hideAllYInputs();
        if (this.value === 'ptm') {
            document.getElementById('y_ptm_input').style.display = 'block';
        } else if (this.value === 'AA') {
            document.getElementById('y_aa_checkboxes').style.display = 'block';
        } else if (this.value === 'sec') {
            document.getElementById('y_sec_checkboxes').style.display = 'block';
        } else if (this.value === 'domain') {
            document.getElementById('y_domain_input').style.display = 'block';
        } else if (this.value === 'protein') {
            document.getElementById('y_protein_input').style.display = 'block';
        }
        updateXOptions(this.value);
});

    // In x change event:
    updateYOptions(this.value);
    
    // In y change event:
    updateXOptions(this.value);
    
    // Select all checkboxes for x AA
    document.getElementById('x_all').addEventListener('change', function() {
        const checkboxes = document.querySelectorAll('#x_aa_checkboxes input[type="checkbox"][name="x_aa_types[]"]');
        checkboxes.forEach(checkbox => checkbox.checked = this.checked);
    });

    // Uncheck "Select All" if any individual box is unchecked
    document.querySelectorAll('#x_aa_checkboxes input[type="checkbox"][name="x_aa_types[]"]').forEach(checkbox => {
        checkbox.addEventListener('change', function() {
            const all = document.querySelectorAll('#x_aa_checkboxes input[type="checkbox"][name="x_aa_types[]"]');
            const allChecked = Array.from(all).every(cb => cb.checked);
            document.getElementById('x_all').checked = allChecked;
        });
    });

    // Select all checkboxes for x Sec
    document.getElementById('x_all').addEventListener('change', function() {
        const checkboxes = document.querySelectorAll('#x_sec_checkboxes input[type="checkbox"][name="x_sec_types[]"]');
        checkboxes.forEach(checkbox => checkbox.checked = this.checked);
    });

    // Uncheck "Select All" if any individual box is unchecked 
    document.querySelectorAll('#x_sec_checkboxes input[type="checkbox"][name="x_sec_types[]"]').forEach(checkbox => {
        checkbox.addEventListener('change', function() {
            const all = document.querySelectorAll('#x_sec_checkboxes input[type="checkbox"][name="x_sec_types[]"]');
            const allChecked = Array.from(all).every(cb => cb.checked); 
            document.getElementById('x_all').checked = allChecked;
        });
    });

    // Select all checkboxes for y AA
    document.getElementById('y_all').addEventListener('change', function() {
        const checkboxes = document.querySelectorAll('#y_aa_checkboxes input[type="checkbox"][name="y_aa_types[]"]');
        checkboxes.forEach(checkbox => checkbox.checked = this.checked);
    });


    // Uncheck "Select All" if any individual box is unchecked
    document.querySelectorAll('#y_aa_checkboxes input[type="checkbox"][name="y_aa_types[]"]').forEach(checkbox => {
        checkbox.addEventListener('change', function() {
            const all = document.querySelectorAll('#y_aa_checkboxes input[type="checkbox"][name="y_aa_types[]"]');
            const allChecked = Array.from(all).every(cb => cb.checked);
            document.getElementById('y_all').checked = allChecked;
        });
    });

    // Select all checkboxes for y Sec
    document.getElementById('y_all').addEventListener('change', function() {
        const checkboxes = document.querySelectorAll('#y_sec_checkboxes input[type="checkbox"][name="y_types[]"]');
        checkboxes.forEach(checkbox => checkbox.checked = this.checked);
    });

    // Uncheck "Select All" if any individual box is unchecked 
    document.querySelectorAll('#y_sec_checkboxes input[type="checkbox"][name="y_types[]"]').forEach(checkbox => {
        checkbox.addEventListener('change', function() {
            const all = document.querySelectorAll('#y_sec_checkboxes input[type="checkbox"][name="y_types[]"]');
            const allChecked = Array.from(all).every(cb => cb.checked);
            document.getElementById('y_all').checked = allChecked;
        });
    });
    
    function setupBulkSwitch(checkboxId, labelId) {
        const checkbox = document.getElementById(checkboxId);
        const label = document.getElementById(labelId);
        if (checkbox && label) {
        label.textContent = checkbox.checked ? 'Bulk' : 'Individual';
        checkbox.addEventListener('change', function() {
          label.textContent = this.checked ? 'Bulk' : 'Individual';
        });
        }
    }

    // Now call for each pair:
    setupBulkSwitch('x_bulk', 'x_switch-label');
    setupBulkSwitch('y_bulk', 'y_switch-label');
    
    // Initialize - hide all input fields
    hideAllXInputs();
    hideAllYInputs();
    hideAllZInputs();
});
