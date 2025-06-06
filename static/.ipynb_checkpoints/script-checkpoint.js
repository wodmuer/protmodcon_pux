document.addEventListener('DOMContentLoaded', function() {
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

    // Function to hide all Z input fields
    function hideAllZInputs() {
        document.getElementById('z_sec_checkboxes').style.display = 'none';
        document.getElementById('z_domain_input').style.display = 'none';
        document.getElementById('z_protein_input').style.display = 'none';
    }
    
    // Show/hide input fields based on z selection
    document.getElementById('z').addEventListener('change', function() {
        // Hide all z inputs first
        hideAllZInputs();
        // Show the appropriate input based on selection
        if (this.value === 'sec') {
            document.getElementById('z_sec_checkboxes').style.display = 'block';
        } else if (this.value === 'domain') {
            document.getElementById('z_domain_input').style.display = 'block';
        } else if (this.value === 'protein') {
            document.getElementById('z_protein_input').style.display = 'block';
        }
    });

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

    // z protein overlay
    document.getElementById('z_protein_text').addEventListener('input', function() {
        renderOverlay('z_protein_text', 'z_protein-overlay', ' ', validIDs);
    });
    document.getElementById('z_protein_text').addEventListener('scroll', function() {
        document.getElementById('z_protein-overlay').scrollLeft = this.scrollLeft;
    });
    
    // Show/hide input fields based on z selection
    document.getElementById('z').addEventListener('change', function() {
        // Hide all z inputs first
        //hideAllZInputs();
        
        // Show the appropriate input based on selection
        if (this.value === 'AA') {
            document.getElementById('z_aa_checkboxes').style.display = 'block';
        } else if (this.value === 'sec') {
            document.getElementById('z_sec_checkboxes').style.display = 'block';
        } else if (this.value === 'domain') {
            document.getElementById('z_domain_input').style.display = 'block';
        } else if (this.value === 'protein') {
            document.getElementById('z_protein_input').style.display = 'block';
        }
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
    document.getElementById('x_aa_all').addEventListener('change', function() {
        const checkboxes = document.querySelectorAll('#x_aa_checkboxes input[type="checkbox"][name="x_types[]"]');
        checkboxes.forEach(checkbox => checkbox.checked = this.checked);
    });

    // Uncheck "Select All" if any individual box is unchecked
    document.querySelectorAll('#x_aa_checkboxes input[type="checkbox"][name="x_types[]"]').forEach(checkbox => {
        checkbox.addEventListener('change', function() {
            const all = document.querySelectorAll('#x_aa_checkboxes input[type="checkbox"][name="x_types[]"]');
            const allChecked = Array.from(all).every(cb => cb.checked);
            /*document.getElementById('x_aa_all').checked = allChecked;*/
        });
    });

    // Select all checkboxes for x Sec
    document.getElementById('x_sec_all').addEventListener('change', function() {
        const checkboxes = document.querySelectorAll('#x_sec_checkboxes input[type="checkbox"][name="x_types[]"]');
        checkboxes.forEach(checkbox => checkbox.checked = this.checked);
    });

    // Uncheck "Select All" if any individual box is unchecked 
    document.querySelectorAll('#x_sec_checkboxes input[type="checkbox"][name="x_types[]"]').forEach(checkbox => {
        checkbox.addEventListener('change', function() {
            const all = document.querySelectorAll('#x_sec_checkboxes input[type="checkbox"][name="x_types[]"]');
            const allChecked = Array.from(all).every(cb => cb.checked); 
            document.getElementById('x_sec_all').checked = allChecked;
        });
    });

    // Select all checkboxes for y AA
    document.getElementById('y_aa_all').addEventListener('change', function() {
        const checkboxes = document.querySelectorAll('#y_aa_checkboxes input[type="checkbox"][name="y_types[]"]');
        checkboxes.forEach(checkbox => checkbox.checked = this.checked);
    });


    // Uncheck "Select All" if any individual box is unchecked
    document.querySelectorAll('#y_aa_checkboxes input[type="checkbox"][name="y_types[]"]').forEach(checkbox => {
        checkbox.addEventListener('change', function() {
            const all = document.querySelectorAll('#y_aa_checkboxes input[type="checkbox"][name="y_types[]"]');
            const allChecked = Array.from(all).every(cb => cb.checked);
            document.getElementById('y_aa_all').checked = allChecked;
        });
    });

    // Select all checkboxes for y Sec
    document.getElementById('y_sec_all').addEventListener('change', function() {
        const checkboxes = document.querySelectorAll('#y_sec_checkboxes input[type="checkbox"][name="y_types[]"]');
        checkboxes.forEach(checkbox => checkbox.checked = this.checked);
    });

    // Uncheck "Select All" if any individual box is unchecked 
    document.querySelectorAll('#y_sec_checkboxes input[type="checkbox"][name="y_types[]"]').forEach(checkbox => {
        checkbox.addEventListener('change', function() {
            const all = document.querySelectorAll('#y_sec_checkboxes input[type="checkbox"][name="y_types[]"]');
            const allChecked = Array.from(all).every(cb => cb.checked);
            document.getElementById('y_sec_all').checked = allChecked;
        });
    });

    // Select all checkboxes for z Sec
    document.getElementById('z_sec_all').addEventListener('change', function() {
        const checkboxes = document.querySelectorAll('#z_sec_checkboxes input[type="checkbox"][name="z_data[]"]');
        checkboxes.forEach(checkbox => checkbox.checked = this.checked);
    });

    // Uncheck "Select All" if any individual box is unchecked 
    document.querySelectorAll('#z_sec_checkboxes input[type="checkbox"][name="z_data[]"]').forEach(checkbox => {
        checkbox.addEventListener('change', function() {
            const all = document.querySelectorAll('#z_sec_checkboxes input[type="checkbox"][name="z_data[]"]');
            const allChecked = Array.from(all).every(cb => cb.checked);
            document.getElementById('z_sec_all').checked = allChecked;
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
    setupBulkSwitch('x_bulk', 'switch-label-x');
    setupBulkSwitch('y_bulk', 'switch-label-y');d


    // Initialize - hide all input fields
    hideAllXInputs();
    hideAllYInputs();
    hideAllZInputs();

});