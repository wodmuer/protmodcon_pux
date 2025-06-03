document.addEventListener('DOMContentLoaded', function() {
    let validPTMs = [];
    let validDomains = [];

    // Mapping for corresponding options between X and Y dropdowns
    const xToYMap = {
        'ptm_name': 'PTM',
        'AA': 'AA',
        'sec': 'sec',
        'domain': 'domain'
    };
    const yToXMap = {
        'PTM': 'ptm_name',
        'AA': 'AA',
        'sec': 'sec',
        'domain': 'domain'
    };

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

    // Function to hide all X input fields
    function hideAllXInputs() {
        document.getElementById('x_ptm_input').style.display = 'none';
        document.getElementById('x_aa_checkboxes').style.display = 'none';
        document.getElementById('x_sec_checkboxes').style.display = 'none';
        document.getElementById('x_domain_input').style.display = 'none';
    }

    // Function to hide all Y input fields
    function hideAllYInputs() {
        document.getElementById('y_ptm_input').style.display = 'none';
        document.getElementById('y_aa_checkboxes').style.display = 'none';
        document.getElementById('y_sec_checkboxes').style.display = 'none';
        document.getElementById('y_domain_input').style.display = 'none';
    }

    async function fetchValidLists() {
        try {
            const [ptms, domains] = await Promise.all([
                fetch('/static/valid_PTMs.json').then(r => r.json()),
                fetch('/static/valid_domains.json').then(r => r.json())
            ]);
            validPTMs = ptms;
            validDomains = domains;
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

    // x Domain overlay
    document.getElementById('x_domain_text').addEventListener('input', function() {
        renderOverlay('x_domain_text', 'domain-overlay', ' ', validDomains);
    });
    document.getElementById('x_domain_text').addEventListener('scroll', function() {
        document.getElementById('domain-overlay').scrollLeft = this.scrollLeft;
    });

    // y PTM overlay
    document.getElementById('y_ptm_text').addEventListener('input', function() {
        renderOverlay('y_ptm_text', 'second_ptm-overlay', ' ', validPTMs);
    });
    document.getElementById('y_ptm_text').addEventListener('scroll', function() {
        document.getElementById('second_ptm-overlay').scrollLeft = this.scrollLeft;
    });

    // y Domain overlay
    document.getElementById('y_domain_text').addEventListener('input', function() {
        renderOverlay('y_domain_text', 'second_domain-overlay', ' ', validDomains);
    });
    document.getElementById('y_domain_text').addEventListener('scroll', function() {
        document.getElementById('second_domain-overlay').scrollLeft = this.scrollLeft;
    });

    // Show/hide input fields based on x selection
    document.getElementById('x').addEventListener('change', function() {
        // Hide all x inputs first
        hideAllXInputs();
        
        // Show the appropriate input based on selection
        if (this.value === 'ptm_name') {
            document.getElementById('x_ptm_input').style.display = 'block';
        } else if (this.value === 'AA') {
            document.getElementById('x_aa_checkboxes').style.display = 'block';
        } else if (this.value === 'sec') {
            document.getElementById('x_sec_checkboxes').style.display = 'block';
        } else if (this.value === 'domain') {
            document.getElementById('x_domain_input').style.display = 'block';
        }
        
        // Update Y options to disable the selected X option
        updateYOptions(this.value);
    });

    // Show/hide input fields based on y selection
    document.getElementById('y').addEventListener('change', function() {
        // Hide all y inputs first
        hideAllYInputs();
        
        // Show the appropriate input based on selection
        if (this.value === 'PTM') {
            document.getElementById('y_ptm_input').style.display = 'block';
        } else if (this.value === 'AA') {
            document.getElementById('y_aa_checkboxes').style.display = 'block';
        } else if (this.value === 'sec') {
            document.getElementById('y_sec_checkboxes').style.display = 'block';
        } else if (this.value === 'domain') {
            document.getElementById('y_domain_input').style.display = 'block';
        }
        
        // Update X options to disable the selected Y option
        updateXOptions(this.value);
    });

    // Select all checkboxes for x AA
    document.getElementById('x_aa_all').addEventListener('change', function() {
        const checkboxes = document.querySelectorAll('#x_aa_checkboxes input[type="checkbox"][name="x_data[]"]');
        checkboxes.forEach(checkbox => checkbox.checked = this.checked);
    });

    // Uncheck "Select All" if any individual box is unchecked
    document.querySelectorAll('#x_aa_checkboxes input[type="checkbox"][name="x_data[]"]').forEach(checkbox => {
        checkbox.addEventListener('change', function() {
            const all = document.querySelectorAll('#x_aa_checkboxes input[type="checkbox"][name="x_data[]"]');
            const allChecked = Array.from(all).every(cb => cb.checked);
            document.getElementById('x_aa_all').checked = allChecked;
        });
    });

    // Select all checkboxes for x Sec
    document.getElementById('x_sec_all').addEventListener('change', function() {
        const checkboxes = document.querySelectorAll('#x_sec_checkboxes input[type="checkbox"][name="x_data[]"]');
        checkboxes.forEach(checkbox => checkbox.checked = this.checked);
    });

    // Uncheck "Select All" if any individual box is unchecked 
    document.querySelectorAll('#x_sec_checkboxes input[type="checkbox"][name="x_data[]"]').forEach(checkbox => {
        checkbox.addEventListener('change', function() {
            const all = document.querySelectorAll('#x_sec_checkboxes input[type="checkbox"][name="x_data[]"]');
            const allChecked = Array.from(all).every(cb => cb.checked); 
            document.getElementById('x_sec_all').checked = allChecked;
        });
    });

    // Select all checkboxes for y AA
    document.getElementById('y_aa_all').addEventListener('change', function() {
        const checkboxes = document.querySelectorAll('#y_aa_checkboxes input[type="checkbox"][name="y_data[]"]');
        checkboxes.forEach(checkbox => checkbox.checked = this.checked);
    });

    // Uncheck "Select All" if any individual box is unchecked
    document.querySelectorAll('#y_aa_checkboxes input[type="checkbox"][name="y_data[]"]').forEach(checkbox => {
        checkbox.addEventListener('change', function() {
            const all = document.querySelectorAll('#y_aa_checkboxes input[type="checkbox"][name="y_data[]"]');
            const allChecked = Array.from(all).every(cb => cb.checked);
            document.getElementById('y_aa_all').checked = allChecked;
        });
    });

    // Select all checkboxes for y Sec
    document.getElementById('y_sec_all').addEventListener('change', function() {
        const checkboxes = document.querySelectorAll('#y_sec_checkboxes input[type="checkbox"][name="y_data[]"]');
        checkboxes.forEach(checkbox => checkbox.checked = this.checked);
    });

    // Uncheck "Select All" if any individual box is unchecked 
    document.querySelectorAll('#y_sec_checkboxes input[type="checkbox"][name="y_data[]"]').forEach(checkbox => {
        checkbox.addEventListener('change', function() {
            const all = document.querySelectorAll('#y_sec_checkboxes input[type="checkbox"][name="y_data[]"]');
            const allChecked = Array.from(all).every(cb => cb.checked);
            document.getElementById('y_sec_all').checked = allChecked;
        });
    });

    // Initialize - hide all input fields
    hideAllXInputs();
    hideAllYInputs();
});

