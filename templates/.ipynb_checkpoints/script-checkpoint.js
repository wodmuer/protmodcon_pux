document.addEventListener('DOMContentLoaded', function() {
    let validPTMs = [];
    let validDomains = [];

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
            html += `<span class="${isValid ? 'token-valid' : 'token-invalid'}">${token}</span>`;
            if (i !== parts.length - 1) html += sep;
        }
        overlay.innerHTML = html;
        overlay.scrollLeft = input.scrollLeft;
    }

    // First_x PTM overlay
    document.getElementById('first_x_ptm_text').addEventListener('input', function() {
        renderOverlay('first_x_ptm_text', 'ptm-overlay', ' ', validPTMs);
    });
    document.getElementById('first_x_ptm_text').addEventListener('scroll', function() {
        document.getElementById('ptm-overlay').scrollLeft = this.scrollLeft;
    });

    // First_x Domain overlay
    document.getElementById('first_x_domain_text').addEventListener('input', function() {
        renderOverlay('first_x_domain_text', 'domain-overlay', ',', validDomains);
    });
    document.getElementById('first_x_domain_text').addEventListener('scroll', function() {
        document.getElementById('domain-overlay').scrollLeft = this.scrollLeft;
    });

    // Second_x PTM overlay
    document.getElementById('second_x_ptm_text').addEventListener('input', function() {
        renderOverlay('second_x_ptm_text', 'second_ptm-overlay', ' ', validPTMs);
    });
    document.getElementById('second_x_ptm_text').addEventListener('scroll', function() {
        document.getElementById('second_ptm-overlay').scrollLeft = this.scrollLeft;
    });

    // Second_x Domain overlay
    document.getElementById('second_x_domain_text').addEventListener('input', function() {
        renderOverlay('second_x_domain_text', 'second_domain-overlay', ',', validDomains);
    });
    document.getElementById('second_x_domain_text').addEventListener('scroll', function() {
        document.getElementById('second_domain-overlay').scrollLeft = this.scrollLeft;
    });

    // Show/hide input fields based on first_x selection
    document.getElementById('first_x').addEventListener('change', function() {
        document.getElementById('first_x_ptm_input').style.display = (this.value === 'PTM') ? 'block' : 'none';
        document.getElementById('first_x_aa_checkboxes').style.display = (this.value === 'AA_all') ? 'block' : 'none';
        document.getElementById('first_x_sec_checkboxes').style.display = (this.value === 'sec_all') ? 'block' : 'none';
        document.getElementById('first_x_domain_input').style.display = (this.value === 'domain') ? 'block' : 'none';
    });

    // Show/hide input fields based on second_x selection
    document.getElementById('second_x').addEventListener('change', function() {
        document.getElementById('second_x_ptm_input').style.display = (this.value === 'PTM') ? 'block' : 'none';
        document.getElementById('second_x_aa_checkboxes').style.display = (this.value === 'AA_all') ? 'block' : 'none';
        document.getElementById('second_x_sec_checkboxes').style.display = (this.value === 'sec_all') ? 'block' : 'none';
        document.getElementById('second_x_domain_input').style.display = (this.value === 'domain') ? 'block' : 'none';
    });

    // Select all checkboxes for first_x AA
    document.getElementById('first_x_aa_all').addEventListener('change', function() {
        const checkboxes = document.querySelectorAll('input[name="first_x_aa[]"]');
        checkboxes.forEach(checkbox => checkbox.checked = this.checked);
    });

    // Select all checkboxes for first_x Sec
    document.getElementById('first_x_sec_all').addEventListener('change', function() {
        const checkboxes = document.querySelectorAll('input[name="first_x_sec[]"]');
        checkboxes.forEach(checkbox => checkbox.checked = this.checked);
    });

    // Select all checkboxes for second_x AA
    document.getElementById('second_x_aa_all').addEventListener('change', function() {
        const checkboxes = document.querySelectorAll('input[name="second_x_aa[]"]');
        checkboxes.forEach(checkbox => checkbox.checked = this.checked);
    });

    // Select all checkboxes for second_x Sec
    document.getElementById('second_x_sec_all').addEventListener('change', function() {
        const checkboxes = document.querySelectorAll('input[name="second_x_sec[]"]');
        checkboxes.forEach(checkbox => checkbox.checked = this.checked);
    });

    // Initial state - hide all input fields
    document.getElementById('first_x_ptm_input').style.display = 'none';
    document.getElementById('first_x_aa_checkboxes').style.display = 'none';
    document.getElementById('first_x_sec_checkboxes').style.display = 'none';
    document.getElementById('first_x_domain_input').style.display = 'none';
    document.getElementById('second_x_ptm_input').style.display = 'none';
    document.getElementById('second_x_aa_checkboxes').style.display = 'none';
    document.getElementById('second_x_sec_checkboxes').style.display = 'none';
    document.getElementById('second_x_domain_input').style.display = 'none';
});