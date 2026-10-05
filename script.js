/**
 * ==============================================================================
 * Bayesian Network for Intelligent Medical Analysis
 * AI/ML Project
 * ==============================================================================
 * This script implements a Bayesian Network inference engine for medical risk estimation.
 * 
 * Target Diseases (Hypothesis Nodes):
 * 1. Flu (Influenza)
 * 2. Pneumonia
 * 
 * Evidence (Symptom Nodes):
 * 1. Fever
 * 2. Cough
 * 3. Headache
 * 4. Fatigue
 * 
 * Network Topology (Directed Acyclic Graph):
 * - Fever    ──→ Flu, Pneumonia
 * - Cough    ──→ Flu, Pneumonia
 * - Headache ──→ Flu (Headache is independent / negligible for Pneumonia)
 * - Fatigue  ──→ Flu, Pneumonia
 * ==============================================================================
 */

// ==============================================================================
// SECTION 1: PROBABILISTIC KNOWLEDGE BASE (EASY TO EXPLAIN DURING VIVA)
// ==============================================================================

/**
 * 1. Prior Probabilities P(Disease)
 * Represents the baseline prevalence of each condition in the target population.
 */
const PRIORS = {
    flu: 0.12,          // P(Flu = True) = 12%
    flu_no: 0.88,       // P(Flu = False) = 88%
    
    pneumonia: 0.04,    // P(Pneumonia = True) = 4%
    pneumonia_no: 0.96  // P(Pneumonia = False) = 96%
};

/**
 * 2. Conditional Probability Tables (CPT)
 * P(Symptom = Present | Disease = Present)
 * Represents the sensitivity/probability of showing a symptom when having the disease.
 */
const CPT_DISEASE_PRESENT = {
    flu: {
        fever: 0.85,     // P(Fever = 1 | Flu = 1)
        cough: 0.80,     // P(Cough = 1 | Flu = 1)
        headache: 0.70,  // P(Headache = 1 | Flu = 1)
        fatigue: 0.85    // P(Fatigue = 1 | Flu = 1)
    },
    pneumonia: {
        fever: 0.90,     // P(Fever = 1 | Pneumonia = 1)
        cough: 0.92,     // P(Cough = 1 | Pneumonia = 1)
        headache: 0.15,  // P(Headache = 1 | Pneumonia = 1) -> Low for lower respiratory
        fatigue: 0.80    // P(Fatigue = 1 | Pneumonia = 1)
    }
};

/**
 * 3. Background Probabilities (False Positive Rates)
 * P(Symptom = Present | Disease = Absent)
 * Represents the probability of showing a symptom due to other background reasons.
 */
const CPT_DISEASE_ABSENT = {
    flu: {
        fever: 0.10,     // P(Fever = 1 | Flu = 0)
        cough: 0.15,     // P(Cough = 1 | Flu = 0)
        headache: 0.20,  // P(Headache = 1 | Flu = 0)
        fatigue: 0.15    // P(Fatigue = 1 | Flu = 0)
    },
    pneumonia: {
        fever: 0.12,     // P(Fever = 1 | Pneumonia = 0)
        cough: 0.18,     // P(Cough = 1 | Pneumonia = 0)
        headache: 0.22,  // P(Headache = 1 | Pneumonia = 0)
        fatigue: 0.18    // P(Fatigue = 1 | Pneumonia = 0)
    }
};

// ==============================================================================
// SECTION 2: BAYESIAN INFERENCE CALCULATION ENGINE
// ==============================================================================

/**
 * Calculates posterior probability P(Disease = 1 | Symptoms) using Bayes' Rule:
 * 
 *                    P(Evidence | Disease = 1) * P(Disease = 1)
 * P(Disease | E) = -------------------------------------------------------------
 *                  P(Evidence | Disease = 1)*P(D=1) + P(Evidence | Disease = 0)*P(D=0)
 * 
 * Under conditional independence (Naive Bayes assumption):
 * P(Evidence | D) = ∏ P(S_i = e_i | D)
 * 
 * @param {string} disease - 'flu' or 'pneumonia'
 * @param {Object} symptoms - Map of symptom name to boolean (true for 'yes', false for 'no')
 * @returns {Object} Full breakdown of prior, likelihood, joint, and posterior
 */
function calculateBayesianPosterior(disease, symptoms) {
    const priorTrue = PRIORS[disease];
    const priorFalse = PRIORS[disease + '_no'];

    const cptTrue = CPT_DISEASE_PRESENT[disease];
    const cptFalse = CPT_DISEASE_ABSENT[disease];

    let likelihoodTrue = 1.0;
    let likelihoodFalse = 1.0;

    // Multiply conditionally independent probabilities for each symptom
    for (const [symptom, isPresent] of Object.entries(symptoms)) {
        const probGivenTrue = cptTrue[symptom];
        const probGivenFalse = cptFalse[symptom];

        if (isPresent) {
            // Symptom is present (e_i = 1)
            likelihoodTrue *= probGivenTrue;
            likelihoodFalse *= probGivenFalse;
        } else {
            // Symptom is absent (e_i = 0) -> Complement probability
            likelihoodTrue *= (1.0 - probGivenTrue);
            likelihoodFalse *= (1.0 - probGivenFalse);
        }
    }

    // Numerator: Joint probability P(Evidence, Disease = 1)
    const numerator = likelihoodTrue * priorTrue;

    // Denominator: Total evidence probability P(Evidence)
    const denominator = numerator + (likelihoodFalse * priorFalse);

    // Posterior Probability P(Disease = 1 | Evidence)
    let posterior = 0;
    if (denominator > 0) {
        posterior = numerator / denominator;
    }

    return {
        prior: priorTrue,
        likelihood: likelihoodTrue,
        jointNumerator: numerator,
        marginalEvidence: denominator,
        posterior: posterior,
        percentage: Math.round(posterior * 100)
    };
}

/**
 * Determines risk category based on percentage:
 * - 0% – 30%   → Low
 * - 31% – 60%  → Moderate
 * - 61% – 100% → High
 */
function getRiskLevel(percentage) {
    if (percentage <= 30) {
        return {
            level: 'Low',
            badgeClass: 'risk-low',
            fillClass: 'var(--risk-low-fill)'
        };
    } else if (percentage <= 60) {
        return {
            level: 'Moderate',
            badgeClass: 'risk-moderate',
            fillClass: 'var(--risk-mod-fill)'
        };
    } else {
        return {
            level: 'High',
            badgeClass: 'risk-high',
            fillClass: 'var(--risk-high-fill)'
        };
    }
}

// ==============================================================================
// SECTION 3: DOM INTERACTION & APPLICATION WORKFLOW
// ==============================================================================

/**
 * Reads current symptom selections from the form
 */
function getSelectedSymptoms() {
    return {
        fever: document.querySelector('input[name="fever"]:checked')?.value === 'yes',
        cough: document.querySelector('input[name="cough"]:checked')?.value === 'yes',
        headache: document.querySelector('input[name="headache"]:checked')?.value === 'yes',
        fatigue: document.querySelector('input[name="fatigue"]:checked')?.value === 'yes'
    };
}

/**
 * Main function triggered by "Analyze Patient" button
 */
function analyzePatient() {
    const symptoms = getSelectedSymptoms();

    // 1. Calculate Bayesian Posteriors
    const fluResult = calculateBayesianPosterior('flu', symptoms);
    const pneumoniaResult = calculateBayesianPosterior('pneumonia', symptoms);

    // 2. Update Result Display
    updateResultUI(fluResult, pneumoniaResult, symptoms);

    // 3. Update Visual Bayesian Network Diagram
    updateDiagramUI(symptoms, fluResult, pneumoniaResult);

    // 4. Update Viva Mathematical Breakdown Log
    updateVivaLog(fluResult, pneumoniaResult);
}

/**
 * Updates UI Cards, Progress Bars, Risk Badges, and Natural Language Explanation
 */
function updateResultUI(flu, pneu, symptoms) {
    // 1. Update Flu Card
    const fluRisk = getRiskLevel(flu.percentage);
    document.getElementById('flu-percent').textContent = `${flu.percentage}%`;
    const fluBadge = document.getElementById('flu-risk-badge');
    fluBadge.textContent = `${fluRisk.level} Risk`;
    fluBadge.className = `risk-badge ${fluRisk.badgeClass}`;
    
    const fluBar = document.getElementById('flu-progress-bar');
    fluBar.style.width = `${flu.percentage}%`;
    fluBar.style.background = fluRisk.fillClass;

    // 2. Update Pneumonia Card
    const pneuRisk = getRiskLevel(pneu.percentage);
    document.getElementById('pneumonia-percent').textContent = `${pneu.percentage}%`;
    const pneuBadge = document.getElementById('pneumonia-risk-badge');
    pneuBadge.textContent = `${pneuRisk.level} Risk`;
    pneuBadge.className = `risk-badge ${pneuRisk.badgeClass}`;
    
    const pneuBar = document.getElementById('pneumonia-progress-bar');
    pneuBar.style.width = `${pneu.percentage}%`;
    pneuBar.style.background = pneuRisk.fillClass;

    // 3. Determine Most Likely Condition
    const winnerEl = document.getElementById('most-likely-condition');
    const subtextEl = document.getElementById('most-likely-subtext');
    const mostLikelyBox = document.getElementById('most-likely-box');

    const activeCount = Object.values(symptoms).filter(Boolean).length;

    let mostLikely = "";
    if (activeCount === 0) {
        winnerEl.textContent = "No Symptoms Reported";
        subtextEl.textContent = "Baseline prior risks apply. Both conditions are at minimal low levels.";
        winnerEl.style.color = "#94a3b8";
    } else if (flu.percentage === pneu.percentage) {
        winnerEl.textContent = "Equal Risk (Flu & Pneumonia)";
        subtextEl.textContent = `Both conditions have an estimated probability of ${flu.percentage}%.`;
        winnerEl.style.color = "#38bdf8";
        mostLikely = "Equal";
    } else if (flu.percentage > pneu.percentage) {
        winnerEl.textContent = "Flu (Influenza)";
        subtextEl.textContent = `Flu probability (${flu.percentage}%) is higher than Pneumonia (${pneu.percentage}%).`;
        winnerEl.style.color = "#38bdf8";
        mostLikely = "Flu";
    } else {
        winnerEl.textContent = "Pneumonia";
        subtextEl.textContent = `Pneumonia probability (${pneu.percentage}%) is higher than Flu (${flu.percentage}%).`;
        winnerEl.style.color = "#38bdf8";
        mostLikely = "Pneumonia";
    }

    // 4. Generate Educational Explanation
    generateExplanation(flu, pneu, symptoms, mostLikely);
}

/**
 * Builds clear natural language explanation of the inference
 */
function generateExplanation(flu, pneu, symptoms, mostLikely) {
    const expText = document.getElementById('explanation-text');
    const activeSymptoms = [];
    if (symptoms.fever) activeSymptoms.push("Fever");
    if (symptoms.cough) activeSymptoms.push("Cough");
    if (symptoms.headache) activeSymptoms.push("Headache");
    if (symptoms.fatigue) activeSymptoms.push("Fatigue");

    if (activeSymptoms.length === 0) {
        expText.innerHTML = "No symptoms were selected. The system displays background prior probabilities (Flu: 12%, Pneumonia: 4%). Both risks remain in the <strong>Low</strong> category.";
        return;
    }

    let explanation = `Based on the observed symptoms (<strong>${activeSymptoms.join(", ")}</strong>), the Bayesian Network calculates: `;

    if (flu.percentage > pneu.percentage) {
        explanation += `a higher probability for <strong>Flu (${flu.percentage}%)</strong> compared to <strong>Pneumonia (${pneu.percentage}%)</strong>. `;
        if (symptoms.headache) {
            explanation += `The presence of <em>Headache</em> strongly supports Flu over Pneumonia because headaches have a high conditional probability for Flu (70%) but a very low correlation with Pneumonia (15%).`;
        } else {
            explanation += `The combination of systemic symptoms matches the prior and conditional likelihood profile for Influenza.`;
        }
    } else if (pneu.percentage > flu.percentage) {
        explanation += `a higher probability for <strong>Pneumonia (${pneu.percentage}%)</strong> compared to <strong>Flu (${flu.percentage}%)</strong>. `;
        if (!symptoms.headache && symptoms.cough && symptoms.fever) {
            explanation += `Severe lower respiratory indicators (Fever & Cough) without upper-respiratory headache significantly elevate the Bayesian posterior for Pneumonia.`;
        }
    } else {
        explanation += `an equivalent probability of <strong>${flu.percentage}%</strong> for both conditions based on the symmetric evidence provided.`;
    }

    expText.innerHTML = explanation;
}

/**
 * Highlights active nodes and animated edges in the SVG Bayesian Network Diagram
 */
function updateDiagramUI(symptoms, fluResult, pneumoniaResult) {
    // 1. Update Symptom Nodes & Edge Highlights
    const symptomKeys = ['fever', 'cough', 'headache', 'fatigue'];

    symptomKeys.forEach(symptom => {
        const isYes = symptoms[symptom];
        
        // Form item highlight
        const itemEl = document.getElementById(`item-${symptom}`);
        if (itemEl) {
            if (isYes) itemEl.classList.add('active');
            else itemEl.classList.remove('active');
        }

        // SVG Node styling
        const svgNode = document.getElementById(`svg-node-${symptom}`);
        const stateText = document.getElementById(`svg-state-${symptom}`);
        if (svgNode && stateText) {
            if (isYes) {
                svgNode.classList.add('active');
                stateText.textContent = "State: YES";
            } else {
                svgNode.classList.remove('active');
                stateText.textContent = "State: No";
            }
        }
    });

    // 2. Update Edge Classes
    setEdgeState('edge-fever-flu', symptoms.fever);
    setEdgeState('edge-fever-pne', symptoms.fever);
    setEdgeState('edge-cough-flu', symptoms.cough);
    setEdgeState('edge-cough-pne', symptoms.cough);
    setEdgeState('edge-headache-flu', symptoms.headache);
    setEdgeState('edge-fatigue-flu', symptoms.fatigue);
    setEdgeState('edge-fatigue-pne', symptoms.fatigue);

    // 3. Update Disease SVG Nodes
    const fluText = document.getElementById('svg-prob-flu');
    const pneText = document.getElementById('svg-prob-pne');
    const fluNode = document.getElementById('svg-node-flu');
    const pneNode = document.getElementById('svg-node-pneumonia');

    if (fluResult && pneumoniaResult) {
        fluText.textContent = `P(Flu): ${fluResult.percentage}%`;
        pneText.textContent = `P(Pneu): ${pneumoniaResult.percentage}%`;

        fluNode.classList.remove('dominant');
        pneNode.classList.remove('dominant');

        if (fluResult.percentage > pneumoniaResult.percentage && fluResult.percentage > 30) {
            fluNode.classList.add('dominant');
        } else if (pneumoniaResult.percentage > fluResult.percentage && pneumoniaResult.percentage > 30) {
            pneNode.classList.add('dominant');
        }
    }
}

function setEdgeState(edgeId, isActive) {
    const edge = document.getElementById(edgeId);
    if (!edge) return;
    if (isActive) {
        edge.classList.add('active-edge');
        edge.setAttribute('marker-end', 'url(#arrow-active)');
    } else {
        edge.classList.remove('active-edge');
        edge.setAttribute('marker-end', 'url(#arrow)');
    }
}

/**
 * Real-time feedback when radio buttons change
 */
function handleSymptomChange() {
    const symptoms = getSelectedSymptoms();
    updateDiagramUI(symptoms);
}

// ==============================================================================
// SECTION 4: SAMPLE CASES & FORM CONTROLS
// ==============================================================================

/**
 * Loads predefined clinical sample test cases
 * 
 * Case 1 – Mild Symptoms: Fever=No, Cough=Yes, Headache=Yes, Fatigue=No
 * Case 2 – Flu-like Symptoms: Fever=Yes, Cough=Yes, Headache=Yes, Fatigue=Yes
 * Case 3 – Respiratory Symptoms: Fever=Yes, Cough=Yes, Headache=No, Fatigue=Yes
 */
function loadSampleCase(caseNumber) {
    let settings = {
        fever: 'no',
        cough: 'no',
        headache: 'no',
        fatigue: 'no'
    };

    if (caseNumber === 1) {
        // Case 1 – Mild Symptoms
        settings = { fever: 'no', cough: 'yes', headache: 'yes', fatigue: 'no' };
    } else if (caseNumber === 2) {
        // Case 2 – Flu-like Symptoms
        settings = { fever: 'yes', cough: 'yes', headache: 'yes', fatigue: 'yes' };
    } else if (caseNumber === 3) {
        // Case 3 – Respiratory Symptoms
        settings = { fever: 'yes', cough: 'yes', headache: 'no', fatigue: 'yes' };
    }

    // Apply values to radio inputs
    for (const [symptom, val] of Object.entries(settings)) {
        const radio = document.getElementById(`${symptom}-${val}`);
        if (radio) radio.checked = true;
    }

    // Trigger analysis automatically
    analyzePatient();
}

/**
 * Resets all inputs and clears result state
 */
function clearForm() {
    const defaultNo = ['fever-no', 'cough-no', 'headache-no', 'fatigue-no'];
    defaultNo.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.checked = true;
    });

    // Reset Results Display
    document.getElementById('flu-percent').textContent = "0%";
    document.getElementById('flu-risk-badge').textContent = "Low Risk";
    document.getElementById('flu-risk-badge').className = "risk-badge risk-low";
    document.getElementById('flu-progress-bar').style.width = "0%";

    document.getElementById('pneumonia-percent').textContent = "0%";
    document.getElementById('pneumonia-risk-badge').textContent = "Low Risk";
    document.getElementById('pneumonia-risk-badge').className = "risk-badge risk-low";
    document.getElementById('pneumonia-progress-bar').style.width = "0%";

    document.getElementById('most-likely-condition').textContent = "Click \"Analyze Patient\" to compute";
    document.getElementById('most-likely-condition').style.color = "#38bdf8";
    document.getElementById('most-likely-subtext').textContent = "Bayesian inference awaiting symptom selection";

    document.getElementById('explanation-text').innerHTML = 
        "Please select patient symptoms and click <strong>\"Analyze Patient\"</strong> or choose a sample case above to generate probabilistic risk assessments.";

    // Reset Diagram
    const symptoms = { fever: false, cough: false, headache: false, fatigue: false };
    updateDiagramUI(symptoms);

    document.getElementById('svg-prob-flu').textContent = "P(Flu): --%";
    document.getElementById('svg-prob-pne').textContent = "P(Pneu): --%";
    document.getElementById('svg-node-flu').classList.remove('dominant');
    document.getElementById('svg-node-pneumonia').classList.remove('dominant');

    // Reset Viva Table
    document.getElementById('viva-lik-flu').textContent = "--";
    document.getElementById('viva-lik-pne').textContent = "--";
    document.getElementById('viva-num-flu').textContent = "--";
    document.getElementById('viva-num-pne').textContent = "--";
    document.getElementById('viva-post-flu').textContent = "--%";
    document.getElementById('viva-post-pne').textContent = "--%";
}

// ==============================================================================
// SECTION 5: VIVA CALCULATION BREAKDOWN
// ==============================================================================

/**
 * Toggles collapsible Viva math section
 */
function toggleVivaDetails() {
    const content = document.getElementById('viva-details');
    const btn = document.getElementById('viva-toggle-btn');
    if (content.classList.contains('open')) {
        content.classList.remove('open');
        btn.classList.remove('rotated');
    } else {
        content.classList.add('open');
        btn.classList.add('rotated');
    }
}

/**
 * Updates the step-by-step numbers in the viva calculation breakdown table
 */
function updateVivaLog(flu, pneu) {
    document.getElementById('viva-lik-flu').textContent = flu.likelihood.toFixed(5);
    document.getElementById('viva-lik-pne').textContent = pneu.likelihood.toFixed(5);
    document.getElementById('viva-num-flu').textContent = flu.jointNumerator.toFixed(5);
    document.getElementById('viva-num-pne').textContent = pneu.jointNumerator.toFixed(5);
    document.getElementById('viva-post-flu').textContent = `${flu.percentage}% (${flu.posterior.toFixed(4)})`;
    document.getElementById('viva-post-pne').textContent = `${pneu.percentage}% (${pneu.posterior.toFixed(4)})`;
}

// ==============================================================================
// INITIALIZATION
// ==============================================================================
document.addEventListener('DOMContentLoaded', () => {
    // Initial diagram state setup
    handleSymptomChange();
});
