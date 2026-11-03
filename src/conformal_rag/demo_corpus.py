"""Demo corpus: source documents, labeled calibration samples, and showcase cases.

The calibration sets deliberately mix blatant and *subtle* hallucinations. CRC
picks its threshold from the hallucination score distribution, so a corpus of
only obvious falsehoods would put the threshold at 0.0 and the demo would
approve everything. Graded subtlety is what makes the guarantee visible.

Facts here are accurate reference statements, used as retrieval sources.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Source documents (indexed by the retriever)
# --------------------------------------------------------------------------

GENERAL_DOCS = [
    "The capital of France is Paris. Paris is the most populous city in France, with a population of over 2 million people.",
    "Python is a high-level programming language created by Guido van Rossum in 1991. It emphasises code readability and supports multiple programming paradigms.",
    "The Eiffel Tower was completed in 1889 for the World's Fair. It stands 330 metres tall and was the tallest structure in the world until 1930.",
    "Water boils at 100 degrees Celsius at sea level. The boiling point falls as altitude rises because atmospheric pressure decreases.",
    "Conformal prediction is a framework for constructing prediction sets with guaranteed coverage. It provides distribution-free uncertainty quantification.",
    "Conformal Risk Control extends conformal prediction to control the expected risk of a decision rule, including the false discovery rate.",
    "FastAPI is a Python web framework created by Sebastian Ramirez. It provides automatic OpenAPI documentation and async request handling.",
    "The e5-base-v2 sentence embedding model produces 768 dimensional vectors and is trained with contrastive learning on text pairs.",
]

MEDICAL_DOCS = [
    "Paracetamol is an analgesic and antipyretic. The maximum recommended daily dose for adults is 4 grams.",
    "Normal human body temperature is approximately 37 degrees Celsius. Rectal measurements run slightly higher than oral measurements.",
    "Penicillin was discovered by Alexander Fleming in 1928. It was the first widely used antibiotic.",
    "Insulin is a hormone produced by the beta cells of the pancreas. It regulates blood glucose levels.",
    "Hypertension in adults is commonly defined as a sustained blood pressure of 140 over 90 millimetres of mercury or higher.",
    "Vitamin C deficiency causes scurvy. Symptoms include bleeding gums, fatigue and poor wound healing.",
    "The ABO blood group system classifies blood into A, B, AB and O types. The Rh factor is a separate antigen.",
    "Aspirin inhibits cyclooxygenase enzymes, which reduces the production of prostaglandins.",
    "The kidneys filter blood and remove waste products through urine. They also regulate fluid and electrolyte balance.",
    "Metformin is commonly used as a first line oral medication for type 2 diabetes.",
]

FINANCE_DOCS = [
    "EBITDA stands for earnings before interest, taxes, depreciation and amortisation. It is used as a proxy for operating cash flow.",
    "The current ratio is current assets divided by current liabilities. A ratio above 1 generally indicates short term liquidity.",
    "The price to earnings ratio is the share price divided by earnings per share.",
    "Return on investment is net profit divided by the cost of the investment, expressed as a percentage.",
    "Compound interest is calculated on the initial principal and also on accumulated interest from previous periods.",
    "Diversification spreads capital across assets to reduce the impact of any single holding on portfolio risk.",
]

# --------------------------------------------------------------------------
# Labeled calibration samples: (context, response, label)
# label 0 = faithful, 1 = hallucination. Graded from blatant to subtle.
# --------------------------------------------------------------------------

GENERAL_SAMPLES = [
    (GENERAL_DOCS[0], "Paris is the capital of France.", 0),
    (GENERAL_DOCS[0], "Paris has a population of over 2 million people.", 0),
    (GENERAL_DOCS[1], "Python was created by Guido van Rossum in 1991.", 0),
    (GENERAL_DOCS[1], "Python emphasises code readability.", 0),
    (GENERAL_DOCS[2], "The Eiffel Tower was completed in 1889.", 0),
    (GENERAL_DOCS[2], "The Eiffel Tower is 330 metres tall.", 0),
    (GENERAL_DOCS[3], "Water boils at 100 degrees Celsius at sea level.", 0),
    (GENERAL_DOCS[3], "The boiling point of water falls as altitude rises.", 0),
    (GENERAL_DOCS[4], "Conformal prediction provides distribution-free uncertainty quantification.", 0),
    (GENERAL_DOCS[5], "Conformal Risk Control can control the false discovery rate.", 0),
    (GENERAL_DOCS[6], "FastAPI was created by Sebastian Ramirez.", 0),
    (GENERAL_DOCS[7], "The e5-base-v2 model produces 768 dimensional vectors.", 0),
    (GENERAL_DOCS[0], "The capital of France is Lyon.", 1),
    (GENERAL_DOCS[0], "Paris has a population of about 8 million people.", 1),
    (GENERAL_DOCS[1], "Python was created by Google in 1991.", 1),
    (GENERAL_DOCS[1], "Python was created by Guido van Rossum in 2005.", 1),
    (GENERAL_DOCS[2], "The Eiffel Tower was completed in 1899 for the World's Fair.", 1),
    (GENERAL_DOCS[2], "The Eiffel Tower stands 300 metres tall.", 1),
    (GENERAL_DOCS[3], "Water boils at 90 degrees Celsius at sea level.", 1),
    (GENERAL_DOCS[3], "The boiling point of water rises as altitude rises.", 1),
    (GENERAL_DOCS[4], "Conformal prediction requires the data to be normally distributed.", 1),
    (GENERAL_DOCS[5], "Conformal Risk Control guarantees zero errors in production.", 1),
    (GENERAL_DOCS[6], "FastAPI was created by Guido van Rossum.", 1),
    (GENERAL_DOCS[7], "The e5-base-v2 model produces 512 dimensional vectors.", 1),
]

MEDICAL_SAMPLES = [
    (MEDICAL_DOCS[0], "The maximum daily dose of paracetamol for adults is 4 grams.", 0),
    (MEDICAL_DOCS[0], "Paracetamol reduces pain and fever.", 0),
    (MEDICAL_DOCS[1], "Normal body temperature is about 37 degrees Celsius.", 0),
    (MEDICAL_DOCS[2], "Penicillin was discovered by Alexander Fleming in 1928.", 0),
    (MEDICAL_DOCS[3], "Insulin is produced by the beta cells of the pancreas.", 0),
    (MEDICAL_DOCS[4], "Hypertension is defined as 140 over 90 or higher.", 0),
    (MEDICAL_DOCS[5], "Vitamin C deficiency causes scurvy.", 0),
    (MEDICAL_DOCS[7], "Aspirin inhibits cyclooxygenase enzymes.", 0),
    (MEDICAL_DOCS[8], "The kidneys filter waste products from the blood.", 0),
    (MEDICAL_DOCS[9], "Metformin is a first line treatment for type 2 diabetes.", 0),
    (MEDICAL_DOCS[0], "The maximum daily dose of paracetamol for adults is 8 grams.", 1),
    (MEDICAL_DOCS[0], "Paracetamol is an antibiotic.", 1),
    (MEDICAL_DOCS[1], "Normal body temperature is 39 degrees Celsius.", 1),
    (MEDICAL_DOCS[2], "Penicillin was discovered by Louis Pasteur in 1928.", 1),
    (MEDICAL_DOCS[3], "Insulin is produced by the alpha cells of the pancreas.", 1),
    (MEDICAL_DOCS[4], "Hypertension is defined as 120 over 80 or higher.", 1),
    (MEDICAL_DOCS[5], "Vitamin C deficiency causes rickets.", 1),
    (MEDICAL_DOCS[7], "Aspirin increases cyclooxygenase activity.", 1),
    (MEDICAL_DOCS[8], "The kidneys produce insulin.", 1),
    (MEDICAL_DOCS[9], "Metformin is used to treat type 1 diabetes.", 1),
]

FINANCE_SAMPLES = [
    (FINANCE_DOCS[0], "EBITDA stands for earnings before interest, taxes, depreciation and amortisation.", 0),
    (FINANCE_DOCS[1], "A current ratio above 1 generally indicates short term liquidity.", 0),
    (FINANCE_DOCS[2], "The price to earnings ratio divides share price by earnings per share.", 0),
    (FINANCE_DOCS[3], "Return on investment is net profit divided by the cost of the investment.", 0),
    (FINANCE_DOCS[4], "Compound interest is calculated on principal and accumulated interest.", 0),
    (FINANCE_DOCS[5], "Diversification spreads capital across assets.", 0),
    (FINANCE_DOCS[0], "EBITDA is used as a proxy for operating cash flow.", 0),
    (FINANCE_DOCS[5], "Diversification reduces the impact of any single holding on portfolio risk.", 0),
    (FINANCE_DOCS[4], "Compound interest is calculated on the initial principal and also on accumulated interest from previous periods.", 0),
    (FINANCE_DOCS[5], "Portfolio risk is reduced by spreading capital.", 0),
    (FINANCE_DOCS[0], "EBITDA includes interest and taxes in its calculation.", 1),
    (FINANCE_DOCS[1], "The current ratio is current liabilities divided by current assets.", 1),
    (FINANCE_DOCS[2], "The price to earnings ratio divides earnings per share by share price.", 1),
    (FINANCE_DOCS[3], "Return on investment is revenue divided by total assets.", 1),
    (FINANCE_DOCS[4], "Compound interest is calculated only on the initial principal.", 1),
    (FINANCE_DOCS[5], "Diversification concentrates capital in a single asset.", 1),
    (FINANCE_DOCS[0], "EBITDA is used as a proxy for total revenue.", 1),
    (FINANCE_DOCS[1], "A current ratio above 1 indicates insolvency.", 1),
    (FINANCE_DOCS[3], "Return on investment ignores the cost of the investment.", 1),
    (FINANCE_DOCS[5], "Diversification increases the impact of a single holding.", 1),
]

# --------------------------------------------------------------------------
# Showcase cases for the UI: one faithful, one subtle, one blatant per domain.
# --------------------------------------------------------------------------

SHOWCASE = {
    # Medicine leads: it has the tightest tolerance, so the first case a
    # visitor runs shows a plainly wrong answer being withheld rather than an
    # approved answer that happens to sit inside the permitted error rate.
    "medicine": {
        "query": "What is the maximum daily dose of paracetamol for adults?",
        "context": MEDICAL_DOCS[0],
        "cases": [
            {"kind": "faithful", "response": "The maximum daily dose of paracetamol for adults is 4 grams."},
            {"kind": "subtle", "response": "The maximum daily dose of paracetamol for adults is 8 grams."},
            {"kind": "blatant", "response": "Paracetamol is an antibiotic."},
        ],
    },
    "finance": {
        "query": "What does a current ratio above 1 indicate?",
        "context": FINANCE_DOCS[1],
        "cases": [
            {"kind": "faithful", "response": "A current ratio above 1 generally indicates short term liquidity."},
            {"kind": "subtle", "response": "A current ratio above 1 generally indicates insolvency."},
            {"kind": "blatant", "response": "The current ratio measures annual revenue growth."},
        ],
    },
    "general": {
        "query": "What is the capital of France and how big is it?",
        "context": GENERAL_DOCS[0],
        "cases": [
            {"kind": "faithful", "response": "Paris is the capital of France."},
            {"kind": "subtle", "response": "Paris has a population of about 8 million people."},
            {"kind": "blatant", "response": "The capital of France is Lyon."},
        ],
    },
}

# Representative questions per domain. The seeder cycles these as the
# calibration "queries" so the router's centroid is built from real questions
# rather than a placeholder string, which is what makes auto-routing work on a
# freshly seeded demo.
ROUTING_QUERIES = {
    "general": [
        "What is the capital of France?",
        "Who created Python and when?",
        "How tall is the Eiffel Tower?",
        "At what temperature does water boil?",
        "What does conformal prediction guarantee?",
        "What does Conformal Risk Control control?",
        "Who created FastAPI?",
        "How many dimensions does e5-base-v2 produce?",
    ],
    "medicine": [
        "What is the maximum daily dose of paracetamol for adults?",
        "What is a normal human body temperature?",
        "Who discovered penicillin and when?",
        "Where is insulin produced in the body?",
        "How is hypertension defined in adults?",
        "What does vitamin C deficiency cause?",
        "How is aspirin's mechanism described?",
        "What do the kidneys do?",
        "What is metformin used to treat?",
    ],
    "finance": [
        "What does EBITDA stand for?",
        "How is the current ratio calculated?",
        "What does a current ratio above 1 indicate?",
        "How is the price to earnings ratio defined?",
        "How is return on investment calculated?",
        "How does compound interest work?",
        "Why does diversification reduce risk?",
    ],
}

# Domain registry for the seed step: (name, alpha, docs, samples)
DOMAINS = {
    "medicine": {"alpha": 0.05, "docs": MEDICAL_DOCS, "samples": MEDICAL_SAMPLES},
    "finance": {"alpha": 0.15, "docs": FINANCE_DOCS, "samples": FINANCE_SAMPLES},
}

MAIN_ALPHA = 0.15
ALL_DOCS = GENERAL_DOCS + MEDICAL_DOCS + FINANCE_DOCS