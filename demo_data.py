"""Demo datasets for the Conformal RAG panel demo.

Each dataset is a list of facts. Responses are generated at three faithfulness
levels so calibration sees a realistic spread rather than only the trivially-zero
fabricated end:

  faithful  restates the context                    -> label 0
  drifted   correct except one corrupted detail     -> label 1
  invented  contradicts or fabricates              -> label 1

The "drifted" level is the important one. Real hallucinations are overwhelmingly
small factual errors, not total fabrications, and a calibration set containing
only fabrications teaches a threshold far lower than production needs.

Scale via build_dataset(facts, n_per_fact) — the demo sizes the calibration set
from the length of the fact list.
"""

from __future__ import annotations

# (topic, context, faithful, drifted, invented)
Fact = tuple[str, str, str, str, str]

# --------------------------------------------------------------------------
# Default: technology and science
# --------------------------------------------------------------------------

TECH_FACTS: list[Fact] = [
    ("einstein",
     "Albert Einstein published the special theory of relativity in 1905 and the general "
     "theory in 1915. He received the Nobel Prize in Physics in 1921 for his "
     "explanation of the photoelectric effect.",
     "Einstein published the special theory of relativity in 1905 and the general theory in 1915.",
     "Einstein published the special theory of relativity in 1905 and the general theory in 1916.",
     "Einstein published both theories of relativity in 1905 and won the Nobel Prize for them in 1921."),

    ("french-revolution",
     "The French Revolution began in 1789 with the storming of the Bastille and ended in "
     "the late 1790s with the rise of Napoleon Bonaparte.",
     "The Revolution began in 1789 with the storming of the Bastille and ended in the late 1790s.",
     "The Revolution began in 1689 with the storming of the Bastille.",
     "The French Revolution began in 1789 and ended when Napoleon Bonaparte was crowned in 1804."),

    ("python",
     "Python was conceived in the late 1980s by Guido van Rossum at CWI in the "
     "Netherlands as a successor to the ABC programming language.",
     "Python was conceived by Guido van Rossum at CWI in the late 1980s.",
     "Python was conceived by Guido van Rossum at Bell Labs in the late 1980s.",
     "Python was created by Bjarne Stroustrup in 1995 as a compiled systems language."),

    ("conformal",
     "Conformal prediction constructs prediction sets with guaranteed coverage without "
     "assuming any distribution for the data.",
     "Conformal prediction gives guaranteed coverage with no distributional assumption.",
     "Conformal prediction requires the data to be normally distributed.",
     "Conformal prediction returns a confidence interval around a single point prediction."),

    ("qdrant",
     "Qdrant is a vector database written in Rust that supports dense, sparse and hybrid "
     "vector search with filtering.",
     "Qdrant is a vector database written in Rust supporting dense, sparse and hybrid search.",
     "Qdrant is a vector database written in Rust released under the MIT licence.",
     "Qdrant is a relational SQL database written in Java for transactional workloads."),

    ("mitochondria",
     "The mitochondrion is the organelle responsible for producing ATP through oxidative "
     "phosphorylation in eukaryotic cells.",
     "The mitochondrion produces ATP through oxidative phosphorylation.",
     "The mitochondrion produces ATP through photosynthesis in plant cells.",
     "Mitochondria are the organelles where protein synthesis occurs in prokaryotes."),

    ("turing",
     "Alan Turing published his paper on the decision procedure for Hilbert's Entscheidungsproblem "
     "in 1936, and proposed the Turing machine as a model of computation.",
     "Turing published his paper on computability and proposed the Turing machine in 1936.",
     "Turing published his paper on computability and proposed the Turing machine in 1946.",
     "Turing invented the computer and led its wartime decryption effort in 1940."),

    ("photosynthesis",
     "Photosynthesis converts light energy into chemical energy stored in glucose, using "
     "carbon dioxide and water, and releasing oxygen as a by-product.",
     "Photosynthesis converts light into chemical energy stored in glucose and releases oxygen.",
     "Photosynthesis converts nitrogen into glucose and releases oxygen.",
     "Photosynthesis consumes oxygen and produces carbon dioxide as a by-product."),

    ("docker",
     "Docker packages applications into containers that share the host operating system "
     "kernel rather than each carrying a full virtual machine.",
     "Docker packages applications into containers that share the host kernel.",
     "Docker packages applications into containers each carrying a full virtual machine.",
     "Docker is a virtual machine hypervisor that emulates dedicated hardware."),

    ("redis",
     "Redis is an in-memory key-value data store used as a database, cache, and message broker. "
     "It supports persistence through snapshots and append-only logs.",
     "Redis is an in-memory key-value store used as a database, cache and message broker.",
     "Redis is a relational database with full ACID transactions by default.",
     "Redis is a distributed filesystem designed for large-scale archival storage."),

    ("dna",
     "DNA stores genetic information in a double helix of two complementary strands, with "
     "bases adenine pairing with thymine and guanine pairing with cytosine.",
     "DNA is a double helix where adenine pairs with thymine and guanine with cytosine.",
     "DNA is a single strand where adenine pairs with cytosine.",
     "DNA was discovered by Mendel in 1865 as a double-stranded molecule."),

    ("transformers",
     "The Transformer architecture relies on self-attention rather than recurrence or "
     "convolution, which allows parallel training and long-range dependency modelling.",
     "The Transformer uses self-attention instead of recurrence, allowing parallel training.",
     "The Transformer uses recurrence with a much larger hidden state than earlier models.",
     "The Transformer removed the need for any attention mechanism by using convolutions."),

    ("sqlite",
     "SQLite is a serverless relational database engine that stores an entire database as a "
     "single file, requiring no separate server process.",
     "SQLite is a serverless database that stores everything in one file with no server.",
     "SQLite requires a running server process but stores data in a single file.",
     "SQLite is a NoSQL key-value store optimized for write throughput."),

    ("numpy",
     "NumPy provides efficient N-dimensional arrays through vectorised operations implemented "
     "in compiled C, avoiding interpreter overhead on large numeric workloads.",
     "NumPy provides N-dimensional arrays with vectorised C-implemented operations.",
     "NumPy is a pure-Python library where vectorised operations are interpreted at runtime.",
     "NumPy is a distributed computing framework for cluster-scale array processing."),

    ("curie",
     "Marie Curie discovered polonium and radium, and was the first person to win a Nobel "
     "Prize in two different sciences: Physics in 1903 and Chemistry in 1911.",
     "Marie Curie won the Nobel Prize in Physics in 1903 and Chemistry in 1911.",
     "Marie Curie won the Nobel Prize in Chemistry in 1903 and Physics in 1911.",
     "Marie Curie was the first person to win two Nobel Prizes in the same science."),

    ("firewall",
     "A network firewall monitors and controls traffic between network segments according "
     "to a rule set, filtering by address, port, protocol, or connection state.",
     "A firewall controls traffic between network segments using a rule set.",
     "A firewall encrypts all traffic passing between two network segments.",
     "A firewall compresses network traffic to reduce bandwidth usage."),
]

# --------------------------------------------------------------------------
# Alternative: medicine and pharmacology
# Recognisable stakes make the abstention behaviour easier to read.
# --------------------------------------------------------------------------

MED_FACTS: list[Fact] = [
    ("metformin",
     "Metformin is a biguanide that lowers blood glucose by decreasing hepatic glucose "
     "production and improving insulin sensitivity. It is not associated with hypoglycaemia "
     "when used alone.",
     "Metformin lowers blood glucose by reducing hepatic glucose production.",
     "Metformin lowers blood glucose by stimulating insulin release from the pancreas.",
     "Metformin is an insulin replacement used to treat type 1 diabetes."),

    ("aspirin",
     "Aspirin irreversibly inhibits cyclooxygenase, reducing thromboxane A2 production. This "
     "mechanism underlies both its analgesic and its antiplatelet effect.",
     "Aspirin irreversibly inhibits cyclooxygenase, reducing thromboxane A2.",
     "Aspirin is a reversible inhibitor of cyclooxygenase.",
     "Aspirin works by blocking opioid receptors in the central nervous system."),

    ("penicillin",
     "Penicillin inhibits bacterial cell wall synthesis by binding penicillin-binding "
     "proteins and blocking the transpeptidase cross-linking step.",
     "Penicillin blocks bacterial cell wall synthesis by inhibiting transpeptidase.",
     "Penicillin inhibits bacterial DNA replication by blocking gyrase.",
     "Penicillin dissolves the bacterial cell membrane causing immediate lysis."),

    ("warfarin",
     "Warfarin inhibits vitamin K epoxide reductase, reducing synthesis of vitamin "
     "K-dependent clotting factors II, VII, IX and X.",
     "Warfarin inhibits vitamin K epoxide reductase, reducing clotting factor synthesis.",
     "Warfarin directly inhibits clotting factor II activity.",
     "Warfarin increases fibrinogen production to reduce bleeding time."),

    ("insulin",
     "Insulin is a peptide hormone secreted by pancreatic beta cells. It lowers blood "
     "glucose by promoting cellular uptake and suppressing hepatic gluconeogenesis.",
     "Insulin is secreted by pancreatic beta cells and lowers blood glucose.",
     "Insulin is secreted by pancreatic alpha cells and lowers blood glucose.",
     "Insulin raises blood glucose by stimulating hepatic glycogenolysis."),

    ("amoxicillin",
     "Amoxicillin is an aminopenicillin antibiotic with activity against gram-positive "
     "cocci and some gram-negative organisms, and is eliminated renally.",
     "Amoxicillin is an aminopenicillin antibiotic eliminated renally.",
     "Amoxicillin is eliminated hepatically via extensive first-pass metabolism.",
     "Amoxicillin is an antiviral effective against influenza and herpes."),

    ("paracetamol",
     "Paracetamol's mechanism is not fully established, though it is thought to act "
     "centrally via cyclooxygenase inhibition and has no significant anti-inflammatory action.",
     "Paracetamol acts centrally and has no significant anti-inflammatory action.",
     "Paracetamol is a potent anti-inflammatory that works by inhibiting leukotriene synthesis.",
     "Paracetamol works by blocking histamine receptors to reduce allergic responses."),

    ("heparin",
     "Heparin potentiates antithrombin, accelerating inactivation of thrombin and factor Xa. "
     "Its effect is monitored by the activated partial thromboplastin time.",
     "Heparin potentiates antithrombin and is monitored with the aPTT.",
     "Heparin directly neutralises thrombin and is monitored by the bleeding time.",
     "Heparin is an oral anticoagulant that inhibits vitamin K epoxide reductase."),

    ("morphine",
     "Morphine is a Schedule II opioid mu-receptor agonist used for severe pain. Respiratory "
     "depression is the primary toxicity and the reason for careful titration.",
     "Morphine is a mu-receptor opioid agonist for severe pain; respiratory depression is the main risk.",
     "Morphine is a kappa-receptor opioid agonist used for mild to moderate pain.",
     "Morphine is a non-opioid analgesic that works through sodium channel blockade."),

    ("epinephrine",
     "Epinephrine is an alpha and beta adrenergic agonist. It raises heart rate and contractility "
     "via beta-1 receptors and causes vasoconstriction via alpha-1 receptors.",
     "Epinephrine raises heart rate via beta-1 receptors and constricts vessels via alpha-1.",
     "Epinephrine lowers heart rate via beta-1 receptors and dilates vessels via alpha-1.",
     "Epinephrine is a beta blocker that reduces heart rate and lowers blood pressure."),

    ("tamoxifen",
     "Tamoxifen is a selective estrogen receptor modulator that acts as an antagonist in "
     "breast tissue and an agonist in bone and endometrium.",
     "Tamoxifen is a selective estrogen receptor modulator, antagonising in breast tissue.",
     "Tamoxifen is a pure estrogen receptor agonist throughout the body.",
     "Tamoxifen inhibits aromatase and blocks all estrogen synthesis."),

    ("ciprofloxacin",
     "Ciprofloxacin is a fluoroquinolone that inhibits bacterial DNA gyrase and "
     "topoisomerase IV, and is associated with tendinitis and QT prolongation.",
     "Ciprofloxacin inhibits DNA gyrase and is associated with tendinitis.",
     "Ciprofloxacin inhibits cell wall synthesis and causes no significant adverse effects.",
     "Ciprofloxacin is a folate antagonist used in the treatment of malaria."),

    ("furosemide",
     "Furosemide is a loop diuretic inhibiting the sodium-potassium-chloride cotransporter in "
     "the thick ascending limb, increasing sodium, potassium, and water excretion.",
     "Furosemide is a loop diuretic inhibiting the NKCC cotransporter.",
     "Furosemide is a thiazide diuretic inhibiting sodium reabsorption in the distal tubule.",
     "Furosemide is an aldosterone antagonist that works by blocking potassium channels."),

    ("ondansetron",
     "Ondansetron is a 5-HT3 receptor antagonist used to prevent nausea and vomiting. A "
     "notable risk is QT prolongation.",
     "Ondansetron blocks 5-HT3 receptors to prevent vomiting and can prolong QT.",
     "Ondansetron is a dopamine D2 antagonist used as an antipsychotic.",
     "Ondansetron is a prokinetic that stimulates gastric motility directly."),

    ("digoxin",
     "Digoxin inhibits the sodium-potassium ATPase, raising intracellular sodium and reducing "
     "sodium-calcium exchange. Toxicity includes arrhythmias and visual disturbances.",
     "Digoxin inhibits the sodium-potassium ATPase and can cause arrhythmias.",
     "Digoxin activates the sodium-potassium ATPase, lowering intracellular sodium.",
     "Digoxin is a potassium-sparing diuretic that reduces urinary potassium loss."),

    ("hepatitis",
     "Hepatitis B is a DNA virus transmitted through blood and body fluids; chronic infection "
     "is associated with cirrhosis and hepatocellular carcinoma.",
     "Hepatitis B is a DNA virus linked to cirrhosis and liver cancer.",
     "Hepatitis B is an RNA virus transmitted primarily through contaminated water.",
     "Hepatitis B confers lifelong natural immunity upon recovery from infection."),
]


DATASETS = {
    "tech": TECH_FACTS,
    "med": MED_FACTS,
}


# Alternate phrasings of the same underlying fact. Asking a RAG system the same
# question three ways yields three different answers in practice, so repeating a
# fact across phrasings is what a real calibration set looks like — not padding.
PHRASINGS = (
    "{topic}",
    "explain {topic}",
    "what is known about {topic}",
)


def build_dataset(facts: list[Fact]):
    """Expand facts into labelled (query, context, response, label) rows.

    Each fact yields one faithful and two hallucinated samples (one drifted, one
    invented) per phrasing. That is a 2/3 hallucination rate, far above the 1-50%
    band guards.validate_label_distribution accepts, so callers downsample with
    balance(). The drifted samples are kept preferentially: they are the
    near-misses a threshold actually has to learn to reject.
    """
    rows = []
    for topic, context, faithful, drifted, invented in facts:
        readable = topic.replace("-", " ")
        for phrasing in PHRASINGS:
            query = phrasing.format(topic=readable)
            rows.append((query, context, faithful, 0))
            rows.append((query, context, drifted, 1))
            rows.append((query, context, invented, 1))
    return rows


def balance(rows, hallucination_rate: float = 0.40):
    """Downsample hallucinations to hit a realistic hallucination rate.

    Guards reject calibration above 50%, and the real-world rate for RAG
    hallucination is roughly 10-20%. Keeping the drifted (subtle) cases
    preferentially over the invented (blatant) ones preserves the score spread
    the threshold is supposed to learn.
    """
    faithful = [r for r in rows if r[3] == 0]
    wrong = [r for r in rows if r[3] == 1]
    target_wrong = int(round(len(faithful) * hallucination_rate / (1 - hallucination_rate)))
    target_wrong = min(target_wrong, len(wrong))
    # Interleave so the surviving hallucinations span both difficulty levels.
    stride = max(1, len(wrong) // target_wrong) if target_wrong else 1
    kept = wrong[::stride][:target_wrong]
    out = faithful + kept
    out.sort(key=lambda r: (r[3], r[0]))
    return out


def split_dataset(rows, held_out_every: int = 2):
    """Interleaved calibration/test split.

    Alternating rather than slicing by prefix, so both sets cover every topic.
    A prefix split would put whole topics on one side and let the threshold
    generalise across topics it never saw.
    """
    calibration, held_out = [], []
    for i, row in enumerate(rows):
        (held_out if i % held_out_every == 0 else calibration).append(row)
    return calibration, held_out


def make_corpus(facts: list[Fact]) -> list[str]:
    """One document per fact, for the retrieval step."""
    return [f[1] for f in facts]