"""Test isolation: never call a real provider from the suite.

A developer's .env may carry a GROQ key. `config.py` loads it at import, which
would switch the judge on for every `create_app()` in the tests: a network call
per uncertain sample, and a dependency on provider availability. The suite
patches scoring instead, so force the judge off before any project import.
`load_dotenv()` does not override variables that are already set.
"""

import os

os.environ["CONFORMAL_RAG_USE_LLM_JUDGE"] = "0"