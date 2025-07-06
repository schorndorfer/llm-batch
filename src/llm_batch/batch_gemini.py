import json
import openai
from datetime import datetime
from pathlib import Path
from typing_extensions import Annotated
from cyclopts import App, Parameter
from llm_batch import (
    __version__,
    console,
    logger,
)

# ---------------------------------------------------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------------------------------------------------
gemini_batch_app = App(help="Gemini batching commands", version=__version__)
