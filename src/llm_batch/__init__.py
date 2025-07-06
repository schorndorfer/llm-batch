import os
import logging
import logging.config
import yaml
from importlib import resources
from dotenv import load_dotenv
from rich.console import Console
from cyclopts import App
from llm_batch import data  # type: ignore

__author__ = """Kellogg Research Support"""
__email__ = "rs@kellogg.northwestern.edu"
__app_name__ = "openaihelper"
__version__ = "0.1.0"


load_dotenv(dotenv_path=os.getcwd())


help_msg = """
Commands to execute LLM batch jobs.
"""
app = App(help=help_msg, version=__version__)


console = Console(style="white on black")


CONFIG = {}
with resources.path(data, "config.yml") as path:
    CONFIG = yaml.load(open(path), Loader=yaml.FullLoader)

# setup logging
logging.config.dictConfig(CONFIG["logging"])
logger = logging.getLogger(__name__)
