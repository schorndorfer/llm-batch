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
openai_batch_app = App(help="OpenAI batching commands", version=__version__)


# ---------------------------------------------------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------------------------------------------------
@openai_batch_app.command()
def send(
    batch_file: Annotated[Path, Parameter(help="Batch file")] = None,  # type: ignore
    description: Annotated[
        str, Parameter("--desc", help="Description of the batch job")
    ] = "batch job from batch",  # type: ignore
):
    """
    Upload a batch file to OpenAI
    """
    client = openai.OpenAI()
    batch_input_file = client.files.create(file=open(batch_file, "rb"), purpose="batch")
    console.print(f"Uploaded batch file: {batch_file}")
    console.print(f"[orange1]{batch_input_file}")
    logger.info(f"Uploaded batch file: {batch_file}")
    logger.info(f"{batch_input_file}")
    batch_create_response = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": f"{description}: {batch_file.name}"},
    )
    console.print(f"Batch created: {batch_create_response.id}")
    logger.info(f"Batch created: {batch_create_response.id}")

# ---------------------------------------------------------------------------------------------------------------------
@openai_batch_app.command()
def fetch(
    batch_id: Annotated[str, Parameter(help="Batch ID")] = None,  # type: ignore
    out: Annotated[Path, Parameter("--out", help="Path to output file")] = Path("."),
    batch_name: Annotated[str, Parameter("--batch", help="Batch name")] = "batch",
):
    """
    Download batch results to a file if the batch job is completed, else job status is displayed.
    """
    client = openai.OpenAI()
    batch_retrieve_response = client.batches.retrieve(batch_id)
    logger.info(batch_retrieve_response)
    console.print(batch_retrieve_response)
    if batch_retrieve_response.status == "completed":
        file_response = client.files.content(batch_retrieve_response.output_file_id)  # type: ignore
        out.mkdir(parents=True, exist_ok=True)
        out_file = out / f"{batch_name}-responses.jsonl"
        out_file.write_text(file_response.text)
        logger.info(f"writing json output to {out_file}")
        console.print(f"[orange1]writing json output to {out_file}")


# ---------------------------------------------------------------------------------------------------------------------
@openai_batch_app.command()
def check(
    limit: Annotated[
        int, Parameter("--limit", help="Limit the number of batches to list")
    ] = 100,
):
    """
    Display all OpenAI batches for your account
    """
    client = openai.OpenAI()
    batches = client.batches.list(limit=limit)
    batches = sorted(batches, key=lambda x: x.created_at)
    for b in batches:
        console.print(b.id, b.status, datetime.fromtimestamp(b.created_at))
