import json
import os
import openai
from datetime import datetime
from pathlib import Path
from typing_extensions import Annotated
from cyclopts import App, Parameter
from anthropic import Anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from llm_batch import (
    __version__,
    console,
    logger,
)

# ---------------------------------------------------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------------------------------------------------
anthropic_batch_app = App(help="Anthropic batching commands", version=__version__)


# ---------------------------------------------------------------------------------------------------------------------
@anthropic_batch_app.command()
def send(
    batch_file: Annotated[Path, Parameter(help="Batch file")] = None,  # type: ignore
):
    """
    Upload a batch file to Anthropic and start processing it.
    """
    request_datas = []
    with open(batch_file, "r") as f:    
        for line in f:
            request_datas.append(json.loads(line))

    requests=[]
    for idx, request_data in enumerate(request_datas):
        body = request_data['body']
        params=MessageCreateParamsNonStreaming(
            model=body['model'],
            max_tokens=body['max_tokens'],
            messages=body['messages'],
        )
        requests.append(
            Request(
                custom_id=f"id-{idx}", 
                params=params
            )
        )
    client = Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY")
    )
    message_batch = client.messages.batches.create(requests=requests)
    console.print(f"[green]Batch {message_batch.id} created successfully.[/green]")

# ---------------------------------------------------------------------------------------------------------------------
@anthropic_batch_app.command()
def check(
    limit: Annotated[
        int, Parameter("--limit", help="Limit the number of batches to list")
    ] = 100,
):
    """
    Display all Anthropic batches for your account
    """
    client = Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY")
    )
    batches = client.messages.batches.list(limit=limit)
    batches = sorted(batches, key=lambda x: x.created_at)
    for b in batches:
        console.print(b.id, b.processing_status, b.created_at)


# ---------------------------------------------------------------------------------------------------------------------
@anthropic_batch_app.command()
def fetch(
    batch_id: Annotated[str, Parameter(help="Batch ID")] = None,  # type: ignore
    out: Annotated[Path, Parameter("--out", help="Path to output file")] = Path("."),
    batch_name: Annotated[str, Parameter("--batch", help="Batch name")] = "batch",
):
    """
    Download batch results to a file if the batch job is completed, else job status is displayed.
    """
    if not out.exists():
        out.mkdir(parents=True)
    out_file = out / f"{batch_name}-responses.jsonl"
    out_file.write_text("")

    client = Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY")
    )
    results = client.messages.batches.results(
        message_batch_id=batch_id,
    )
    results_json = [result.to_json() for result in results]
    out_file.write_text(",\n".join(results_json))
    
    console.print(f"[orange1]writing json output to {out_file}")
    logger.info(f"writing json output to {out_file}")
    