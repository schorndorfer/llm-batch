import os
import json
import litellm
import jinja2
import yaml
import fitz
import logging
import logging.config
from pathlib import Path
from itertools import product
from datetime import datetime
from typing import Dict, List
from typing_extensions import Annotated
from cyclopts import App, Parameter
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from llm_batch import __version__, CONFIG, console, logger, app
from llm_batch.batch_openai import openai_batch_app
from llm_batch.batch_anthropic import anthropic_batch_app
from llm_batch.batch_gemini import gemini_batch_app


# ---------------------------------------------------------------------------------------------------------------------
# App setup
# --------------------------------------------------------------------------------------------------------------------

batch_app = App(help="Batching commands", version=__version__)
app.command(batch_app, name="batch")
batch_app.command(openai_batch_app, name="openai")
batch_app.command(anthropic_batch_app, name="anthropic")
batch_app.command(gemini_batch_app, name="gemini")

utils_app = App(help="Utility commands", version=__version__)
app.command(utils_app, name="utils")


# ---------------------------------------------------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------------------------------------------------
def extract_combinations(dict_of_lists: Dict[str, List]) -> List[Dict[str, str]]:
    """
    Given a dict of lists, return all possible combinations (cartesian product).
    """
    keys = list(dict_of_lists.keys())
    value_lists = list(product(*dict_of_lists.values()))
    combinations = []
    for values in value_lists:
        comb = dict(zip(keys, values))
        combinations.append(comb)
    return combinations


# ---------------------------------------------------------------------------------------------------------------------
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
def completion_with_backoff(chat_params, console) -> Dict:
    response = litellm.completion(**chat_params)
    return response.json()  # type: ignore



# ---------------------------------------------------------------------------------------------------------------------
# Commands: batch
# ---------------------------------------------------------------------------------------------------------------------
@batch_app.command()
def make(
    in_dir: Annotated[Path, Parameter(help="Path to input files")] = Path("."),
    out: Annotated[Path, Parameter(help="Path to output file")] = Path("."),
    batch_name: Annotated[str, Parameter("--batch", help="Batch name")] = "batch",
) -> None:
    """
    Make a batch requests file.
    """
    json_files = [f for f in in_dir.glob("*.json")]
    if not json_files:
        console.print("[red]No JSON files found in the input directory.[/red]")
        return

    # Create the output file
    if not out.exists():
        out.mkdir(parents=True)
    out_file = out / f"{batch_name}-requests.jsonl"
    out_file.write_text("")

    # Loop through the JSON request files
    requests = []
    for f in json_files:
        try:
            console.print(f"[green]Found JSON file:[/green] {f}")
            request_body = json.loads(f.read_text())
            if "request" in request_body:
                request_body = request_body["request"]
            request = {
                "custom_id": f"id_{f.name}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": request_body,
            }
            requests.append(request)
        except json.JSONDecodeError as e:
            console.print(f"[red]Error decoding JSON in file {f}: {e}[/red]")
            continue
    out_file.write_text("\n".join([json.dumps(r) for r in requests]))
    console.print(f"Batch file created: {out_file}")

# ---------------------------------------------------------------------------------------------------------------------
# Commands: utils
# ---------------------------------------------------------------------------------------------------------------------
@utils_app.command()
def config() -> None:
    "Display configuration parameters"
    console.print(CONFIG)
    console.print(
        f"OPENAI_API_KEY: {os.environ.get('OPENAI_API_KEY', 'Not Set')[:12]}***"
    )
    console.print(
        f"ANTHROPIC_API_KEY: {os.environ.get('ANTHROPIC_API_KEY', 'Not Set')[:12]}***"
    )
    console.print(
        f"GEMINI_API_KEY: {os.environ.get('GEMINI_API_KEY', 'Not Set')[:5]}***"
    )


# ---------------------------------------------------------------------------------------------------------------------
@utils_app.command()
def pdf2text(
    in_dir: Annotated[Path, Parameter(help="Path to input PDF files")] = None,  # type: ignore
    out: Annotated[Path, Parameter(help="Path to output text files")] = Path("."),
    start: Annotated[int, Parameter(help="Start page")] = 0,
    end: Annotated[int, Parameter(help="End page")] = 10_000_000,
):
    """
    Extract text from a collection of PDF files and write each output to a text file.
    """
    assert end >= start
    if not out.exists():
        out.mkdir(parents=True)

    for pdf in in_dir.glob("*.pdf"):
        console.print(f"processing pdf file: {pdf.name}")
        logging.info(f"extracting text from: {pdf.name}")
        try:
            doc = fitz.open(pdf)
            textfile = out / f"{pdf.stem}.txt"
            pages = [page for page in doc if start <= page.number <= end]  # type: ignore
            textfile.write_text(
                chr(12).join([page.get_text(sort=True) for page in pages])  # type: ignore
            )
        except Exception as e:
            logger.error(f"exception: {type(e)}: {e}")
            continue


# ---------------------------------------------------------------------------------------------------------------------
# Commands: template
# ---------------------------------------------------------------------------------------------------------------------
@app.command()
def template(
    template: Annotated[Path, Parameter(help="Prompt template")],
    data: Annotated[Path, Parameter(help="Template data")],
    out: Annotated[Path, Parameter(help="Output directory for the responses")] = Path(
        "."
    ),
    execute: Annotated[
        bool, Parameter(help="Run the template and make synchronous API calls")
    ] = False,
) -> None:
    """
    Generate prompts from a template and data file, and optionally make API calls.
    The template should be a Jinja2 template, and the data file should be a YAML file
    containing the parameters for the template.
    """
    # validate input parameters
    assert template.is_file(), f"Template file {template} does not exist"
    assert data.is_file(), f"Data file {data} does not exist"
    if out.is_file():
        raise ValueError(f"Output path {out} is a file, expected a directory")
    if not out.exists():
        os.makedirs(out)
    if execute:
        console.print(
            "[bold yellow]Running in execute mode, API calls will be made[/bold yellow]"
        )
    else:
        console.print(
            "[bold yellow]Running in dry-run mode, no API calls will be made[/bold yellow]"
        )

    # set up template environment
    environment = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(template.parent)),
        undefined=jinja2.StrictUndefined,
    )
    t = environment.get_template(template.name)

    # load the template parameters
    yaml_data = yaml.safe_load(open(data, "r"))

    # extract combinations and render the template for each combination
    for idx, combination in enumerate(extract_combinations(yaml_data)):

        # render the template with the current combination
        chat_params = json.loads(t.render(**combination), strict=False)

        # create the output file
        model_name = chat_params.get("model", "unknown_model").replace("/", "_")
        model_dir = out / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        out_file = model_dir / f"{datetime.now().timestamp()}.json"
        out_file.write_text("")
        with open(out_file, "a") as f:
            f.write("\n{\n")
            f.write('"template_params": ')
            json.dump(combination, f, indent=2)
            f.write(",\n")
            f.write('"request": ')
            json.dump(chat_params, f, indent=2)

        if not execute:
            response = None
        else:
            try:
                response = completion_with_backoff(chat_params, console=console)
            except Exception as e:
                console.print(
                    f"[bold red]Error processing combination {idx+1:04d}: {e}[/bold red]"
                )
                logger.error(f"Error processing combination {idx+1:04d}: {e}")
                continue

            # write the combination, params, and respoonse to the output file
            with open(out_file, "a") as f:
                f.write(",\n")
                f.write('"response": ')
                json.dump(response, f, indent=2)

        with open(out_file, "a") as f:
            f.write("\n}\n")

        message = f"Executed combination {idx+1:05d} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{'-'*60}"
        console.print(f"[bold green]{message}[/bold green]")
        logger.info(message)
