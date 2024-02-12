from typing import Optional

import typer
from typing_extensions import Annotated

import inference_cli.lib
from inference_cli.benchmark import benchmark_app
from inference_cli.cloud import cloud_app
from inference_cli.server import server_app

app = typer.Typer()
app.add_typer(server_app, name="server")
app.add_typer(cloud_app, name="cloud")
app.add_typer(benchmark_app, name="benchmark")


@app.command()
def infer(
    input_reference: Annotated[
        str,
        typer.Option(
            "--input",
            "-i",
            help="URL or local path of image / directory with images or video to run inference on.",
        ),
    ],
    model_id: Annotated[
        str,
        typer.Option(
            "--model_id",
            "-m",
            help="Model ID in format project/version.",
        ),
    ],
    api_key: Annotated[
        Optional[str],
        typer.Option(
            "--api-key",
            "-a",
            help="Roboflow API key for your workspace. If not given - env variable `ROBOFLOW_API_KEY` will be used",
        ),
    ] = None,
    host: Annotated[
        str,
        typer.Option("--host", "-h", help="Host to run inference on."),
    ] = "http://localhost:9001",
    output_location: Annotated[
        Optional[str],
        typer.Option(
            "--output_location",
            "-o",
            help="Location where to save the result (path to directory)",
        ),
    ] = None,
    display: Annotated[
        bool,
        typer.Option(
            "--display/--no-display",
            "-D/-d",
            help="Boolean flag to decide if visualisations should be displayed on the screen",
        ),
    ] = False,
    visualise: Annotated[
        bool,
        typer.Option(
            "--visualise/--no-visualise",
            "-V/-v",
            help="Boolean flag to decide if visualisations should be preserved",
        ),
    ] = True,
    visualisation_config: Annotated[
        Optional[str],
        typer.Option(
            "--visualisation_config",
            "-c",
            help="Location of yaml file with visualisation config",
        ),
    ] = None,
    model_config: Annotated[
        Optional[str],
        typer.Option(
            "--model_config", "-mc", help="Location of yaml file with model config"
        ),
    ] = None,
):
    typer.echo(
        f"Running inference on {input_reference}, using model: {model_id}, and host: {host}"
    )
    inference_cli.lib.infer(
        input_reference=input_reference,
        model_id=model_id,
        api_key=api_key,
        host=host,
        output_location=output_location,
        display=display,
        visualise=visualise,
        visualisation_config=visualisation_config,
        model_configuration=model_config,
    )


if __name__ == "__main__":
    app()
