import tensorflow as tf
import uvicorn
from fastapi.staticfiles import StaticFiles
from .app import app
import pathlib

def run_APanalysis_webserver(
        model,
        dataset = None,
        dataset_info = None,
        host: str = "localhost",
        port: int = 8000,
        log_level: str = "info"
    ):
    # Setting the model and dataset
    app.model = model
    app.dataset = dataset
    app.dataset_info = dataset_info

    # Starting the server
    app.mount("/", StaticFiles(directory=pathlib.Path(__file__).parents[0].joinpath('static').resolve(), html=True), name="react_build")
    uvicorn.run(app, host=host, port=port, log_level=log_level)
