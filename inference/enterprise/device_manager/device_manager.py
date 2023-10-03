from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI

from inference.core.env import METRICS_INTERVAL
from inference.core.version import __version__
from inference.enterprise.device_manager.command_handler import (
    Command,
    fetch_commands,
    handle_command,
)
from inference.enterprise.device_manager.metrics_service import report_metrics

app = FastAPI(
    title="Roboflow Device Manager",
    description="The device manager enables remote control and monitoring of Roboflow inference server containers",
    version=__version__,
    terms_of_service="https://roboflow.com/terms",
    contact={
        "name": "Roboflow Inc.",
        "url": "https://roboflow.com/contact",
        "email": "help@roboflow.com",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
    root_path="/",
)


@app.get("/")
def root():
    return {
        "name": "Roboflow Device Manager",
        "version": __version__,
        "terms_of_service": "https://roboflow.com/terms",
        "contact": {
            "name": "Roboflow Inc.",
            "url": "https://roboflow.com/contact",
            "email": "help@roboflow.com",
        },
        "license_info": {
            "name": "Apache 2.0",
            "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
        },
    }


@app.post("/exec_command")
async def exec_command(command: Command):
    info = handle_command(command)
    return {"status": "ok", "data": info}


scheduler = BackgroundScheduler()
scheduler.add_job(fetch_commands, "interval", seconds=4)
scheduler.add_job(report_metrics, "interval", seconds=METRICS_INTERVAL)
scheduler.start()
