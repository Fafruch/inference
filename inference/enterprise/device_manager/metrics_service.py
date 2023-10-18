import time

from inference.core.devices.utils import GLOBAL_DEVICE_ID
from inference.core.env import API_KEY, METRICS_INTERVAL, TAGS
from inference.core.logger import logger
from inference.core.managers.metrics import get_model_metrics, get_system_info
from inference.core.version import __version__
from inference.enterprise.device_manager.container_service import (
    get_container_by_id,
    get_container_ids,
    get_latest_inferences,
)
from inference.enterprise.device_manager.helpers import (
    get_cache_model_items,
    get_device_id,
)

from inference.enterprise.device_manager import pubsub


def aggregate_model_stats(container_id):
    """
    Aggregate statistics for models within a specified container.

    This function retrieves and aggregates performance metrics for all models
    associated with the given container within a specified time interval.

    Args:
        container_id (str): The unique identifier of the container for which
            model statistics are to be aggregated.

    Returns:
        list: A list of dictionaries, where each dictionary represents a model's
        statistics with the following keys:
        - "dataset_id" (str): The ID of the dataset associated with the model.
        - "version" (str): The version of the model.
        - "api_key" (str): The API key that was used to make an inference against this model
        - "metrics" (dict): A dictionary containing performance metrics for the model:
            - "num_inferences" (int): Number of inferences made
            - "num_errors" (int): Number of errors
            - "avg_inference_time" (float): Average inference time in seconds

    Notes:
        - The function calculates statistics over a time interval defined by
          the global constant METRICS_INTERVAL, passed in when starting up the container.
    """
    now = time.time()
    start = now - METRICS_INTERVAL
    models = []
    api_keys = get_cache_model_items().get(container_id, dict()).keys()
    for api_key in api_keys:
        model_ids = get_cache_model_items().get(container_id, dict()).get(api_key, [])
        for model_id in model_ids:
            model = {
                "dataset_id": model_id.split("/")[0],
                "version": model_id.split("/")[1],
                "api_key": api_key,
                "metrics": get_model_metrics(
                    container_id, model_id, min=start, max=now
                ),
            }
            models.append(model)
    return models


def build_container_stats():
    """
    Build statistics for containers and their associated models.

    Returns:
        list: A list of dictionaries, where each dictionary represents statistics
        for a container and its associated models with the following keys:
        - "uuid" (str): The unique identifier (UUID) of the container.
        - "startup_time" (float): The timestamp representing the container's startup time.
        - "models" (list): A list of dictionaries representing statistics for each
          model associated with the container (see `aggregate_model_stats` for format).

    Notes:
        - This method relies on a singleton `container_service` for container information.
    """
    containers = []
    for id in get_container_ids():
        container = get_container_by_id(id)
        if container:
            container_stats = {}
            models = aggregate_model_stats(id)
            container_stats["uuid"] = container.id
            container_stats["version"] = container.version
            container_stats["startup_time"] = container.startup_time
            container_stats["models"] = models
            if container.status == "running":
                container_stats["status"] = "running"
            elif container.status == "exited":
                container_stats["status"] = "stopped"
            elif container.status == "paused":
                container_stats["status"] = "idle"
            elif container.status == "restarting" or container.status == "stopping":
                container_stats["status"] = "processing"
            else:
                container_stats["status"] = "unknown"
            containers.append(container_stats)
    return containers


def aggregate_device_stats():
    """
    Aggregate statistics for the device.
    """
    window_start_timestamp = str(int(time.time()))
    all_data = {
        "api_key": API_KEY,
        "timestamp": window_start_timestamp,
        "device": {
            "id": get_device_id(),
            "name": GLOBAL_DEVICE_ID,
            "type": f"roboflow-inference-server=={__version__}",
            "tags": TAGS,
            "system_info": get_system_info(),
            "containers": build_container_stats(),
        },
    }
    return all_data


def send_metrics():
    """
    Report metrics to Roboflow.

    This function aggregates statistics for the device and its containers and
    sends them to Roboflow.
    """
    all_data = aggregate_device_stats()
    logger.info(str(all_data))
    pubsub.dispatch(pubsub.METRICS_TOPIC, all_data)


def send_latest_inferences():
    inferences = get_latest_inferences()
    pubsub.dispatch(pubsub.STREAM_TOPIC, inferences)
