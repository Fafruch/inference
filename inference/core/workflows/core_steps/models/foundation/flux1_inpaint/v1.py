from typing import List, Literal, Optional, Type, TypeVar, Union

import supervision as sv
from pydantic import ConfigDict, Field

import io
import requests
import numpy as np
import torch
from PIL import Image

from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    BATCH_OF_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    IMAGE_KIND,
    FLOAT_KIND,
    INTEGER_KIND,
    STRING_KIND,
    StepOutputImageSelector,
    StepOutputSelector,
    WorkflowImageSelector,
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

T = TypeVar("T")
K = TypeVar("K")

DETECTIONS_CLASS_NAME_FIELD = "class_name"

# TODO
LONG_DESCRIPTION = """
"""

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_SEED = np.iinfo(np.int32).max


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Stability AI Inpainting Model",
            "version": "v1",
            "short_description": "Stability AI Inpainting",
            "long_description": LONG_DESCRIPTION,
            # TODO: Is it a correct license?
            "license": "Apache-2.0",
            "block_type": "model",
        },
        protected_namespaces=(),
    )

    type: Literal["roboflow_core/stability_ai_inpaint@v1"]
    image: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
        title="Image",
        description="The image to infer on",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )
    boxes: Optional[
        StepOutputSelector(
            kind=[
                BATCH_OF_INSTANCE_SEGMENTATION_PREDICTION_KIND,
            ]
        )
    ] = Field(  # type: ignore
        description="Boxes (from other model predictions) to ground SAM2",
        examples=["$steps.object_detection_model.predictions"],
        default=None,
    )
    prompt: Union[WorkflowParameterSelector(kind=[STRING_KIND]), str] = Field(
        description="",  # TODO
    )
    stability_ai_api_key: Union[WorkflowParameterSelector(kind=[STRING_KIND]), str] = Field(
        description="",  # TODO
    )
    # strength: Union[
    #     Optional[float], WorkflowParameterSelector(kind=[FLOAT_KIND])
    # ] = Field(
    #     # TODO: Add min/max (<0; 1>)
    #     default=0.85,
    #     description="Indicates extent to transform the reference `image`. Must be between 0 and 1. `image` is used as a starting point and more noise is added the higher the `strength`.",
    # )
    # num_inference_steps: Union[
    #     Optional[int], WorkflowParameterSelector(kind=[INTEGER_KIND])
    # ] = Field(
    #     # TODO: Add min/max (<1; 50>)
    #     default=20,
    #     description="The number of denoising steps. More denoising steps usually lead to a higher quality image",
    # )
    seed: Union[
        Optional[int], WorkflowParameterSelector(kind=[INTEGER_KIND])
    ] = Field(
        # TODO: Add min/max (<1; MAX_SEED>)
        default=111,
        description="Randomize seed",
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="image",
                kind=[IMAGE_KIND],
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


class Flux1InpaintingBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        image: WorkflowImageData,
        boxes: sv.Detections,
        prompt: str,
        stability_ai_api_key: str,
        seed: int,
    ) -> BlockResult:
        # TODO: Can we handle it in a more user-friendly way?
        if boxes.mask is None:
            return {"image": image}

        # TODO: Do we need to copy it?
        copied_image = Image.fromarray(image.numpy_image.copy())
        copied_image.save('image.jpg')
        image_stream = io.BytesIO()
        copied_image.save(image_stream, format='JPEG')  # Save as JPEG to the stream
        image_stream.seek(0)  # Reset the stream position to the beginning

        common_mask = ((np.sum(boxes.mask, axis=0) > 0).astype(int) * 255).astype(np.uint8)
        mask_image = Image.fromarray(common_mask)
        mask_image.save('mask.jpg')
        mask_image_stream = io.BytesIO()
        mask_image.save(mask_image_stream, format='JPEG')  # Save as JPEG to the stream
        mask_image_stream.seek(0)  # Reset the stream position to the beginning

        response = requests.post(
            f"https://api.stability.ai/v2beta/stable-image/edit/inpaint",
            headers={
                "authorization": f"Bearer {stability_ai_api_key}",
                "accept": "image/*"
            },
            files={
                # "image": open("./dog-wearing-vr-goggles.png", "rb"),
                # "mask": open("./mask.png", "rb"),
                'image': ('image.jpg', image_stream, 'image/jpeg'),
                'mask': ('mask.jpg', mask_image_stream, 'image/jpeg')
            },
            data={
                "prompt": prompt,
                "output_format": "jpeg",
            },
        )

        if response.status_code == 200:
            response_image = Image.open(io.BytesIO(response.content))

            result_image = WorkflowImageData(
                parent_metadata=image.parent_metadata,
                workflow_root_ancestor_metadata=image.workflow_root_ancestor_metadata,
                numpy_image=np.array(response_image),
            )

            return {"image": result_image}
        else:
            # TODO: Different exception
            raise Exception(str(response.json()))
