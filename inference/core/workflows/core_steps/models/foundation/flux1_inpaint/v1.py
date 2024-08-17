from typing import List, Literal, Optional, Type, TypeVar, Union

import supervision as sv
from pydantic import ConfigDict, Field

from diffusers import StableDiffusionXLInpaintPipeline
import numpy as np
import torch

# TODO: Do we need to define requests? Probably not
from inference.core.entities.requests.sam2 import (
    Box,
    Sam2Prompt,
    Sam2PromptSet,
    Sam2SegmentationRequest,
)
# TODO: To remove
from inference.core.entities.responses.inference import (
    InferenceResponseImage,
    InstanceSegmentationInferenceResponse,
    InstanceSegmentationPrediction,
    Point,
)
# TODO: Do we need to define responses?
from inference.core.entities.responses.sam2 import Sam2SegmentationPrediction
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
            "name": "Flux 1 Inpainting Model",
            "version": "v1",
            "short_description": "Flux 1 Inpainting",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
        },
        protected_namespaces=(),
    )

    type: Literal["roboflow_core/flux1_inpaint@v1"]
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
    strength: Union[
        Optional[float], WorkflowParameterSelector(kind=[FLOAT_KIND])
    ] = Field(
        # TODO: Add min/max (<0; 1>)
        default=0.85,
        description="Indicates extent to transform the reference `image`. Must be between 0 and 1. `image` is used as a starting point and more noise is added the higher the `strength`.",
    )
    num_inference_steps: Union[
        Optional[int], WorkflowParameterSelector(kind=[INTEGER_KIND])
    ] = Field(
        # TODO: Add min/max (<1; 50>)
        default=20,
        description="The number of denoising steps. More denoising steps usually lead to a higher quality image",
    )
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
        num_inference_steps: int,
        strength: float,
        seed: int,
    ) -> BlockResult:
        # TODO: Can we handle it in a more user-friendly way?
        if boxes.mask is None:
            return {"image": image}

        # TODO: Do we need to copy it?
        copied_image = image.numpy_image.copy()
        common_mask = (np.sum(boxes.mask, axis=0) > 0).astype(int)

        # TODO: Add support also for other devices
        pipe = StableDiffusionXLInpaintPipeline.from_pretrained("stabilityai/sdxl-turbo", safety_checker=None).to("cpu")

        generator = torch.Generator().manual_seed(seed)

        result = pipe(
            prompt=prompt,
            image=copied_image,
            mask_image=common_mask,
            # TODO: Do we need to resize it first?
            # width=width,
            # height=height,
            # TODO: Add support for guidance_scale
            # guidance_scale=8.0,
            strength=strength,
            generator=generator,
            num_inference_steps=num_inference_steps
        ).images[0]

        result_image = WorkflowImageData(
            parent_metadata=image.parent_metadata,
            workflow_root_ancestor_metadata=image.workflow_root_ancestor_metadata,
            numpy_image=result,
        )

        return {"image": result_image}
