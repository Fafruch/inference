from typing import List, Literal, Optional, Type, TypeVar, Union

import supervision as sv

import numpy as np
import torch
from pydantic import ConfigDict, Field
from diffusers import AutoPipelineForInpainting, FluxInpaintPipeline, StableDiffusionXLInpaintPipeline
from PIL import Image

from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
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

LONG_DESCRIPTION = """
Intelligently modify images by filling in or replacing specified areas with new content based on the content of a "mask" image.

** Dedicated inference server required (GPU recomended) **

The "mask" is provided by explicitly passing in a separate image via the mask parameter (for example passed in from the Segment Anything 2 Model).
You can select from multiple available inpainting models.
"""

MAX_SEED = np.iinfo(np.int32).max

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Inpainting Model",
            "version": "v1",
            "short_description": "Inpainting",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
        },
        protected_namespaces=(),
    )

    type: Literal["roboflow_core/inpainting@v1"]
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
        description="What you wish to see in the output image. A strong, descriptive prompt that clearly defines elements, colors, and subjects will lead to better results.",
    )
    model_selection: Union[
        Literal["black-forest-labs/FLUX.1-schnell", "diffusers/sdxl-1.0-inpainting-0.1", "stabilityai/sdxl-turbo"],
        WorkflowParameterSelector(kind=[STRING_KIND])
    ] = Field(
        default="black-forest-labs/FLUX.1-schnell",
        description="What you wish to see in the output image. A strong, descriptive prompt that clearly defines elements, colors, and subjects will lead to better results.",
    )
    negative_prompt: Union[Optional[WorkflowParameterSelector(kind=[STRING_KIND])], str] = Field(
        description="A blurb of text describing what you do not wish to see in the output image. This is an advanced feature.",
        default=None,
    )
    strength: Union[
        Optional[float], WorkflowParameterSelector(kind=[FLOAT_KIND])
    ] = Field(
        default=0.85,
        description="Indicates extent to transform the reference `image`. Must be between 0 and 1. `image` is used as a starting point and more noise is added the higher the `strength`.",
    )
    num_inference_steps: Union[
        Optional[int], WorkflowParameterSelector(kind=[INTEGER_KIND])
    ] = Field(
        default=20,
        description="The number of denoising steps. More denoising steps usually lead to a higher quality image",
    )
    seed: Union[
        Optional[int], WorkflowParameterSelector(kind=[INTEGER_KIND])
    ] = Field(
        default=42,
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


class InpaintingBlockV1(WorkflowBlock):

    def __init__(
        self,
        model_manager: ModelManager,
        api_key: Optional[str],
        step_execution_mode: StepExecutionMode,
    ):
        self._model_manager = model_manager
        self._api_key = api_key
        self._step_execution_mode = step_execution_mode


    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["model_manager", "api_key", "step_execution_mode"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        image: WorkflowImageData,
        boxes: sv.Detections,
        prompt: str,
        model_selection: Literal["black-forest-labs/FLUX.1-schnell", "diffusers/sdxl-1.0-inpainting-0.1", "stabilityai/sdxl-turbo"],
        negative_prompt: Optional[str],
        num_inference_steps: Optional[int],
        strength: Optional[float],
        seed: Optional[int],
    ) -> BlockResult:
        if self._step_execution_mode is StepExecutionMode.LOCAL:
            return self.run_locally(
                image=image,
                boxes=boxes,
                prompt=prompt,
                model_selection=model_selection,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                strength=strength,
                seed=seed,
            )
        elif self._step_execution_mode is StepExecutionMode.REMOTE:
            raise NotImplementedError(
                "Remote execution is not supported for Inpainting. Run a local or dedicated inference server to use this block (GPU recommended)."
            )
        else:
            raise ValueError(
                f"Unknown step execution mode: {self._step_execution_mode}"
            )

    def run_locally(
        self,
        image: WorkflowImageData,
        boxes: sv.Detections,
        prompt: str,
        model_selection: Literal["black-forest-labs/FLUX.1-schnell", "diffusers/sdxl-1.0-inpainting-0.1", "stabilityai/sdxl-turbo"],
        negative_prompt: Optional[str],
        num_inference_steps: Optional[int],
        strength: Optional[float],
        seed: Optional[int],
    ):
        if boxes.mask is None:
            return {"image": image}

        image_copy = image.numpy_image.copy()
        mask_image = (np.sum(boxes.mask, axis=0) > 0).astype(np.uint8)
        width, height = Image.fromarray(image_copy).size
        generator = torch.Generator().manual_seed(seed)
        pipe = self._get_pipeline_based_model_selection(model_selection)

        result = pipe(
            prompt=prompt,
            image=image_copy,
            mask_image=mask_image,
            width=width,
            height=height,
            strength=strength,
            generator=generator,
            num_inference_steps=num_inference_steps,
            negative_prompt=negative_prompt
        ).images[0]

        result_image = WorkflowImageData(
            parent_metadata=image.parent_metadata,
            workflow_root_ancestor_metadata=image.workflow_root_ancestor_metadata,
            numpy_image=np.array(result),
        )

        return {"image": result_image}

    def _get_pipeline_based_model_selection(self, model_selection: Literal[
        "black-forest-labs/FLUX.1-schnell", "diffusers/sdxl-1.0-inpainting-0.1", "stabilityai/sdxl-turbo"]):
        default_torch_dtype = torch.bfloat16
        default_model_pipeline = FluxInpaintPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=default_torch_dtype
        ).to(DEVICE)

        return {
            "black-forest-labs/FLUX.1-schnell": default_model_pipeline,
            "diffusers/sdxl-1.0-inpainting-0.1": AutoPipelineForInpainting.from_pretrained(
                "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                torch_dtype=default_torch_dtype,
                variant="fp16")
            .to(DEVICE),
            "stabilityai/sdxl-turbo": StableDiffusionXLInpaintPipeline.from_pretrained(
                "stabilityai/sdxl-turbo",
                torch_dtype=default_torch_dtype,
                safety_checker=None)
            .to(DEVICE),
        }[model_selection]
