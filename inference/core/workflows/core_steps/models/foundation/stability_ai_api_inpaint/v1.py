from typing import List, Literal, Optional, Type, TypeVar, Union

import supervision as sv
from pydantic import ConfigDict, Field

import io
import requests
import numpy as np
from PIL import Image

from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    BATCH_OF_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    IMAGE_KIND,
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
STABILITY_AI_IMAGE_MAX_PIXELS = 9437184

LONG_DESCRIPTION = """
Intelligently modify images by filling in or replacing specified areas with new content based on the content of a "mask" image.

The "mask" is provided in one of two ways:

1. Explicitly passing in a separate image via the mask parameter (for example passed in from the Segment Anything 2 Model).
2. Derived from the alpha channel of the image parameter.
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Stability AI API Inpainting Model",
            "version": "v1",
            "short_description": "Stability AI Inpainting",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
        },
        protected_namespaces=(),
    )

    type: Literal["roboflow_core/stability_ai_api_inpaint@v1"]
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
    stability_ai_api_key: Union[WorkflowParameterSelector(kind=[STRING_KIND]), str] = Field(
        description="Use your Stability API key to authentication requests",
    )
    negative_prompt: Union[Optional[WorkflowParameterSelector(kind=[STRING_KIND])], str] = Field(
        description="A blurb of text describing what you do not wish to see in the output image. This is an advanced feature.",
        default=None,
    )
    grow_mask: Union[
        Optional[int], WorkflowParameterSelector(kind=[INTEGER_KIND])
    ] = Field(
        default=5,
        description="Grows the edges of the mask outward in all directions by the specified number of pixels. The expanded area around the mask will be blurred, which can help smooth the transition between inpainted content and the original image."
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


class StabilityAIAPIInpaintingModelBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        image: WorkflowImageData,
        boxes: sv.Detections,
        prompt: str,
        stability_ai_api_key: str,
        negative_prompt: str,
        grow_mask: int,
        seed: int,
    ) -> BlockResult:
        image_stream = self._get_image_stream(image)
        mask_stream = self._get_mask_stream(boxes)

        response = self._get_response_from_stability_ai_api(
            stability_ai_api_key,
            prompt,
            image_stream,
            mask_stream,
            negative_prompt,
            grow_mask,
            seed
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
            raise ValueError(f"Stability AI API error: {str(response.json())}")

    def _get_image_stream(self, image: WorkflowImageData) -> io.BytesIO:
        image_copy = Image.fromarray(image.numpy_image.copy())
        resized_image = self._resize_image(image_copy)
        image_stream = self._get_stream_from_image(resized_image)

        return image_stream

    def _get_mask_stream(self, boxes: sv.Detections) -> Union[io.BytesIO, None]:
        if boxes.mask is None:
            return None

        mask_image = self._map_boxes_mask_to_image(boxes)
        resized_mask_image = self._resize_image(mask_image)
        mask_stream = self._get_stream_from_image(resized_mask_image)

        return mask_stream

    def _map_boxes_mask_to_image(self, boxes: sv.Detections) -> Image:
        return Image.fromarray(((np.sum(boxes.mask, axis=0) > 0).astype(int) * 255).astype(np.uint8))

    def _get_stream_from_image(self, image: Image) -> io.BytesIO:
        image_stream = io.BytesIO()
        image.save(image_stream, format='JPEG')
        image_stream.seek(0)

        return image_stream

    def _resize_image(self, image: Image) -> Image:
        image_copy = image.copy()
        original_width, original_height = image_copy.size

        current_pixels = original_width * original_height
        scaling_factor = (STABILITY_AI_IMAGE_MAX_PIXELS / current_pixels) ** 0.5

        new_width = int(original_width * scaling_factor) - 1
        new_height = int(original_height * scaling_factor) - 1

        if scaling_factor < 1:
            return image_copy.resize((new_width, new_height), Image.LANCZOS)

        return image_copy

    def _get_response_from_stability_ai_api(self, stability_ai_api_key: str, prompt: str, image_stream: io.BytesIO, mask_stream: io.BytesIO, negative_prompt: str, grow_mask: int, seed: int) -> requests.Response:
        headers = {
            "authorization": f"Bearer {stability_ai_api_key}",
            "accept": "image/*",
        }

        files = {
            "image": ("image.jpg", image_stream, "image/jpeg"),
        }
        if mask_stream is not None:
            files["mask"] = ("mask.jpg", mask_stream, "image/jpeg")

        data = {
            "prompt": prompt,
            "output_format": "jpeg",
            "negative_prompt": negative_prompt,
            "grow_mask": grow_mask,
            "seed": seed,
        }

        return requests.post(
            f"https://api.stability.ai/v2beta/stable-image/edit/inpaint",
            headers=headers,
            files=files,
            data=data,
        )
