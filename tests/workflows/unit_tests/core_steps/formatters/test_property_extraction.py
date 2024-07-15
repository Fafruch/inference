import pytest

from inference.core.entities.responses.inference import (
    ClassificationInferenceResponse,
    InferenceResponseImage,
    ClassificationPrediction,
)
from inference.core.workflows.core_steps.common.query_language.entities.operations import (
    OperationsChain,
)
from inference.core.workflows.core_steps.formatters.property_definition import (
    PropertyDefinitionBlock,
)


@pytest.mark.asyncio
async def test_property_extraction_block() -> None:
    # given
    data = ClassificationInferenceResponse(
        image=InferenceResponseImage(width=128, height=256),
        predictions=[
            ClassificationPrediction(
                **{"class": "cat", "class_id": 0, "confidence": 0.6}
            ),
            ClassificationPrediction(
                **{"class": "dog", "class_id": 1, "confidence": 0.4}
            ),
        ],
        top="cat",
        confidence=0.6,
        parent_id="some",
    ).model_dump(by_alias=True, exclude_none=True)
    operations = OperationsChain.model_validate(
        {
            "operations": [
                {
                    "type": "ClassificationPropertyExtract",
                    "property_name": "top_class",
                },
                {
                    "type": "LookupTable",
                    "lookup_table": {"cat": "cat-mutated"},
                },
            ]
        }
    ).operations
    step = PropertyDefinitionBlock()

    # when
    result = await step.run(data=data, operations=operations)

    # then
    assert result == {"output": "cat-mutated"}
