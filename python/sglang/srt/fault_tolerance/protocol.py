from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

DPRank = Annotated[int, Field(strict=True, ge=0)]
RequestId = Annotated[str, Field(strict=True)]


class RetryParams(BaseModel):
    pass


class ScaleDownParams(BaseModel):
    removed_dp_ranks: list[DPRank]


class RetryRequest(BaseModel):
    instruction: Literal["retry"]
    params: RetryParams = Field(default_factory=RetryParams)
    request_id: RequestId = ""


class ScaleDownRequest(BaseModel):
    instruction: Literal["scale_down"]
    params: ScaleDownParams
    request_id: RequestId = ""


FaultToleranceApplyRequest = Annotated[
    Union[RetryRequest, ScaleDownRequest],
    Field(discriminator="instruction"),
]

_APPLY_REQUEST_ADAPTER = TypeAdapter(FaultToleranceApplyRequest)
_VALIDATION_ERROR_MESSAGES = {
    "json_invalid": "Invalid JSON format",
    "dict_type": "Request body must be a JSON object.",
    "union_tag_not_found": "'instruction' is required.",
    "model_type": "'params' must be an object.",
    "missing": "'removed_dp_ranks' must be a list of integers.",
    "list_type": "'removed_dp_ranks' must be a list of integers.",
    "int_type": "'removed_dp_ranks' must be a list of integers.",
    "greater_than_equal": "'removed_dp_ranks' contains a rank out of range.",
    "string_type": "'request_id' must be a string.",
}


def parse_apply_request(body: bytes) -> FaultToleranceApplyRequest:
    try:
        return _APPLY_REQUEST_ADAPTER.validate_json(body)
    except ValidationError as exc:
        error = exc.errors()[0]
        if error["type"] == "union_tag_invalid":
            message = f"Invalid instruction: '{error['ctx']['tag']}'."
        else:
            message = _VALIDATION_ERROR_MESSAGES.get(error["type"], error["msg"])
        raise ValueError(message) from None
