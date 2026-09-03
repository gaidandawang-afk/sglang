from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

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
