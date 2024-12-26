from typing import List, Union
from fastapi import APIRouter
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

router = APIRouter()


class InputChat(BaseModel):
    """Input for the chat endpoint."""

    messages: List[Union[HumanMessage, AIMessage, SystemMessage]] = Field(
        ...,
        description="The chat messages representing the current conversation.",
    )


@router.get("/retrieve/")
async def read_items():
    # Example of using the InputChat model
    example_input = InputChat(
        messages=[
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
            SystemMessage(content="System initialized."),
        ]
    )
    return example_input.dict()
