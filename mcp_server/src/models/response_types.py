"""Response type definitions for Graphiti MCP Server."""

from typing import Any

from pydantic import BaseModel


class ErrorResponse(BaseModel):
    error: str


class SuccessResponse(BaseModel):
    message: str


class NodeResult(BaseModel):
    uuid: str
    name: str
    labels: list[str]
    created_at: str | None = None
    summary: str | None = None
    group_id: str
    attributes: dict[str, Any] = {}


class NodeSearchResponse(BaseModel):
    message: str
    nodes: list[NodeResult]


class FactSearchResponse(BaseModel):
    message: str
    facts: list[dict[str, Any]]


class EpisodeSearchResponse(BaseModel):
    message: str
    episodes: list[dict[str, Any]]


class StatusResponse(BaseModel):
    status: str
    message: str
