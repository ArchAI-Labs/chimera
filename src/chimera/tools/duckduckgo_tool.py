from crewai.tools import BaseTool
from langchain_community.tools import DuckDuckGoSearchRun
from typing import Type
from pydantic import BaseModel, Field

# ============================================================
# DUCKDUCKGO CUSTOM SEARCH TOOL
# ------------------------------------------------------------
# This module defines a simple custom CrewAI tool that integrates
# the DuckDuckGo search functionality through LangChain.
#
# It allows agents to perform web searches using DuckDuckGo
# directly inside CrewAI workflows.
# ============================================================


class DuckDuckGoSearchSchema(BaseModel):
    """Input for DuckDuckGoSearchRun."""

    query: str = Field(description="The search query to look for on the web.")


class MyCustomDuckDuckGoTool(BaseTool):
    name: str = "Custom DuckDuckGo Search"
    description: str = "A custom tool to search the web for a given query."
    args_schema: Type[BaseModel] = DuckDuckGoSearchSchema

    _duckduckgo_tool: DuckDuckGoSearchRun = DuckDuckGoSearchRun()

    def _run(self, query: str) -> str:
        """Perform the search using the internal DuckDuckGo tool"""
        response = self._duckduckgo_tool.run(tool_input=query)
        return response
