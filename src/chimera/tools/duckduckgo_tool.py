from crewai.tools import BaseTool
from langchain_community.tools import DuckDuckGoSearchRun
from typing import Type
# CORREZIONE: Importa da pydantic, non da pydantic.v1
from pydantic import BaseModel, Field

# Questo schema ora usa Pydantic V2, che è ciò che CrewAI si aspetta
class DuckDuckGoSearchSchema(BaseModel):
    """Input for DuckDuckGoSearchRun."""
    query: str = Field(description="The search query to look for on the web.")

class MyCustomDuckDuckGoTool(BaseTool):
    name: str = "Custom DuckDuckGo Search"
    description: str = "A custom tool to search the web for a given query."
    args_schema: Type[BaseModel] = DuckDuckGoSearchSchema
    
    _duckduckgo_tool: DuckDuckGoSearchRun = DuckDuckGoSearchRun()

    def _run(self, query: str) -> str:
        """Usa il tool di ricerca con la query fornita."""
        response = self._duckduckgo_tool.run(tool_input=query)
        return response