"""
DuckDuckGo search tool wrapper for LlamaIndex
Migrated from CrewAI implementation
"""
from typing import Any, Optional
from llama_index.core.tools import FunctionTool


class MyCustomDuckDuckGoTool:
    """
    Custom DuckDuckGo search tool compatible with LlamaIndex
    """
    
    def __init__(self, max_results: int = 5):
        """
        Initialize the DuckDuckGo search tool
        
        Args:
            max_results: Maximum number of search results to return
        """
        self.max_results = max_results
        self._initialize_search()
    
    def _initialize_search(self):
        """Initialize the DuckDuckGo search backend"""
        try:
            from ddgs import DDGS
            self.ddgs = DDGS()
        except ImportError:
            raise ImportError(
                "duckduckgo_search is required. "
                "Install it with: pip install duckduckgo-search"
            )
    
    def run(self, query: str) -> str:
        """
        Execute a DuckDuckGo search
        
        Args:
            query: Search query string
            
        Returns:
            Formatted string with search results
        """
        try:
            results = list(self.ddgs.text(query, max_results=self.max_results))
            
            if not results:
                return f"No results found for query: {query}"
            
            formatted_results = [f"Search Results for '{query}':\n"]
            
            for idx, result in enumerate(results, 1):
                title = result.get('title', 'No title')
                href = result.get('href', 'No URL')
                body = result.get('body', 'No description')
                
                formatted_results.append(
                    f"\n{idx}. {title}\n"
                    f"   URL: {href}\n"
                    f"   {body}\n"
                )
            
            return '\n'.join(formatted_results)
            
        except Exception as e:
            return f"Error performing search: {str(e)}"
    
    def as_tool(self) -> FunctionTool:
        """
        Convert to LlamaIndex FunctionTool
        
        Returns:
            FunctionTool instance
        """
        return FunctionTool.from_defaults(
            fn=self.run,
            name="duckduckgo_search",
            description=(
                "Search the web using DuckDuckGo. "
                "Input should be a search query string. "
                "Returns formatted search results with titles, URLs, and descriptions."
            )
        )