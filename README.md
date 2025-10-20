# Chimera

---

## Overview

Chimera is an advanced AI-driven project designed to automate the creation of LinkedIn content through a fully integrated content pipeline. It harnesses the power of the CrewAI framework to orchestrate multiple specialized AI agents, each responsible for distinct content generation tasks such as editorial planning, technical and product content creation, post writing, and visual asset generation. This automation pipeline leverages advanced language models (LLMs), retrieval-augmented generation (RAG) techniques, semantic vector search, and various external tools to generate, enrich, and manage content effectively.

The architectural scope includes seamless integration of external knowledge bases (via a Qdrant vector database), web search APIs (SerperDev or DuckDuckGo), AI-driven image generation (DALL-E), web scraping, and file management utilities. The system is configurable through YAML files and environment variables, allowing end users or developers to adapt the pipeline to different domains, topics, and publication schedules.

Within the broader landscape of AI content generation platforms, this repository focuses specifically on the LinkedIn content creation use case, providing a modular, extensible, and maintainable framework to automate social media content workflows. It targets content marketers, social media managers, AI practitioners, and developers looking to build or customize automated content pipelines using state-of-the-art AI and vector search technology.

### Internal Module Roles

- **LinkedInCrew**: The core orchestration class defining all agents, tasks, and tools composing the content creation pipeline. It dynamically assembles the agents and tasks into a sequential crew workflow based on user inputs.

- **Agents**: Specialized AI entities each encapsulating specific roles such as content management, general or product expertise, copywriting, design, and planning. Agents use different LLM configurations and toolsets to fulfill their responsibilities.

- **Tasks**: Discrete pipeline stages representing workflow units such as editorial planning, content generation, post writing, and visual creation. Tasks are linked to particular agents.

- **Tools**: External integrations providing capabilities like web search, image generation, downloading, scraping, and semantic knowledge base access. Tools are injected into agents to augment their abilities.

- **Utilities**: Helper functions and configuration management for environment loading, memory directory checks, LLM setup, and output formatting.

- **Storage Components**: QdrantStorage class and related modules manage vector embeddings storage, search, and update operations, enabling retrieval-augmented generation for knowledge-based content.

- **Main Runner**: The entry point script handling user interaction, input validation, directory setup, crew instantiation, and pipeline execution.

---

## Technology Stack

- **Programming Language**: Python 3.x, chosen for its rich ecosystem in AI and data processing.

- **Core Framework**: CrewAI – a framework facilitating AI orchestration through declarative definitions of agents, tasks, crews, and processes.

- **Vector Database**: Qdrant – used for storing and retrieving vector embeddings to enable semantic search and knowledge retrieval.

- **LLM Providers**: Supports multiple backends including OpenAI, Google, Anthropic, Groq, Ollama – configurable via environment variables.

- **Search APIs**: SerperDev API and DuckDuckGo (via LangChain integration) for real-time web search capabilities.

- **Image Generation**: DALL-E (via custom DallETool) for AI-powered image creation.

- **Memory Storage**: 
  - Long-term memory stored in SQLite databases.
  - Short-term and entity memories implemented over Qdrant vector storage.

- **Utilities and Libraries**:
  - Python-dotenv for environment configuration.
  - Panel for UI chat interface output.
  - Pydantic for data validation schemas.
  - LangChain for text splitting and query handling.
  - wget for image downloading.

---

## Directory Structure

```
├── src/
│   ├── chimera/
│   │   ├── crew.py                  # Core crew class with agents, tasks, and crew orchestration
│   │   ├── main.py                  # Entry point script for user input and crew execution
│   │   ├── tools/
│   │   │   ├── dalle_tool.py        # DALL-E image generation and download tools
│   │   │   ├── duckduckgo_tool.py  # Custom DuckDuckGo web search tool
│   │   │   ├── qdrant_tool.py       # Knowledge base insertion and search tools
│   │   ├── utils/
│   │   │   ├── utils.py             # Utility functions: output printing, memory checks, LLM config
│   │   │   ├── storage_config.py    # Factory functions for long-term, short-term, entity memory
│   │   │   ├── storage_qdrant.py    # QdrantStorage class managing vector DB interactions
│   ├── config/
│   │   ├── agents.yaml              # YAML configuration for agent parameters
│   │   ├── tasks.yaml               # YAML configuration for task parameters
├── output/                         # Default directory for storing generated text and images
├── memory/                         # Directory used for persistent memory storage (SQLite DB)
├── .env                           # Environment configuration file for API keys, model settings
├── README.md                      # Project documentation (this file)
```

---

## Getting Started

### Prerequisites and Setup

- **Python Version**: Ensure Python 3.8 or later is installed.

- **Dependency Installation**: Use `pip` or a virtual environment manager to install required dependencies, including CrewAI, LangChain, Qdrant client, Panel, and others as specified in `requirements.txt` (assumed present).

- **Environment Variables**: 
  - Create a `.env` file at the project root.
  - Configure API keys (e.g., `SERPER_API_KEY`, `OPENAI_API_KEY`), LLM provider parameters (`PROVIDER`, `MODEL`, `MANAGER_MODEL`), Qdrant connection details (`QDRANT_MODE`, `QDRANT_HOST`), and other settings like temperature and token limits.
  - Example:
    ```
    PROVIDER=openai
    MODEL=gpt-4.1-mini
    MANAGER_MODEL=gpt-4.1
    SERPER_API_KEY=your_serper_key_here
    QDRANT_MODE=memory
    ```

- **Configuration Files**: Ensure `config/agents.yaml` and `config/tasks.yaml` are present and properly configured to define agent and task behaviors.

- **Directory Preparation**: The system auto-creates required memory and output directories if missing.

### Cloning the Repository

Clone the project repository with:

```
git clone <repository_url>
```

Replace `<repository_url>` with the actual source URL.

### Module Usage Overview

- **LinkedInCrew Module (`crew.py`)**: 
  - Import and instantiate `LinkedInCrew` with user inputs such as expert type, topic, number of posts, and frequency.
  - Use the `crew()` method to assemble the workflow.
  - Call `kickoff()` on the crew to start the content generation pipeline.
  - This module acts as the central orchestrator binding agents, tasks, and tools.

- **Agents and Tasks**: 
  - Agents are configured factory methods returning CrewAI `Agent` instances with assigned LLMs and tools.
  - Tasks encapsulate stages like editorial planning and post creation, linked to respective agents.
  - They are automatically registered via decorators and used within the crew workflow.

- **Tools**: 
  - Web search tools switch dynamically based on API key availability.
  - Image generation and download tools enable visual content automation.
  - Knowledge base tools manage semantic search and data insertion leveraging Qdrant.
  - Tools are imported and injected into agents as dependencies.

- **Utilities**: 
  - Functions for LLM configuration and directory management are available for reuse.
  - Memory management utilities provide interfaces to long-term and short-term storage systems.

- **Main Runner (`main.py`)**: 
  - Handles user interaction for pipeline parameters.
  - Sets up output directories with timestamped naming for traceability.
  - Instantiates and runs the LinkedInCrew pipeline.
  - Saves output to organized markdown files.

### Minimal Configuration and Usage Example

1. Set environment variables through `.env` or export in shell.

2. Prepare the `agents.yaml` and `tasks.yaml` configuration files to define agent behaviors.

3. Run the main script to start the pipeline:

- User enters expert type (generalist or product expert), topic, number of posts, and posting frequency.

- The system builds the appropriate crew setup and executes tasks sequentially.

- Outputs (textual content and images) are saved in timestamped directories under `output/`.

This setup requires no code changes to switch LLM providers or tools; simply updating environment variables and configuration files suffices.

---

## Functional Analysis

### 1. Main Responsibilities of the System

The system’s core responsibility is to automate LinkedIn content creation by orchestrating AI agents that generate different facets of content in a structured pipeline. It manages the flow from editorial planning to technical/product content generation, post writing, visual design, and scheduling. It provides foundational services such as:

- LLM configuration and management.
- Integration of external tools (search, scraping, image generation).
- Semantic knowledge base management via vector search.
- Task orchestration with process control.
- Input handling and output persistence.

### 2. Problems the System Solves

- **Content Generation Scalability**: Automates the production of high-quality LinkedIn posts and associated visual assets, saving time and effort for content creators.

- **Knowledge Integration**: Uses retrieval-augmented generation with Qdrant vector search to enhance content factuality and relevance by leveraging stored domain knowledge.

- **Multi-Modal Content**: Combines text and images generated through AI to produce engaging social media content.

- **Configurability and Extensibility**: Enables users to tailor content workflows for different expertise domains (generalist vs. product-focused) and publication schedules.

- **Tool Abstraction**: Wraps diverse external services as modular tools seamlessly integrated into agents.

### 3. Interaction of Modules and Components

- **Agents and Tasks**: Agents execute tasks assigned to them, using configured LLMs and tools. Tasks represent discrete workflow units and are linked to agents via factory methods.

- **Crew Orchestration**: The `LinkedInCrew` class aggregates agents and tasks according to input parameters into a `Crew` instance, configuring process flow (sequential) and verbosity.

- **Tools and Storage**: Tools provide capabilities such as web search, image generation, scraping, and knowledge base operations. These are injected into agents to expand their functionalities.

- **Memory Layers**: Long-term memory (SQLite) and short-term/entity memories (Qdrant) provide persistent and ephemeral data storage for knowledge and context.

- **Main Script**: Acts as the user interface layer, collecting inputs, preparing environment and directories, and triggering the crew execution.

This modular architecture promotes loose coupling and high cohesion, enabling independent development and testing of agents, tools, and tasks.

### 4. User-Facing vs. System-Facing Functionalities

- **User-Facing**:
  - Command-line interface via `main.py` for input collection (expert type, topic, number of posts).
  - Output directories containing generated text content and images.
  - Informative print messages guiding the user through pipeline execution.

- **System-Facing**:
  - CrewAI orchestration of agents and tasks.
  - Underlying tool integrations for web search, scraping, image generation.
  - Background management of semantic knowledge base and memory abstractions.
  - Environment-driven configuration loading and management.

The clear distinction ensures users interact mainly with the input/output layers while the system manages complex AI workflows transparently.

---

## Architectural Patterns and Design Principles Applied

- **Factory Pattern**: Agents, tasks, and LLM configurations are instantiated through decorated factory methods, enabling dynamic creation based on configuration files and runtime parameters.

- **Decorator Pattern**: Use of `@agent`, `@task`, `@tool`, and `@crew` decorators registers components with CrewAI, promoting extensibility and declarative code organization.

- **Strategy Pattern**: Different agents embody distinct content generation strategies — e.g., a generalist expert versus a product expert using RAG techniques.

- **Configuration-Driven Architecture**: Decouples code from deployment settings via YAML config files and environment variables, facilitating flexible adaptation.

- **Sequential Pipeline Pattern**: Defines a linear task execution order ensuring predictable processing flow and dependency management.

- **Idempotency Pattern**: Uses deterministic stable IDs for knowledge chunks to avoid duplicate entries in the vector database.

- **Single Responsibility Principle**: Each agent and task has a focused role, enhancing maintainability.

- **Dependency Injection**: Tools and LLM instances are injected into agents, reducing coupling and enhancing testability.

---

## Code Quality Analysis

Based on SonarQube and software analyst reports, the code quality metrics are as follows:

- **Bugs**: No critical bugs reported; code is largely free from blocking defects.

- **Vulnerabilities**: No high-severity security vulnerabilities identified, indicating a reasonably secure codebase.

- **Code Smells**: Some occurrences of code smells such as hardcoded strings, duplicated code blocks (especially LLM configuration), and print statements used instead of structured logging.

- **Code Coverage**: No explicit test coverage data available; absence of testing scripts suggests limited automated testing presence.

- **Duplication**: Minor duplication detected, particularly in LLM configuration setup and environment variable handling.

**Implications**:

- The codebase is modular and well-structured, supporting maintainability.

- Lack of comprehensive testing and structured logging may affect reliability and diagnosability in production environments.

- Some technical debt exists in configuration handling and error management that could impact scalability and robustness.

---

## Weaknesses and Areas for Improvement

- [ ] **Refactor LLM Configuration**: Abstract repeated LLM instantiation code into reusable helper functions to eliminate duplication and reduce maintenance overhead.

- [ ] **Externalize Configuration Paths**: Replace hardcoded YAML configuration file paths with environment variables or CLI parameters to increase flexibility.

- [ ] **Replace Print Statements with Logging**: Integrate a structured logging framework with configurable log levels (info, warning, error) to improve monitoring and debugging.

- [ ] **Enhance Error Handling**: Add comprehensive try-except blocks around critical operations such as Qdrant client interactions and external API calls to improve fault tolerance.

- [ ] **Validate User Inputs and Environment Variables**: Implement input validation routines to prevent invalid runtime parameters and catch potential misconfigurations early.

- [ ] **Fix Environment Variable Typo**: Correct the likely typo `MANGER_MODEL` to `MANAGER_MODEL` to avoid runtime misconfigurations.

- [ ] **Introduce Asynchronous Processing**: Evaluate and implement async or parallel task execution to enhance throughput and responsiveness.

- [ ] **Increase Test Coverage**: Develop unit and integration tests targeting agents, tasks, and tools, particularly for knowledge base interactions and pipeline orchestration.

- [ ] **Modularize Tool Injection**: Decouple tools from class attributes where possible, enabling more flexible agent composition and easier testing.

- [ ] **Document Configuration and Usage**: Provide detailed documentation on environment variables, YAML configuration formats, and usage examples to aid onboarding and maintenance.

- [ ] **Secure API Key Management**: Implement secure storage and access patterns for sensitive credentials, especially for deployment environments.

- [ ] **CI/CD Integration**: Establish continuous integration pipelines for automated testing, code quality checks, and deployment readiness.

---

## Further Areas of Investigation

- **Performance and Scalability**: Analyze bottlenecks in sequential task execution and explore parallelization or distributed processing options.

- **Semantic Search Optimization**: Fine-tune text chunking parameters and vector indexing strategies in Qdrant for improved retrieval accuracy and speed.

- **Advanced Agent Collaboration**: Investigate adding delegation or dynamic task assignment among agents to enhance pipeline flexibility.

- **Extensibility to Other Social Platforms**: Research adapting the pipeline architecture for other content platforms (e.g., Twitter, Instagram).

- **Security Auditing**: Conduct thorough security reviews focusing on API key management, data privacy, and network communication.

- **Comprehensive Testing**: Measure current test coverage and identify critical areas lacking tests, such as error scenarios and edge cases.

---

## Attribution

Generated with the support of [ArchAI](https://github.com/ArchAI-Labs/code_explainer), an automated documentation system.