# Chimera

<br>

![logo](https://github.com/ArchAI-Labs/chimera/blob/main/img/chimera.png)

<br>

---

## Overview

The `chimera` repository is a powerful, modular platform designed to automate the end-to-end creation of expert-level LinkedIn content campaigns. It leverages advanced AI techniques, including Large Language Models (LLMs), agent-oriented workflows, and integration with external APIs and knowledge bases, to orchestrate the planning, research, writing, and scheduling of LinkedIn posts. The system is highly configurable, supporting both generalist and product-focused content strategies, and is suitable for marketing professionals, content creators, agencies, and organizations aiming to scale and standardize their LinkedIn content output.

**Key capabilities include:**
- Automated creation of editorial strategies tailored to a user’s topic and audience.
- Generation of research-backed LinkedIn posts, each complete with references and best-practice formatting.
- Production of AI-generated visual prompts for tools like DALL-E to create engaging post imagery.
- Construction of content calendars, mapping posts to optimal publishing dates and performance metrics.
- Extensible integration with web search, image generation, and vector-based knowledge retrieval.

**Role of internal modules:**
- **Orchestration Modules:** `LinkedInCrew` and `LinkedInWorkflow` serve as the central orchestrators, managing workflow steps, agent/tool assignment, and event-driven data handoff.
- **Agent/Tool Adapters:** Configured via YAML and environment variables, these enable plug-and-play integration with web search, knowledge bases, and image generation services.
- **Utility and Configuration Modules:** Provide robust file I/O, configuration validation, environment management, and fallback logic for missing dependencies.

---

## Technology Stack

- **Language:** Python 3.x
- **Frameworks:** 
  - [llama_index](https://github.com/jerryjliu/llama_index): For agent orchestration, memory management, workflow/pipeline definition.
- **Libraries:**
  - `requests`, `BeautifulSoup`: Web scraping and HTML parsing.
  - `PyYAML (yaml)`: Configuration file parsing.
  - `python-dotenv (dotenv)`: Environment variable management.
  - `Qdrant`: Vector search database for knowledge management.
  - `OpenAI/DALL-E`: AI-powered image generation.
  - `duckduckgo_search`/`ddgs`: Web search API integration.
- **Tools:**
  - **Custom Utilities:** `utils.utils`, `utils.storage_config`, `utils.memory_config` for LLM config, memory abstraction, and output management.
  - **External Service Adapters:** `tools.duckduckgo_tool`, `tools.dalle_tool`, `tools.qdrant_tool` for external research and content generation.

---

## Directory Structure

The project is structured for clarity and modularity, facilitating easy extension and maintenance.

```
chimera/
├── src/
│   └── chimera/
│       ├── main.py                   # Application entry point and CLI interface
│       ├── crew_llamaindex.py        # Core workflow orchestration and pipeline logic
│       ├── tools/
│       │   ├── duckduckgo_tool.py    # DuckDuckGo web search integration
│       │   ├── dalle_tool.py         # DALL-E image generation and download adapters
│       │   └── qdrant_tool.py        # Qdrant knowledge base integration
│       ├── utils/
│       │   ├── utils.py              # Utility functions (LLM config, file I/O, etc.)
│       │   ├── storage_config.py     # Memory/knowledge base configuration helpers
│       │   └── memory_config.py      # Centralized Qdrant/embedding config and CRUD
│       └── ...
├── config/
│   ├── agents.yaml                   # Agent configuration (roles, goals, tools)
│   ├── tasks.yaml                    # Task configuration (descriptions, expected output)
├── templates/
│   └── linkedin_post_preview.html    # HTML template for post preview rendering
├── output/                           # Generated output (strategy, posts, images, calendars)
│   └── run_*/                        # Timestamped output directories per pipeline run
└── README.md                         # Project documentation (this file)
```

---

## Getting Started

### Prerequisites

- **Python 3.8 or newer** is required.
- Install all dependencies (preferably in a virtual environment):

  - llama_index
  - requests
  - beautifulsoup4
  - pyyaml
  - python-dotenv
  - qdrant-client
  - duckduckgo-search or ddgs
  - openai

  Install with pip:

  ```
  pip install llama_index requests beautifulsoup4 pyyaml python-dotenv qdrant-client duckduckgo-search openai
  ```

- **External Service Access:**  
  - To enable all features, you need valid API keys for OpenAI (DALL-E), and (optionally) Qdrant and DuckDuckGo (if running in certain environments).

- **Configuration Files:**  
  - Ensure `config/agents.yaml` and `config/tasks.yaml` exist with appropriate agent/task definitions.
  - Prepare a `.env` file (see below for required variables).

### Cloning the Repository

Clone the repository from GitHub:

```
git clone https://github.com/ArchAI-Labs/chimera.git
cd chimera
```

### Environment Setup

1. **Create a `.env` file** in the root directory with entries such as:

    ```
    PROVIDER=ollama
    MODEL=llama3
    OPENAI_API_KEY=your-openai-key
    QDRANT_COLLECTION=linkedin_knowledge
    PRODUCT_SITES=https://example.com/product1,https://example.com/product2
    ```

    - `PROVIDER`: LLM provider (`ollama`, `openai`, or `anthropic`).
    - `MODEL`: Model name for the LLM.
    - `OPENAI_API_KEY`: Required for DALL-E image generation.
    - `PRODUCT_SITES`: (Optional) Comma-separated URLs for product knowledge ingestion.

2. **Install dependencies** as described above.

### Running the Pipeline

- Launch the CLI interface:
  ```
  cd src/chimera
  python main.py
  ```
- The system will guide you through input collection (expert type, topic, number of posts, posting frequency), validate your environment, and execute the content generation workflow.
- Outputs will be saved in a timestamped `output/run_*/` directory with all generated files (strategy, posts, images, calendar).

### Module Usage

#### `crew_llamaindex.py` (Pipeline Orchestration)

- **Purpose:** Drives the entire content creation process, manages agents, tools, and workflow execution.
- **Integration:** Import and instantiate `LinkedInCrew` with input parameters, then invoke `run()` or `kickoff()` to start the pipeline.
- **Configuration:** Relies on both environment variables and YAML config files for flexibility.

#### `main.py` (CLI/Entrypoint)

- **Purpose:** Provides the user interface for collecting input, validating configuration, and triggering the pipeline.
- **Usage:** Run as a script; it manages environment setup, input collection, and invokes the orchestrator.

#### `tools/` (External API Adapters)

- **Purpose:** Encapsulate integration with external services like DuckDuckGo, DALL-E, and Qdrant.
- **Usage:** Used internally by agents via the `FunctionTool` adapter pattern; can be extended for new tools.

#### `utils/` (Utility and Config Management)

- **Purpose:** Utility functions for environment validation, file I/O, LLM configuration, and memory/knowledge management.
- **Usage:** Imported as dependencies across orchestration and tool modules.

**Example: External integration**

To use the workflow as a library, import `LinkedInCrew` and run the pipeline with your parameters:

- Ensure `.env` and YAML configs are set up.
- Prepare your input dictionary (e.g., `{"topic": "AI in Healthcare", "num_posts": 5, ...}`).
- Instantiate `LinkedInCrew(inputs=your_inputs)` and call `run()`.

---

## Functional Analysis

### 1. Main Responsibilities of the System

- **Automated Content Campaign Generation:**  
  Orchestrates the full lifecycle of a LinkedIn campaign—from editorial planning, through post creation and visual prompt generation, to scheduling and artifact output.
- **Agent-Driven Workflow:**  
  Delegates subtasks to specialized agents (e.g., research, writing, design, planning), each configured via YAML and empowered by a suite of tools.
- **Research Integration:**  
  Enables both web-based and knowledge base-backed research, ensuring content is authoritative and relevant.
- **Output Management:**  
  Produces ready-to-use Markdown, HTML, and text artifacts for user review and publishing.

### 2. Problems the System Solves

- Reduces manual effort and expertise required to plan, research, and produce LinkedIn campaigns.
- Ensures content is up-to-date by integrating web and internal research sources.
- Bridges the gap between textual and visual content creation via AI-powered image prompts.
- Standardizes campaign output for consistency, quality, and repeatability.

### 3. Interaction of Modules and Components

- **User Input → Orchestrator (`LinkedInCrew`):**  
  All workflow parameters are collected at the start and injected as context.
- **Config Loading:**  
  YAML configs are loaded for agents and tasks, providing instructions and context for each workflow step.
- **Tool Initialization:**  
  Tools (web search, file writing, image generation, knowledge base) are wrapped as `FunctionTool` objects and injected into agent instances.
- **Agent Execution:**  
  Each workflow step invokes the appropriate agent, passing prompts and context, and handling results as typed events.
- **Event-Driven Workflow:**  
  Steps are decorated with `@step` and pass typed event objects between them, ensuring strong interface contracts.
- **Output Artifacts:**  
  Final outputs are saved to disk, and HTML previews are generated for easy user validation.

### 4. User-Facing vs. System-Facing Functionalities

- **User-Facing:**  
  - CLI for input and confirmation.
  - Output files (Markdown, HTML previews, images, calendar) for publishing and review.
  - Clear logging and user feedback during each step.
- **System-Facing:**  
  - Tool and agent abstractions for modularity and extensibility.
  - Config and environment management for portability.
  - Error handling, fallback logic, and health checks for robustness.

### Interface/Abstract Classes

- **Event Base Class:**  
  All workflow step outputs inherit from a common `Event`, ensuring consistent data handoff and enabling type safety across the workflow.
- **FunctionTool Wrapper:**  
  Every tool is standardized using the `FunctionTool` interface, enforcing a consistent callable signature for agent consumption.

---

## Architectural Patterns and Design Principles Applied

- **Facade Pattern:**  
  `LinkedInCrew` serves as the central orchestrator, simplifying the management of agents, tools, and workflow for external callers.
- **Factory Pattern:**  
  LLMs, agents, and tools are instantiated based on configuration and environment, enabling provider-agnostic operation.
- **Adapter Pattern:**  
  Tools are wrapped as `FunctionTool` objects, allowing interchangeable use regardless of backend.
- **Event Sourcing/DTO Pattern:**  
  Workflow step outputs are strongly-typed event classes, ensuring clarity and traceability of data flow.
- **Null Object Pattern:**  
  Fallback tool implementations ensure the system degrades gracefully if dependencies are missing.
- **Config-Driven Architecture:**  
  Behavior is determined by YAML files and environment variables, not hardcoded logic.
- **Pipeline/Workflow Pattern:**  
  Each workflow step is atomic, composable, and event-driven, supporting robust orchestration and easy extension.
- **Separation of Concerns:**  
  Clear boundaries between configuration, workflow orchestration, agent/tool logic, and utility functions.
- **Dependency Injection:**  
  Tools, agents, and configs are passed as dependencies, supporting testability and modularity.
- **Defensive Programming:**  
  Extensive error handling, output validation, and fallback mechanisms are in place.

---

## Code Quality Analysis

**SonarQube and Analyst-Derived Metrics:**

- **Bugs:**  
  - No explicit critical bugs identified in the current review; however, aggressive suppression of warnings may hide important issues.
- **Vulnerabilities:**  
  - No direct vulnerabilities detected, but warning suppression and lack of input validation (e.g., for URLs, file paths) could expose latent security risks.
- **Code Smells:**  
  - Warning suppression at the global level is a significant code smell, as it may mask deprecations, performance issues, or runtime failures.
  - Manual parsing of LLM output using regular expressions is fragile and may break with changes in LLM output format.
  - Scattered fallback logic for tool implementations increases the risk of maintenance errors and interface divergence.
- **Code Coverage:**  
  - No explicit tests are visible in the provided files; key areas such as config loading, event parsing, and fallback logic likely lack automated test coverage.
- **Duplication:**  
  - Some redundancy in fallback logic for tool initialization and error handling, but overall code duplication appears moderate.

**Implications:**
- **Maintainability:**  
  While modular and clear, maintainability could be hampered by warning suppression, fragile parsing, and distributed fallback logic.
- **Reliability:**  
  Fallbacks and defensive programming increase robustness, but risks remain due to error silencing.
- **Security:**  
  Absence of strict input validation and warning suppression may allow security issues to go undetected.
- **Scalability:**  
  The system is architecturally scalable due to modularity, but testing and validation should be strengthened to support large-scale deployments.

---

## Weaknesses and Areas for Improvement

The following concrete TODOs are derived from code analysis and SonarQube/code quality findings:

- [ ] **Remove Global Warning Suppression:**  
      Limit warning suppression to specific, well-understood cases. Ensure all other warnings are logged or surfaced during development and runtime.
- [ ] **Centralize Fallback Logic:**  
      Move all dummy/null tool implementations into a dedicated module to ensure maintainability and interface consistency.
- [ ] **Enforce Structured LLM Outputs:**  
      Require LLMs to return outputs in a structured format (e.g., JSON), and update parsing logic to use schema validation instead of regular expressions.
- [ ] **Validate and Document Environment Variables:**  
      Implement startup validation for all required environment variables and provide clear documentation or schema files.
- [ ] **Add Automated Test Coverage:**  
      Develop unit and integration tests for config loading, workflow step transitions, event parsing, and fallback tool logic.
- [ ] **Replace Print Statements with Structured Logging:**  
      Integrate a logging framework to standardize error and progress reporting, with support for log levels and output to files.
- [ ] **Improve Input Validation:**  
      Sanitize all user-supplied URLs, file names, and paths to prevent injection and file system attacks.
- [ ] **Refactor Tool Initialization:**  
      Use a factory or registry pattern to deduplicate and clarify tool creation and health checking.
- [ ] **Optimize Web Scraping:**  
      Consider asynchronous or batched scraping to improve ingestion performance and avoid blocking on slow websites.
- [ ] **Enhance Documentation:**  
      Expand high-level and module-level documentation, including usage examples, configuration references, and system diagrams.
- [ ] **Expand Code Comments and Docstrings:**  
      Ensure all public classes and functions include descriptive docstrings for clarity and maintainability.
- [ ] **Monitor for Dependency Updates:**  
      Track updates and deprecations in external APIs and upgrade dependencies regularly to maintain compatibility and security.

---

## Further Areas of Investigation

- **Performance and Scalability:**  
  Investigate the performance of web scraping and large-scale knowledge ingestion, especially for large or slow product sites. Benchmark Qdrant ingestion and retrieval at scale.
- **Security Hardening:**  
  Review all input surfaces for injection risks, add validation/sanitization, and audit file I/O for path traversal vulnerabilities.
- **Test Coverage Analysis:**  
  Identify all untested modules and prioritize test development for the most complex or fragile code paths.
- **Monitoring and Health Checks:**  
  Develop infrastructure for monitoring agent/tool health, workflow execution status, and error rates for production deployments.
- **Extensibility for New Channels:**  
  Explore integration with other social media platforms, direct publishing APIs, or analytics feedback loops.
- **User Management and Access Control:**  
  Consider adding multi-user support, authentication, and role-based access control for collaborative environments.
- **Advanced Analytics:**  
  Investigate adding campaign performance tracking, feedback collection, and self-improving content optimization loops.

---

## Attribution

Generated with the support of [ArchAI](https://github.com/ArchAI-Labs), an automated documentation system.
