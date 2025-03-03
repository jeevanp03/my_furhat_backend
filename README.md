Below is the updated README with the changes requested. The reference to the non-existent top-level tools.py has been removed, and the instructions now clarify the hybrid dependency management flow using Poetry and pip.

---

# Conversational Agent Project

This project implements an advanced conversational agent that can:

- Ingest documents (e.g., PDFs) and store them in a Chroma vector database for **retrieval-augmented generation (RAG)**.
- Dynamically route queries to web search or vectorstore retrieval.
- Integrate with external APIs (Foursquare, OpenStreetMap, Overpass, IPData) for location-based or IP-based queries.
- Generate natural language responses via Large Language Models (LLMs) such as HuggingFace Transformers or Llama Cpp.

It includes **FastAPI** endpoints for synchronous/asynchronous interaction, **LangGraph**-based state machines for multi-step conversation flows, and optional classification tools.

---

## Table of Contents

- [Conversational Agent Project](#conversational-agent-project)
  - [Table of Contents](#table-of-contents)
  - [Project Structure](#project-structure)
  - [Requirements \& Installation](#requirements--installation)
  - [Configuration](#configuration)
    - [Creating Your .env File](#creating-your-env-file)
    - [Using direnv for Environment Management](#using-direnv-for-environment-management)
  - [Usage](#usage)
    - [Running the FastAPI Server](#running-the-fastapi-server)
    - [Interactive Conversations \& Tests](#interactive-conversations--tests)
  - [Managing Dependencies with Poetry and pip](#managing-dependencies-with-poetry-and-pip)
    - [Activating the Poetry Environment](#activating-the-poetry-environment)
    - [Additional Troubleshooting Steps](#additional-troubleshooting-steps)
  - [Key Components](#key-components)
  - [Notes](#notes)
  - [Contributing](#contributing)

---

## Project Structure

A simplified view of the repository (showing the most important directories and files):

```
.
├── furhat_skills/               # (Optional) Skills or scripts for Furhat robot integration
├── middleware/
│   └── main.py                  # FastAPI or other APIs for the middleware layer
├── my_furhat_backend/
│   ├── agents/
│   │   ├── document_agent.py            # State-graph-based DocumentAgent for RAG conversations
│   │   ├── test_2_conversational_agent.py  # Extended test workflow with dynamic routing & grading
│   │   └── test_conversational_agent.py    # Basic RAG-based conversation test
│   ├── api_clients/
│   │   ├── foursquare_client.py    # Foursquare API client
│   │   ├── osm_client.py           # OSM (Nominatim) API client
│   │   ├── overpass_client.py      # Overpass API client
│   │   └── ipdata_client.py        # IPData API client
│   ├── config/
│   │   └── settings.py             # Loads environment variables (dotenv)
│   ├── ingestion/
│   │   └── CMRPublished.pdf        # Example PDF for ingestion (or other docs)
│   ├── llm_tools/
│   │   └── tools.py                # Defines location-based search tools referencing api_clients
│   ├── models/
│   │   ├── chatbot_factory.py      # Factory to build HuggingFace/Llama-based chatbots
│   │   ├── classifier.py           # Zero-shot classifier
│   │   ├── llm_factory.py          # Creates HuggingFace/Llama LLMs
│   │   └── model_pipeline.py       # Manages HF pipelines (auth, loading, etc.)
│   ├── RAG/
│   │   └── rag_flow.py             # RAG logic (loading, chunking, storing in Chroma)
│   ├── utils/
│   │   └── util.py                 # Utility functions for text cleaning & formatting
│   └── main.py                     # FastAPI app with /ask, /transcribe, /response endpoints
├── tests/                          # Additional tests or integration tests
├── .env                            # Environment variables (API keys, etc.) – typically in .gitignore
├── .envrc                          # direnv configuration file
├── pyproject.toml                  # Poetry or other build-system configuration
├── poetry.lock                     # Poetry lockfile
└── README.md                       # Project documentation (this file)
```

---

## Requirements & Installation

This project uses a hybrid dependency management approach: some dependencies are installed via pip into your environment, and others are managed by Poetry (tracked in the poetry.lock file).

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/yourproject.git
   cd yourproject
   ```

2. **Install Dependencies via Poetry**  
   If you haven't already, install Poetry globally. Then run:

   ```bash
   poetry lock
   poetry install
   ```

   This will create or update the `poetry.lock` file and install the dependencies into Poetry’s virtual environment.

3. **(Optional) Install Additional pip Dependencies**  
   If there are any extra dependencies provided via a `requirements.txt` file, you can install them after activating the Poetry environment:
   ```bash
   pip install -r requirements.txt
   ```

---

## Configuration

### Creating Your .env File

Because the `.env` file is typically listed in `.gitignore`, you’ll need to create your own locally:

1. **Create a `.env` file** in the root directory of the project (same level as `pyproject.toml`).
2. **Add your API keys and environment variables.** For example:
   ```bash
   FSQ_KEY=<YOUR_FOURSQUARE_API_KEY>
   IP_KEY=<YOUR_IPDATA_API_KEY>
   HF_KEY=<YOUR_HUGGINGFACE_API_KEY>
   ```
3. **Save the file.** This file will be loaded at runtime by `my_furhat_backend/config/settings.py`.

### Using direnv for Environment Management

If you use **direnv** to automatically load environment variables:

- You might see an error like:
  ```
  direnv: error /Users/{home_directory}/my_furhat_backend/.envrc is blocked. Run `direnv allow` to approve its content
  ```
- To approve the content of `.envrc`, run:

  ```bash
  direnv allow
  ```

  This command will load the file and export the necessary environment variables.

- Your `.envrc` may include:
  ```bash
  export PATH="$HOME/.local/bin:$PATH"
  ```
  This ensures that your local binaries are in your PATH.

---

## Usage

### Running the FastAPI Server

To start the server (after ensuring your environment is activated):

```bash
poetry run uvicorn my_furhat_backend.main:app --host 0.0.0.0 --port 8000
```

Or, if using pip/venv:

```bash
uvicorn my_furhat_backend.main:app --host 0.0.0.0 --port 8000
```

**Endpoints:**

- `POST /ask` — Synchronously processes a user query and returns an answer.
- `POST /transcribe` — Asynchronously handles transcriptions (stores response for later retrieval).
- `GET /response` — Fetches the latest response generated by the agent.

### Interactive Conversations & Tests

There are multiple ways to test or interact with the conversation agent:

1. **Basic RAG-based Conversation**

   ```bash
   poetry run python my_furhat_backend/agents/test_conversational_agent.py
   ```

   This script starts a conversation loop using document ingestion and retrieval.

2. **Extended Workflow with Grading & Routing**

   ```bash
   poetry run python my_furhat_backend/agents/test_2_conversational_agent.py
   ```

   This version adds dynamic routing (vector store vs. web search) and grading chains (document/answer grading).

3. **Middleware Service**  
   If you have a middleware service at `middleware/main.py`, run:
   ```bash
   poetry run python middleware/main.py
   ```
   Or, if it's a FastAPI service:
   ```bash
   uvicorn middleware.main:app --reload
   ```

---

## Managing Dependencies with Poetry and pip

This project employs a hybrid dependency strategy:

- **Poetry:** Manages the core dependencies (tracked in `poetry.lock`).
- **pip:** Some additional dependencies may be installed via pip.

**Correct Flow:**

1. Install Poetry and run:
   ```bash
   poetry lock
   poetry install
   ```
2. Activate the Poetry environment (see next section).
3. If needed, after activating, install extra dependencies from `requirements_poetry.txt`:
   ```bash
   pip install -r requirements.txt
   ```

### Activating the Poetry Environment

When using **Poetry**, you typically run:

```bash
poetry shell
```

If that command does not work as expected or you prefer a manual approach, activate the environment with:

```bash
source $(poetry env info --path)/bin/activate
```

This command retrieves the virtual environment's path from `poetry env info --path` and activates it.

### Additional Troubleshooting Steps

- **Project Directory:** Ensure you’re in the directory containing your `pyproject.toml` file, as some Poetry commands require it.
- **Check for Aliases:** Verify there is no shell alias or function named `poetry` or `shell` interfering with the command.
- **Reinstall/Upgrade Poetry:** Although version 2.0.1 should work, if issues persist, consider reinstalling or updating Poetry.
- **Manual Activation:** If `poetry shell` fails, using `source $(poetry env info --path)/bin/activate` should let you work within your Poetry virtual environment.

---

## Key Components

- **`main.py` in `my_furhat_backend/`**: FastAPI app exposing `/ask`, `/transcribe`, and `/response` endpoints.
- **`rag_flow.py` in `RAG/`**: Handles document ingestion, chunking, storage in a Chroma vector store, and retrieval with optional cross-encoder reranking.
- **`document_agent.py`**: Defines a **StateGraph** workflow for capturing user input, retrieving context, checking uncertainty, and generating responses.
- **`chatbot_factory.py`, `llm_factory.py`, `model_pipeline.py`**: Provide factory methods and pipelines for constructing language models (HuggingFace or Llama Cpp) and chatbots.
- **`llm_tools/tools.py`**: Integration points (tools) that the agent can invoke for location or place queries, referencing the various API clients in `api_clients/`.
- **`middleware/main.py`**: Additional FastAPI or bridging service to connect the conversation agent with other systems.
- **`tests/`**: Contains additional tests or integration checks.

## Notes

- Currently the tools in `llm_tools/tools.py` and the apis in `api_clients` are not being used. They were created at the early stages of the project
  when the original scope of the project was as an agent that would act as a concierge.

---

## Contributing

Contributions are welcome! If you have ideas or bug fixes:

1. Fork this repository.
2. Create a new branch for your changes.
3. Submit a pull request describing your enhancements.

---

**Enjoy using your Conversational Agent!** If you have any questions or run into issues, feel free to open an issue or reach out for support.
