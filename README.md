# <img src="img/pyagent-smith_logo.png" style="height:1em; vertical-align: middle;"> pyagent-smith

Implementation of Langchain Agent that can search web for information. Uses Ollama - models, which can be easily changed to OpenAI, AzureOpenAI, etc. models
Notebooks directory demonstrates the usage.

[TBD] finish web cralwer tool

## Installation

### Install as a Package

Install the package from the repository:

```bash
# Install in development mode (editable)
pip install -e .

# Or install normally
pip install .
```

### Post-Installation Setup

After installation, you need to install Playwright browsers:

```bash
# Install Playwright browsers (required for web crawling)
playwright install chromium
```

Or install all browsers:

```bash
playwright install
```

## Ollama

* prerequisites
  * [installation](https://ollama.com/download/linux)
* code

llama3.1:8b chosen based on <https://skywork.ai/blog/llm/ollama-models-list-2025-100-models-compared/#by-task>

```python
# Full setup (check, start, pull model)
python scripts/run_ollama.py

# Check if Ollama is running
python scripts/run_ollama.py --check-only

# Show Ollama status and resource usage (CPU, RAM)
python scripts/run_ollama.py --status

# Start Ollama server only
python scripts/run_ollama.py --start-only

# Stop Ollama server
python scripts/run_ollama.py --stop

# Note: If Ollama is installed as a systemd service (common on Linux),
# you may need to use sudo to start/stop it:
#   sudo systemctl start ollama
#   sudo systemctl stop ollama
#   sudo systemctl status ollama

# Pull a specific model
python scripts/run_ollama.py --pull-only deepseek-r1:7b

# Custom model and settings
python scripts/run_ollama.py --model mistral --base-url http://localhost:11434
```

## LangSmith Observability

LangSmith provides observability and tracing for your agent. Monitor LLM interactions, tool invocations, and agent decisions.

### Setup

1. **Get your API key**: Sign up at <https://smith.langchain.com> and get your API key

2. **Configure via environment variables** (recommended):

   ```bash
   export LANGSMITH_API_KEY=your_api_key_here
   export LANGSMITH_PROJECT=my-project  # Optional
   ```

3. **Or add to `config.json`**:

   ```json
   {
     "langsmith": {
       "api_key": "your_api_key_here",
       "project": "my-project",
       "endpoint": "https://api.smith.langchain.com",
       "tracing_enabled": true
     }
   }
   ```

### Usage

LangSmith tracing is enabled by default when an API key is configured

### Viewing Traces

Visit <https://smith.langchain.com> to:

* View trace trees with all LLM calls and tool invocations
* Monitor performance (latency, token usage, costs)
* Debug issues by inspecting inputs/outputs
* Filter traces by tags and metadata
* Set up alerts for errors or performance issues

See `notebooks/agent_usage_example.ipynb` for more examples.
