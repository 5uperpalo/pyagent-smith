import os
from typing import Optional, Dict, Any
from pyagent_smith.llm.utils import load_config_json


def get_langsmith_settings(path: Optional[str] = None) -> Dict[str, Any]:
    """Get LangSmith configuration from environment variables or config file.

    Args:
        path: Optional path to config.json file

    Returns:
        Dictionary with LangSmith settings (api_key, project, endpoint, workspace_id, tracing_enabled)
    """
    data = load_config_json(path).get("langsmith", {}) or {}

    # Environment variables take precedence over config file
    api_key = os.getenv("LANGSMITH_API_KEY") or data.get("api_key")
    project = os.getenv("LANGSMITH_PROJECT") or data.get("project")
    endpoint = os.getenv("LANGSMITH_ENDPOINT") or data.get("endpoint", "https://eu.api.smith.langchain.com")
    workspace_id = os.getenv("LANGSMITH_WORKSPACE_ID") or data.get("workspace_id")

    # Check if tracing is enabled (default: True if API key is set)
    tracing_env = os.getenv("LANGSMITH_TRACING", "").lower()
    if tracing_env:
        tracing_enabled = tracing_env in ("true", "1", "yes")
    else:
        tracing_enabled = data.get("tracing_enabled", bool(api_key))

    return {
        "api_key": api_key.strip() if api_key else None,
        "project": project.strip() if project else None,
        "endpoint": endpoint.strip() if endpoint else None,
        "workspace_id": workspace_id.strip() if workspace_id else None,
        "tracing_enabled": tracing_enabled,
    }


def setup_langsmith_tracing(
    api_key: Optional[str] = None,
    project: Optional[str] = None,
    endpoint: Optional[str] = None,
    workspace_id: Optional[str] = None,
    path: Optional[str] = None,
) -> bool:
    """Set up LangSmith tracing by configuring environment variables.

    This function sets the necessary environment variables for LangSmith tracing.
    LangChain will automatically pick up these variables and start tracing.

    Args:
        api_key: Optional LangSmith API key (overrides config/env)
        project: Optional project name (overrides config/env)
        endpoint: Optional LangSmith endpoint (overrides config/env)
        workspace_id: Optional workspace ID (required for org-scoped keys)
        path: Optional path to config.json file

    Returns:
        True if tracing is successfully configured, False otherwise
    """
    settings = get_langsmith_settings(path)

    # Use provided parameters or fall back to settings
    # Strip whitespace from API key if provided directly
    final_api_key = (api_key.strip() if api_key else None) or settings.get("api_key")
    final_project = project or settings.get("project")
    final_endpoint = endpoint or settings.get("endpoint", "https://eu.api.smith.langchain.com")
    final_workspace_id = workspace_id or settings.get("workspace_id")
    # If explicitly calling setup_langsmith_tracing, enable tracing unless explicitly disabled
    tracing_enabled = settings.get("tracing_enabled", True)

    if not final_api_key:
        print("Warning: LangSmith API key not found. Tracing will not be enabled.")
        print("Set LANGSMITH_API_KEY environment variable or add to config.json")
        return False

    # Set environment variables for LangChain to pick up
    # CRITICAL: LANGCHAIN_TRACING_V2 must be set to "true" for tracing to work
    # This is the primary flag that LangChain checks to enable tracing
    if tracing_enabled:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        os.environ.pop("LANGCHAIN_TRACING_V2", None)

    os.environ["LANGSMITH_API_KEY"] = final_api_key
    os.environ["LANGSMITH_TRACING"] = "true" if tracing_enabled else "false"

    if final_project:
        os.environ["LANGSMITH_PROJECT"] = final_project
    else:
        # Remove project from env if it was set but we're not using it
        os.environ.pop("LANGSMITH_PROJECT", None)

    if final_endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = final_endpoint

    # Workspace ID is required for org-scoped API keys
    if final_workspace_id:
        os.environ["LANGSMITH_WORKSPACE_ID"] = final_workspace_id
    else:
        # Remove workspace ID if it was set but we're not using it
        os.environ.pop("LANGSMITH_WORKSPACE_ID", None)

    print("✓ LangSmith tracing configured")
    if final_project:
        print(f"  Project: {final_project}")
    if final_workspace_id:
        print(f"  Workspace ID: {final_workspace_id}")
    print(f"  Endpoint: {final_endpoint}")
    print(f"  Tracing: {'enabled' if tracing_enabled else 'disabled'}")

    return True
