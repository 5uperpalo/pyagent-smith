from typing import Optional
import os
from typing import Dict
from langchain_openai import AzureChatOpenAI

from pyagent_smith.llm.utils import load_config_json


def get_azure_openai_settings(path: Optional[str] = None) -> Dict[str, str]:
    data = load_config_json(path).get("azure_openai", {}) or {}
    # Env overrides take precedence over file values
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or data.get("endpoint", "")
    subscription_key = os.getenv("AZURE_OPENAI_API_KEY") or data.get("subscription_key", "")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION") or data.get("api_version", "")
    deployment = (
        os.getenv("AZURE_OPENAI_DEPLOYMENT") or os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT") or data.get("deployment", "")
    )

    return {
        "endpoint": endpoint.strip(),
        "subscription_key": subscription_key.strip(),
        "api_version": api_version.strip(),
        "deployment": deployment.strip(),
    }


def create_azure_chat_llm() -> Optional[AzureChatOpenAI]:
    settings = get_azure_openai_settings()
    endpoint = settings.get("endpoint")
    api_key = settings.get("api_key")
    deployment = settings.get("deployment")
    api_version = settings.get("api_version") or "2024-05-01-preview"

    if not endpoint or not api_key or not deployment:
        return None

    return AzureChatOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
        azure_deployment=deployment,
        temperature=0.2,
    )
