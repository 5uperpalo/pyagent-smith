from typing import Optional, Dict, Union, List
import os
import subprocess
import time
import requests
import psutil
from langchain_ollama import ChatOllama

from pyagent_smith.llm.utils import load_config_json


def get_ollama_settings(path: Optional[str] = None) -> Dict[str, str]:
    """Get Ollama configuration from environment variables or config file.

    Args:
        path: Optional path to config.json file

    Returns:
        Dictionary with Ollama settings (base_url, model, temperature)
    """
    data = load_config_json(path).get("ollama", {}) or {}
    # Env overrides take precedence over file values
    base_url = os.getenv("OLLAMA_BASE_URL") or data.get("base_url", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL") or data.get("model", "llama2")
    temperature = os.getenv("OLLAMA_TEMPERATURE")

    # Parse temperature if it's a string
    if temperature:
        try:
            temperature = float(temperature)
        except ValueError:
            temperature = None

    if temperature is None:
        temperature = data.get("temperature", 0.2)

    return {
        "base_url": base_url.strip(),
        "model": model.strip(),
        "temperature": float(temperature) if temperature else 0.2,
    }


def create_ollama_chat_llm(
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: Optional[float] = None,
) -> Optional[ChatOllama]:
    """Create a ChatOllama instance for use with LangChain agents.

    Args:
        model: Optional model name (overrides config/env)
        base_url: Optional base URL for Ollama API (overrides config/env)
        temperature: Optional temperature setting (overrides config/env)

    Returns:
        ChatOllama instance if configuration is valid, None otherwise
    """
    settings = get_ollama_settings()

    # Use provided parameters or fall back to settings
    model_name = model or settings.get("model")
    ollama_base_url = base_url or settings.get("base_url")
    temp = temperature if temperature is not None else settings.get("temperature", 0.2)

    if not model_name:
        return None

    try:
        return ChatOllama(
            model=model_name,
            base_url=ollama_base_url,
            temperature=temp,
        )
    except Exception as e:
        # Log error if needed
        print(f"Failed to create Ollama LLM: {e}")
        return None


def is_ollama_running(base_url: Optional[str] = None) -> bool:
    """Check if Ollama server is running and accessible.

    Args:
        base_url: Optional base URL for Ollama API. If not provided, uses settings.

    Returns:
        True if Ollama is running and accessible, False otherwise
    """
    if base_url is None:
        settings = get_ollama_settings()
        base_url = settings.get("base_url", "http://localhost:11434")

    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def _is_ollama_service() -> bool:
    """Check if Ollama is installed as a systemd service."""
    try:
        result = subprocess.run(
            ["systemctl", "is-enabled", "ollama"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def start_ollama_server(background: bool = True) -> Union[subprocess.Popen, bool, None]:
    """Start Ollama server locally.

    Args:
        background: If True, start in background. If False, run in foreground.

    Returns:
        subprocess.Popen object if started in background, None otherwise
    """
    try:
        # Check if ollama command exists
        result = subprocess.run(
            ["which", "ollama"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode != 0:
            print("Error: 'ollama' command not found. Please install Ollama first.")
            print("\nQuick installation (Linux/WSL2):")
            print("  curl -fsSL https://ollama.com/install.sh | sh")
            print("\nOr see docs/INSTALL_OLLAMA.md for detailed instructions.")
            print("Visit https://ollama.ai for more information.")
            return None

        # Check if already running
        if is_ollama_running():
            print("Ollama is already running.")
            return None

        # Check if Ollama is installed as a systemd service
        if _is_ollama_service():
            print("Starting Ollama service (requires sudo)...")
            print("Note: Ollama is installed as a systemd service.")
            print("You may need to run: sudo systemctl start ollama")

            # Try to start via systemctl
            try:
                result = subprocess.run(
                    ["sudo", "systemctl", "start", "ollama"],
                    timeout=10
                )
                if result.returncode == 0:
                    # Wait a bit for server to start
                    time.sleep(2)
                    if is_ollama_running():
                        print("✓ Ollama service started successfully.")
                        return True  # Return True instead of Popen for service
                    else:
                        print("⚠ Service start command succeeded but server not responding yet.")
                        return True
                else:
                    print("⚠ Failed to start Ollama service. Please run manually:")
                    print("  sudo systemctl start ollama")
                    return None
            except subprocess.TimeoutExpired:
                print("⚠ Timeout starting Ollama service. Please check manually:")
                print("  sudo systemctl start ollama")
                return None
            except Exception as e:
                print(f"⚠ Error starting Ollama service: {e}")
                print("Please run manually: sudo systemctl start ollama")
                return None

        # Not a service, run directly as subprocess
        print("Starting Ollama server...")
        if background:
            # Redirect output to /dev/null to avoid pipe issues
            try:
                devnull = subprocess.DEVNULL
            except AttributeError:
                import os
                devnull = open(os.devnull, 'wb')

            process = subprocess.Popen(
                ["ollama", "serve"],
                stdout=devnull,
                stderr=devnull,
            )
            # Wait a bit for server to start
            time.sleep(2)
            if is_ollama_running():
                print("Ollama server started successfully.")
                return process
            else:
                print("Warning: Ollama process started but server not responding yet.")
                return process
        else:
            # Run in foreground (blocking)
            subprocess.run(["ollama", "serve"])
            return None
    except FileNotFoundError:
        print("Error: 'ollama' command not found. Please install Ollama first.")
        print("\nQuick installation (Linux/WSL2):")
        print("  curl -fsSL https://ollama.com/install.sh | sh")
        print("\nOr see docs/INSTALL_OLLAMA.md for detailed instructions.")
        print("Visit https://ollama.ai for more information.")
        return None
    except Exception as e:
        print(f"Error starting Ollama server: {e}")
        return None


def ensure_ollama_running(base_url: Optional[str] = None, auto_start: bool = True) -> bool:
    """Ensure Ollama server is running, optionally starting it if not.

    Args:
        base_url: Optional base URL for Ollama API
        auto_start: If True, automatically start Ollama if not running

    Returns:
        True if Ollama is running, False otherwise
    """
    if is_ollama_running(base_url):
        return True

    if auto_start:
        print("Ollama is not running. Attempting to start...")
        process = start_ollama_server(background=True)
        if process:
            # Wait a bit longer for server to be ready
            for _ in range(10):
                time.sleep(1)
                if is_ollama_running(base_url):
                    return True
            print("Warning: Ollama started but may not be ready yet.")
            return False
        return False
    else:
        print("Ollama is not running. Please start it manually with: ollama serve")
        return False


def list_ollama_models(base_url: Optional[str] = None) -> List[Dict]:
    """List all downloaded Ollama models.

    Args:
        base_url: Optional base URL for Ollama API. If not provided, uses settings.

    Returns:
        List of dictionaries containing model information (name, size, modified_at, etc.)
        Returns empty list if Ollama is not running or if there's an error.
    """
    if base_url is None:
        settings = get_ollama_settings()
        base_url = settings.get("base_url", "http://localhost:11434")

    try:
        response = requests.get(f"{base_url}/api/tags", timeout=10)
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])
            return models
        return []
    except Exception as e:
        print(f"Error listing models: {e}")
        return []


def _check_model_available(model: str, base_url: str) -> bool:
    """Check if a model is already available locally."""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            return any(model in name for name in model_names)
    except Exception:
        pass
    return False


def pull_ollama_model(model: str, base_url: Optional[str] = None) -> bool:
    """Pull/download an Ollama model if it's not already available.

    Args:
        model: Name of the model to pull (e.g., "llama2", "mistral")
        base_url: Optional base URL for Ollama API

    Returns:
        True if model is available (was already present or successfully pulled), False otherwise
    """
    if base_url is None:
        settings = get_ollama_settings()
        base_url = settings.get("base_url", "http://localhost:11434")

    # Check if model is already available
    if _check_model_available(model, base_url):
        print(f"Model '{model}' is already available.")
        return True

    # Ensure Ollama is running
    if not ensure_ollama_running(base_url, auto_start=True):
        print("Cannot pull model: Ollama server is not running.")
        return False

    # Pull the model
    print(f"Pulling model '{model}'... This may take a while...")
    try:
        result = subprocess.run(
            ["ollama", "pull", model],
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout for large models
        )
        if result.returncode == 0:
            print(f"Model '{model}' pulled successfully.")
            return True
        print(f"Error pulling model '{model}': {result.stderr}")
        return False
    except subprocess.TimeoutExpired:
        print(f"Timeout while pulling model '{model}'. The model may still be downloading.")
        return False
    except Exception as e:
        print(f"Error pulling model '{model}': {e}")
        return False


def setup_ollama_local(
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    auto_start: bool = True,
    auto_pull: bool = True,
) -> bool:
    """Complete setup for running Ollama locally: check/start server and pull model if needed.

    Args:
        model: Model name to use (defaults to config)
        base_url: Base URL for Ollama API (defaults to config)
        auto_start: Automatically start Ollama if not running
        auto_pull: Automatically pull model if not available

    Returns:
        True if setup successful, False otherwise
    """
    settings = get_ollama_settings()
    model_name = model or settings.get("model", "llama2")
    ollama_base_url = base_url or settings.get("base_url", "http://localhost:11434")

    print(f"Setting up Ollama locally (model: {model_name}, base_url: {ollama_base_url})...")

    # Ensure server is running
    if not ensure_ollama_running(ollama_base_url, auto_start=auto_start):
        return False

    # Pull model if needed
    if auto_pull:
        if not pull_ollama_model(model_name, ollama_base_url):
            print(f"Warning: Model '{model_name}' may not be available.")
            return False

    print("Ollama setup complete!")
    return True


def stop_ollama_server() -> bool:
    """Stop Ollama server if it's running.

    Returns:
        True if Ollama was stopped (or wasn't running), False if stop failed
    """
    try:
        # Try systemctl first (if installed as a service)
        result = subprocess.run(
            ["systemctl", "is-active", "ollama"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            # It's running as a service, try to stop it
            stop_result = subprocess.run(
                ["sudo", "systemctl", "stop", "ollama"],
                timeout=10
            )
            if stop_result.returncode == 0:
                print("✓ Ollama service stopped.")
                return True

        # Try service command (alternative)
        result = subprocess.run(
            ["service", "ollama", "status"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if "running" in result.stdout.lower() or result.returncode == 0:
            stop_result = subprocess.run(
                ["sudo", "service", "ollama", "stop"],
                timeout=10
            )
            if stop_result.returncode == 0:
                print("✓ Ollama service stopped.")
                return True
        
        # Try pkill (works for any ollama process)
        result = subprocess.run(
            ["sudo", "pkill", "-f", "ollama"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        # Check if it worked
        time.sleep(1)
        if not is_ollama_running():
            print("✓ Ollama stopped.")
            return True
        else:
            print("⚠ Warning: Stop command executed but Ollama may still be running.")
            print("  You may need to run: sudo pkill -9 -f ollama")
            return False
            
    except FileNotFoundError:
        # systemctl/service commands not available, try pkill
        result = subprocess.run(
            ["sudo", "pkill", "-f", "ollama"],
            capture_output=True,
            text=True,
            timeout=10
        )
        time.sleep(1)
        if not is_ollama_running():
            print("✓ Ollama stopped.")
            return True
        return False
    except subprocess.TimeoutExpired:
        print("⚠ Timeout while stopping Ollama. It may require manual intervention.")
        return False
    except Exception as e:
        print(f"⚠ Error stopping Ollama: {e}")
        print("  You may need to stop it manually with: sudo pkill -f ollama")
        return False


def get_ollama_resource_usage() -> Optional[Dict[str, float]]:
    """Get CPU and memory usage of Ollama processes.

    Returns:
        Dictionary with 'cpu_percent', 'memory_mb', 'memory_percent', and 'process_count',
        or None if Ollama is not running
    """
    try:
        ollama_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info.get('cmdline', [])
                if cmdline and 'ollama' in ' '.join(cmdline).lower():
                    ollama_processes.append(psutil.Process(proc.info['pid']))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        if not ollama_processes:
            return None
        
        total_cpu = sum(p.cpu_percent(interval=0.1) for p in ollama_processes)
        total_memory_mb = sum(p.memory_info().rss / 1024 / 1024 for p in ollama_processes)
        
        # Get system memory for percentage calculation
        try:
            system_memory = psutil.virtual_memory()
            memory_percent = (total_memory_mb / (system_memory.total / 1024 / 1024)) * 100
        except:
            memory_percent = 0.0
        
        return {
            'cpu_percent': total_cpu,
            'memory_mb': total_memory_mb,
            'memory_percent': memory_percent,
            'process_count': len(ollama_processes),
        }
    except Exception as e:
        print(f"Error getting resource usage: {e}")
        return None
