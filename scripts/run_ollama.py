#!/usr/bin/env python3
"""Script to run Ollama locally and set it up for use with pyagent-smith."""

import sys
import argparse
from pathlib import Path

# Add parent directory to path to import pyagent_smith
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyagent_smith.llm import (
    setup_ollama_local,
    is_ollama_running,
    start_ollama_server,
    pull_ollama_model,
    get_ollama_settings,
    stop_ollama_server,
    get_ollama_resource_usage,
    list_ollama_models,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run and setup Ollama locally for pyagent-smith"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name to use (e.g., llama2, mistral). Defaults to config.json",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        help="Base URL for Ollama API. Defaults to http://localhost:11434",
    )
    parser.add_argument(
        "--no-auto-start",
        action="store_true",
        help="Don't automatically start Ollama if not running",
    )
    parser.add_argument(
        "--no-auto-pull",
        action="store_true",
        help="Don't automatically pull model if not available",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check if Ollama is running, don't start it",
    )
    parser.add_argument(
        "--start-only",
        action="store_true",
        help="Only start Ollama server, don't pull models",
    )
    parser.add_argument(
        "--pull-only",
        type=str,
        metavar="MODEL",
        help="Only pull the specified model, don't start server",
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Stop Ollama server if it's running",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show Ollama status and resource usage (CPU, RAM)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all downloaded Ollama models",
    )

    args = parser.parse_args()

    # Handle specific actions
    if args.stop:
        if stop_ollama_server():
            sys.exit(0)
        else:
            sys.exit(1)

    if args.status:
        base_url = args.base_url or get_ollama_settings().get("base_url", "http://localhost:11434")
        is_running = is_ollama_running(base_url)
        if is_running:
            print("✓ Ollama is running and accessible.")
            print(f"  Base URL: {base_url}")

            # Get resource usage
            usage = get_ollama_resource_usage()
            if usage:
                print("\nResource Usage:")
                print(f"  Processes: {usage['process_count']}")
                print(f"  CPU: {usage['cpu_percent']:.1f}%")
                print(f"  Memory: {usage['memory_mb']:.1f} MB ({usage['memory_percent']:.1f}% of system)")
            else:
                print("  (Could not retrieve resource usage)")
        else:
            print("✗ Ollama is not running.")
        sys.exit(0 if is_running else 1)

    if args.check_only:
        base_url = args.base_url or get_ollama_settings().get("base_url", "http://localhost:11434")
        if is_ollama_running(base_url):
            print("✓ Ollama is running and accessible.")
            sys.exit(0)
        else:
            print("✗ Ollama is not running.")
            sys.exit(1)

    if args.start_only:
        process = start_ollama_server(background=True)
        if process:
            print("✓ Ollama server started.")
            sys.exit(0)
        else:
            print("✗ Failed to start Ollama server.")
            sys.exit(1)

    if args.list_models:
        base_url = args.base_url or get_ollama_settings().get("base_url", "http://localhost:11434")
        if not is_ollama_running(base_url):
            print("✗ Ollama is not running. Please start it first.")
            sys.exit(1)
        
        models = list_ollama_models(base_url)
        if models:
            print(f"\nDownloaded Ollama models ({len(models)}):")
            print("-" * 60)
            for model in models:
                name = model.get("name", "unknown")
                size = model.get("size", 0)
                # Format size in GB
                size_gb = size / (1024 ** 3) if size > 0 else 0
                modified = model.get("modified_at", "")
                print(f"  • {name}")
                if size_gb > 0:
                    print(f"    Size: {size_gb:.2f} GB")
                if modified:
                    print(f"    Modified: {modified}")
                print()
        else:
            print("No models downloaded yet.")
            print("Use --pull-only <model_name> to download a model.")
        sys.exit(0)

    if args.pull_only:
        base_url = args.base_url or get_ollama_settings().get("base_url", "http://localhost:11434")
        if pull_ollama_model(args.pull_only, base_url):
            print(f"✓ Model '{args.pull_only}' is ready.")
            sys.exit(0)
        else:
            print(f"✗ Failed to pull model '{args.pull_only}'.")
            sys.exit(1)

    # Full setup
    success = setup_ollama_local(
        model=args.model,
        base_url=args.base_url,
        auto_start=not args.no_auto_start,
        auto_pull=not args.no_auto_pull,
    )

    if success:
        print("\n✓ Ollama is ready to use!")
        print("\nYou can now use it in your code:")
        print("  from pyagent_smith.llm import create_ollama_chat_llm")
        print("  llm = create_ollama_chat_llm()")
        sys.exit(0)
    else:
        print("\n✗ Setup incomplete. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
