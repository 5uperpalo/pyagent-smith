import json
from typing import Dict, Any
from pathlib import Path
from typing import Optional


def load_config_json(path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from a JSON file.

    Args:
        path: Optional path to config.json file. If not provided, looks for
              config.json in the same directory as this file.

    Returns:
        Dictionary with configuration data, or empty dict if file not found
    """
    if path:
        config_path = Path(path)
    else:
        # Look for config.json in the parent directory (pyagent-smith root)
        config_path = Path(__file__).parent.parent / "config.json"

    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
