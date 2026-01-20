import json
from typing import Dict, Any
from pathlib import Path
from typing import Optional


def load_config_json(path: Optional[str] = None) -> Dict[str, Any]:
    # Check for an absolute config path specified by environment variable
    if path:
        config_path = Path(path)
    else:
        config_path = Path(__file__).with_name("config.json")

    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {}