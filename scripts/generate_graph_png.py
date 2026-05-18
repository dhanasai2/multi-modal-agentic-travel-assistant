import base64
import os
import sys
from pathlib import Path

import requests

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from travel_agent.graph import build_workflow  # noqa: E402


def main() -> None:
    output_dir = Path(__file__).resolve().parents[1]
    workflow = build_workflow()
    mermaid = workflow.compile().get_graph().draw_mermaid()

    (output_dir / "graph.mmd").write_text(mermaid, encoding="utf-8")

    encoded = base64.urlsafe_b64encode(mermaid.encode("utf-8")).decode("utf-8")
    url = f"https://mermaid.ink/img/{encoded}"
    png_data = requests.get(url, timeout=20).content
    (output_dir / "graph.png").write_bytes(png_data)
    print("Generated graph.mmd and graph.png")


if __name__ == "__main__":
    main()
