from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Iterable, Mapping, Any


def write_html_report(rows: Iterable[Mapping[str, Any]], path: str | Path) -> str:
    """Write a simple HTML summary report for evaluation rows."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    row_list = list(rows)
    if not row_list:
        body = "<p>No evaluation rows were supplied.</p>"
    else:
        columns = list(row_list[0].keys())
        rows_html = []
        for row in row_list:
            cells = "".join(f"<td>{escape(str(row.get(col, '')))}</td>" for col in columns)
            rows_html.append(f"<tr>{cells}</tr>")
        header = "".join(f"<th>{escape(str(col))}</th>" for col in columns)
        body = (
            "<table>"
            f"<thead><tr>{header}</tr></thead>"
            f"<tbody>{''.join(rows_html)}</tbody>"
            "</table>"
        )
    html = f"""<!DOCTYPE html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\" />
    <title>Module 5 evaluation report</title>
    <style>
      body {{ font-family: sans-serif; margin: 2rem; }}
      table {{ border-collapse: collapse; width: 100%; }}
      th, td {{ border: 1px solid #ccc; padding: 0.5rem; text-align: left; }}
      th {{ background: #f0f0f0; }}
    </style>
  </head>
  <body>
    <h1>Module 5 evaluation report</h1>
    {body}
  </body>
</html>
"""
    target.write_text(html, encoding="utf-8")
    return html
