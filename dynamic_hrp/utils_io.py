from pathlib import Path
from typing import Union
import pandas as pd

def save_table_markdown(df: pd.DataFrame, out_dir: Union[str, Path], name: str) -> None:
    """
    Save a DataFrame as a Markdown table (Times New Roman, 12 pt font).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.md"

    table_md = df.to_markdown(index=True)
    # styled_md = (
    #     "<div style='font-family:\"Times New Roman\"; font-size:12pt;'>\n"
    #     + table_md +
    #     "\n</div>"
    # )

    out_path.write_text(table_md, encoding="utf-8")
    print(f"Markdown table saved to: {out_path}")