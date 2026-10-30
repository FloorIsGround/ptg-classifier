"""Export a copy with saved AI codes.

Good: uv run --frozen python scripts/export_ai_workbook.py --profile /private/review.private.json
Bad: overwrite the source or start another inference run to export existing decisions.
"""
from ptg.export.ai_workbook import main


if __name__ == "__main__":
    raise SystemExit(main())
