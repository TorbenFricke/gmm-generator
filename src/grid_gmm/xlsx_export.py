"""Export a Gaussian model as a single-row xlsx sheet (one column per parameter).

The layout has four parameter sets (``Average[1..4]``, ``Workday[1..4, 1..96]``, ...). This package
models one component, so only set 1 is filled; all other columns, including the trend columns, are
written as empty cells. ``Average[1]`` and ``Deviation[1]`` are mu and sigma in percent of P_max
(e.g. 47.35 for a ratio of 0.4735); the factors stay plain ratios. ``Cos`` is 1 (purely active load)
and ``Correlation`` is 0.

The file is written with the standard library (``zipfile``), so no Excel library is required.
"""

from __future__ import annotations

import re
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Optional, Union
from xml.sax.saxutils import escape

from grid_gmm.constants import INTERVALS_PER_DAY, MONTHS_PER_YEAR
from grid_gmm.factors import FactorProfiles
from grid_gmm.model import GaussianComponent

#: Number of parameter sets in the layout (only set 1 is filled).
NUM_SETS = 4
COS_PHI = 1.0
#: Average and Deviation are exported in percent of P_max.
PERCENT = 100.0
CORRELATION = 0.0

#: Words in a file name that describe the file rather than the load, dropped from the model name.
_FILENAME_NOISE = frozenset(
    {"hourly", "quarterly", "15min", "converted", "profile", "profiles", "config", "fitted", "raw", "kw", "w"}
)

CellValue = Optional[Union[str, float]]


def xlsx_columns() -> list[str]:
    """All column names of the export layout, in order."""
    cols = ["Name", "Cos", "Indicator", "Correlation"]
    cols += [f"Average[{s}]" for s in range(1, NUM_SETS + 1)]
    cols += [f"Deviation[{s}]" for s in range(1, NUM_SETS + 1)]
    for prefix, length in (("Workday", INTERVALS_PER_DAY), ("Weekend", INTERVALS_PER_DAY), ("Month", MONTHS_PER_YEAR)):
        cols += [f"{prefix}[{s},{i}]" for s in range(1, NUM_SETS + 1) for i in range(1, length + 1)]
    cols += [f"TrendWorkday[{i}]" for i in range(1, INTERVALS_PER_DAY + 1)]
    cols += [f"TrendWeekend[{i}]" for i in range(1, INTERVALS_PER_DAY + 1)]
    cols += [f"TrendMonth[{i}]" for i in range(1, MONTHS_PER_YEAR + 1)]
    return cols


def model_name_from_path(path: str | Path) -> str:
    """Readable model name from an input file name, e.g. ``electric_cooking_hourly.csv`` -> ``Electric Cooking``.

    Descriptive words such as "hourly", "converted" or "config" are dropped; words of up to two
    letters are written in capitals (``hp`` -> ``HP``).
    """
    words = [w for w in re.split(r"[\s_\-.]+", Path(path).stem) if w]
    kept = [w for w in words if w.lower() not in _FILENAME_NOISE] or words
    return " ".join(w.upper() if len(w) <= 2 else w.capitalize() for w in kept)


def indicator_from_name(name: str) -> str:
    """The first word of the model name."""
    parts = name.split()
    return parts[0] if parts else ""


def xlsx_row(component: GaussianComponent, factors: FactorProfiles, name: str) -> dict[str, CellValue]:
    """Values of the single data row, keyed by column name (columns not listed stay empty)."""
    row: dict[str, CellValue] = {
        "Name": name,
        "Cos": COS_PHI,
        "Indicator": indicator_from_name(name),
        "Correlation": CORRELATION,
        "Average[1]": component.mu_ratio * PERCENT,
        "Deviation[1]": component.sigma_ratio * PERCENT,
    }
    for prefix, values in (
        ("Workday", factors.weekday_factors),
        ("Weekend", factors.weekend_factors),
        ("Month", factors.month_factors),
    ):
        for i, value in enumerate(values, start=1):
            row[f"{prefix}[1,{i}]"] = float(value)
    return row


def save_model_to_xlsx(
    component: GaussianComponent,
    factors: FactorProfiles,
    filepath: str | Path,
    name: str | None = None,
) -> Path:
    """Write the model as an xlsx file with a header row and one data row.

    Args:
        name: Model name; defaults to the component name. Its first word becomes the indicator.
    """
    columns = xlsx_columns()
    row = xlsx_row(component, factors, name if name is not None else component.name)
    unknown = set(row) - set(columns)
    if unknown:  # pragma: no cover - guards against layout/row drift
        raise ValueError(f"Row has columns that are not in the layout: {sorted(unknown)}")

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_single_sheet(path, [list(columns), [row.get(col) for col in columns]])
    return path


# --- minimal SpreadsheetML writer ------------------------------------------------------------


def _column_letter(index: int) -> str:
    """0-based column index -> Excel column letters (0 -> A, 26 -> AA)."""
    letters = ""
    index += 1
    while index:
        index, rem = divmod(index - 1, 26)
        letters = chr(65 + rem) + letters
    return letters


def _cell_xml(ref: str, value: CellValue) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return f'<c r="{ref}" t="inlineStr"><is><t xml:space="preserve">{escape(value)}</t></is></c>'
    return f'<c r="{ref}"><v>{float(value)!r}</v></c>'


def _write_single_sheet(path: Path, rows: Sequence[Sequence[CellValue]]) -> None:
    sheet_rows = []
    for r, values in enumerate(rows, start=1):
        cells = "".join(_cell_xml(f"{_column_letter(c)}{r}", v) for c, v in enumerate(values))
        sheet_rows.append(f'<row r="{r}">{cells}</row>')

    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    rel_ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    pkg_rel_ns = "http://schemas.openxmlformats.org/package/2006/relationships"
    office_rel = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    header = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'

    parts = {
        "[Content_Types].xml": (
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/worksheets/sheet1.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            "</Types>"
        ),
        "_rels/.rels": (
            f'<Relationships xmlns="{pkg_rel_ns}">'
            f'<Relationship Id="rId1" Type="{office_rel}/officeDocument" Target="xl/workbook.xml"/>'
            "</Relationships>"
        ),
        "xl/workbook.xml": (
            f'<workbook xmlns="{ns}" xmlns:r="{rel_ns}">'
            '<sheets><sheet name="GM" sheetId="1" r:id="rId1"/></sheets></workbook>'
        ),
        "xl/_rels/workbook.xml.rels": (
            f'<Relationships xmlns="{pkg_rel_ns}">'
            f'<Relationship Id="rId1" Type="{office_rel}/worksheet" Target="worksheets/sheet1.xml"/>'
            "</Relationships>"
        ),
        "xl/worksheets/sheet1.xml": f'<worksheet xmlns="{ns}"><sheetData>{"".join(sheet_rows)}</sheetData></worksheet>',
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, xml in parts.items():
            zf.writestr(name, header + xml)
