"""Tests of the one-row xlsx export."""

import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pytest

from grid_gmm.cli import main
from grid_gmm.factors import FactorProfiles
from grid_gmm.model import GaussianComponent
from grid_gmm.xlsx_export import model_name_from_path, save_model_to_xlsx, xlsx_columns

_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _read_rows(path: Path) -> list[dict[str, str]]:
    """Return {column letter: text} per row, from the sheet written by the exporter."""
    with zipfile.ZipFile(path) as zf:
        root = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
    rows = []
    for row in root.iterfind(".//m:row", _NS):
        cells = {}
        for c in row.iterfind("m:c", _NS):
            col = "".join(ch for ch in c.get("r") if ch.isalpha())
            text = c.find("m:is/m:t", _NS) if c.get("t") == "inlineStr" else c.find("m:v", _NS)
            cells[col] = text.text
        rows.append(cells)
    return rows


def _as_dict(path: Path) -> dict[str, str]:
    header, data = _read_rows(path)
    return {header[col]: data[col] for col in data}


def test_columns_match_layout():
    cols = xlsx_columns()
    assert len(cols) == 1032
    assert cols[:5] == ["Name", "Cos", "Indicator", "Correlation", "Average[1]"]
    assert cols[cols.index("Workday[1,96]") + 1] == "Workday[2,1]"
    assert cols[cols.index("Weekend[4,96]") + 1] == "Month[1,1]"
    assert cols[-1] == "TrendMonth[12]"


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("input_profiles/electric_cooking_hourly.csv", "Electric Cooking"),
        ("hp_archetype_terrace_converted.csv", "HP Archetype Terrace"),
        ("results/electric_cooking_config.json", "Electric Cooking"),
        ("hourly.csv", "Hourly"),
    ],
)
def test_model_name_from_path(filename, expected):
    assert model_name_from_path(filename) == expected


def test_row_values(tmp_path: Path):
    component = GaussianComponent(p_max=3.0, mu_ratio=0.5, sigma_ratio=0.25, name="ignored")
    weekday = np.linspace(0, 0.5, 96)
    factors = FactorProfiles(np.full(12, 0.9), weekday, weekday[::-1])
    path = save_model_to_xlsx(component, factors, tmp_path / "gm.xlsx", name="Electric Cooking")

    values = _as_dict(path)
    assert values["Name"] == "Electric Cooking"
    assert values["Indicator"] == "Electric"
    assert float(values["Cos"]) == 1.0
    assert float(values["Correlation"]) == 0.0
    assert float(values["Average[1]"]) == pytest.approx(50.0)
    assert float(values["Deviation[1]"]) == pytest.approx(25.0)
    assert [float(values[f"Workday[1,{i}]"]) for i in range(1, 97)] == pytest.approx(weekday)
    assert [float(values[f"Weekend[1,{i}]"]) for i in range(1, 97)] == pytest.approx(weekday[::-1])
    assert [float(values[f"Month[1,{i}]"]) for i in range(1, 13)] == pytest.approx(np.full(12, 0.9))
    # Everything else stays empty.
    assert len(values) == 6 + 96 + 96 + 12
    assert "Average[2]" not in values and "Workday[2,1]" not in values and "TrendMonth[1]" not in values


def test_cli_save_xlsx_uses_input_file_name(tmp_path: Path, profile_csv: Path):
    xlsx = tmp_path / "out.xlsx"
    args = ["--csv", str(tmp_path / "r.csv"), "--plot", str(tmp_path / "r.png"), "--runs", "10", "--connections", "1"]
    assert main([*args, "--from-year-profile", str(profile_csv), "--save-xlsx", str(xlsx)]) == 0
    values = _as_dict(xlsx)
    assert values["Name"] == model_name_from_path(profile_csv)
    assert values["Indicator"] == values["Name"].split()[0]
