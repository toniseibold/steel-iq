"""Recreating business cases must remove routes deleted from the master workbook."""

from pathlib import Path

import pandas as pd
import pytest

from steelo.adapters.repositories.json_repository import PrimaryFeedstockJsonRepository
from steelo.data.recreation_functions import recreate_primary_feedstock_data
from steelo.domain.models import PrimaryFeedstock


def test_recreation_removes_deleted_charcoal_routes(tmp_path: Path) -> None:
    json_path = tmp_path / "primary_feedstocks.json"
    workbook = tmp_path / "master.xlsx"
    rows = [
        {
            "Business case": business_case,
            "Metallic charge": charge,
            "Reductant": "",
            "Side": "Input",
            "Type": "feedstock",
            "Metric type": "Materials",
            "Vector": charge,
            "Value": 1.5,
            "Unit": "t/t",
        }
        for business_case in ["iron_bf", "iron_charcoal", "iron_charcoal+ccs"]
        for charge in ["io_low", "io_mid", "io_high"]
    ]
    pd.DataFrame(rows).to_excel(workbook, sheet_name="Bill of Materials", index=False)
    original = recreate_primary_feedstock_data(json_path, workbook)
    assert len(original.list()) == 9

    remaining_rows = [dict(row, Value=1.7) for row in rows if row["Business case"] == "iron_bf"]
    pd.DataFrame(remaining_rows).to_excel(workbook, sheet_name="Bill of Materials", index=False)
    rebuilt = recreate_primary_feedstock_data(json_path, workbook)

    for feedstocks in [rebuilt.list(), PrimaryFeedstockJsonRepository(json_path).list()]:
        assert len(feedstocks) == 3
        assert {pf.technology for pf in feedstocks} == {"bf"}
        assert all(pf.required_quantity_per_ton_of_product == 1.7 for pf in feedstocks)


def test_failed_workbook_read_preserves_existing_feedstocks(tmp_path: Path) -> None:
    json_path = tmp_path / "primary_feedstocks.json"
    repo = PrimaryFeedstockJsonRepository(json_path)
    repo.add(PrimaryFeedstock(technology="BF", metallic_charge="io_low", reductant="coke"))
    original = json_path.read_bytes()

    with pytest.raises(FileNotFoundError):
        recreate_primary_feedstock_data(json_path, tmp_path / "missing.xlsx")

    assert json_path.read_bytes() == original
