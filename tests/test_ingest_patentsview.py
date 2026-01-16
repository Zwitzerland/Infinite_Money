from __future__ import annotations

from unittest.mock import Mock, patch

from hedge_fund.data.ingest.patentsview import PatentsViewConfig, fetch_patentsview


def test_patentsview_ingest() -> None:
    response = Mock()
    response.json.return_value = {
        "patents": [
            {"patent_title": "demo", "patent_date": "2020-01-01", "patent_number": "1"}
        ]
    }
    response.raise_for_status = Mock()
    with patch("hedge_fund.data.ingest.patentsview.requests.post", return_value=response):
        result = fetch_patentsview(
            PatentsViewConfig(
                query={"_gte": {"patent_date": "2020-01-01"}},
                fields=["patent_title", "patent_date", "patent_number"],
            )
        )

    assert len(result.records) == 1
