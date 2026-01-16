from __future__ import annotations

from unittest.mock import Mock, patch

from hedge_fund.data.ingest.ofac import fetch_ofac_sdn


def test_ofac_ingest_parses_rows() -> None:
    csv_payload = "a,b,c\n1,2,3\n"
    response = Mock()
    response.text = csv_payload
    response.raise_for_status = Mock()
    with patch("hedge_fund.data.ingest.ofac.requests.get", return_value=response):
        result = fetch_ofac_sdn()

    assert len(result.records) == 2
