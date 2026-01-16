from __future__ import annotations

from datetime import datetime, timezone

from hedge_fund.data.news.normalize import RawArticle, normalize_article


def test_normalize_article_hash() -> None:
    raw = RawArticle(
        source="GDELT",
        url="https://example.test/path?x=1",
        published_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        language="en",
        title="Test Title",
        snippet="Snippet",
        payload={},
    )
    article = normalize_article(raw)
    assert article.article_id
    assert article.url == "https://example.test/path"
