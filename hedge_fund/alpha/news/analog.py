"""Historical analog retrieval for news events."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from hedge_fund.data.news.models import Article


@dataclass(frozen=True)
class AnalogResult:
    """Analog retrieval result."""

    article_id: str
    score: float


def retrieve_analogs(
    *,
    query: Article,
    corpus: Iterable[Article],
    top_k: int = 5,
) -> tuple[AnalogResult, ...]:
    """Retrieve top-K analog articles by TF-IDF cosine similarity."""
    documents = [query.title + " " + (query.snippet or "")]
    corpus_list = list(corpus)
    documents.extend([c.title + " " + (c.snippet or "") for c in corpus_list])
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(documents)
    scores = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
    ranked = sorted(
        (
            AnalogResult(article_id=corpus_list[i].article_id, score=float(score))
            for i, score in enumerate(scores)
        ),
        key=lambda item: item.score,
        reverse=True,
    )
    return tuple(ranked[:top_k])
