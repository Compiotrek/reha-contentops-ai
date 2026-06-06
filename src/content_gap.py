"""MVP content-gap alerting for repeated unmatched content requests.

This module intentionally produces review signals, not autonomous content
decisions. It clusters only processed items that were already routed as
`track_only` content requests, then emits alerts when repeated demand crosses
the configured threshold.
"""

from collections import Counter, deque
from dataclasses import dataclass
from hashlib import sha256

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import (
    CONTENT_GAP_MIN_REQUESTS,
    CONTENT_GAP_REFINE_SEMANTIC_SIMILARITY_THRESHOLD,
    CONTENT_GAP_SEMANTIC_NEIGHBORHOOD_COHESION_THRESHOLD,
    CONTENT_GAP_SEMANTIC_SIMILARITY_THRESHOLD,
    CONTENT_GAP_SEMANTIC_SEED_SIMILARITY_THRESHOLD,
    CONTENT_GAP_SIMILARITY_THRESHOLD,
    CONTENT_GAP_STRONG_TEXT_SIMILARITY_THRESHOLD,
)
from src.match import get_embeddings
from src.schemas import ProcessedFeedback
from src.text_signals import extract_topic_tokens


CONTENT_GAP_GENERIC_TOPICS = {
    "advanced",
    "anfaenger",
    "beginner",
    "coordination",
    "endurance",
    "exercises",
    "general",
    "intermediate",
    "mobility",
    "mobilization",
    "none",
    "relaxation",
    "stability",
    "strength",
}
GENERAL_BODY_REGIONS = {"", "general"}


@dataclass(frozen=True)
class ContentGapAlert:
    alert_id: str
    request_count: int
    theme: str
    message_ids: list[str]
    example_messages: list[str]
    body_regions: list[str]
    therapy_goals: list[str]
    difficulty_requested: list[str]
    equipment: list[str]
    recommended_action: str = "review_content_gap"


def detect_content_gap_alerts(
    processed_items: list[ProcessedFeedback],
    min_requests: int = CONTENT_GAP_MIN_REQUESTS,
    similarity_threshold: float = CONTENT_GAP_SIMILARITY_THRESHOLD,
    clustering_mode: str = "auto",
) -> list[ContentGapAlert]:
    candidates = _content_gap_candidates(processed_items)
    if len(candidates) < min_requests:
        return []

    signature_texts = [_topic_signature_text(item) for item in candidates]
    clusters = _cluster_content_gap_texts(
        texts=signature_texts,
        similarity_threshold=similarity_threshold,
        clustering_mode=clustering_mode,
    )
    clusters = _refine_semantic_clusters(
        candidates,
        clusters,
        min_requests,
        signature_texts,
    )
    alerts = _alerts_from_clusters(candidates, clusters, min_requests)
    if clustering_mode == "auto" and not alerts:
        lexical_clusters = _cluster_lexical_texts(signature_texts, similarity_threshold)
        alerts = _alerts_from_clusters(candidates, lexical_clusters, min_requests)
    alerts.sort(key=lambda alert: (-alert.request_count, alert.alert_id))
    return alerts


def _alerts_from_clusters(
    candidates: list[ProcessedFeedback],
    clusters: list[list[int]],
    min_requests: int,
) -> list[ContentGapAlert]:
    alerts = []
    for cluster in clusters:
        if len(cluster) < min_requests:
            continue
        items = _remove_topic_outliers([candidates[index] for index in cluster])
        if len(items) >= min_requests:
            alerts.append(_build_alert(items))
    return alerts


def _cluster_content_gap_texts(
    texts: list[str],
    similarity_threshold: float,
    clustering_mode: str,
) -> list[list[int]]:
    if clustering_mode == "semantic":
        return _cluster_semantic_texts(texts)
    if clustering_mode == "lexical":
        return _cluster_lexical_texts(texts, similarity_threshold)
    if clustering_mode != "auto":
        raise ValueError(f"Unsupported content gap clustering mode: {clustering_mode}")

    try:
        return _cluster_semantic_texts(texts)
    except Exception:
        return _cluster_lexical_texts(texts, similarity_threshold)


def _content_gap_candidates(
    processed_items: list[ProcessedFeedback],
) -> list[ProcessedFeedback]:
    return [
        item
        for item in processed_items
        if item.decision.match_status == "track_only"
        and "content_request" in item.classification.labels
        and not item.classification.safety_flag
    ]


def _cluster_semantic_texts(
    texts: list[str],
    semantic_threshold: float = CONTENT_GAP_SEMANTIC_SIMILARITY_THRESHOLD,
) -> list[list[int]]:
    if not texts:
        return []
    if len(texts) == 1:
        return [[0]]

    embeddings = get_embeddings(texts)
    similarities = cosine_similarity(embeddings)
    strict_clusters = _average_linkage_clusters(similarities, semantic_threshold)
    seed_clusters = _average_linkage_clusters(
        similarities,
        CONTENT_GAP_SEMANTIC_SEED_SIMILARITY_THRESHOLD,
    )
    neighborhood_clusters = _nearest_neighbor_seed_clusters(
        similarities,
        min_cluster_size=CONTENT_GAP_MIN_REQUESTS,
        cohesion_threshold=CONTENT_GAP_SEMANTIC_NEIGHBORHOOD_COHESION_THRESHOLD,
    )
    return _dedupe_overlapping_clusters(
        [*strict_clusters, *seed_clusters, *neighborhood_clusters],
        similarities,
    )


def _cluster_lexical_texts(
    texts: list[str],
    similarity_threshold: float,
) -> list[list[int]]:
    if not texts:
        return []
    if len(texts) == 1:
        return [[0]]

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(4, 8),
        lowercase=True,
        min_df=1,
    )
    topic_tokens = [extract_topic_tokens(text) for text in texts]
    lexical_texts = [
        " ".join(sorted(_meaningful_gap_topics(tokens))) or text
        for text, tokens in zip(texts, topic_tokens)
    ]
    matrix = vectorizer.fit_transform(lexical_texts)
    similarities = cosine_similarity(matrix)

    def is_neighbor(index: int, neighbor_index: int) -> bool:
        return _is_same_gap_topic(
            score=similarities[index][neighbor_index],
            similarity_threshold=similarity_threshold,
            tokens_a=topic_tokens[index],
            tokens_b=topic_tokens[neighbor_index],
        )

    return _connected_components_by_neighbor(texts, is_neighbor)


def _connected_components(similarities, threshold: float) -> list[list[int]]:
    def is_neighbor(index: int, neighbor_index: int) -> bool:
        return similarities[index][neighbor_index] >= threshold

    return _connected_components_by_neighbor(
        range(similarities.shape[0]),
        is_neighbor,
    )


def _average_linkage_clusters(similarities, threshold: float) -> list[list[int]]:
    item_count = (
        similarities.shape[0] if hasattr(similarities, "shape") else len(similarities)
    )
    clusters = [[index] for index in range(item_count)]
    while True:
        best_pair = None
        best_score = threshold
        for left_index in range(len(clusters)):
            for right_index in range(left_index + 1, len(clusters)):
                score = _average_cluster_similarity(
                    similarities,
                    clusters[left_index],
                    clusters[right_index],
                )
                if score > best_score:
                    best_score = score
                    best_pair = (left_index, right_index)
        if best_pair is None:
            break

        left_index, right_index = best_pair
        merged = sorted([*clusters[left_index], *clusters[right_index]])
        clusters = [
            cluster
            for index, cluster in enumerate(clusters)
            if index not in {left_index, right_index}
        ]
        clusters.append(merged)
    return sorted(clusters, key=lambda cluster: cluster[0])


def _average_cluster_similarity(similarities, left: list[int], right: list[int]) -> float:
    scores = [
        similarities[left_index][right_index]
        for left_index in left
        for right_index in right
    ]
    return sum(scores) / len(scores)


def _nearest_neighbor_seed_clusters(
    similarities,
    min_cluster_size: int,
    cohesion_threshold: float,
) -> list[list[int]]:
    item_count = (
        similarities.shape[0] if hasattr(similarities, "shape") else len(similarities)
    )
    if item_count < min_cluster_size:
        return []

    clusters: list[list[int]] = []
    for index in range(item_count):
        nearest_indexes = sorted(
            (neighbor_index for neighbor_index in range(item_count) if neighbor_index != index),
            key=lambda neighbor_index: similarities[index][neighbor_index],
            reverse=True,
        )
        cluster = sorted([index, *nearest_indexes[: min_cluster_size - 1]])
        if _cluster_cohesion(similarities, cluster) >= cohesion_threshold:
            clusters.append(cluster)
    return clusters


def _dedupe_overlapping_clusters(
    clusters: list[list[int]],
    similarities,
) -> list[list[int]]:
    unique_clusters = {
        tuple(sorted(cluster))
        for cluster in clusters
        if cluster
    }
    alert_sized_clusters = [
        list(cluster)
        for cluster in unique_clusters
        if len(cluster) >= CONTENT_GAP_MIN_REQUESTS
    ]
    smaller_clusters = [
        list(cluster)
        for cluster in unique_clusters
        if len(cluster) < CONTENT_GAP_MIN_REQUESTS
    ]
    ranked_clusters = [
        *sorted(
            alert_sized_clusters,
            key=lambda cluster: (-_cluster_cohesion(similarities, cluster), -len(cluster)),
        ),
        *sorted(smaller_clusters, key=lambda cluster: (-len(cluster), cluster[0])),
    ]
    selected_clusters: list[list[int]] = []
    used_indexes: set[int] = set()
    for cluster in ranked_clusters:
        if used_indexes.intersection(cluster):
            continue
        selected_clusters.append(cluster)
        used_indexes.update(cluster)
    return sorted(selected_clusters, key=lambda cluster: cluster[0])


def _cluster_cohesion(similarities, cluster: list[int]) -> float:
    if len(cluster) < 2:
        return 1.0
    scores = [
        similarities[left_index][right_index]
        for position, left_index in enumerate(cluster)
        for right_index in cluster[position + 1 :]
    ]
    return sum(scores) / len(scores)


def _connected_components_by_neighbor(items, is_neighbor) -> list[list[int]]:
    visited = set()
    clusters: list[list[int]] = []
    item_count = len(items)
    for start_index in range(item_count):
        if start_index in visited:
            continue
        cluster = []
        queue = deque([start_index])
        visited.add(start_index)
        while queue:
            index = queue.popleft()
            cluster.append(index)
            neighbors = [
                neighbor_index
                for neighbor_index in range(item_count)
                if neighbor_index not in visited and is_neighbor(index, neighbor_index)
            ]
            for neighbor_index in neighbors:
                visited.add(neighbor_index)
                queue.append(neighbor_index)
        clusters.append(sorted(cluster))
    return clusters


def _has_shared_topic(tokens_a: set[str], tokens_b: set[str]) -> bool:
    meaningful_a = _meaningful_gap_topics(tokens_a)
    meaningful_b = _meaningful_gap_topics(tokens_b)
    return bool(meaningful_a and meaningful_b and meaningful_a.intersection(meaningful_b))


def _meaningful_gap_topics(tokens: set[str]) -> set[str]:
    return {token for token in tokens if token not in CONTENT_GAP_GENERIC_TOPICS}


def _is_same_gap_topic(
    score: float,
    similarity_threshold: float,
    tokens_a: set[str],
    tokens_b: set[str],
) -> bool:
    if _has_shared_topic(tokens_a, tokens_b):
        return True
    return score >= max(
        similarity_threshold,
        CONTENT_GAP_STRONG_TEXT_SIMILARITY_THRESHOLD,
    )


def _build_alert(items: list[ProcessedFeedback]) -> ContentGapAlert:
    message_ids = [item.message.message_id for item in items]
    return ContentGapAlert(
        alert_id=_alert_id(message_ids),
        request_count=len(items),
        theme=_theme(items),
        message_ids=message_ids,
        example_messages=[item.message.user_message for item in items[:3]],
        body_regions=_metadata_values(items, "body_region"),
        therapy_goals=_metadata_values(items, "therapy_goal"),
        difficulty_requested=_metadata_values(items, "difficulty_requested"),
        equipment=_metadata_values(items, "equipment"),
    )


def _remove_topic_outliers(
    items: list[ProcessedFeedback],
) -> list[ProcessedFeedback]:
    if len(items) < CONTENT_GAP_MIN_REQUESTS + 1:
        return items

    kept_items = []
    for item in items:
        other_items = [other for other in items if other is not item]
        if _is_topic_outlier(item, other_items):
            continue
        kept_items.append(item)
    return kept_items


def _is_topic_outlier(
    item: ProcessedFeedback,
    other_items: list[ProcessedFeedback],
) -> bool:
    body_region = item.classification.body_region or ""
    if body_region in GENERAL_BODY_REGIONS:
        return False

    same_region_count = sum(
        1
        for other in other_items
        if (other.classification.body_region or "") == body_region
    )
    if same_region_count > 0:
        return False

    item_topics = _meaningful_gap_topics(
        extract_topic_tokens(_topic_signature_text(item))
    )
    other_topic_counts = Counter(
        topic
        for other in other_items
        for topic in _meaningful_gap_topics(
            extract_topic_tokens(_topic_signature_text(other))
        )
    )
    repeated_other_topics = {
        topic for topic, count in other_topic_counts.items() if count >= 2
    }
    if not repeated_other_topics:
        return False
    return not item_topics.intersection(repeated_other_topics)


def _content_gap_text(item: ProcessedFeedback) -> str:
    classification = item.classification
    parts = [
        item.message.user_message,
        classification.evidence_quote,
    ]
    if _is_semantic_summary(classification.classifier_source):
        parts.append(classification.summary)
    request_theme = _trusted_request_theme(item)
    if request_theme:
        parts.extend([request_theme] * 3)
    metadata_parts = [
        classification.body_region,
        classification.therapy_goal,
        classification.difficulty_requested,
        classification.equipment,
        classification.position,
    ]
    parts.extend(value for value in metadata_parts if value)
    topic_tokens = sorted(extract_topic_tokens(" ".join(parts)))
    if topic_tokens:
        parts.append("topic: " + " ".join(topic_tokens))
    return " ".join(part for part in parts if part)


def _topic_signature_text(item: ProcessedFeedback) -> str:
    classification = item.classification
    parts = []
    request_theme = _trusted_request_theme(item)
    if request_theme:
        parts.extend([request_theme] * 4)
    parts.extend(
        value
        for value in [
            classification.body_region,
            classification.therapy_goal,
            classification.difficulty_requested,
            classification.equipment,
            classification.position,
        ]
        if value
    )
    parts.extend(sorted(extract_topic_tokens(item.message.user_message)))
    return " ".join(parts) or item.message.user_message


def _refine_semantic_clusters(
    candidates: list[ProcessedFeedback],
    clusters: list[list[int]],
    min_requests: int,
    signature_texts: list[str],
) -> list[list[int]]:
    refined_clusters: list[list[int]] = []
    for cluster in clusters:
        if len(cluster) < min_requests:
            refined_clusters.append(cluster)
            continue

        signature_subset = [signature_texts[index] for index in cluster]
        semantic_splits = _cluster_signature_texts_semantically(signature_subset)
        large_semantic_splits = [
            [cluster[index] for index in split_cluster]
            for split_cluster in semantic_splits
            if len(split_cluster) >= min_requests
        ]
        if large_semantic_splits and len(large_semantic_splits) != 1:
            refined_clusters.extend(large_semantic_splits)
            continue

        split_clusters = _cluster_lexical_texts(
            signature_subset,
            CONTENT_GAP_SIMILARITY_THRESHOLD,
        )
        large_splits = [
            [cluster[index] for index in split_cluster]
            for split_cluster in split_clusters
            if len(split_cluster) >= min_requests
        ]
        if large_splits:
            refined_clusters.extend(large_splits)
        elif len(split_clusters) > 1 and len(cluster) > min_requests:
            continue
        else:
            refined_clusters.append(cluster)
    return refined_clusters


def _cluster_signature_texts_semantically(texts: list[str]) -> list[list[int]]:
    try:
        return _cluster_semantic_texts(
            texts,
            semantic_threshold=CONTENT_GAP_REFINE_SEMANTIC_SIMILARITY_THRESHOLD,
        )
    except Exception:
        return []


def _theme(items: list[ProcessedFeedback]) -> str:
    request_theme_counts = Counter(
        _trusted_request_theme(item)
        for item in items
        if _trusted_request_theme(item)
    )
    if request_theme_counts:
        return request_theme_counts.most_common(1)[0][0]

    topic_counts = Counter(
        topic
        for item in items
        for topic in extract_topic_tokens(_content_gap_text(item))
    )
    ranked_topics = [
        topic for topic, _count in topic_counts.most_common() if len(topic) >= 4
    ]
    return ", ".join(ranked_topics[:3]) if ranked_topics else "content request cluster"


def _trusted_request_theme(item: ProcessedFeedback) -> str | None:
    if item.classification.metadata_source in {"llm_metadata", "llm_classifier"}:
        return item.classification.request_theme
    if item.classification.classifier_source in {"llm", "hybrid_llm"}:
        return item.classification.request_theme
    return None


def _is_semantic_summary(classifier_source: str | None) -> bool:
    return classifier_source in {"llm", "hybrid_llm"}


def _metadata_values(items: list[ProcessedFeedback], field: str) -> list[str]:
    counts = Counter(
        value
        for item in items
        if (value := getattr(item.classification, field)) is not None
    )
    return [value for value, _count in counts.most_common()]


def _alert_id(message_ids: list[str]) -> str:
    digest = sha256(";".join(sorted(message_ids)).encode("utf-8")).hexdigest()[:10]
    return f"gap_{digest}"
