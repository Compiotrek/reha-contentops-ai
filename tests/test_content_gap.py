from src.content_gap import (
    _average_linkage_clusters,
    _cluster_semantic_texts,
    _nearest_neighbor_seed_clusters,
    detect_content_gap_alerts,
)
from src.schemas import (
    ClassifiedFeedback,
    DecisionResult,
    FeedbackMessage,
    ProcessedFeedback,
)


def test_detect_content_gap_alerts_for_repeated_track_only_requests() -> None:
    items = [
        _content_request("gap_001", "Ich will Handstandübungen für Anfänger."),
        _content_request("gap_002", "Gibt es Handstand Progressions ohne Equipment?"),
        _content_request("gap_003", "Bitte mehr Übungen um einen Handstand zu lernen."),
        _content_request("gap_004", "Ich suche eine Handstand Vorbereitung an der Wand."),
        _content_request("gap_005", "Könnt ihr Handstandübungen für zuhause ergänzen?"),
    ]

    alerts = detect_content_gap_alerts(items)

    assert len(alerts) == 1
    assert alerts[0].request_count == 5
    assert alerts[0].recommended_action == "review_content_gap"
    assert alerts[0].message_ids == [
        "gap_001",
        "gap_002",
        "gap_003",
        "gap_004",
        "gap_005",
    ]


def test_detect_content_gap_alerts_ignores_existing_content_and_log_only() -> None:
    items = [
        _content_request("gap_001", "Ich will Handstandübungen für Anfänger."),
        _content_request("gap_002", "Gibt es Handstand Progressions ohne Equipment?"),
        _content_request("gap_003", "Bitte mehr Übungen um einen Handstand zu lernen."),
        _content_request("gap_004", "Ich suche eine Handstand Vorbereitung an der Wand."),
        _content_request(
            "gap_existing",
            "Könnt ihr Handstandübungen für zuhause ergänzen?",
            match_status="existing_content",
            final_action="link_existing_content",
        ),
        _non_content_feedback("gap_log", "Das Video ist super erklärt."),
    ]

    alerts = detect_content_gap_alerts(items)

    assert alerts == []


def test_detect_content_gap_alerts_ignores_possible_duplicates() -> None:
    items = [
        _content_request(
            "gap_001",
            "Ich will Handstandübungen für zuhause.",
            match_status="possible_duplicate",
            final_action="aggregate_possible_duplicate",
        ),
        _content_request(
            "gap_002",
            "Gibt es Handstand Progressions ohne Equipment?",
            match_status="possible_duplicate",
            final_action="aggregate_possible_duplicate",
        ),
        _content_request("gap_003", "Bitte mehr Übungen um einen Handstand zu lernen."),
        _content_request("gap_004", "Ich suche eine Handstand Vorbereitung an der Wand."),
        _content_request("gap_005", "Könnt ihr Handstandübungen für zuhause ergänzen?"),
    ]

    alerts = detect_content_gap_alerts(items)

    assert alerts == []


def test_detect_content_gap_alerts_does_not_merge_unrelated_track_only_topics() -> None:
    items = [
        _content_request("gap_001", "Ich will Atemübungen für zuhause."),
        _content_request("gap_002", "Gibt es Atemtraining ohne Geräte?"),
        _content_request("gap_003", "Bitte mehr Übungen für ruhige Atmung."),
        _content_request("gap_004", "Ich suche einfache Atemübungen im Sitzen."),
        _content_request("gap_005", "Könnt ihr Atemübungen für Entspannung ergänzen?"),
        _content_request("gap_006", "Ich brauche Knieübungen ohne Geräte im Liegen."),
        _content_request("gap_007", "Ich suche leichte Knieübungen mit Theraband."),
        _content_request("gap_008", "Bitte kurze Übungen fürs Knie zwischendurch."),
    ]

    alerts = detect_content_gap_alerts(items)

    assert len(alerts) == 1
    assert alerts[0].request_count == 5
    assert alerts[0].message_ids == [
        "gap_001",
        "gap_002",
        "gap_003",
        "gap_004",
        "gap_005",
    ]


def test_detect_content_gap_alerts_can_use_semantic_embeddings(monkeypatch) -> None:
    items = [
        _content_request("gap_001", "Ich will Atemübungen für zuhause."),
        _content_request("gap_002", "Breathing drills for calm evenings."),
        _content_request("gap_003", "Bitte Inhalte für ruhige Atmung."),
        _content_request("gap_004", "Mehr Atemtraining gegen Stress."),
        _content_request("gap_005", "Guided breathing routine without equipment."),
        _content_request("gap_other", "Ich suche Knieübungen ohne Geräte."),
    ]
    embeddings_by_text = {
        "gap_001": [1.0, 0.0],
        "gap_002": [0.98, 0.02],
        "gap_003": [0.97, 0.03],
        "gap_004": [0.96, 0.04],
        "gap_005": [0.95, 0.05],
        "gap_other": [0.0, 1.0],
    }

    def fake_embeddings(texts):
        result = []
        for text in texts:
            if "knie" in text:
                result.append(embeddings_by_text["gap_other"])
            elif "breathing calm" in text:
                result.append(embeddings_by_text["gap_002"])
            elif "inhalt" in text:
                result.append(embeddings_by_text["gap_003"])
            elif "gegen stress" in text:
                result.append(embeddings_by_text["gap_004"])
            elif "guided" in text:
                result.append(embeddings_by_text["gap_005"])
            else:
                result.append(embeddings_by_text["gap_001"])
        return result

    monkeypatch.setattr(
        "src.content_gap.get_embeddings",
        fake_embeddings,
    )

    alerts = detect_content_gap_alerts(items, clustering_mode="semantic")

    assert len(alerts) == 1
    assert alerts[0].request_count == 5
    assert "gap_other" not in alerts[0].message_ids


def test_semantic_content_gap_threshold_accepts_moderate_similarity(
    monkeypatch,
) -> None:
    items = [
        _content_request("gap_001", "Ich suche Übungen fürs Handgelenk."),
        _content_request("gap_002", "Wrist stretches for desk work."),
        _content_request("gap_003", "Bitte Handgelenk Mobility ohne Geräte."),
        _content_request("gap_004", "Handgelenk Übungen für Mausarm."),
        _content_request("gap_005", "Simple wrist routine between meetings."),
        _content_request("gap_other", "Ich will Atemübungen gegen Stress."),
    ]
    embeddings_by_text = {
        "gap_001": [1.0, 0.0, 0.0],
        "gap_002": [0.66, 0.75, 0.0],
        "gap_003": [0.72, 0.69, 0.0],
        "gap_004": [0.7, 0.71, 0.0],
        "gap_005": [0.65, 0.76, 0.0],
        "gap_other": [0.0, 0.0, 1.0],
    }

    def fake_embeddings(texts):
        result = []
        for text in texts:
            if "atem" in text:
                result.append(embeddings_by_text["gap_other"])
            elif "desk" in text:
                result.append(embeddings_by_text["gap_002"])
            elif "mobility" in text:
                result.append(embeddings_by_text["gap_003"])
            elif "mausarm" in text:
                result.append(embeddings_by_text["gap_004"])
            elif "meetings" in text:
                result.append(embeddings_by_text["gap_005"])
            else:
                result.append(embeddings_by_text["gap_001"])
        return result

    monkeypatch.setattr(
        "src.content_gap.get_embeddings",
        fake_embeddings,
    )

    alerts = detect_content_gap_alerts(items, clustering_mode="semantic")

    assert len(alerts) == 1
    assert alerts[0].request_count == 5
    assert "gap_other" not in alerts[0].message_ids


def test_content_gap_alert_uses_request_theme_for_semantic_cluster(
    monkeypatch,
) -> None:
    items = [
        _content_request(
            "gap_001",
            "Ich will Fußgewölbe Übungen für Läufer.",
            request_theme="plantar fascia foot arch",
        ),
        _content_request(
            "gap_002",
            "Gibt es was für Plantarfaszie nach dem Joggen?",
            request_theme="plantar fascia foot arch",
        ),
        _content_request(
            "gap_003",
            "Bitte Fußsohle Mobilisation ohne Ball.",
            request_theme="plantar fascia foot arch",
        ),
        _content_request(
            "gap_004",
            "Ich suche Übungen für Fersensporn Entlastung.",
            request_theme="plantar fascia foot arch",
        ),
        _content_request(
            "gap_005",
            "Könnt ihr Fußgewölbe Stabilität ergänzen?",
            request_theme="plantar fascia foot arch",
        ),
        _content_request(
            "gap_other",
            "Ich suche Kiefer Lockerung.",
            request_theme="jaw relaxation tmj",
        ),
    ]
    embeddings_by_theme = {
        "plantar fascia foot arch": [1.0, 0.0],
        "jaw relaxation tmj": [0.0, 1.0],
    }

    def fake_embeddings(texts):
        return [
            embeddings_by_theme[
                "jaw relaxation tmj"
                if "jaw relaxation tmj" in text
                else "plantar fascia foot arch"
            ]
            for text in texts
        ]

    monkeypatch.setattr("src.content_gap.get_embeddings", fake_embeddings)

    alerts = detect_content_gap_alerts(items, clustering_mode="semantic")

    assert len(alerts) == 1
    assert alerts[0].theme == "plantar fascia foot arch"
    assert alerts[0].message_ids == [
        "gap_001",
        "gap_002",
        "gap_003",
        "gap_004",
        "gap_005",
    ]


def test_content_gap_refines_mixed_semantic_cluster(monkeypatch) -> None:
    items = [
        *[
            _content_request(
                f"wrist_{index}",
                f"Handgelenk Mobility Wunsch {index}.",
                request_theme="wrist mobility",
            )
            for index in range(5)
        ],
        *[
            _content_request(
                f"breath_{index}",
                f"Atemtraining Wunsch {index}.",
                request_theme="breathing relaxation",
            )
            for index in range(5)
        ],
    ]
    monkeypatch.setattr(
        "src.content_gap.get_embeddings",
        lambda texts: [[1.0, 0.0] for _text in texts],
    )

    alerts = detect_content_gap_alerts(items, clustering_mode="semantic")

    assert len(alerts) == 2
    assert sorted(alert.theme for alert in alerts) == [
        "breathing relaxation",
        "wrist mobility",
    ]


def test_average_linkage_does_not_merge_through_single_bridge() -> None:
    similarities = [
        [1.0, 0.8, 0.2],
        [0.8, 1.0, 0.8],
        [0.2, 0.8, 1.0],
    ]

    clusters = _average_linkage_clusters(similarities, threshold=0.7)

    assert clusters == [[0, 1], [2]]


def test_semantic_clustering_keeps_loose_but_coherent_seed(monkeypatch) -> None:
    group_similarity = 0.60
    shared = group_similarity**0.5
    unique = (1 - group_similarity) ** 0.5
    group_vectors = [
        [shared, *(unique if index == dimension else 0.0 for dimension in range(5)), 0.0]
        for index in range(5)
    ]
    other_vector = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    monkeypatch.setattr(
        "src.content_gap.get_embeddings",
        lambda texts: [*group_vectors, other_vector],
    )

    clusters = _cluster_semantic_texts(
        ["jaw"] * 5 + ["neck"],
        semantic_threshold=0.64,
    )

    assert [0, 1, 2, 3, 4] in clusters
    assert [5] in clusters


def test_nearest_neighbor_seed_cluster_finds_coherent_minimum_group() -> None:
    similarities = [
        [1.0, 0.62, 0.58, 0.57, 0.56, 0.10],
        [0.62, 1.0, 0.55, 0.54, 0.53, 0.10],
        [0.58, 0.55, 1.0, 0.56, 0.54, 0.10],
        [0.57, 0.54, 0.56, 1.0, 0.55, 0.10],
        [0.56, 0.53, 0.54, 0.55, 1.0, 0.10],
        [0.10, 0.10, 0.10, 0.10, 0.10, 1.0],
    ]

    clusters = _nearest_neighbor_seed_clusters(
        similarities,
        min_cluster_size=5,
        cohesion_threshold=0.50,
    )

    assert [0, 1, 2, 3, 4] in clusters


def test_nearest_neighbor_seed_cluster_rejects_weak_group_cohesion() -> None:
    similarities = [
        [1.0, 0.70, 0.70, 0.70, 0.70],
        [0.70, 1.0, 0.20, 0.20, 0.20],
        [0.70, 0.20, 1.0, 0.20, 0.20],
        [0.70, 0.20, 0.20, 1.0, 0.20],
        [0.70, 0.20, 0.20, 0.20, 1.0],
    ]

    clusters = _nearest_neighbor_seed_clusters(
        similarities,
        min_cluster_size=5,
        cohesion_threshold=0.50,
    )

    assert clusters == []


def test_content_gap_alert_removes_specific_body_region_outlier(
    monkeypatch,
) -> None:
    items = [
        _content_request(
            "neck_001",
            "Ich suche Nackenentspannung im Sitzen.",
            body_region="neck",
            request_theme="neck relaxation",
        ),
        *[
            _content_request(
                f"jaw_{index}",
                f"Kiefer Entspannung Wunsch {index}.",
                body_region="jaw",
                request_theme="jaw relaxation",
            )
            for index in range(5)
        ],
    ]

    def fake_embeddings(texts):
        return [
            [0.98, 0.02] if "neck relaxation" in text else [1.0, 0.0]
            for text in texts
        ]

    monkeypatch.setattr("src.content_gap.get_embeddings", fake_embeddings)

    alerts = detect_content_gap_alerts(items, clustering_mode="semantic")

    assert len(alerts) == 1
    assert alerts[0].request_count == 5
    assert alerts[0].message_ids == [
        "jaw_0",
        "jaw_1",
        "jaw_2",
        "jaw_3",
        "jaw_4",
    ]


def _content_request(
    message_id: str,
    user_message: str,
    match_status: str = "track_only",
    final_action: str = "aggregate_content_request",
    request_theme: str | None = None,
    body_region: str | None = None,
) -> ProcessedFeedback:
    item = _processed_item(
        message_id=message_id,
        user_message=user_message,
        labels=["content_request"],
        match_status=match_status,
        final_action=final_action,
    )
    item.classification.request_theme = request_theme
    if body_region is not None:
        item.classification.body_region = body_region
    if request_theme is not None:
        item.classification.metadata_source = "llm_metadata"
    return item


def _non_content_feedback(message_id: str, user_message: str) -> ProcessedFeedback:
    return _processed_item(
        message_id=message_id,
        user_message=user_message,
        labels=["praise"],
        match_status="log_only",
        final_action="aggregate_praise",
    )


def _processed_item(
    message_id: str,
    user_message: str,
    labels: list[str],
    match_status: str,
    final_action: str,
) -> ProcessedFeedback:
    message = FeedbackMessage(message_id=message_id, user_message=user_message)
    classification = ClassifiedFeedback(
        message_id=message_id,
        user_message=user_message,
        labels=labels,
        safety_flag=False,
        evidence_quote=user_message,
        summary=user_message,
        confidence=0.9,
        routing="process",
    )
    decision = DecisionResult(
        routing="process",
        match_status=match_status,
        reason="Test decision.",
        final_action=final_action,
        review_required=False,
        priority="none",
    )
    return ProcessedFeedback(
        message=message,
        classification=classification,
        matches=[],
        decision=decision,
        classifier_mode="hybrid",
        matcher_mode="embeddings",
        routing="process",
        top_matches=[],
        match_status=match_status,
        decision_reason="Test decision.",
        final_action=final_action,
        review_required=False,
        priority="none",
    )
