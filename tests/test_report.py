from src.report import generate_daily_report


def test_daily_report_note_for_mock_classifier(tmp_path) -> None:
    output_path = tmp_path / "report.md"

    generate_daily_report([], str(output_path), classifier="mock")

    assert (
        "This report uses deterministic counts from processed mock-classified data."
        in output_path.read_text(encoding="utf-8")
    )


def test_daily_report_note_for_llm_classifier(tmp_path) -> None:
    output_path = tmp_path / "report.md"

    generate_daily_report([], str(output_path), classifier="llm")

    assert (
        "This report uses deterministic counts from processed LLM-classified data."
        in output_path.read_text(encoding="utf-8")
    )
