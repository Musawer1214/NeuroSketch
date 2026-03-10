from hussain_livetorch_architect.live_demo import normalize_demo_formats


def test_demo_formats_add_svg_and_json() -> None:
    out = normalize_demo_formats("dot,pdf")
    parts = out.split(",")
    assert "svg" in parts
    assert "json" in parts
    assert "pdf" in parts


def test_demo_formats_keep_existing() -> None:
    out = normalize_demo_formats("json,svg,d2")
    parts = out.split(",")
    assert parts.count("json") == 1
    assert parts.count("svg") == 1
