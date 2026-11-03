from config import ABLATE, _build_ablation_config, ablation_snapshot

_ALL_FIELDS = [
    "ablate_4gram", "ablate_rescore", "ablate_embed_match",
    "ablate_frag_norm", "ablate_window", "ablate_ooc",
    "ablate_chunk_summary", "ablate_community",
]


def test_ablate_default_all_false():
    for f in _ALL_FIELDS:
        assert getattr(ABLATE, f) is False


def test_snapshot_empty_when_all_off(monkeypatch):
    for f in _ALL_FIELDS:
        monkeypatch.setattr(ABLATE, f, False)
    assert ablation_snapshot() == {}


def test_snapshot_only_true_fields(monkeypatch):
    monkeypatch.setattr(ABLATE, "ablate_4gram", True)
    monkeypatch.setattr(ABLATE, "ablate_window", True)
    monkeypatch.setattr(ABLATE, "ablate_ooc", False)
    assert ablation_snapshot() == {"ablate_4gram": True, "ablate_window": True}


def test_env_parsing(monkeypatch):
    monkeypatch.setenv("ABLATE_4GRAM", "1")
    monkeypatch.setenv("ABLATE_OOC", "true")
    monkeypatch.setenv("ABLATE_WINDOW", "0")
    monkeypatch.setenv("ABLATE_COMMUNITY", "yes")
    monkeypatch.setenv("ABLATE_RESCORE", "off")
    monkeypatch.setenv("ABLATE_FRAG_NORM", "on")
    cfg = _build_ablation_config()
    assert cfg.ablate_4gram is True
    assert cfg.ablate_ooc is True
    assert cfg.ablate_window is False
    assert cfg.ablate_community is True
    assert cfg.ablate_rescore is False
    assert cfg.ablate_frag_norm is True
