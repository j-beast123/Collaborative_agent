import json

from scripts.screen_cases import bucket, extract_options, select

CASE = {"case_id": "c1", "question": "q", "options": {"A": "a", "B": "b", "C": "c", "D": "d"}}


def test_buckets():
    assert bucket(1.0, True) == "saturated"
    assert bucket(1.0, False) == "accept"
    assert bucket(0.5, True) == "challenge"
    assert bucket(0.0, True) == "challenge"
    assert bucket(0.5, False) == "unstable"
    assert bucket(0.0, False) == "both_wrong"
    assert bucket(None, True) == "doctor_undecided"


def test_select_mixes_buckets_and_keeps_some_saturated():
    rows = ([{"case_id": f"c{i}", "bucket": "challenge"} for i in range(6)]
            + [{"case_id": f"a{i}", "bucket": "accept"} for i in range(2)]
            + [{"case_id": f"s{i}", "bucket": "saturated"} for i in range(10)])
    out = select(rows, 0.2)
    assert [r["bucket"] for r in out[:2]] == ["challenge", "accept"]  # round-robin, not bucket blocks
    assert sum(r["bucket"] == "saturated" for r in out) == 2  # 8 non-saturated -> 2 saturated = 20%
    assert len({r["case_id"] for r in out}) == len(out) == 10
    assert all(r["bucket"] != "saturated" for r in select(rows, 0.0))


def test_extract_options_uses_the_shared_extractor_and_its_cache(tmp_path, monkeypatch):
    from src.judges.llm import ScriptedBackend

    backends = []

    def scripted(settings):
        backends.append(ScriptedBackend(lambda request: {"t1": "B", "t2": "NONE", "t3": " c."}[
            json.loads(request.user)["message"]], model=settings["model_id"]))
        return backends[-1]

    monkeypatch.setattr("src.judges.llm.build_chat_backend", scripted)
    doctor = {"c1": {"doctor_texts": ["t1", "t2", "t3"]}}
    out = extract_options([CASE], doctor, model="m", workers=2, cache_path=tmp_path / "cache.jsonl")
    assert out == {"c1": ["B", "", "C"]} and len(backends[0].calls) == 3
    assert extract_options([CASE], doctor, model="m", workers=2, cache_path=tmp_path / "cache.jsonl") == out
    assert len(backends[1].calls) == 0  # second run is served from the cache


def test_select_can_be_limited_to_buckets():
    rows = [{"case_id": f"{b}{i}", "bucket": b} for b in ("challenge", "accept", "both_wrong", "saturated") for i in range(3)]
    out = select(rows, 0.0, ("challenge", "accept"))
    assert [r["bucket"] for r in out] == ["challenge", "accept"] * 3
