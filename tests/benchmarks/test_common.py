import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
common = pytest.importorskip("benchmarks.common")


def test_read_records_ignores_an_unterminated_tail(tmp_path):
    f = tmp_path / "results.jsonl"
    f.write_text(json.dumps({"a": 1}) + "\n" + json.dumps({"a": 2}) + "\n" + '{"a": 3, "b"')
    assert common.read_records(f) == [{"a": 1}, {"a": 2}]


def test_read_records_rejects_a_malformed_terminated_line(tmp_path):
    f = tmp_path / "results.jsonl"
    f.write_text(json.dumps({"a": 1}) + "\n" + "{oops\n")
    with pytest.raises(ValueError, match=":2: malformed record"):
        common.read_records(f)


def test_drop_incomplete_tail_truncates_only_an_unterminated_line(tmp_path):
    f = tmp_path / "results.jsonl"
    complete = json.dumps({"a": 1}) + "\n"
    f.write_text(complete + '{"a": 2')
    common.drop_incomplete_tail(f)
    assert f.read_text() == complete
    common.drop_incomplete_tail(f)  # already complete: unchanged
    assert f.read_text() == complete
    common.drop_incomplete_tail(tmp_path / "missing.jsonl")  # no file: nothing to do
    g = tmp_path / "only_tail.jsonl"
    g.write_text('{"a"')
    common.drop_incomplete_tail(g)
    assert g.read_text() == ""
