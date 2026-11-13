import json

from zubaan.provenance import package_versions, run_metadata, write_json


def test_package_versions_marks_missing_packages():
    versions = package_versions(("zubaan", "surely-not-installed-pkg"))
    assert versions["zubaan"]
    assert versions["surely-not-installed-pkg"] is None


def test_run_metadata_reads_data_source(tmp_path):
    meta = tmp_path / "flores_meta.json"
    meta.write_text(json.dumps({"source": "meta"}), encoding="utf-8")
    out = run_metadata(["some/model"], meta)
    assert out["data"]["source"] == "meta"
    assert out["model_revisions"] == {"some/model": None}
    assert {"python", "platform", "packages", "created_at"} <= out.keys()


def test_write_json_keeps_urdu_readable(tmp_path):
    path = write_json({"text": "اردو"}, tmp_path / "sub" / "out.json")
    assert "اردو" in path.read_text(encoding="utf-8")
