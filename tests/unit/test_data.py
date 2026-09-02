"""Dataset version stats, version allocation, DVC fallback, upload failures."""
from unittest.mock import patch

import pandas as pd


def test_column_stats_numeric_and_categorical():
    from app.data.service import column_stats

    df = pd.DataFrame({"a": [1, 2, None, 4], "b": ["x", "y", None, "x"], "c": [None, None, None, None]})
    stats = column_stats(df)
    assert "a" in stats and stats["a"]["mean"] is not None
    assert "b" in stats and stats["b"]["n_unique"] == 2
    assert stats["a"]["missing_pct"] == 25.0


def test_upload_fails_on_empty_file(client, api_key):
    r = client.post(
        "/datasets",
        headers={"Authorization": f"Bearer {api_key}"},
        files={"file": ("empty.csv", b"", "text/csv")},
        data={"name": "t", "target_column": "y"},
    )
    # Our route returns 400 for empty content (caught before service)
    assert r.status_code in (400, 500)


def test_version_allocation_increments(client, api_key, db):
    # Mock DVC IO to avoid real subprocess/S3; simulate two uploads of same dataset name
    csv1 = b"Id,SalePrice,x\n1,100,1\n2,200,2\n"
    csv2 = b"Id,SalePrice,x\n1,110,1\n2,210,2\n3,300,3\n"
    from unittest.mock import patch

    with patch("app.data.service.dvc_io.dvc_add", side_effect=["md5a", "md5b"]), patch(
        "app.data.service.dvc_io.dvc_push", return_value=None
    ), patch("app.data.service.dvc_io.dvc_remote_key", return_value="files/md5/ab/cd"):
        # Patch REPO write to temp so we don't touch real data/raw
        with patch("app.data.service.Path.write_bytes"), patch("app.data.service.Path.mkdir"):
            r1 = client.post(
                "/datasets",
                headers={"Authorization": f"Bearer {api_key}"},
                files={"file": ("train.csv", csv1, "text/csv")},
                data={"name": "dup-test", "target_column": "SalePrice"},
            )
            assert r1.status_code == 200, r1.text
            assert r1.json()["version"] == 1
            r2 = client.post(
                "/datasets",
                headers={"Authorization": f"Bearer {api_key}"},
                files={"file": ("train.csv", csv2, "text/csv")},
                data={"name": "dup-test", "target_column": "SalePrice"},
            )
            assert r2.status_code == 200, r2.text
            assert r2.json()["version"] == 2


def test_version_content_fallback_remote_then_cache(client, db):
    # get_version -> version_content uses dvc_io.read_content which tries remote then cache
    from app.core.models import Dataset, DatasetVersion

    ds = Dataset(name="fallback-test")
    db.add(ds)
    db.commit()
    dv = DatasetVersion(
        dataset_id=ds.id,
        version=1,
        dvc_md5="abc123",
        storage_key="files/md5/ab/c123",
        original_filename="train.csv",
        n_rows=2,
        n_cols=2,
        column_stats={},
    )
    db.add(dv)
    db.commit()
    dv_id = ds.id

    # Simulate remote miss then cache hit
    with patch("app.data.dvc_io.fetch_remote_content", side_effect=Exception("no remote")), patch(
        "app.data.dvc_io.cache_path_for"
    ) as mock_cache:
        mock_cache.return_value.read_bytes.return_value = b"a,b\n1,2\n"
        r = client.get(f"/datasets/{dv_id}/versions/1/content")
        assert r.status_code == 200
        assert b"a,b" in r.content


def test_dvc_push_failure_returns_500(client, api_key):
    csv = b"Id,SalePrice,x\n1,100,1\n"
    with patch("app.data.service.dvc_io.dvc_add", side_effect=RuntimeError("dvc add failed")), patch(
        "app.data.service.Path.write_bytes"
    ), patch("app.data.service.Path.mkdir"):
        r = client.post(
            "/datasets",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": ("train.csv", csv, "text/csv")},
            data={"name": "fail-dvc", "target_column": "SalePrice"},
        )
        assert r.status_code == 500
