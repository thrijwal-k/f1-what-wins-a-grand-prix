import pytest


@pytest.fixture()
def data_dir(tmp_path, monkeypatch):
    """Point every script at an empty temporary data folder."""
    monkeypatch.setenv("F1_DATA_DIR", str(tmp_path))
    return tmp_path
