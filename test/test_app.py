from app.app import read_usage
from unittest.mock import mock_open, patch


def test_home(client):
    result = client.get("/")
    assert result.status_code == 200


@patch("builtins.open", new_callable=mock_open, read_data='test readme')
def test_read_usage(mock_file, client):
    result = client.get("/usage")
    assert result.status_code == 200
    assert result.data.decode() == "test readme"
    mock_file.assert_called_once_with('README.md', 'r')
