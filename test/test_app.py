from app.app import home, read_usage
from unittest.mock import mock_open, patch


def test_home():
    assert home() == ("this sentence is already halfway over,"
                      " and still hasn't said anything at all")


@patch("builtins.open", new_callable=mock_open, read_data='test readme')
def test_read_usage(mock_file):
    assert read_usage() == "test readme"
    mock_file.assert_called_once_with('README.md', 'r')
