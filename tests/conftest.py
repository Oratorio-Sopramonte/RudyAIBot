import pytest
from unittest.mock import MagicMock
from qdrant_client import QdrantClient
from datapizza.clients.google import GoogleClient

@pytest.fixture
def mock_qdrant_client(mocker):
    """
    Mocks the QdrantClient to avoid actual database connections.
    """
    mock_client = MagicMock(spec=QdrantClient)
    # Mock search response structure if needed, or leave generic for now
    return mock_client

@pytest.fixture
def mock_gemini_client(mocker):
    """
    Mocks the GoogleClient (Gemini) to avoid actual API calls.
    """
    mock_client = MagicMock(spec=GoogleClient)
    # Default behavior: return a dummy response
    mock_client.run.return_value = "This is a mock response from Gemini."
    return mock_client

@pytest.fixture
def sample_pdf_data():
    """
    Returns a dummy string simulating PDF content.
    """
    return "This is a sample PDF content regarding the Oratorio rules. Opening hours are 8am to 8pm."

@pytest.fixture
def mock_embedding_model(mocker):
    """
    Mocks the embedding model to avoid loading heavy weights.
    """
    mock = MagicMock()
    # Mock embed to return a list of floats
    mock.embed.return_value = [[0.1] * 384] # 384 is a common dimension
    return mock
