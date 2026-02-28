"""Tests for RAGSystem.query() and integration with VectorStore / AIGenerator."""
import pytest
import anthropic
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_rag(mocker, tmp_path):
    mocker.patch("rag_system.VectorStore", return_value=mocker.MagicMock())
    mock_ai = mocker.MagicMock()
    mock_ai.generate_response.return_value = "Here is your answer."
    mocker.patch("rag_system.AIGenerator", return_value=mock_ai)
    # Also patch DocumentProcessor so startup doesn't fail
    mocker.patch("rag_system.SessionManager", return_value=mocker.MagicMock())

    from config import Config
    from rag_system import RAGSystem

    config = Config(
        ANTHROPIC_API_KEY="fake",
        CHROMA_PATH=str(tmp_path / "chroma"),
    )
    system = RAGSystem(config)
    # Ensure tool_manager returns empty sources by default
    system.tool_manager = mocker.MagicMock()
    system.tool_manager.get_tool_definitions.return_value = []
    system.tool_manager.get_last_sources.return_value = []
    return system


# ---------------------------------------------------------------------------
# Passing tests
# ---------------------------------------------------------------------------

def test_query_returns_response_and_sources(mock_rag):
    result = mock_rag.query("What is Python?", session_id="s1")
    assert isinstance(result, tuple)
    response, sources = result
    assert isinstance(response, str)
    assert isinstance(sources, list)


def test_query_wraps_prompt_with_prefix(mock_rag):
    mock_rag.query("What is Python?", session_id="s1")
    call_kwargs = mock_rag.ai_generator.generate_response.call_args
    query_arg = call_kwargs[1].get("query") or call_kwargs[0][0]
    assert "What is Python?" in query_arg


def test_api_error_returns_friendly_message(mock_rag):
    """Fix 2: AuthenticationError is caught in RAGSystem.query() and returns a friendly tuple."""
    mock_rag.ai_generator.generate_response.side_effect = anthropic.AuthenticationError(
        message="Invalid API key",
        response=MagicMock(status_code=401),
        body={},
    )
    response, sources = mock_rag.query("What is a variable?", session_id="s1")
    assert "authentication error" in response.lower()
    assert sources == []


def test_sources_reset_after_query(mock_rag):
    mock_rag.query("What is Python?", session_id="s1")
    mock_rag.tool_manager.reset_sources.assert_called_once()


def test_session_history_updated(mock_rag):
    mock_rag.query("What is a loop?", session_id="sess42")
    mock_rag.session_manager.add_exchange.assert_called_once()
    args = mock_rag.session_manager.add_exchange.call_args[0]
    assert args[0] == "sess42"
    assert "What is a loop?" in args[1]


# ---------------------------------------------------------------------------
# Bug-reveal test — XFAIL (BUG 1)
# ---------------------------------------------------------------------------

def test_add_metadata_none_fields_succeeds(real_vector_store, sample_course_no_optionals):
    """Fix 1: None instructor/course_link are stored as empty strings, not rejected."""
    real_vector_store.add_course_metadata(sample_course_no_optionals)
    assert real_vector_store.course_catalog.count() == 1
