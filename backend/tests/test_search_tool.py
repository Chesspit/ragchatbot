"""Tests for CourseSearchTool.execute() and VectorStore.add_course_content()."""
import pytest
from unittest.mock import MagicMock
from vector_store import SearchResults


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_search_results(docs, metas, distances=None):
    return SearchResults(
        documents=docs,
        metadata=metas,
        distances=distances or [0.1] * len(docs),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def search_tool(mock_vector_store):
    from search_tools import CourseSearchTool
    return CourseSearchTool(mock_vector_store)


# ---------------------------------------------------------------------------
# Happy-path / mock-based tests (all PASS pre-fix)
# ---------------------------------------------------------------------------

def test_execute_returns_formatted_results(search_tool, mock_vector_store):
    mock_vector_store.search.return_value = make_search_results(
        docs=["Intro to Python."],
        metas=[{"course_title": "Python 101", "lesson_number": 1}],
    )
    result = search_tool.execute(query="Python basics")
    assert "Python 101" in result
    assert "Lesson 1" in result
    assert "Intro to Python." in result


def test_execute_empty_results(search_tool, mock_vector_store):
    mock_vector_store.search.return_value = SearchResults.empty("")
    result = search_tool.execute(query="variables")
    assert "No relevant content found" in result


def test_execute_search_error(search_tool, mock_vector_store):
    mock_vector_store.search.return_value = SearchResults.empty("Search error: timeout")
    result = search_tool.execute(query="variables")
    assert "Search error: timeout" in result


def test_execute_with_filters(search_tool, mock_vector_store):
    mock_vector_store.search.return_value = SearchResults.empty("")
    search_tool.execute(query="loops", course_name="Python 101", lesson_number=3)
    mock_vector_store.search.assert_called_once_with(
        query="loops", course_name="Python 101", lesson_number=3
    )


def test_last_sources_populated(search_tool, mock_vector_store):
    mock_vector_store.search.return_value = make_search_results(
        docs=["Content."],
        metas=[{"course_title": "ML Course", "lesson_number": 2}],
    )
    search_tool.execute(query="neural networks")
    assert search_tool.last_sources == ["ML Course - Lesson 2"]


def test_lesson_number_none_in_metadata(search_tool, mock_vector_store):
    """Chunks without lesson_number should render gracefully (no 'Lesson None')."""
    mock_vector_store.search.return_value = make_search_results(
        docs=["General overview content."],
        metas=[{"course_title": "Intro Course"}],  # no lesson_number key
    )
    result = search_tool.execute(query="overview")
    assert "None" not in result
    assert "Intro Course" in result


def test_tool_definition_format(search_tool):
    defn = search_tool.get_tool_definition()
    assert defn["name"] == "search_course_content"
    assert "query" in defn["input_schema"]["required"]


def test_search_empty_collection(real_vector_store):
    """Searching an empty ChromaDB collection returns SearchResults, not an exception."""
    results = real_vector_store.search(query="anything")
    assert isinstance(results, SearchResults)


# ---------------------------------------------------------------------------
# Bug-reveal test — XFAIL (BUG 1)
# ---------------------------------------------------------------------------

def test_add_course_content_none_lesson(real_vector_store, sample_chunk_no_lesson):
    """Fix 1: Chunk with lesson_number=None is stored without the key (not rejected)."""
    real_vector_store.add_course_content([sample_chunk_no_lesson])
    assert real_vector_store.course_content.count() == 1
