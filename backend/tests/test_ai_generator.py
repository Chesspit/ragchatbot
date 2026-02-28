"""Tests for AIGenerator.generate_response() and tool-calling logic."""
import pytest
import anthropic
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Response builders
# ---------------------------------------------------------------------------

def make_text_response(text="Answer."):
    r = MagicMock()
    r.stop_reason = "end_turn"
    block = MagicMock()
    block.type = "text"
    block.text = text
    r.content = [block]
    return r


def make_tool_use_response(tool_name="search_course_content", tool_input=None, tool_id="tu_1"):
    r = MagicMock()
    r.stop_reason = "tool_use"
    # spec mirrors the real ToolUseBlock: has type/name/input/id but NO .text
    block = MagicMock(spec=["type", "name", "input", "id"])
    block.type = "tool_use"
    block.name = tool_name
    block.input = tool_input or {"query": "test"}
    block.id = tool_id
    r.content = [block]
    return r


def make_malformed_tool_use_response():
    """stop_reason=tool_use but content has no tool_use blocks."""
    r = MagicMock()
    r.stop_reason = "tool_use"
    block = MagicMock()
    block.type = "text"
    block.text = "unexpected text"
    r.content = [block]
    return r


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ai_generator(mocker):
    mocker.patch("ai_generator.anthropic.Anthropic", return_value=MagicMock())
    from ai_generator import AIGenerator
    gen = AIGenerator(api_key="fake", model="claude-test")
    return gen


@pytest.fixture
def mock_tool_manager():
    tm = MagicMock()
    tm.execute_tool.return_value = "Search results: Python is great."
    return tm


# ---------------------------------------------------------------------------
# Existing tests
# ---------------------------------------------------------------------------

def test_direct_text_response(ai_generator):
    ai_generator.client.messages.create.return_value = make_text_response("Hello!")
    result = ai_generator.generate_response("What is Python?")
    assert result == "Hello!"


def test_tool_use_calls_execute_tool(ai_generator, mock_tool_manager):
    ai_generator.client.messages.create.side_effect = [
        make_tool_use_response(),
        make_text_response("Based on search: Python is a language."),
    ]
    result = ai_generator.generate_response(
        query="Explain Python",
        tools=[{"name": "search_course_content"}],
        tool_manager=mock_tool_manager,
    )
    mock_tool_manager.execute_tool.assert_called_once()
    assert "Python" in result


def test_tool_results_in_message_history(ai_generator, mock_tool_manager):
    """Second API call must include a role=user message with type=tool_result."""
    ai_generator.client.messages.create.side_effect = [
        make_tool_use_response(tool_id="tu_42"),
        make_text_response("Final answer."),
    ]
    ai_generator.generate_response(
        query="question",
        tools=[{"name": "search_course_content"}],
        tool_manager=mock_tool_manager,
    )
    second_call_kwargs = ai_generator.client.messages.create.call_args_list[1][1]
    messages = second_call_kwargs.get("messages", [])
    tool_result_messages = [
        m for m in messages
        if m.get("role") == "user"
        and isinstance(m.get("content"), list)
        and any(c.get("type") == "tool_result" for c in m["content"])
    ]
    assert len(tool_result_messages) == 1


def test_conversation_history_in_system(ai_generator):
    ai_generator.client.messages.create.return_value = make_text_response()
    ai_generator.generate_response("query", conversation_history="User asked X.")
    call_kwargs = ai_generator.client.messages.create.call_args[1]
    assert "User asked X." in call_kwargs["system"]


def test_auth_error_returns_friendly_message(ai_generator):
    """Fix 2: AuthenticationError is caught and returns a friendly string."""
    # AIGenerator itself doesn't catch API errors — RAGSystem does.
    # This test documents that the error still propagates out of AIGenerator.
    ai_generator.client.messages.create.side_effect = anthropic.AuthenticationError(
        message="Invalid API key",
        response=MagicMock(status_code=401),
        body={},
    )
    with pytest.raises(anthropic.AuthenticationError):
        ai_generator.generate_response("test question")


def test_empty_content_returns_friendly_message(ai_generator):
    """Fix 3: Empty content list returns a friendly string instead of IndexError."""
    r = MagicMock()
    r.stop_reason = "end_turn"
    r.content = []
    ai_generator.client.messages.create.return_value = r
    result = ai_generator.generate_response("question")
    assert result == "Received empty response from AI service."


def test_tool_use_none_tool_manager(ai_generator):
    """Fix 3: tool_use with no tool_manager returns a safe fallback string."""
    ai_generator.client.messages.create.return_value = make_tool_use_response()
    result = ai_generator.generate_response(
        "What is Python?",
        tools=[{"name": "search_course_content"}],
        tool_manager=None,
    )
    assert isinstance(result, str)
    assert "tool manager" in result.lower()


# ---------------------------------------------------------------------------
# New tests — sequential tool calling
# ---------------------------------------------------------------------------

def test_two_sequential_tool_rounds(ai_generator, mock_tool_manager):
    """Full 2-round flow: 3 API calls, execute_tool called twice, tools excluded on last call."""
    mock_tool_manager.execute_tool.side_effect = ["Result A", "Result B"]
    ai_generator.client.messages.create.side_effect = [
        make_tool_use_response(tool_id="tu_1"),   # initial call
        make_tool_use_response(tool_id="tu_2"),   # round 0 — Claude searches again
        make_text_response("Final answer after 2 rounds."),  # round 1 — text
    ]
    result = ai_generator.generate_response(
        query="Complex question",
        tools=[{"name": "search_course_content"}],
        tool_manager=mock_tool_manager,
    )

    assert result == "Final answer after 2 rounds."
    assert ai_generator.client.messages.create.call_count == 3
    assert mock_tool_manager.execute_tool.call_count == 2

    # Round 0 (second API call, index 1): tools must be present
    round0_kwargs = ai_generator.client.messages.create.call_args_list[1][1]
    assert "tools" in round0_kwargs

    # Round 1 (third API call, index 2): tools must be absent — forced text response
    round1_kwargs = ai_generator.client.messages.create.call_args_list[2][1]
    assert "tools" not in round1_kwargs


def test_execute_tool_exception_appended_as_error_result(ai_generator, mock_tool_manager):
    """execute_tool exception must not propagate; error is sent as a tool_result so Claude can respond."""
    mock_tool_manager.execute_tool.side_effect = RuntimeError("Connection timeout")
    ai_generator.client.messages.create.side_effect = [
        make_tool_use_response(tool_id="tu_err"),
        make_text_response("I couldn't retrieve that information."),
    ]
    result = ai_generator.generate_response(
        query="Search something",
        tools=[{"name": "search_course_content"}],
        tool_manager=mock_tool_manager,
    )

    # No exception propagated
    assert result == "I couldn't retrieve that information."

    # The second API call must include the error as a tool_result
    second_call_kwargs = ai_generator.client.messages.create.call_args_list[1][1]
    messages = second_call_kwargs["messages"]
    tool_result_msgs = [
        m for m in messages
        if m.get("role") == "user" and isinstance(m.get("content"), list)
    ]
    assert len(tool_result_msgs) == 1
    error_result = tool_result_msgs[0]["content"][0]
    assert error_result["tool_use_id"] == "tu_err"
    assert error_result.get("is_error") is True
    assert "Connection timeout" in error_result["content"]


def test_empty_tool_results_returns_error(ai_generator, mock_tool_manager):
    """stop_reason=tool_use but no tool_use blocks in content → specific error, no second API call."""
    ai_generator.client.messages.create.return_value = make_malformed_tool_use_response()
    result = ai_generator.generate_response(
        query="question",
        tools=[{"name": "search_course_content"}],
        tool_manager=mock_tool_manager,
    )

    assert "malformed" in result.lower() or "no tool_use" in result.lower()
    assert ai_generator.client.messages.create.call_count == 1
    mock_tool_manager.execute_tool.assert_not_called()


def test_max_rounds_exhausted_returns_specific_error(ai_generator, mock_tool_manager):
    """When the final response is still tool_use, the error is distinct from 'empty response'."""
    ai_generator.client.messages.create.side_effect = [
        make_tool_use_response(tool_id="tu_1"),  # initial
        make_tool_use_response(tool_id="tu_2"),  # round 0
        make_tool_use_response(tool_id="tu_3"),  # round 1 (final) — still tool_use
    ]
    result = ai_generator.generate_response(
        query="question",
        tools=[{"name": "search_course_content"}],
        tool_manager=mock_tool_manager,
    )

    assert result == "Could not generate a text response within the allowed tool-use rounds."
    # Must be distinct from the empty-content error
    assert result != "Received empty response from AI service."
    assert ai_generator.client.messages.create.call_count == 3
