import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # MAX_TOOL_ROUNDS counts tool-execution cycles, not total API calls.
    # Total API calls = 1 (initial in generate_response) + MAX_TOOL_ROUNDS.
    MAX_TOOL_ROUNDS = 2

    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Tool Selection Rules (follow strictly):
1. **Lesson list / course outline / "what lessons" / "show me the syllabus"** → MUST use `get_course_outline`. Never use `search_course_content` for these queries.
2. **Questions about specific course content, concepts, or explanations** → use `search_course_content`
3. **General knowledge questions unrelated to a specific course** → answer directly, no tool
4. **Complex multi-part queries** (e.g., "find a course on the same topic as lesson 4 of course X") → use up to 2 sequential tool calls: first retrieve what you need, then search using that information.

Sequential Tool Use:
- You may make up to 2 tool calls in separate request rounds before giving your final answer.
- After receiving the results of your first tool call, evaluate whether you have enough information to answer completely. If not, make one additional tool call.
- Your final response must always be a natural language answer — never a tool call.
- If a tool returns no results or an error, say so clearly and answer with what you have.

When using `get_course_outline`, present:
- The course title and link
- Every lesson as: "Lesson N: <title>"

Response rules:
- Fact-based answers only — do not guess or invent lesson titles
- If a tool yields no results, say so clearly
- No meta-commentary: no "based on the search results", no explanation of tool usage


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }

    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string. API errors are not caught here —
            they propagate to the caller (e.g. RAGSystem).
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content
        }

        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        # Get response from Claude
        response = self.client.messages.create(**api_params)

        # Handle tool execution if needed
        if response.stop_reason == "tool_use":
            if tool_manager:
                return self._handle_tool_execution(response, api_params, tool_manager)
            return "Unable to search: tool manager not configured."

        return self._extract_text(response)

    def _extract_text(self, response) -> str:
        """Extract the text from a response, returning a descriptive error string on failure."""
        if not response.content:
            return "Received empty response from AI service."
        first = response.content[0]
        if hasattr(first, "text"):
            return first.text
        if hasattr(first, "type") and first.type == "tool_use":
            return "Could not generate a text response within the allowed tool-use rounds."
        return "Received unexpected response format from AI service."

    def _handle_tool_execution(self, initial_response, initial_api_params: Dict[str, Any], tool_manager):
        """
        Handle sequential tool execution for up to MAX_TOOL_ROUNDS rounds.
        Each round is a separate API request, allowing Claude to reason about
        previous results before deciding whether to search again.

        Loop invariant: at the start of each iteration, `messages` holds the
        complete conversation so far (user turn + alternating assistant/tool-result
        pairs from prior rounds).
        """
        messages = initial_api_params["messages"].copy()
        messages.append({"role": "assistant", "content": initial_response.content})

        current_response = initial_response
        tools = initial_api_params.get("tools")

        for round_num in range(self.MAX_TOOL_ROUNDS):
            # Execute all tool calls in the current response
            tool_results = []
            for block in current_response.content:
                if block.type == "tool_use":
                    try:
                        result = tool_manager.execute_tool(block.name, **block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result
                        })
                    except Exception as exc:
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": f"Error executing tool: {exc}",
                            "is_error": True
                        })

            if not tool_results:
                return "Received malformed tool_use response: no tool_use blocks found."

            messages.append({"role": "user", "content": tool_results})

            allow_another_round = round_num < self.MAX_TOOL_ROUNDS - 1

            round_params = {
                **self.base_params,
                "messages": messages,
                "system": initial_api_params["system"]
            }
            # Keep tools available if more rounds remain so Claude can search again.
            # Omitting tools on the final round forces a text response.
            if allow_another_round and tools:
                round_params["tools"] = tools
                round_params["tool_choice"] = {"type": "auto"}

            current_response = self.client.messages.create(**round_params)

            # Stop early if Claude gave a text answer.
            # On the final round, tools are excluded so this always fires.
            if current_response.stop_reason != "tool_use":
                break

            # Claude wants another tool call — add its response and continue
            messages.append({"role": "assistant", "content": current_response.content})

        return self._extract_text(current_response)
