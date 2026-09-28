"""Tool failures are paired error observations, not graph-ending exceptions."""

import sys
import unittest
from pathlib import Path
from typing import Annotated, TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import ToolInvocationError

from logic.agent import tool_error_message, tool_node
from utils.auth_utils import MissingUserContextError


class TestToolErrorHarness(unittest.IsolatedAsyncioTestCase):
    async def test_parallel_failure_keeps_success_and_model_continues(self):
        @tool
        async def broken() -> str:
            """Raise a data formatting error."""
            raise ValueError("private row detail")

        @tool
        async def working() -> str:
            """Return an independent successful result."""
            return "independent result"

        class State(TypedDict):
            messages: Annotated[list, add_messages]

        async def model(state: State):
            if len(state["messages"]) == 1:
                return {"messages": [AIMessage(content="", tool_calls=[
                    {"name": "broken", "args": {}, "id": "bad"},
                    {"name": "working", "args": {}, "id": "good"},
                ])]}
            return {"messages": [AIMessage(content="The first tool failed; the other succeeded.")]}

        graph = StateGraph(State)
        graph.add_node("model", model)
        graph.add_node("tools", ToolNode([broken, working], handle_tool_errors=tool_error_message))
        graph.set_entry_point("model")
        graph.add_conditional_edges("model", lambda state: "tools" if state["messages"][-1].tool_calls else "end", {"tools": "tools", "end": END})
        graph.add_edge("tools", "model")
        result = await graph.compile().ainvoke({"messages": [HumanMessage(content="Do both")]})
        observations = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        self.assertEqual({m.tool_call_id for m in observations}, {"bad", "good"})
        self.assertEqual(next(m for m in observations if m.tool_call_id == "bad").status, "error")
        self.assertEqual(next(m for m in observations if m.tool_call_id == "good").content, "independent result")
        self.assertNotIn("private row detail", str(observations))
        self.assertIsInstance(result["messages"][-1], AIMessage)
        self.assertIn("failed", result["messages"][-1].content)

    def test_real_agent_node_uses_error_handler(self):
        self.assertIs(tool_node._handle_tool_errors, tool_error_message)

    def test_missing_authenticated_identity_still_fails_closed(self):
        with self.assertRaises(MissingUserContextError):
            tool_error_message(MissingUserContextError("no user"))

    def test_invalid_tool_arguments_remain_actionable(self):
        error = ToolInvocationError("broken", ValueError("invalid argument"), {})
        self.assertIn("工具参数", tool_error_message(error))


if __name__ == "__main__":
    unittest.main()
