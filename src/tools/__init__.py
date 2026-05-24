from tools.memory import memTools
from tools.todos import todo_tools
from tools.reminder import reminder_tools
from tools.call_user import call_tools
from tools.eye import screen_tools
from tools.email_tools import email_tools
from tools.search import search_tools
from tools.screenshot import screenshot_tools
from tools.rag import rag_tools

import logging

_import_logger = logging.getLogger("tools-init")

_BUILTIN_TOOLS: list = (
    todo_tools +
    reminder_tools +
    call_tools +
    memTools +
    email_tools +
    screen_tools +
    search_tools +
    screenshot_tools +
    rag_tools
)

_import_logger.debug(
    "[tools-init] Built-in tools loaded | count=%d | names=%s",
    len(_BUILTIN_TOOLS),
    [t.name for t in _BUILTIN_TOOLS],
)

ALL_TOOLS = list(_BUILTIN_TOOLS)
