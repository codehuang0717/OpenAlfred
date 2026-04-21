from tools.memory import memTools
from tools.todos import todo_tools
from tools.reminder import reminder_tools
from tools.call_user import call_tools
from tools.eye import search_screen_history, get_current_screen_context
from tools.email_tools import email_tools

ALL_TOOLS = (
    todo_tools + 
    reminder_tools + 
    call_tools + 
    memTools +
    email_tools +
    [search_screen_history, get_current_screen_context]
)
