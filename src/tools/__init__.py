from tools.memory import memTools
from tools.todos import todo_tools
from tools.reminder import reminder_tools
from tools.call_user import call_tools
from tools.eye import screen_tools
from tools.email_tools import email_tools
from tools.browser import browser_tools
from tools.search import search_tools

ALL_TOOLS = (
    todo_tools + 
    reminder_tools + 
    call_tools + 
    memTools +
    email_tools +
    browser_tools +
    screen_tools +
    search_tools
)

