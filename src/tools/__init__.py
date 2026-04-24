from tools.memory import memTools
from tools.todos import todo_tools
from tools.reminder import reminder_tools
from tools.call_user import call_tools
from tools.eye import screen_tools
from tools.email_tools import email_tools
from tools.browser import browser_tools

ALL_TOOLS = (
    todo_tools + 
    reminder_tools + 
    call_tools + 
    memTools +
    email_tools +
    browser_tools +
    screen_tools
)

# ─── Tool Groups for Dynamic Selection (Strategy 2A) ─────────────────────
# Each group maps a set of intent keywords to the corresponding tools.
# The keyword router in nodes.py uses these to select only relevant tools
# per turn, drastically reducing token overhead.

TOOL_GROUPS = {
    "todos": {
        "keywords": ["任务", "代办", "待办", "todo", "清单", "日程", "计划", "安排", "完成", "进度"],
        "tools": todo_tools,
    },
    "reminders": {
        "keywords": ["提醒", "闹钟", "叫醒", "定时", "reminder", "起床", "alarm"],
        "tools": reminder_tools,
    },
    "email": {
        "keywords": ["邮件", "邮箱", "email", "收件", "发件", "inbox", "写信", "回复邮件"],
        "tools": email_tools,
    },
    "screen": {
        "keywords": ["屏幕", "看看我", "在干嘛", "在做什么", "看一下屏幕", "screen", "我的电脑"],
        "tools": screen_tools,
    },
    "browser": {
        "keywords": ["浏览器", "网站", "打开网页", "browser", "moodle", "上网"],
        "tools": browser_tools,
    },
    "call": {
        "keywords": ["打电话", "拨打", "呼叫", "call", "电话"],
        "tools": call_tools,
    },
    "memory": {
        "keywords": ["记住", "记忆", "偏好", "习惯", "记得", "memory"],
        "tools": memTools,
    },
}

# Default tools when no keyword matches (most common scenario: tasks + reminders)
DEFAULT_TOOLS = todo_tools + reminder_tools + memTools
