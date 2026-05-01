"""
OpenAlfred — Backward-compatible re-export layer.

All database functions have been refactored into the `db/` package.
This file re-exports everything so that existing `from database import xxx`
statements continue to work without modification.
"""

# ─── Re-export everything from the db package ─────────────────────────────

from db.connection import DATABASE_PATH, AUDIO_CACHE_DIR, init_db

from db.todo import (
    get_all_todos,
    add_todo,
    update_todo,
    delete_todo,
    get_todo_by_id,
    get_pending_todo_notifications,
    mark_todo_notification_sent,
)

from db.reminder import (
    add_reminder,
    get_pending_reminders,
    mark_reminder_sent,
    update_reminder,
    get_all_reminders,
    delete_reminder,
)

from db.user import (
    create_user,
    get_user_by_username,
    get_user_by_id,
    update_user_last_login,
    get_active_user,
)

from db.settings import set_setting, get_setting

from db.thread import get_thread_memory, set_thread_memory

from db.supervisor_state import (
    get_supervisor_state,
    update_supervisor_state,
    reset_supervisor_state,
)

from db.email_creds import (
    set_email_credentials,
    get_email_credentials,
    delete_email_credentials,
)
