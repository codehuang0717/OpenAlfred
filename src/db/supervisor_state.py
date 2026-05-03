"""
Supervisor state repository — distraction tracking persistence.
"""

from typing import Optional
from datetime import datetime, timezone
from db.connection import get_db


async def get_supervisor_state(user_id: str = "default") -> Optional[dict]:
    async with get_db() as db:
        async with db.execute(
            "SELECT * FROM supervisor_sessions WHERE user_id = ?",
            (user_id,)
        ) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None

async def update_supervisor_state(
    user_id: str,
    is_distracted: bool,
    distraction_start_time: Optional[str] = None,
    last_alert_time: Optional[str] = None,
    consecutive_distractions: int = 0,
    last_decision: Optional[str] = None
):
    now = datetime.now(timezone.utc).isoformat()
    async with get_db() as db:
        # 1. Try to update existing session
        query_update = """
            UPDATE supervisor_sessions SET
                is_distracted = :is_distracted,
                distraction_start_time = COALESCE(:distraction_start_time, distraction_start_time),
                last_alert_time = COALESCE(:last_alert_time, last_alert_time),
                consecutive_distractions = :consecutive_distractions,
                last_decision = COALESCE(:last_decision, last_decision),
                updated_at = :updated_at
            WHERE user_id = :user_id
        """
        params = {
            "user_id": user_id,
            "is_distracted": 1 if is_distracted else 0,
            "distraction_start_time": distraction_start_time,
            "last_alert_time": last_alert_time,
            "consecutive_distractions": consecutive_distractions,
            "last_decision": last_decision,
            "updated_at": now
        }
        cursor = await db.execute(query_update, params)
        
        # 2. If no row was updated, insert a new one
        if cursor.rowcount == 0:
            query_insert = """
                INSERT INTO supervisor_sessions (
                    user_id, is_distracted, distraction_start_time, 
                    last_alert_time, consecutive_distractions, last_decision, updated_at
                ) VALUES (:user_id, :is_distracted, :distraction_start_time, :last_alert_time, :consecutive_distractions, :last_decision, :updated_at)
            """
            await db.execute(query_insert, params)
            
        await db.commit()

async def reset_supervisor_state(user_id: str):
    now = datetime.now(timezone.utc).isoformat()
    async with get_db() as db:
        await db.execute("""
            UPDATE supervisor_sessions SET
                is_distracted = 0,
                distraction_start_time = NULL,
                consecutive_distractions = 0,
                updated_at = ?
            WHERE user_id = ?
        """, (now, user_id))
        await db.commit()
