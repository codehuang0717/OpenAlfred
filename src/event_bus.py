"""
EventBus — Redis-backed event system for OpenAlfred.

Provides:
  - Pub/Sub for real-time inter-process communication
  - Sorted Set delayed queue for scheduled events (reminders)
  - Graceful degradation when Redis is unavailable
"""

import json
import asyncio
from enum import Enum
from datetime import datetime, timezone
from typing import AsyncGenerator, Optional, Any

import redis.asyncio as aioredis
from utils.logger import get_logger

logger = get_logger("event_bus")

# ─── Event Types ──────────────────────────────────────────────────────────

class EventType(str, Enum):
    # Todo events
    TODO_CREATED = "todo.created"
    TODO_UPDATED = "todo.updated"
    TODO_DELETED = "todo.deleted"

    # Reminder events
    REMINDER_CREATED = "reminder.created"
    REMINDER_UPDATED = "reminder.updated"
    REMINDER_DELETED = "reminder.deleted"
    REMINDER_DUE = "reminder.due"
    REMINDER_SENT = "reminder.sent"

    # Todo notification events
    TODO_NOTIFICATION_DUE = "todo.notification_due"

    # Supervisor events
    SUPERVISOR_STATE_CHANGED = "supervisor.state_changed"


# ─── Channel constants ───────────────────────────────────────────────────

CHANNEL_PREFIX = "openalfred:"
DELAYED_QUEUE_KEY = "openalfred:delayed_events"


# ─── EventBus ─────────────────────────────────────────────────────────────

class EventBus:
    """Async Redis-backed event bus with Pub/Sub and delayed queue."""

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self._redis_url = redis_url
        self._redis: Optional[aioredis.Redis] = None
        self._connected = False
        self._connect_lock = asyncio.Lock()

    async def _ensure_connected(self):
        """Internal helper to ensure Redis is connected before any operation."""
        if self._connected and self._redis:
            return
        
        async with self._connect_lock:
            # Double check after acquiring lock
            if self._connected and self._redis:
                return
            await self.connect()

    async def connect(self):
        """Initialize the Redis connection pool."""
        try:
            self._redis = aioredis.from_url(
                self._redis_url,
                decode_responses=True,
                max_connections=20,
            )
            await self._redis.ping()
            self._connected = True
            logger.info(f"EventBus connected to Redis at {self._redis_url}")
        except Exception as e:
            self._connected = False
            logger.warning(f"EventBus failed to connect to Redis: {e}. Running in degraded mode.")

    async def close(self):
        """Close the Redis connection."""
        if self._redis:
            await self._redis.aclose()
            self._connected = False
            logger.info("EventBus disconnected from Redis")

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ─── Pub/Sub ──────────────────────────────────────────────────────

    async def publish(self, event_type: EventType, data: dict | None = None):
        """Publish an event to the Redis Pub/Sub channel."""
        await self._ensure_connected()
        if not self._connected:
            logger.warning(f"EventBus: Cannot publish {event_type.value} - NOT CONNECTED to Redis")
            return

        channel = f"{CHANNEL_PREFIX}{event_type.value}"
        payload = json.dumps({
            "type": event_type.value,
            "data": data or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        try:
            await self._redis.publish(channel, payload)
            logger.debug(f"Published {event_type.value}: {data}")
        except Exception as e:
            logger.warning(f"Failed to publish event {event_type.value}: {e}")

    async def subscribe(self, *patterns: str) -> AsyncGenerator[dict, None]:
        """Subscribe to event channels matching the given patterns."""
        await self._ensure_connected()
        if not self._connected:
            # Block forever if not connected (caller should handle reconnection)
            while True:
                await asyncio.sleep(60)
                return

        pubsub = self._redis.pubsub()
        channel_patterns = [f"{CHANNEL_PREFIX}{p}" for p in patterns]

        try:
            await pubsub.psubscribe(*channel_patterns)
            logger.info(f"Subscribed to patterns: {patterns}")

            async for message in pubsub.listen():
                if message["type"] == "pmessage":
                    try:
                        payload = json.loads(message["data"])
                        yield payload
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON in event: {message['data']}")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Subscription error: {e}")
        finally:
            await pubsub.punsubscribe(*channel_patterns)
            await pubsub.aclose()

    # ─── Delayed Queue (Sorted Set) ──────────────────────────────────

    async def schedule(self, event_type: EventType, data: dict, execute_at: str):
        """Schedule a delayed event using Redis Sorted Set."""
        await self._ensure_connected()
        if not self._connected:
            logger.warning(f"EventBus: Cannot schedule {event_type.value} - NOT CONNECTED to Redis")
            return

        try:
            dt = datetime.fromisoformat(execute_at.replace("Z", "+00:00"))
            score = dt.timestamp()
        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid execute_at '{execute_at}': {e}")
            return

        payload = json.dumps({
            "type": event_type.value,
            "data": data,
            "scheduled_for": execute_at,
        })

        try:
            await self._redis.zadd(DELAYED_QUEUE_KEY, {payload: score})
            logger.debug(f"Scheduled {event_type.value} for {execute_at}")
        except Exception as e:
            logger.warning(f"Failed to schedule event: {e}")

    async def unschedule(self, event_type: EventType, data_match: dict):
        """Remove a scheduled event matching the given type and data fields."""
        await self._ensure_connected()
        if not self._connected:
            return

        try:
            # Scan all entries (for small sets this is fine)
            entries = await self._redis.zrange(DELAYED_QUEUE_KEY, 0, -1)
            for entry in entries:
                try:
                    parsed = json.loads(entry)
                    if parsed.get("type") != event_type.value:
                        continue
                    entry_data = parsed.get("data", {})
                    if all(entry_data.get(k) == v for k, v in data_match.items()):
                        await self._redis.zrem(DELAYED_QUEUE_KEY, entry)
                        logger.debug(f"Unscheduled {event_type.value} matching {data_match}")
                        return
                except json.JSONDecodeError:
                    continue
        except Exception as e:
            logger.warning(f"Failed to unschedule event: {e}")

    async def get_due_events(self) -> list[dict]:
        """Retrieve and atomically remove all events that are now due."""
        await self._ensure_connected()
        if not self._connected:
            return []

        now_score = datetime.now(timezone.utc).timestamp()

        try:
            # Atomically get and remove due events using a Lua script
            lua_script = """
            local entries = redis.call('ZRANGEBYSCORE', KEYS[1], '-inf', ARGV[1])
            if #entries > 0 then
                redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', ARGV[1])
            end
            return entries
            """
            entries = await self._redis.eval(lua_script, 1, DELAYED_QUEUE_KEY, str(now_score))

            results = []
            for entry in entries or []:
                try:
                    parsed = json.loads(entry)
                    results.append(parsed)
                except json.JSONDecodeError:
                    continue

            if results:
                logger.info(f"Retrieved {len(results)} due events from delayed queue")
            return results

        except Exception as e:
            logger.warning(f"Failed to get due events: {e}")
            return []


# ─── Singleton ────────────────────────────────────────────────────────────

event_bus = EventBus()
