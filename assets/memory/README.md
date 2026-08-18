# OpenAlfred L1 Local Memory

OpenAlfred stores long-term user memory as local Markdown files under
`memory/{user_id}/`. These files are managed by `agent/src/logic/memory_manager.py`
and are isolated per user.

## Architecture

```text
load_context_node
  -> MemoryManager.build_injection_text(user_id)
      -> profile.md
      -> preferences.md
  -> system_instruction

agent_node
  -> bound memory tools
      -> get_user_profile
      -> get_user_memory_category
      -> update_user_memory

extract_knowledge_node
  -> read existing L1 memories
  -> LLM structured fact extraction
  -> append categorized facts to Markdown files
```

## Files

| File | Purpose | Injection |
|------|---------|-----------|
| `profile.md` | Core user facts: name, identity, important dates, stable background | Injected every turn |
| `preferences.md` | User preferences, likes/dislikes, interests, working style | Injected every turn |
| `relationship.md` | Relationship context and interaction history | Read on demand by tool |
| `learned_patterns.md` | Behavioral patterns, habits, recurring workflows | Read on demand by tool |

`_templates/` contains the default files copied when a user memory directory is
created for the first time.

## Prompt Injection

Only `profile.md` and `preferences.md` are injected automatically:

```text
[用户长期记忆]
---
## 用户基本信息
- ...

## 用户偏好
- ...
---
```

Relationship and behavior-pattern memories are intentionally kept out of the
default prompt and are loaded only when the model calls `get_user_memory_category`.

## Write Paths

Memory can be written in two ways:

1. `update_user_memory` appends an explicit fact to one category.
2. `extract_knowledge_node` periodically reviews recent conversation turns and
   appends durable facts to the matching Markdown file.

Both paths write local Markdown only.

## Relevant Config

```bash
# Extract knowledge every N turns
EXTRACTION_INTERVAL=3

# Ignore very short user messages during extraction
EXTRACTION_MIN_MSG_LENGTH=8
```

## Debugging

```bash
# View extraction logs
cat logs/graph-nodes.log

# View user memory files
cat memory/{user_id}/profile.md
cat memory/{user_id}/preferences.md
cat memory/{user_id}/relationship.md
cat memory/{user_id}/learned_patterns.md
```
