# Agent completion contract

HTTP success and an absent tool call do not establish a completed answer.
`logic.agent_outcome.classify_response` validates each complete generation before
the graph dispatches tools or performs memory extraction.

| Outcome | Graph action |
| --- | --- |
| Valid tool calls with a completion signal | Execute the paired tools once, then prepare context again |
| Completed visible answer or refusal | Extract eligible user knowledge, then finish |
| Length limit, empty answer, invalid calls, missing signal, provider failure | Checkpoint a visible diagnostic; `fail_run` raises `AgentRunError` |

The separate failure node is deliberate: raising inside `agent_node` would lose
its diagnostic update, while returning an error message alone would mark the
server run successful. Failure messages have no executable tool calls. The
history API exports their outcome so reloads and reconciliation retain errors.
Earlier successful tool operations are not rolled back. There is no automatic
model substitution, generation retry, continuation, or side-effect replay.

## Budget and provider protocol

`CONTEXT_OUTPUT_RESERVE` is both the generation cap and the input planner's output
reserve. The default is 16384 tokens, including reasoning and the visible answer.
With the default 65536 total window and 2048 safety margin, estimated input plus
tool schemas must fit 47104 tokens. This is a ceiling, not a minimum charge;
providers can still truncate complex reasoning and the failure must remain visible.

MiMo chat explicitly enables thinking. `services.mimo_chat.MiMoChatOpenAI`
preserves streamed and non-streamed `reasoning_content`, and returns it with
assistant tool-call messages in later requests. Budget calculations include it;
chat rendering, excerpts and rolling-summary evidence exclude it. See the
[MiMo protocol requirements](https://mimo.mi.com/docs/zh-CN/quick-start/usage-guide/text-generation/deep-thinking).
Old checkpoints that discarded reasoning cannot recover it retroactively; no
fabricated reasoning is inserted. A provider rejecting such history reports a
visible failure; starting a new conversation establishes a complete protocol history.

Provider-reported usage logs include reasoning tokens, finish reason, generation
limit and outcome. No reasoning text is logged. `tests/test_agent_outcomes.py`
tests the SDK wire format, failure checkpoints, and exactly-once tool dispatch.
