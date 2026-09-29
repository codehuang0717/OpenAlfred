# Timezones, chat runs, and Windows startup

## Time authority

The browser sends its IANA timezone (`Intl.DateTimeFormat().resolvedOptions().timeZone`)
with each chat run and saves it through authenticated `PUT /api/settings/timezone`.
The saved setting is user-scoped, not a server-wide `TIMEZONE` environment variable.
Voice runs snapshot the saved timezone at entry; tools reuse that snapshot throughout
the run. Background supervision reads the account setting when a cycle starts.
Missing or invalid timezones are explicit errors, never London/server-time defaults.

The model receives local current time, UTC offset, and zone name. Tools interpret
naive local dates in that zone; explicit offsets remain authoritative. Storage and
HTTP todo input use UTC/offset-aware timestamps. DST gaps and repeated local times
require clarification. Existing records are not silently shifted.

Device timezone is not geographic location: IP/VPN location is not used. A user who
travels should check their device timezone. Account background work uses the most
recent timezone reported by a signed-in device.

## Chat lifecycle

Frontend streams belong to account/thread sessions, not mounted pages. Navigation
unsubscribes the view without stopping the run. Only Stop explicitly cancels the
server run. Stream requests use `on_disconnect: continue`; after reload, the client
checks for an active run and polls persisted history until completion. This recovery
shows saved checkpoints, not guaranteed token-by-token replay.

## Windows startup

Use root `./start-all.ps1`. The former `src/services/start-all.ps1` forwards to it.
`-CheckOnly` verifies pinned Screenpipe model sizes and hashes without capture.
Normal startup also verifies models before launching services. Use
`./start-all.ps1 -RepairModels -CheckOnly` in the same terminal that normally starts
the app to repair invalid files through verified downloads without launching capture;
omit `-CheckOnly` to repair and then start. Download/checksum failures stop startup.
`-SupervisorOnly` uses the same supervisor launcher as the complete stack.
The launcher replaces only this repository's supervisor process tree. Capture
still requires the existing local account binding and enabled recording preference.
Diagnostics go to `logs/supervisor-stdout.log` and `logs/supervisor-stderr.log`.

## Verification and references

- Backend: `uv run python -m unittest discover -s tests -p "test_*.py"`.
- Frontend: `node --test src/lib/*.test.ts`, `npx tsc --noEmit`, `npm run build`.
- Manual: start a long reply, switch threads, return, then exercise explicit Stop.
- [LangChain runtime context](https://docs.langchain.com/oss/python/langchain/runtime)
- [Browser timezone resolution](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Intl/DateTimeFormat/resolvedOptions)
