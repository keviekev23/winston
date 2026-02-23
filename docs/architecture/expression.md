# Expression Engine

> **Parent:** `STEERING.md` Section 2
> **Related:** `docs/architecture/brain.md`, `docs/architecture/mqtt_topics.md`

## Visual Expression (MacBook Display)

Simple animated eyes (two circles) expressing the following states:

| State | Visual Behavior | Trigger |
|-------|----------------|---------|
| **Neutral** | Relaxed eyes, slow idle blink | Default resting state |
| **Listening** | Wide eyes, oriented toward user | User is speaking |
| **Thinking** | Eyes look up-and-right, slower blink | Waiting for cloud response |
| **Uncertain** | Eyes narrow slightly, subtle tilt | Low-confidence perception |
| **Excitement** | Eyes widen, quick double-blink | User asks for help, positive moment |
| **Acknowledging** | Quick blink + slight nod motion | User gives a command, agent confirms |
| **Playful** | Asymmetric eye squint, bouncy motion | Light moment, joke, fun suggestion |
| **Waiting/Patient** | Soft half-lidded eyes, gentle sway | User is busy (hands full, cooking) |

Eye state transitions are driven by the cerebellum. No lip-sync with speech required.

## Audio Expression

- **TTS:** ElevenLabs API (cloud). Character voice selected during setup.
- **Speaker:** MacBook built-in speakers
- **Non-verbal sounds:** Optional subtle audio cues (acknowledgment chirp, alert tone) — explore in iteration
