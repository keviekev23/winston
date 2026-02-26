"""
MQTT topic constants — mirrors docs/architecture/mqtt_topics.md exactly.
Import these everywhere; never hardcode topic strings.
"""

PREFIX = "souschef"


class Perception:
    TRANSCRIPT    = f"{PREFIX}/perception/speech/transcript"
    SENTIMENT     = f"{PREFIX}/perception/speech/sentiment"
    VAD           = f"{PREFIX}/perception/speech/vad"
    DIARIZATION   = f"{PREFIX}/perception/speech/diarization"
    SCENE_SNAPSHOT = f"{PREFIX}/perception/scene/snapshot"
    SCENE_CHANGE   = f"{PREFIX}/perception/scene/change"
    SCENE_CONTEXT  = f"{PREFIX}/perception/scene/context"   # text summary + image_path for cerebrum


class Brain:
    CEREBELLUM_REACTION   = f"{PREFIX}/brain/cerebellum/reaction"
    CEREBRUM_RESPONSE     = f"{PREFIX}/brain/cerebrum/response"
    CEREBRUM_CLARIFICATION = f"{PREFIX}/brain/cerebrum/clarification"
    CEREBRUM_DELEGATION   = f"{PREFIX}/brain/cerebrum/delegation"
    PLAN_CREATED          = f"{PREFIX}/brain/cerebrum/plan/created"
    PLAN_STEP_UPDATE      = f"{PREFIX}/brain/cerebrum/plan/step_update"
    PLAN_REPLAN           = f"{PREFIX}/brain/cerebrum/plan/replan"
    PLAN_COMPLETED        = f"{PREFIX}/brain/cerebrum/plan/completed"
    SESSION_START         = f"{PREFIX}/brain/session/start"
    SESSION_END           = f"{PREFIX}/brain/session/end"
    SESSION_SUMMARY       = f"{PREFIX}/brain/session/summary"


class Expression:
    EYES_STATE   = f"{PREFIX}/expression/eyes/state"
    TTS_SPEAK    = f"{PREFIX}/expression/tts/speak"
    TTS_STATUS   = f"{PREFIX}/expression/tts/status"
    SOUND_EFFECT = f"{PREFIX}/expression/sound/effect"


class Memory:
    PROFILE_UPDATE   = f"{PREFIX}/memory/profile/update"
    PROFILE_QUERY    = f"{PREFIX}/memory/profile/query"
    SPACES_OBJECT_UPDATE = f"{PREFIX}/memory/spaces/object_update"
    SPACES_QUERY     = f"{PREFIX}/memory/spaces/query"
    EPISODE_SAVE     = f"{PREFIX}/memory/episode/save"
    EPISODE_QUERY    = f"{PREFIX}/memory/episode/query"
    EPISODE_RESULT   = f"{PREFIX}/memory/episode/result"
    DREAMING_TRIGGER = f"{PREFIX}/memory/dreaming/trigger"
    DREAMING_STATUS  = f"{PREFIX}/memory/dreaming/status"
    DREAMING_UPDATES = f"{PREFIX}/memory/dreaming/updates"


class System:
    HEALTH         = f"{PREFIX}/system/health"
    CONFIG         = f"{PREFIX}/system/config"
    MEMORY_MONITOR = f"{PREFIX}/system/debug/memory_monitor"
