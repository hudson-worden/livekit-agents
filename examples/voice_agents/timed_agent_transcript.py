import asyncio
import logging
from collections.abc import AsyncGenerator, AsyncIterable

from dotenv import load_dotenv

from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.agents import (
    Agent,
    AgentSession,
    AgentStateChangedEvent,
    JobContext,
    RoomOutputOptions,
    WorkerOptions,
    cli,
)
from livekit.agents.voice import ConversationItemAddedEvent, UserInputTranscribedEvent
from livekit.agents.voice.agent import ModelSettings
from livekit.agents.voice.io import PlaybackFinishedEvent, TimedString
from livekit.plugins import elevenlabs, deepgram, openai, silero

logger = logging.getLogger("my-worker")
logger.setLevel(logging.INFO)

load_dotenv()


# This example shows how to obtain timing information from both:
# 1. TTS-aligned transcripts (agent output): supported for Cartesia and ElevenLabs TTS (word level timestamps)
#    and non-streaming TTS with StreamAdapter (sentence level timestamps).
# 2. User input transcripts: start_time and end_time from STT providers that support timing information.


# Just for demonstration
agent_turn_start: float | None = None  # When then most recent agent turn started
agent_turn_end: float | None = None  # When the most recent agent turn ended


class MyAgent(Agent):
    def __init__(self):
        super().__init__(instructions="You are a helpful assistant.")

        self._closing_task: asyncio.Task[None] | None = None

    async def transcription_node(
        self, text: AsyncIterable[str | TimedString], model_settings: ModelSettings
    ) -> AsyncGenerator[str | TimedString, None]:
        """
        Can't use this b/c it doesn't handle interruptions. It just yields everything
        from the LLM generation.
        """
        async for chunk in text:
            if isinstance(chunk, TimedString):
                pass
                # logger.info(f"TimedString: '{chunk}' ({chunk.start_time} - {chunk.end_time})")
            yield chunk


async def entrypoint(ctx: JobContext):
    session = AgentSession(
        stt=deepgram.STT(),
        llm=openai.LLM(),
        tts=elevenlabs.TTS(),
        vad=silero.VAD.load(),
        # enable TTS-aligned transcript, can be configured at the Agent level as well
        use_tts_aligned_transcript=True,
    )
    last_user_input_transcribed_event: UserInputTranscribedEvent | None = None
    last_user_conversation_item_event: ConversationItemAddedEvent | None = None

    @session.on("user_input_transcribed")
    def _on_user_input_transcribed(event: UserInputTranscribedEvent):
        nonlocal last_user_input_transcribed_event
        timing_info = ""
        if event.start_time is not None and event.end_time is not None:
            duration = event.end_time - event.start_time
            timing_info = f" (start: {event.start_time:.3f}s, end: {event.end_time:.3f}s, duration: {duration:.3f}s)"

        logger.info(
            f"User {'FINAL' if event.is_final else 'INTERIM'} transcript: '{event.transcript}'{timing_info}"
        )
        if event.is_final:
            last_user_input_transcribed_event = event
            annotate_conversation_items()

    @session.on("conversation_item_added")
    def _on_conversation_item_added(event: ConversationItemAddedEvent):
        if event.item.role != "user":
            return

        nonlocal last_user_conversation_item_event
        last_user_conversation_item_event = event
        annotate_conversation_items()
        logger.info(f"User added conversation item: {event.item}")

    def annotate_conversation_items():
        nonlocal last_user_input_transcribed_event, last_user_conversation_item_event
        if last_user_input_transcribed_event and last_user_conversation_item_event:
            if (
                last_user_input_transcribed_event.transcript
                == last_user_conversation_item_event.item.content[0]
            ):
                logger.info(
                    "Annotating conversation item with timing info",
                    extra={
                        "start_time": last_user_input_transcribed_event.start_time,
                        "end_time": last_user_input_transcribed_event.end_time,
                    },
                )
                last_user_conversation_item_event, last_user_input_transcribed_event = None, None

    await session.start(agent=MyAgent(), room=ctx.room)

    @session.on("agent_state_changed")
    def _handle_agent_state_changed(event: AgentStateChangedEvent):
        logger.info(f"Agent state changed: {event.old_state} -> {event.new_state}")


    @session.output.audio.on("playback_finished")
    def _handle_playback_finished(event: PlaybackFinishedEvent):
        logger.info(f"Playback finished: {event.playback_position} ({'interrupted' if event.interrupted else 'completed'})")
        if event.synchronized_transcript:
            logger.info(f"Synchronized transcript: {event.synchronized_transcript}")

    await session.start(
        agent=MyAgent(),
        room=ctx.room,
        room_output_options=RoomOutputOptions(
            sync_transcription=True,
            transcription_enabled=True,
        ),
    )

    session.generate_reply(instructions="say hello to the user")


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            agent_name="regal-agent-dev",
            entrypoint_fnc=entrypoint,
        )
    )
