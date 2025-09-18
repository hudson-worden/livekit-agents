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
from livekit.agents.types import TOPIC_TRANSCRIPTION
from livekit.agents.voice import UserInputTranscribedEvent
from livekit.agents.voice.agent import ModelSettings
from livekit.agents.voice.io import TimedString
from livekit.plugins import elevenlabs, deepgram, openai, silero
from livekit.rtc.data_stream import TextStreamReader

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

    @session.on("user_input_transcribed")
    def _on_user_input_transcribed(event: UserInputTranscribedEvent):
        timing_info = ""
        if event.start_time is not None and event.end_time is not None:
            duration = event.end_time - event.start_time
            timing_info = f" (start: {event.start_time:.3f}s, end: {event.end_time:.3f}s, duration: {duration:.3f}s)"

        logger.info(
            f"User {'FINAL' if event.is_final else 'INTERIM'} transcript: '{event.transcript}'{timing_info}"
        )

    await session.start(agent=MyAgent(), room=ctx.room)

    @session.on("agent_state_changed")
    def _handle_agent_state_changed(event: AgentStateChangedEvent):
        logger.info(f"Agent state changed: {event.old_state} -> {event.new_state}")

    def log_transcripts(reader: TextStreamReader, participant_id: str):
        logger.info(f"Transcribing text: {participant_id}")
        stream_id = reader.info.stream_id
        segment_id = reader.info.attributes.get("segment_id", None)
        if not segment_id:
            logger.warning("No segment id found for text stream")
            return

        async def _log():
            track_id = reader.info.attributes.get("track_id", None)
            async for chunk in reader:
                is_final = chunk.attributes.get("final", "false").lower() == "true"
                content = chunk.text
                logger.info(
                    f"[{participant_id}] ({'FINAL' if is_final else 'INTERIM'}) {content} (stream: {stream_id}, segment: {segment_id}, track: {track_id})"
                )

            # update the final flag
            final = reader.info.attributes.get("final", "null")
            logger.info(
                f"[{participant_id}] FINAL '' (stream: {stream_id}, segment: {segment_id}, track: {track_id}, final: {final})"
            )

        asyncio.create_task(_log())

    await session.start(
        agent=MyAgent(),
        room=ctx.room,
        room_output_options=RoomOutputOptions(
            sync_transcription=True,
            transcription_enabled=True,
        ),
    )
    session._room_io._room.register_text_stream_handler(TOPIC_TRANSCRIPTION, log_transcripts)

    # ctx.room.register_text_stream_handler(
    #     TOPIC_TRANSCRIPTION,
    #     log_transcripts
    # )

    session.generate_reply(instructions="say hello to the user")


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            agent_name="regal-agent-dev",
            entrypoint_fnc=entrypoint,
        )
    )
