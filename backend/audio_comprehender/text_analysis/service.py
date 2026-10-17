"""Публичная точка входа в модуль текстового анализа: CallTranscript на входе,
структурированная оценка консультации (TextAnalysis) на выходе."""

from audio_comprehender.asr.types import CallTranscript
from audio_comprehender.text_analysis.llm_client import get_client, get_model_uri
from audio_comprehender.text_analysis.parsing import TextAnalysis, parse_text_analysis
from audio_comprehender.text_analysis.prompts import SYSTEM_PROMPT


def analyze_transcript_with_llm(transcript: CallTranscript) -> TextAnalysis:
    response = get_client().chat.completions.create(
        model=get_model_uri(),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": transcript.as_dialogue_text()},
        ],
        temperature=0.2,
    )
    return parse_text_analysis(response.choices[0].message.content)
