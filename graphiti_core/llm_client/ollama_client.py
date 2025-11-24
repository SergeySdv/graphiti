import json
import logging
from typing import Any

from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from .openai_client import OpenAIClient

logger = logging.getLogger(__name__)


class OllamaClient(OpenAIClient):
    """
    Custom OpenAIClient for Ollama/Local providers that avoids using 'responses.parse'.
    It uses standard 'chat.completions.create' with JSON mode and prompt engineering.
    """

    def _clean_keys(self, obj: Any) -> Any:
        """Recursively clean dictionary keys by removing trailing colons."""
        if isinstance(obj, dict):
            new_obj = {}
            for k, v in obj.items():
                new_key = k.rstrip(':').strip()
                new_obj[new_key] = self._clean_keys(v)
            return new_obj
        elif isinstance(obj, list):
            return [self._clean_keys(item) for item in obj]
        else:
            return obj

    async def _create_structured_completion(
        self,
        model: str,
        messages: list[ChatCompletionMessageParam],
        temperature: float | None,
        max_tokens: int,
        response_model: type[BaseModel],
        reasoning: str | None = None,
        verbosity: str | None = None,
    ) -> Any:
        """Create a structured completion using standard chat completions with JSON mode."""

        # Check if model starts with gpt-5/o1/o3 (reasoning models) to adjust temp
        is_reasoning_model = (
            model.startswith('gpt-5') or model.startswith('o1') or model.startswith('o3')
        )

        logger.debug(f'OllamaClient generating structured response for model {model}')

        # Prepare messages with schema injection
        modified_messages = [m.copy() if isinstance(m, dict) else m for m in messages]

        # Generate schema string
        try:
            schema = response_model.model_json_schema()
            schema_str = json.dumps(schema, indent=2)

            instruction = (
                f'\n\nIMPORTANT: You must respond with a valid JSON object matching this schema:\n'
                f'{schema_str}\n'
                f'Do not include any text before or after the JSON.'
            )

            if modified_messages:
                last_msg = modified_messages[-1]
                if isinstance(last_msg, dict) and 'content' in last_msg:
                    last_msg['content'] = str(last_msg['content']) + instruction

        except Exception as e:
            logger.warning(f'Failed to inject schema into prompt: {e}')

        # Call Ollama with JSON mode
        response = await self.client.chat.completions.create(
            model=model,
            messages=modified_messages,
            temperature=temperature if not is_reasoning_model else None,
            max_tokens=max_tokens,
            response_format={'type': 'json_object'},
        )

        class MockResponse:
            def __init__(self, content: str | None):
                self.output_text = content

        content = response.choices[0].message.content

        if not content:
            return MockResponse(content)

        # Post-processing for common schema mismatches
        try:
            data = json.loads(content)

            # Fix 1: Clean keys (remove trailing colons)
            data = self._clean_keys(data)

            # Fix 2: 'entities' -> 'extracted_entities'
            if (
                isinstance(data, dict)
                and 'entities' in data
                and 'extracted_entities' not in data
                and 'ExtractedEntities' in response_model.__name__
            ):
                logger.info("Patching response: renaming 'entities' to 'extracted_entities'")
                # Use dict methods to avoid type checker confusion if possible, or cast
                data['extracted_entities'] = data.pop('entities')

            content = json.dumps(data)
        except Exception as e:
            logger.warning(f'Failed to post-process JSON: {e}')

        return MockResponse(content)
