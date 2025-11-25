from typing import Any, Optional
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel
from graphiti_core.llm_client.openai_client import OpenAIClient
import json
import logging
import re

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
        """Create a structured completion using standard chat completions with JSON schema support."""
        
        # Check if model starts with gpt-5/o1/o3 (reasoning models) to adjust temp
        is_reasoning_model = (
            model.startswith('gpt-5') or model.startswith('o1') or model.startswith('o3')
        )
        
        logger.debug(f"OllamaClient generating structured response for model {model}")
        
        # Prepare messages - copy to avoid side effects
        modified_messages = [m.copy() if isinstance(m, dict) else m for m in messages]
        
        # Use native JSON schema support via response_format
        try:
            json_schema = response_model.model_json_schema()
            
            # Ollama expects `format` key in native API, but `response_format` in OpenAI API.
            # The structure for OpenAI API `response_format` when using JSON schema is:
            # { "type": "json_schema", "json_schema": { ... } }
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "schema": json_schema,
                    "strict": True
                }
            }
        except Exception as e:
            logger.warning(f"Failed to generate JSON schema for {response_model.__name__}: {e}")
            # Fallback to simple json_object if schema generation fails
            response_format = {"type": "json_object"}

        # Call Ollama with structured output
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=modified_messages,
                temperature=temperature if not is_reasoning_model else None,
                max_tokens=max_tokens,
                response_format=response_format, 
            )
        except Exception as e:
            logger.error(f"Ollama API call failed: {e}")
            raise

        class MockResponse:
            def __init__(self, content: str | None):
                self.output_text = content
                self.refusal = None # Initialize refusal to prevent AttributeError

        content = response.choices[0].message.content
        
        if not content:
            return MockResponse(content)
        
        # Post-processing for common schema mismatches & robustness
        try:
            # Strip Markdown if present (even with schema, some models chatter)
            if content.strip().startswith("```json"):
                match = re.search(r"```json(.*?)```", content, re.DOTALL)
                if match:
                    content = match.group(1)
            elif content.strip().startswith("```"):
                match = re.search(r"```(.*?)```", content, re.DOTALL)
                if match:
                    content = match.group(1)

            data = json.loads(content)

            # Fix 1: Clean keys (remove trailing colons)
            data = self._clean_keys(data)
            
            # Fix 2: 'entities' -> 'extracted_entities'
            # Even with schema, models sometimes hallucinate the top level key if it's nested.
            if isinstance(data, dict) and 'entities' in data and 'extracted_entities' not in data and 'ExtractedEntities' in response_model.__name__:
                logger.info("Patching response: renaming 'entities' to 'extracted_entities'")
                data['extracted_entities'] = data.pop('entities')
            
            content = json.dumps(data)
        except Exception as e:
            logger.warning(f"Failed to post-process JSON: {e}")
        
        return MockResponse(content)