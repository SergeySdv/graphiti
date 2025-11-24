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
        """Create a structured completion using standard chat completions with JSON mode."""
        
        # Check if model starts with gpt-5/o1/o3 (reasoning models) to adjust temp
        is_reasoning_model = (
            model.startswith('gpt-5') or model.startswith('o1') or model.startswith('o3')
        )
        
        logger.debug(f"OllamaClient generating structured response for model {model}")
        
        # Prepare messages with schema injection
        modified_messages = [m.copy() if isinstance(m, dict) else m for m in messages]
        
        # Generate schema string
        try:
            schema = response_model.model_json_schema()
            schema_str = json.dumps(schema, indent=2)
            
            instruction = (
                f"\n\nIMPORTANT: You must respond with a valid JSON object matching this schema:\n"
                f"{schema_str}\n"
                f"Do not include any text before or after the JSON."
            )
            
            if modified_messages:
                last_msg = modified_messages[-1]
                if isinstance(last_msg, dict) and 'content' in last_msg:
                    last_msg['content'] = str(last_msg['content']) + instruction
                
        except Exception as e:
            logger.warning(f"Failed to inject schema into prompt: {e}")

        # Call Ollama with JSON mode
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=modified_messages,
                temperature=temperature if not is_reasoning_model else None,
                max_tokens=max_tokens,
                response_format={'type': 'json_object'},
            )
        except Exception as e:
            logger.error(f"Ollama API call failed: {e}")
            raise

        class MockResponse:
            def __init__(self, content: str | None):
                self.output_text = content
                
        content = response.choices[0].message.content
        
        if not content:
            return MockResponse(content)
        
        # Post-processing for common schema mismatches
        try:
            # Try to find JSON if there is extra text
            if content.strip().startswith("```json"):
                match = re.search(r"```json(.*?)```", content, re.DOTALL)
                if match:
                    content = match.group(1)
            elif content.strip().startswith("```"):
                match = re.search(r"```(.*?)```", content, re.DOTALL)
                if match:
                    content = match.group(1)

            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                # Fallback: Try to repair truncated JSON
                logger.warning(f"JSON decode failed. Attempting repair on content length {len(content)}")
                
                # Basic repair strategies for truncated JSON
                stripped_content = content.strip()
                
                # 1. Try adding missing closing braces/brackets
                open_braces = stripped_content.count('{')
                close_braces = stripped_content.count('}')
                open_brackets = stripped_content.count('[')
                close_brackets = stripped_content.count(']')
                
                repaired_content = stripped_content
                
                # Simple check for trailing comma in lists/objects
                if repaired_content.rstrip().endswith(','):
                     repaired_content = repaired_content.rstrip()[:-1]

                # If it looks like it cut off in a string, try to close the string
                # This is hard to do perfectly without a parser, but we can try simple heuristics
                # If the last non-whitespace char is not a closer, and we have unclosed quotes...
                # Counting quotes is tricky due to escapes.
                
                # Let's just try closing structures first
                repaired_content += ']' * (open_brackets - close_brackets)
                repaired_content += '}' * (open_braces - close_braces)
                
                try:
                    data = json.loads(repaired_content)
                    logger.info("Successfully repaired truncated JSON by closing braces/brackets.")
                except json.JSONDecodeError:
                    # 2. If that failed, maybe it cut off inside a string?
                    # Try checking if the error is "Unterminated string"
                    # We can try appending '"' then closing braces
                    repaired_content_str = stripped_content + '"'
                    repaired_content_str += ']' * (open_brackets - close_brackets)
                    repaired_content_str += '}' * (open_braces - close_braces)
                    
                    try:
                        data = json.loads(repaired_content_str)
                        logger.info("Successfully repaired truncated JSON by closing string and braces.")
                    except json.JSONDecodeError:
                        logger.error("Failed to repair JSON.")
                        raise  # Re-raise original error if repair fails

            
            # Fix 1: Clean keys (remove trailing colons)
            data = self._clean_keys(data)
            
            # Fix 2: 'entities' -> 'extracted_entities'
            if isinstance(data, dict) and 'entities' in data and 'extracted_entities' not in data and 'ExtractedEntities' in response_model.__name__:
                logger.info("Patching response: renaming 'entities' to 'extracted_entities'")
                data['extracted_entities'] = data.pop('entities')
            
            content = json.dumps(data)
        except Exception as e:
            logger.warning(f"Failed to post-process JSON: {e}")
        
        return MockResponse(content)
