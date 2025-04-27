import re
import os
import json
import httpx
import jinja2
import logging
import platform
import requests

from pathlib import Path
from pydantic import BaseModel
from typing import Optional, Union, Dict, Any, List

logger = logging.getLogger(__name__)

class GenerateRequest(BaseModel):
    model: str
    prompt: str
    system: Optional[str] = None
    template: Optional[str] = None
    context: Optional[List[int]] = None
    stream: bool = False
    raw: Optional[bool] = None
    format: Optional[str] = None
    options: Optional[Dict[str, Any]] = None

class GenerateResponse(BaseModel):
    model: str
    created_at: str
    response: str
    done: bool
    context: Optional[List[int]] = None
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None

class LLMServiceError(Exception):
    """Base exception for LLMService errors"""
    pass

class ConnectionError(LLMServiceError):
    """Failed to connect to the LLM service"""
    pass

class ResponseError(LLMServiceError):
    """Error in the response from the LLM service"""
    pass

class LLMService:
    def __init__(
        self,
        model: str = "llama2",
        host: Optional[str] = None,
        system_prompt_template: Optional[str] = None,
        timeout: int = 60,
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the LLMService with connection parameters and model configuration.
        
        Args:
            model: The model to use for generating commit messages
            host: The host URL of the LLM service (e.g., "http://localhost:11434")
            system_prompt_template: System prompt to guide the model's behavior
            timeout: Timeout for requests in seconds
            openai_api_key: API key for OpenAI services (optional)
            openai_base_url: Base URL for OpenAI-compatible API (optional)
            **kwargs: Additional arguments to pass to the HTTP client
        """
        self.model = model
        
        self.template_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(Path("src/template")),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Load system prompt template
        self.system_prompt_template = self.template_env.get_template(
            system_prompt_template or "system_prompt.j2"
        )
        self.timeout = timeout
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self.openai_base_url = openai_base_url or os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1')
        
        # Initialize HTTP client
        self._client = httpx.Client(
            base_url=self._parse_host(host or os.getenv('LLM_HOST', 'http://localhost:11434')),
            timeout=timeout,
            headers={
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'User-Agent': f'llm-service/{platform.python_version()}'
            },
            **kwargs
        )

    def _render_system_prompt(self,
                              diff_content: str,
                              file_changes: Dict[str, List[str]]
                              ) -> str:
        """Render system prompt from template"""
        return self.system_prompt_template.render(
            diff_content=diff_content,
            new_files=file_changes.get('new_files', []),
            modified_files=file_changes.get('modified_files', []),
            deleted_files=file_changes.get('deleted_files', [])
        )

    def _parse_host(self, host: str) -> str:
        """Parse and normalize the host URL"""
        if not host.startswith(('http://', 'https://')):
            host = f'http://{host}'
        return host.rstrip('/')

    def _clean_commit_message(self, message: str) -> str:
        """Clean up the generated commit message"""
        # Remove code block markers if present
        message = re.sub(r'^```.*?\n', '', message)  # Remove opening ```
        message = re.sub(r'\n```$', '', message)    # Remove closing ```
        message = message.replace('`', '')          # Remove all backticks
        return message.strip()

    def _build_user_prompt(self,
                           diff_content: str, 
                           context: Optional[str] = None,
                           file_changes: Dict[str, List[str]] = None) -> str:
        user_prompt = []
    
        if context:
            user_prompt.append(f"Additional Context: {context}")

        remaining_diff = diff_content[15000:]
        if remaining_diff:
            user_prompt.append(f"Full Diff Content:\n{remaining_diff}")
        
        user_prompt.append("Based on the above system instructions and this additional context, "
                            "generate the concise, clear and final commit message in English.")
        return "\n\n".join(user_prompt)

    def generate_commit_message(self,
                                diff_content: str,
                                context: Optional[str] = None,
                                file_changes: Optional[Dict[str, List[str]]] = None,
                                stream: bool = False,
                                **options
                                ) -> Union[str, GenerateResponse]:
        """
        Generate a commit message based on the provided diff content.
        
        Args:
            diff_content: The git diff content to analyze
            context: Additional context about the changes
            file_changes: Dictionary of file changes (from get_status_changes)
            stream: Whether to stream the response (only for local models)
            **options: Additional model options (temperature, top_p, etc.)
            
        Returns:
            The generated commit message as a string
            Or a GenerateResponse object if streaming
            
        Raises:
            ConnectionError: If unable to connect to the LLM service
            ResponseError: If the LLM service returns an error
        """

        self.system_prompt = self._render_system_prompt(
            diff_content=diff_content,
            file_changes=file_changes
        )
        
        # build user's prompt according to some specific content
        user_prompt = self._build_user_prompt(
            diff_content=diff_content,
            context=context,
            file_changes=file_changes
        )

        # Use OpenAI API if model starts with 'gpt' and OpenAI is explicitly requested
        if self.model.startswith('gpt') and self.openai_api_key:
            return self._generate_with_openai(user_prompt, diff_content)
        else:
            return self._generate_with_local_llm(user_prompt, stream, **options)

    def _generate_with_local_llm(
        self,
        prompt: str,
        stream: bool = False,
        **options
    ) -> Union[str, GenerateResponse]:
        """Generate commit message using local LLM service"""
        try:
            response = self._client.post(
                '/api/generate',
                json={
                    'model': self.model,
                    'prompt': prompt,
                    'system': self.system_prompt,
                    'stream': stream,
                    'options': options,
                }
            )
            response.raise_for_status()
            
            if stream:
                return GenerateResponse(**response.json())
            else:
                data = response.json()
                return self._clean_commit_message(data.get('response', ''))
                
        except httpx.ConnectError as e:
            raise ConnectionError(f"Failed to connect to LLM service: {e}") from e
        except httpx.HTTPStatusError as e:
            raise ResponseError(f"LLM service error: {e.response.text}") from e
        except json.JSONDecodeError as e:
            raise ResponseError("Invalid JSON response from LLM service") from e

    def _generate_with_openai(self, prompt: str) -> str:
        """Generate commit message using OpenAI API"""
        if not self.openai_api_key:
            logger.warning('OPENAI_API_KEY environment variable not set, cannot call API')
            raise ConnectionError("OpenAI API key not configured")

        try:
            response = requests.post(
                f"{self.openai_base_url}/chat/completions",
                headers={
                    'Authorization': f'Bearer {self.openai_api_key}',
                    'Content-Type': 'application/json',
                },
                json={
                    'model': self.model if self.model.startswith('gpt') else 'gpt-4',
                    'messages': [
                        {
                            'role': 'system',
                            'content': self.system_prompt
                        },
                        {
                            'role': 'user',
                            'content': prompt
                        }
                    ],
                    'temperature': 0.7,
                    'max_tokens': 2000,
                },
                timeout=self.timeout
            )

            if response.status_code == 200:
                result = response.json()
                commit_message = result['choices'][0]['message']['content']
                return self._clean_commit_message(commit_message)
            else:
                error_msg = f"API call failed: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise ResponseError(error_msg)
                
        except requests.exceptions.RequestException as e:
            error_msg = f"Error calling API: {str(e)}"
            logger.error(error_msg)
            raise ConnectionError(error_msg) from e

    def close(self):
        """Close the HTTP client"""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
