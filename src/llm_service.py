import os
import json
import httpx
import platform

from pydantic import BaseModel
from typing import Optional, Union, Dict, Any, List

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
    def __init__(self,
                 model: str = "llama2",
                 host: Optional[str] = None,
                 system_prompt: Optional[str] = None,
                 template: Optional[str] = None,
                 timeout: int = 60,
                 **kwargs ):
        """
        Initialize the LLMService with connection parameters and model configuration.
        
        Args:
            model: The model to use for generating commit messages
            host: The host URL of the LLM service (e.g., "http://localhost:11434")
            system_prompt: System prompt to guide the model's behavior
            template: Template for formatting the commit message
            timeout: Timeout for requests in seconds
            **kwargs: Additional arguments to pass to the HTTP client
        """
        self.model = model
        self.system_prompt = system_prompt or (
            "You are an expert software developer. "
            "Generate concise, clear commit messages based on the provided diff content. "
            "Follow conventional commit style when appropriate: "
            "<type>(<scope>): <subject>\n\n<body>\n\n<footer>"
        )
        self.template = template
        self.timeout = timeout
        
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

    def _parse_host(self, host: str) -> str:
        """Parse and normalize the host URL"""
        if not host.startswith(('http://', 'https://')):
            host = f'http://{host}'
        return host.rstrip('/')

    def generate_commit_message(self,
                                diff_content: str,
                                context: Optional[str] = None,
                                stream: bool = False,
                                **options) -> Union[str, GenerateResponse]:
        """
        Generate a commit message based on the provided diff content.
        
        Args:
            diff_content: The git diff content to analyze
            context: Additional context about the changes
            stream: Whether to stream the response
            **options: Additional model options (temperature, top_p, etc.)
            
        Returns:
            The generated commit message as a string (if stream=False)
            Or a GenerateResponse object (if stream=True)
            
        Raises:
            ConnectionError: If unable to connect to the LLM service
            ResponseError: If the LLM service returns an error
        """
        # Combine the diff content with any additional context
        prompt = f"Diff content:\n{diff_content}\n\n"
        if context:
            prompt += f"Additional context:\n{context}\n\n"
        prompt += "Generate a concise, clear commit message based on the above changes."

        try:
            response = self._client.post(
                '/api/generate',
                json={
                    'model': self.model,
                    'prompt': prompt,
                    'system': self.system_prompt,
                    'template': self.template,
                    'stream': stream,
                    'options': options,
                    # 'max_tokens': 2000,
                }
            )
            response.raise_for_status()
            
            if stream:
                return GenerateResponse(**response.json())
            else:
                data = response.json()
                return data.get('response', '').strip()
                
        except httpx.ConnectError as e:
            raise ConnectionError(f"Failed to connect to LLM service: {e}") from e
        except httpx.HTTPStatusError as e:
            raise ResponseError(f"LLM service error: {e.response.text}") from e
        except json.JSONDecodeError as e:
            raise ResponseError("Invalid JSON response from LLM service") from e

    def close(self):
        """Close the HTTP client"""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()