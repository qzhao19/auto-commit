import os
import sys
import logging
import argparse

from pathlib import Path
from typing import Optional, Dict, Union, Any
from src.llm_service import LLMService, GenerateResponse
from src.git_service import GitService
from src.logger import setup_logging

logger = logging.getLogger(__name__)

class CommitMessageGenerator:
    """Coordinate with GitService and LLMService to generate a commit 
       message that complies with the specifications.
    """
    def __init__(
        self,
        repo_path: Optional[Union[str, Path]] = None,
        model: str = "llama2",
        llm_host: Optional[str] = None,
        system_prompt_template: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        timeout: int = 60,
        **llm_kwargs
    ):
        """
        Initialize the commit message generator.

        Args:
            repo_path: Path to the Git repository (supports str or Path objects).
            model: Name of the LLM model (default is llama2).
            llm_host: Address of the local LLM service.
            system_prompt_template: Path to a custom system prompt template.
            openai_api_key: OpenAI API key.
            openai_base_url: OpenAI API address.
            timeout: Request timeout in seconds.
            **llm_kwargs: Additional parameters passed to LLMService.
        """
        self.git_svc = GitService(
            repo_path=str(repo_path) if repo_path else None
        )
        
        self.llm_svc = LLMService(
            model=model,
            host=llm_host,
            system_prompt_template=system_prompt_template,
            timeout=timeout,
            openai_api_key=openai_api_key,
            openai_base_url=openai_base_url,
            **llm_kwargs
        )

    def generate(
        self,
        context: Optional[str] = None,
        stream: bool = False,
        diff_options: Optional[Dict[str, Any]] = None,
        llm_options: Optional[Dict[str, Any]] = None
    ) -> Union[str, GenerateResponse]:
        """
        Generate a commit message that complies with the Conventional Commits specification.

        Args:
            context: Additional context for the changes.
            stream: Whether to use streaming output.
            diff_options: Parameters passed to get_staged_changes_diff.
            llm_options: Parameters passed to the LLM generator.

        Returns:
            The generated commit message content.

        Raises:
            ValueError: When there are no staged changes.
            RuntimeError: When commit message generation fails.
        """
        diff_options = diff_options or {}
        llm_options = llm_options or {}
        
        try:
            # get git diff
            diff = self.git_svc.get_staged_changes_diff(**diff_options)
            if not diff.strip():
                raise ValueError("No staged changes detected (empty diff)")
                
            changes = self.git_svc.get_status_changes()
            
            # generate commit message
            return self.llm_svc.generate_commit_message(
                diff_content=diff,
                context=context,
                file_changes=changes,
                stream=stream,
                **llm_options
            )
            
        except ValueError as e:
            logger.warning(f"Validation failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(
                "Commit generation failed",
                exc_info=True,
                extra={
                    "context": context,
                    "diff_length": len(diff) if diff else 0
                }
            )
            raise RuntimeError(
                f"Failed to generate commit message: {str(e)}"
            ) from e

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.llm_svc.close()

def main():    
    parser = argparse.ArgumentParser(
        description="Generate commit messages using AI based on git changes"
    )
    
    parser.add_argument(
        "--repo",
        type=str,
        help="Git repository path (default: current directory)"
    )
    parser.add_argument(
        "--model",
        default="llama2",
        help="LLM model (llama2/gpt-3.5-turbo/gpt-4)"
    )
    parser.add_argument(
        "--host",
        default="http://localhost:11434",
        help="Local LLM service host"
    )
    parser.add_argument(
        "--openai-key",
        help="OpenAI API key (default: uses OPENAI_API_KEY env var)",
        default=os.getenv("OPENAI_API_KEY")
    )
    parser.add_argument(
        "--context",
        help="Additional context for the commit"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Request timeout in seconds"
    )
    
    args = parser.parse_args()
    setup_logging()
    
    try:
        generator = CommitMessageGenerator(
            repo_path=args.repo,
            model=args.model,
            llm_host=args.host,
            openai_api_key=args.openai_key,
            timeout=args.timeout
        )
        message = generator.generate(context=args.context)

        print("\nGenerated commit message:")
        print("-----------------------")
        print(message)
        print("-----------------------")
        
    except ValueError as e:
        logger.error(f"Invalid argument: {e}")
        sys.exit(2)
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
