import os
import sys
import dotenv
import logging
import argparse

from pathlib import Path
from typing import Optional, Dict, Union, Any
from src.llm_service import LLMService, GenerateResponse
from src.git_service import GitService
from src.logger import setup_logging

setup_logging()

logger = logging.getLogger(__name__)

dotenv.load_dotenv()

class CommitMessageGenerator:
    """Coordinate with GitService and LLMService to generate a commit 
       message that complies with the specifications.
    """
    def __init__(self,
                 repo_path: Optional[Union[str, Path]],
                 **llm_kwargs):
        """
        Initialize the commit message generator.

        Args:
            repo_path: Path to the Git repository (supports str or Path objects).
            **llm_kwargs: Additional parameters passed to LLMService.
        """
        if not repo_path:
            logger.error("Param 'repo_path' is mandatory.")
        
        llm_options = {
            "max_tokens": int(os.getenv("MAX_TOKENS")),
            "temperature": float(os.getenv("TEMPERATURE"))
        }

        self.git_svc = GitService(repo_path=str(repo_path))
        
        self.llm_svc = LLMService(
            model=os.getenv("LLM_MODEL"),
            host=os.getenv("LLM_HOST"),
            system_prompt_template=os.getenv("SYSTEM_PROMPT"),
            timeout=int(os.getenv("TIMEOUT")),
            llm_options = llm_options,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_base_url=os.getenv("OPENAI_API_BASE"),
            **llm_kwargs
        )

    def generate(self,
                 context: Optional[str] = None,
                 stream: bool = False,
                 diff_options: Optional[Dict[str, Any]] = None,
                 ) -> Union[str, GenerateResponse]:
        """
        Generate a commit message that complies with the Conventional Commits specification.

        Args:
            context: Additional context for the changes.
            stream: Whether to use streaming output.
            diff_options: Parameters passed to get_staged_changes_diff.

        Returns:
            The generated commit message content.

        Raises:
            ValueError: When there are no staged changes.
            RuntimeError: When commit message generation fails.
        """
        diff_options = diff_options or {}
        try:
            # get raw git diff
            self.git_svc.get_staged_changes_diff(**diff_options)

            diff = self.git_svc.get_enhanced_diff()
            if not diff.strip():
                logger.error("No staged changes detected (empty diff)")
                raise

            changes = self.git_svc.get_status_changes()
            
            # generate commit message
            return self.llm_svc.generate_commit_message(
                diff_content=diff,
                context=context,
                file_changes=changes,
                stream=stream,
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
        required=True,
        help="Git repository path (default: current directory)"
    )
    parser.add_argument(
        "--context",
        help="Additional context for the commit"
    )
    
    args = parser.parse_args()

    try:
        generator = CommitMessageGenerator(
            repo_path=args.repo
        )
        message = generator.generate(context=args.context)
        sys.stdout.write(message)

    except ValueError as e:
        logger.error(f"Invalid argument: {e}")
        sys.exit(2)
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
