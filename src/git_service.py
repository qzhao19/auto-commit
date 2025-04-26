import subprocess
import logging
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

class GitService:
    """
    A utility class for analyzing Git repository changes including staged changes and file status.
    
    This class provides methods to:
    - Retrieve and parse staged changes diff
    - Categorize and analyze file status changes
    """

    def __init__(self, git_repo_path: Optional[str] = None):
        """
        Initialize the GitService.
        
        Args:
            git_repo_path: Path to the Git repository (default: current directory)
        """
        self.git_repo_path = git_repo_path

    def get_staged_changes_diff(self,
                                extra_args: Optional[List[str]] = None,
                                encoding: str = 'utf-8',
                                errors: str = 'strict'
                                ) -> Optional[str]:
        """
        Retrieve the diff of staged changes in the Git repository.
        
        Args:
            extra_args: Additional git diff arguments
            encoding: Output encoding (default: 'utf-8')
            errors: Encoding error handling (default: 'strict')
        
        Returns:
            Diff content as string if successful, None otherwise
        """
        base_cmd = ['git']
        
        if self.git_repo_path:
            base_cmd.extend(['-C', self.git_repo_path])
        
        diff_cmd = base_cmd + ['diff', '--staged']
        
        if extra_args:
            diff_cmd.extend(extra_args)
        
        try:
            result = subprocess.run(
                diff_cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding=encoding,
                errors=errors
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get staged changes. Command: {' '.join(e.cmd)}\nError: {e.stderr}")
        except FileNotFoundError:
            logger.error("Git command not found. Please ensure Git is installed and in PATH")
        except Exception as e:
            logger.error(f"Unexpected error while getting staged changes: {str(e)}")
        
        return None

    def get_status_changes(self,
                           porcelain_version: str = "v1",
                           include_untracked: bool = True,
                           encoding: str = 'utf-8'
                           ) -> Optional[Dict[str, List[str]]]:
        """
        Retrieve and categorize changed files in the Git repository by their status.
        
        Args:
            porcelain_version: Git porcelain version format ('v1' or 'v2')
            include_untracked: Whether to include untracked files
            encoding: Output encoding (default: 'utf-8')
        
        Returns:
            Dictionary with categorized files or None if error occurs
        """
        base_cmd = ['git']
        if self.git_repo_path:
            base_cmd.extend(['-C', self.git_repo_path])
        
        status_cmd = base_cmd + ['status', f'--porcelain={porcelain_version}']
        if include_untracked:
            status_cmd.append('--untracked-files=all')
        
        try:
            result = subprocess.run(
                status_cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding=encoding
            )
            
            changes = {'added': [],
                       'modified': [],
                       'deleted': [],
                       'renamed': [],
                       'untracked': []
                    }

            for line in result.stdout.splitlines():
                status = line[:2].strip()
                filename = line[3:]
                
                if status == 'A':
                    changes['added'].append(filename)
                elif status == 'M':
                    changes['modified'].append(filename)
                elif status == 'D':
                    changes['deleted'].append(filename)
                elif status == 'R':
                    changes['renamed'].append(filename)
                elif status == '??':
                    changes['untracked'].append(filename)
            
            if not include_untracked:
                del changes['untracked']
                
            return changes

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get Git status. Command: {' '.join(e.cmd)}\nError: {e.stderr}")
        except FileNotFoundError:
            logger.error("Git command not found. Please ensure Git is installed and in PATH")
        except Exception as e:
            logger.error(f"Unexpected error while getting Git status: {str(e)}")
        
        return None
