import subprocess
import logging
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)


class GitDiffParser:
    """Helper class to parse and categorize git diff output"""
    
    @staticmethod
    def parse(diff_content: str) -> Dict[str, List[Dict]]:
        """
        Parse raw git diff output into structured format
        
        Returns:
            {
                "files": [
                    {
                        "path": "file.txt",
                        "changes": [
                            {"type": "added", "content": "+new line"},
                            {"type": "removed", "content": "-old line"}
                        ]
                    }
                ],
                "stats": {
                    "files_changed": 3,
                    "insertions": 10,
                    "deletions": 5
                }
            }
        """
        if not diff_content:
            return {"files": [], "stats": {}}
        
        files = []
        current_file = None
        stats = {"files_changed": 0, "insertions": 0, "deletions": 0}
        
        for line in diff_content.splitlines():
            # File header
            if line.startswith("diff --git"):
                if current_file:
                    files.append(current_file)
                file_path = line.split(" b/")[-1]
                current_file = {"path": file_path, "changes": []}
                stats["files_changed"] += 1
            # Code changes
            elif line.startswith("+") and not line.startswith("+++"):
                current_file["changes"].append({"type": "added", "content": line[1:]})
                stats["insertions"] += 1
            elif line.startswith("-") and not line.startswith("---"):
                current_file["changes"].append({"type": "removed", "content": line[1:]})
                stats["deletions"] += 1
            elif line.startswith(" ") and current_file:
                current_file["changes"].append({"type": "context", "content": line[1:]})
        
        if current_file:
            files.append(current_file)
            
        return {"files": files, "stats": stats}


class GitService:
    """
    A utility class for analyzing Git repository changes including staged changes and file status.
    
    This class provides methods to:
    - Retrieve and parse staged changes diff
    - Categorize and analyze file status changes
    """

    def __init__(self, repo_path: Optional[str] = None):
        """
        Initialize the GitService.
        
        Args:
            repo_path: Path to the Git repository (default: current directory)
        """
        self.repo_path = repo_path
        self.diff_parser = GitDiffParser()
    
    def _run_git_command(self, 
                        cmd_args: List[str],
                        encoding: str = 'utf-8',
                        errors: str = 'strict') -> Optional[str]:
        """Internal helper to run git commands"""
        base_cmd = ['git']
        
        if self.repo_path:
            base_cmd.extend(['-C', str(self.repo_path)])
        
        full_cmd = base_cmd + cmd_args
        
        try:
            result = subprocess.run(
                full_cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding=encoding,
                errors=errors
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            logger.error(f"Git command failed: {' '.join(e.cmd)}\nError: {e.stderr}")
        except FileNotFoundError:
            logger.error("Git command not found. Ensure Git is installed and in PATH")
        except Exception as e:
            logger.error(f"Unexpected error running Git command: {str(e)}")
        return None

    def get_staged_changes_diff(self,
                              extra_args: Optional[List[str]] = None,
                              encoding: str = 'utf-8') -> Dict[str, List[Dict]]:
        """
        Get structured staged changes diff with categorized changes.
        
        Args:
            extra_args: Additional git diff arguments
            encoding: Output encoding
            
        Returns:
            Parsed diff structure with categorized changes
            {
                "files": [
                    {
                        "path": "file.txt", 
                        "changes": [
                            {"type": "added/removed/context", "content": "line content"}
                        ]
                    }
                ],
                "stats": {
                    "files_changed": int,
                    "insertions": int,
                    "deletions": int
                }
            }
        """
        diff_args = ['diff', '--staged', '--color=never']
        if extra_args:
            diff_args.extend(extra_args)
        self.raw_diff = self._run_git_command(diff_args, encoding=encoding)
        
    def get_enhanced_diff(self) -> str:
        """
        Generate enhanced diff output optimized for LLM processing.
        
        Returns:
            Formatted string with categorized changes suitable for commit messages
        """
        diff_data = self.diff_parser.parse(self.raw_diff)
        if not diff_data or not diff_data["files"]:
            return ""
            
        output = []
        summary = diff_data["stats"]
        
        # Add summary header
        output.append(
            f"Summary: {summary['files_changed']} files changed, "
            f"{summary['insertions']} insertions(+), "
            f"{summary['deletions']} deletions(-)\n"
        )
        
        # Add detailed changes per file
        for file in diff_data["files"]:
            file_header = f"\nFile: {file['path']}\n"
            changes = []
            
            for change in file["changes"]:
                if change["type"] == "added":
                    changes.append(f"[+] {change['content']}")
                elif change["type"] == "removed":
                    changes.append(f"[-] {change['content']}")
                # Context lines can be optionally included
            
            if changes:
                output.append(file_header)
                output.extend(changes[:20])  # Limit to 20 changes per file
        
        return "\n".join(output)

    def get_status_changes(self,
                           porcelain_version: str = "v1",
                           include_untracked: bool = False) -> Dict[str, List[str]]:
        """
        Get categorized file status changes.
        
        Args:
            porcelain_version: Git porcelain format version
            include_untracked: Whether to include untracked files
            
        Returns:
            Dictionary of categorized file changes
        """
        status_args = ['status', f'--porcelain={porcelain_version}']
        if include_untracked:
            status_args.append('--untracked-files=all')
            
        raw_status = self._run_git_command(status_args)
        if not raw_status:
            return {}
            
        changes = {
            'added': [],
            'modified': [],
            'deleted': [],
            'renamed': [],
            'untracked': []
        }
        
        for line in raw_status.splitlines():
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
            changes.pop('untracked', None)
            
        return changes
