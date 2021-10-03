import subprocess


# Reference: https://stackoverflow.com/a/21901260
def git_revision_hash() -> str:
    """Return the git commit hash of the current directory.

    Note:
        Will return a `fatal: not a git repository` string if the command fails.
    """
    try:
        string = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT
        )
    except subprocess.CalledProcessError as err:
        string = err.output
    return string.decode("ascii").strip()


# Alias.
git_commit_hash = git_revision_hash
