import os

# Set this to your real Obsidian vault folder path.
# Example Linux/WSL: "/home/youruser/ObsidianVault"
# Example Windows:   "C:\\Users\\youruser\\Documents\\ObsidianVault"
VAULT_PATH = os.getenv("OBSIDIAN_VAULT_PATH", "").strip()

if not VAULT_PATH:
    raise RuntimeError(
        "Set OBSIDIAN_VAULT_PATH environment variable to your Obsidian vault folder path.\n"
        "Example:\n"
        "  export OBSIDIAN_VAULT_PATH=/home/youruser/ObsidianVault"
    )
