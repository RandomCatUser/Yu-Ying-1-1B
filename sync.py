import json
import os
import sys
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import RepositoryNotFoundError


def state_repo(api):
    return os.environ.get("STATE_REPO") or f"{api.whoami()['name']}/heoles1-train-state"


def pull(folder):
    api = HfApi()
    repo = state_repo(api)
    try:
        files = api.list_repo_files(repo, repo_type="model")
    except RepositoryNotFoundError:
        print(f"no state repo {repo} yet, starting fresh")
        return
    if not any(f.startswith(folder + "/") for f in files):
        print(f"nothing stored for {folder}, starting fresh")
        return
    snapshot_download(repo, repo_type="model", allow_patterns=[folder + "/*"], local_dir=".")
    print(f"pulled {folder} from {repo}")


def push(folder):
    state_file = os.path.join(folder, "state.json")
    if not os.path.exists(state_file):
        print(f"no state.json in {folder}, nothing to push")
        return
    api = HfApi()
    repo = state_repo(api)
    api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
    with open(state_file) as f:
        state = json.load(f)
    has_ckpt = os.path.exists(os.path.join(folder, "ckpt.pt"))
    ignore = ["__pycache__"]
    if has_ckpt and not state["done"]:
        ignore.append("*.safetensors")
    api.upload_folder(
        folder_path=folder,
        path_in_repo=folder,
        repo_id=repo,
        repo_type="model",
        ignore_patterns=ignore,
        commit_message=f"{folder} step {state['step']}/{state['total']}",
    )
    print(f"pushed {folder} to {repo}")


if __name__ == "__main__":
    action, folder = sys.argv[1], sys.argv[2]
    {"pull": pull, "push": push}[action](folder)
