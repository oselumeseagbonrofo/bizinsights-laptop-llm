"""
Deploy BizInsights SME Customer Support LLM to Hugging Face Spaces
"""
import sys
import os
import argparse
from pathlib import Path
from huggingface_hub import HfApi, login

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

SPACE_NAME = "bizinsights-sme-assistant"
ROOT_DIR = Path(__file__).resolve().parent
HF_SPACE_DIR = ROOT_DIR / "hf_space"


def main():
    parser = argparse.ArgumentParser(description="Deploy BizInsights Assistant to Hugging Face Spaces")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face User Access Token (with Write permission)")
    parser.add_argument("--space-name", type=str, default=SPACE_NAME, help="Space repository name")
    args = parser.parse_args()

    print("=" * 64)
    print("  Deploying BizInsights Assistant to Hugging Face Spaces")
    print("=" * 64)

    token = args.token or os.environ.get("HF_TOKEN")
    if token:
        login(token=token, add_to_git_credential=True)
        api = HfApi(token=token)
    else:
        api = HfApi()

    # 1. Verify authentication & check token role
    try:
        user_info = api.whoami()
        username = user_info["name"]
        auth = user_info.get("auth", {})
        access_token_info = auth.get("accessToken", {})
        role = access_token_info.get("role", "unknown")
        token_name = access_token_info.get("displayName", "default")
        print(f"[OK] Authenticated as Hugging Face user: '{username}'")
        print(f"[INFO] Active token: '{token_name}' (role: {role})")

        if role == "read":
            print("\n" + "!" * 64)
            print("  [ACTION REQUIRED] Write Permission Needed")
            print("!" * 64)
            print("  Your current Hugging Face token is 'read-only', which cannot")
            print("  create or push new Spaces under your account.")
            print("\n  To fix this in 30 seconds:")
            print("  1. Go to: https://huggingface.co/settings/tokens")
            print("  2. Click 'Create new token'")
            print("  3. Set Token type to: 'Write' (or check 'Create and manage Spaces')")
            print("  4. Copy your token and re-run this script:")
            print(f"     python deploy_to_hf.py --token <YOUR_WRITE_TOKEN>")
            print("!" * 64 + "\n")
            sys.exit(1)

    except Exception as e:
        print(f"[ERROR] Hugging Face authentication failed: {e}")
        print("Please provide a token using: python deploy_to_hf.py --token <YOUR_WRITE_TOKEN>")
        sys.exit(1)

    repo_id = f"{username}/{args.space_name}"
    print(f"[INFO] Target Space: {repo_id}")

    # 2. Create the space if it does not exist
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="docker",
            exist_ok=True,
            private=False
        )
        print(f"[OK] Space repository '{repo_id}' ready on Hugging Face.")
    except Exception as e:
        print(f"[ERROR] Failed to create Space repo: {e}")
        sys.exit(1)

    # 3. Upload all deployment files from hf_space/
    print(f"[INFO] Uploading deployment bundle from {HF_SPACE_DIR.name}...")
    try:
        api.upload_folder(
            folder_path=str(HF_SPACE_DIR),
            repo_id=repo_id,
            repo_type="space",
            commit_message="Deploy BizInsights SME Assistant with mobile UI and Q8_0 GGUF",
        )
        print("[OK] All deployment files successfully uploaded!")
    except Exception as e:
        print(f"[ERROR] Failed to upload files: {e}")
        sys.exit(1)

    space_url = f"https://huggingface.co/spaces/{repo_id}"
    app_url = f"https://{username.replace('_', '-')}-{args.space_name.replace('_', '-')}.hf.space"

    print("\n" + "=" * 64)
    print("  [SUCCESS] DEPLOYMENT INITIATED TO HUGGING FACE SPACES!")
    print("=" * 64)
    print(f"  Space Dashboard : {space_url}")
    print(f"  Direct Web App  : {app_url}")
    print("=" * 64)
    print("  Hugging Face is building the Docker container and initializing the model.")
    print("  The build typically completes in 2-4 minutes.")
    print("  Once ready, anyone in the world can access your assistant from any")
    print("  mobile phone or computer without needing to be on your local network!")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    main()
