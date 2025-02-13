import os
import requests
from git import Repo
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
BASE_DIR = os.getenv("BASE_DIR")
MODEL_DIR = "./models"  # Load model from local directory

# Load the GPT-J 6B model and tokenizer from local storage
print("Loading GPT-J 6B model from local storage...")
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, use_safetensors=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)


def generate_commit_message(repo_name, changes):
    """
    Generate a random commit message using predefined templates.
    """
    templates = [
        f"Automated update for {repo_name}: {changes}",
        f"{repo_name}: {changes} made using GPT.",
        f"Small tweaks to {repo_name}: {changes}",
        f"Cleaned up code or updated docs for {repo_name}: {changes}"
    ]
    return random.choice(templates)


def fetch_repos():
    """
    Fetch all public repositories for the authenticated user.
    """
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}"}
    response = requests.get("https://api.github.com/user/repos?type=public", headers=headers)
    response.raise_for_status()
    return [repo["clone_url"] for repo in response.json()]


def sync_repo(repo_url):
    """
    Clone the repository if not already cloned; otherwise, pull the latest changes.
    """
    repo_name = repo_url.split("/")[-1].replace(".git", "")
    repo_path = os.path.join(BASE_DIR, repo_name)
    
    if not os.path.exists(repo_path):
        print(f"Cloning {repo_name}...")
        Repo.clone_from(repo_url, repo_path)
    else:
        print(f"Pulling latest changes for {repo_name}...")
        repo = Repo(repo_path)
        repo.remotes.origin.pull()

    return repo_path


def generate_edit(file_content):
    """
    Generate a small edit using the GPT-J 6B model.
    """
    input_ids = tokenizer(file_content, return_tensors="pt").input_ids
    output = model.generate(input_ids, max_length=len(input_ids) + 10, do_sample=True, temperature=0.7)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text


def update_repo(repo_path):
    """
    Update a repository by making small improvements and committing the changes.
    """
    print(f"Updating repository at {repo_path}...")
    repo = Repo(repo_path)
    modified = False

    # Example: Update README.md
    readme_path = os.path.join(repo_path, "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r") as file:
            original_content = file.read()
        updated_content = generate_edit(original_content)

        # Check if the generated edit is significantly different from the original
        if updated_content != original_content and len(updated_content) > len(original_content) * 0.8:
            with open(readme_path, "w") as file:
                file.write(updated_content)
            repo.git.add("README.md")
            modified = True

            # Generate a commit message
            commit_message = generate_commit_message(repo_name=repo.working_tree_dir.split('/')[-1], changes="Auto-edited README")
            repo.index.commit(commit_message)
            repo.remote(name="origin").push()
            print(f"Changes pushed to {repo.working_tree_dir} with message: '{commit_message}'")

    if not modified:
        print(f"No changes made to {repo_path}")


def process_repos():
    """
    Main function to process all repositories by fetching, syncing, and updating them.
    """
    print("Starting repository updates...")
    
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)

    try:
        repos = fetch_repos()
        for repo_url in repos:
            repo_path = sync_repo(repo_url)
            update_repo(repo_path)
    except Exception as e:
        print(f"Error occurred: {e}")


if __name__ == "__main__":
    process_repos()