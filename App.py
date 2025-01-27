

import os
import openai
import requests
from git import Repo
import random


# Set OpenAI API Key
openai.api_key = OPENAI_API_KEY


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
    Use OpenAI to suggest small improvements for a given file's content.
    """
    prompt = f"Make small improvements to this code or documentation without changing its functionality:\n\n{file_content}"
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=500
    )
    return response.choices[0].text.strip()


def summarize_changes(original_content, updated_content):
    """
    Use OpenAI to summarize the changes made between two versions of a file.
    """
    prompt = (
        "Summarize the changes made between the following two versions of a file:\n\n"
        "Original:\n" + original_content[:500] + "\n\n"
        "Updated:\n" + updated_content[:500] + "\n\n"
        "Write a short description:"
    )
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=50
    )
    return response.choices[0].text.strip()


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
        if updated_content != original_content:
            with open(readme_path, "w") as file:
                file.write(updated_content)
            repo.git.add("README.md")
            modified = True
            changes = summarize_changes(original_content, updated_content)
            commit_message = generate_commit_message(repo_name=repo.working_tree_dir.split('/')[-1], changes=changes)
            repo.index.commit(commit_message)
            origin = repo.remote(name="origin")
            origin.push()
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
