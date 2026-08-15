#!/usr/bin/env python
"""
Automated GitHub repository setup script
Requirements: git installed, GitHub CLI (gh) installed, authenticated
"""
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a shell command and report results"""
    print(f"\n{'='*70}")
    print(f"▶ {description}")
    print(f"{'='*70}")
    print(f"Command: {cmd}\n")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        if result.returncode != 0:
            print(f"❌ Command failed with exit code {result.returncode}")
            return False
        print("✓ Success")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    project_dir = Path(".")
    repo_name = "urban-heat-island-ml"
    repo_description = "Urban Heat Island ML Analysis with Ensemble Learning and Streamlit Deployment"
    
    print("\n" + "="*70)
    print("🚀 GITHUB REPOSITORY SETUP")
    print("="*70)
    print(f"Project: {repo_name}")
    print(f"Location: {project_dir.absolute()}")
    
    # Check requirements
    print("\n" + "="*70)
    print("📋 CHECKING REQUIREMENTS")
    print("="*70)
    
    git_ok = run_command("git --version", "Check Git installation")
    if not git_ok:
        print("\n❌ Git not found. Install from https://git-scm.com/download/win")
        sys.exit(1)
    
    gh_ok = run_command("gh --version", "Check GitHub CLI installation")
    if not gh_ok:
        print("\n⚠️  GitHub CLI not found. Install from https://cli.github.com")
        print("   Or use manual setup in GITHUB_SETUP.md")
        use_gh = input("\nContinue without GitHub CLI? (y/n): ").lower()
        if use_gh != 'y':
            sys.exit(1)
    
    # Initialize repository
    print("\n" + "="*70)
    print("📦 INITIALIZING REPOSITORY")
    print("="*70)
    
    run_command("git init", "Initialize Git repository")
    run_command('git config user.name "Urban Heat Island ML"', "Set Git user name")
    run_command('git config user.email "ml@example.com"', "Set Git user email")
    run_command("git add .", "Stage all files")
    run_command(
        'git commit -m "Initial commit: Urban Heat Island ML with Streamlit deployment"',
        "Create initial commit"
    )
    
    # Create GitHub repository
    if gh_ok:
        print("\n" + "="*70)
        print("🌐 CREATING GITHUB REPOSITORY")
        print("="*70)
        
        cmd = f'gh repo create {repo_name} --private --source=. --remote=origin --push'
        if run_command(cmd, f"Create private repository '{repo_name}' and push"):
            print("\n" + "="*70)
            print("✅ REPOSITORY CREATED SUCCESSFULLY!")
            print("="*70)
            run_command(
                f"gh repo view {repo_name} --web",
                "Open repository in browser"
            )
        else:
            print("\n❌ Failed to create repository with GitHub CLI")
            print("   Try manual setup in GITHUB_SETUP.md")
    else:
        print("\n" + "="*70)
        print("📝 MANUAL SETUP REQUIRED")
        print("="*70)
        print(f"""
1. Create repository at: https://github.com/new
   - Name: {repo_name}
   - Description: {repo_description}
   - Privacy: Private
   - Click "Create repository"

2. Add remote (copy URL from GitHub):
   git remote add origin https://github.com/YOUR_USERNAME/{repo_name}.git
   git branch -M main
   git push -u origin main

3. When prompted, use Personal Access Token:
   - Create at: https://github.com/settings/tokens
   - Select 'repo' scope
   - Paste token in terminal

See GITHUB_SETUP.md for detailed instructions.
        """)

if __name__ == "__main__":
    main()
