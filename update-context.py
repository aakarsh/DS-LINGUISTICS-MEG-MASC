#!/usr/bin/env python3

import os

INCLUDED_EXTENSIONS = {'.py', '.org', '.txt', '.md', '.toml', '.ini'}

EXCLUDED_DIRS = {'.git', 'venv', '__pycache__', '.context', '.venv-container-uc3n991', '.venv' , 'uc2n561-pytroch-container', 'TUE-RL-AGENTS-PROJECT', 'system'}

def main():
    """
    Finds relevant project files, reads their content, and combines them
    into a single context file for an LLM.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    output_file = os.path.join(script_dir, 'llm_context.txt')

    context_parts = []

    print(f"Scanning project root: {project_root}")

    for root, dirs, files in os.walk(project_root, topdown=True):
        # Exclude specified directories from traversal
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]

        for file in files:
            # Check if the file extension is in our included list
            if any(file.endswith(ext) for ext in INCLUDED_EXTENSIONS):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, project_root)

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Add a header to separate files in the context
                        header = f"--- FILE: {relative_path} ---\n"
                        context_parts.append(header + content)
                        print(f"  + Added {relative_path}")
                except Exception as e:
                    print(f"  ! Error reading {relative_path}: {e}")

    # Combine all parts and write to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(context_parts))

    print(f"\nContext successfully created at: {output_file}")

if __name__ == "__main__":
    main()
