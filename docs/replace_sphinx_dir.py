import os

# Define the directory to search in
search_directory = "../"

# Define the word to search for and the replacement
search_word = "SPHINX_DIRECTORY"
replacement_word = "/project/data/VerticaPy/docs"


# Function to search and replace within a file
def search_replace_in_file(file_path):
    with open(file_path, "r") as file:
        content = file.read()
        updated_content = content.replace(search_word, replacement_word)

    with open(file_path, "w") as file:
        file.write(updated_content)


# Recursively search for Python files and perform the search and replace
for root, dirs, files in os.walk(search_directory):
    for file in files:
        if (file.endswith(".py") or file.endswith(".rst")) and not file.endswith(
            "replace_sphinx_dir.py"
        ):
            file_path = os.path.join(root, file)
            search_replace_in_file(file_path)
            print(f"Updated: {file_path}")

print("Search and replace complete.")
