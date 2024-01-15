# Fix the Header Links for Home/User Guide/Api Reference etc

import os
import re

def find_html_files(folder_path):
    html_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".html"):
                html_files.append(os.path.join(root, file))
    return html_files

def relative_path_to_notebooks(notebooks_path, html_file_path):
    notebooks_path = os.path.abspath(notebooks_path)
    html_file_path = os.path.abspath(html_file_path)
    
    # Count the number of directories between the HTML file and the notebooks folder
    relative_path = os.path.relpath(html_file_path, notebooks_path)
    num_directories = len(relative_path.split(os.path.sep)) - 1
    
    # Print the corresponding number of "../"
    relative_path_to_print = "../" * (num_directories + 1)
    return relative_path_to_print

def replace_top_button_link(html_file_path, replacement):
    with open(html_file_path, 'r') as file:
        content = file.read()

    # Replace the first pattern
    updated_content, count1 = re.subn(r'<a class="top-button" href="\./', f'<a class="top-button" href="{replacement}', content)

    # Replace the second pattern
    updated_content, count2 = re.subn(r'<a href="./home.html">', f'<a href="{replacement}home.html">', updated_content)

    # Write the updated content back to the file
    with open(html_file_path, 'w') as file:
        file.write(updated_content)

    # Check if patterns were found and print a message
    if count1 > 0:
        print(f"Replaced Top Header links in {html_file_path}")
    if count2 > 0:
        print(f"Replaced Logo links in {html_file_path}")

def main():
    notebooks_path = "build/html/notebooks/" 
    print("Starting the crawl")
    html_files = find_html_files(notebooks_path)
    for html_file in html_files:
        relative_path = relative_path_to_notebooks(notebooks_path, html_file)
        replace_top_button_link(html_file, relative_path)

if __name__ == "__main__":
    main()
