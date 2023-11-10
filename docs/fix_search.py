import os


# Function to modify HTML content
def modify_html_content(html_content, search_loc):
    action_start = "class='sidebar-search-container sidebar-search-container_top' method='get' action='"
    action_end = "'"
    action_index_start = html_content.find(action_start)
    action_index_end = html_content.find(
        action_end, action_index_start + len(action_start)
    )
    new_action_value = action_start + search_loc + action_end
    modified_html_content = (
        html_content[:action_index_start]
        + new_action_value
        + html_content[action_index_end + len(action_end) :]
    )
    return modified_html_content


# Search pattern for extracting URL value
search_title_start = 'title="Search" href="'
search_title_end = '"'

# Iterate through all HTML files in 'build/' directory and its subdirectories
for root, _, files in os.walk("build/"):
    for filename in files:
        if filename.endswith(".html"):
            file_path = os.path.join(root, filename)
            with open(file_path, "r") as f:
                html_content = f.read()

            search_title_index_start = html_content.find(search_title_start)
            if search_title_index_start != -1:
                search_title_index_end = html_content.find(
                    search_title_end, search_title_index_start + len(search_title_start)
                )
                search_loc = html_content[
                    search_title_index_start
                    + len(search_title_start) : search_title_index_end
                ]
                modified_content = modify_html_content(html_content, search_loc)

                # Write the modified content back to the file
                with open(file_path, "w") as f:
                    f.write(modified_content)

                print(f"Modified HTML content in {file_path}")
            else:
                print(f"No modification needed in {file_path}")
