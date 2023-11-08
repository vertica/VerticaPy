import os
import re

# Function to find and store logo_loc value
def find_logo_loc(content):
    img_pattern = re.compile(r'<img class="sidebar-logo" src=[\'"]([^\'"]+)[\'"][^>]*>')
    match = img_pattern.search(content)
    if match:
        logo_loc = match.group(1)
        return logo_loc
    return None

# Function to replace src in img tag
def replace_src(img_tag, new_src):
    img_pattern = re.compile(r'<img[^>]*src=[\'"][^\'"]+[\'"][^>]*alt=[\'"]Clickable Image[\'"][^>]*style=[\'"]width:[^\'"]+; min-width:[^\'"]+[\'"][^>]*>')
    modified_img_tag = img_pattern.sub(f'<img src="{new_src}" alt="Clickable Image" style="width:200px; min-width:200px">', img_tag)
    return modified_img_tag


# Directory to search HTML files in
search_directory = 'build/'

# Iterate through all HTML files in the directory and its subdirectories
for root, _, files in os.walk(search_directory):
    for filename in files:
        if filename.endswith('.html'):
            file_path = os.path.join(root, filename)
            with open(file_path, 'r') as f:
                content = f.read()

            # Find the logo_loc value
            logo_loc = find_logo_loc(content)
            if logo_loc:
                # Find and replace the src attribute
                modified_content = replace_src(content, logo_loc)

                # Write the modified content back to the same file
                with open(file_path, 'w') as f:
                    f.write(modified_content)

                print(f"Fixed logo src in {file_path}")

print("Logo src fix complete.")
