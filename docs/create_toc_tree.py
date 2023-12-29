import os
import re
from typing import Optional

from bs4 import BeautifulSoup

# Main Functions


def count_elements_in_toc_tree(html_content: str) -> int:
    """
    ...
    """
    soup = BeautifulSoup(html_content, "html.parser")
    toc_tree = soup.find("div", class_="toc-tree")

    if toc_tree:
        # Find all ul elements inside the toc-tree
        ul_elements = toc_tree.find_all("ul")
        return len(ul_elements)
    else:
        return 0


def extract_headings(html_content: str) -> Optional[list]:
    """
    ...
    """
    soup = BeautifulSoup(html_content, "html.parser")
    headings = []

    for section in soup.find_all("section"):
        section_id = section.get("id")
        h2 = section.find("h2")
        if h2 and h2.find("a", class_="headerlink") is not None:
            heading = {
                "section_id": section_id,
                "heading": h2.text.strip().replace("#", ""),
                "reference": h2.find("a", class_="headerlink").get("href"),
                "subheadings": [],
            }

            # Find h3 headings within the current h2 heading
            h3_list = section.find_all("h3")
            for h3 in h3_list:
                subheading = {
                    "heading": h3.text.strip().replace("#", ""),
                    "reference": h3.find("a", class_="headerlink").get("href"),
                }
                heading["subheadings"].append(subheading)

            headings.append(heading)
    if not len(headings) == 0:
        headings.pop(0)
        return headings
    else:
        return None


def create_html_list(headings: list) -> str:
    """
    ...
    """

    # for h2_heading in headings:
    #     print(f"Section ID: {h2_heading['section_id']}, Heading: {h2_heading['heading']}, Reference: {h2_heading['reference']}")
    #     for h3_heading in h2_heading['subheadings']:
    #         print(f"  Subheading: {h3_heading['heading']}, Reference: {h3_heading['reference']}")

    # Create HTML structure

    html_structure = ""

    for h2_heading in headings:
        html_structure += f'<li class = ""><a class="reference internal" href="{h2_heading["reference"]}">{h2_heading["heading"]}</a>\n'
        count = 0
        for h3_heading in h2_heading["subheadings"]:
            if count == 0:
                html_structure += "<ul>"
            html_structure += f'<li class = ""><a class="reference internal" href="{h3_heading["reference"]}">{h3_heading["heading"]}</a></li>\n'
            count += 1
        # html_structure += '</ul></li>\n'

    # html_structure += '</ul>'

    # print(html_structure)
    return html_structure


# Example usage
# file_path = 'verticapy.vDataFrame.bar.html'
# file_path = 'verticapy.machine_learning.vertica.linear_model.LinearRegression.html'


def get_headers(file_path: str) -> Optional[str]:
    """
    ...
    """
    with open(file_path, "r", encoding="utf-8") as file:
        html_content = file.read()
    if count_elements_in_toc_tree(html_content) > 2:
        headings = extract_headings(html_content)
        create_html_list(headings)
    else:
        print(f"Not applicable to:{file_path}")


def process_html_file(file_path: str, replace_content: str) -> None:
    """
    ...
    """

    # Read the HTML file
    with open(file_path, "r") as file:
        content = file.read()

    # Process the content (modify as needed)
    modified_content = process_content(content, replace_content)

    # Write the modified content back to the file
    if modified_content is not None:
        with open(file_path, "w") as file:
            file.write(modified_content)


def process_content(content: str, replace_content: str):
    """
    ...
    """

    # Define the pattern to search for
    # pattern = r'<a class="reference internal" href="#">.*?</a><ul>'
    pattern = r'<a class="reference internal" href="#">.*?</a><ul>.*?</ul>'

    # Find all occurrences of the pattern
    matches = re.findall(pattern, content, re.DOTALL)

    # Print a statement to confirm if matches are found
    if matches:
        print("Matches found!")
    else:
        print("No matches found.")

    # Process each match and replace it with the desired content
    html_content = None
    for match in matches:
        # print(match)
        modified_match = f"""
            {match}
            {replace_content}        
        """
        html_content = content.replace(match, modified_match)

    if html_content:
        return html_content
    else:
        return None


def get_ignore_file_names(folder_path: str) -> list:
    # Get a list of all files in the folder
    files = os.listdir(folder_path)

    # Filter files with .rst extension
    rst_files = [file for file in files if file.endswith(".rst")]

    html_files = [file.replace(".rst", ".html") for file in rst_files]
    # Print the list of .rst files
    html_files.append("index.html")  # For Jupyter Notebooks
    return html_files


if __name__ == "__main__":
    search_directory = "build/"
    rst_ignore_directory = "./source/"

    # Iterate through all HTML files in the directory and its subdirectories
    files_to_ignore = get_ignore_file_names(rst_ignore_directory)
    for root, _, files in os.walk(search_directory):
        for filename in files:
            if filename.endswith(".html") and filename not in files_to_ignore:
                file_path = os.path.join(root, filename)
                with open(file_path, "r", encoding="utf-8") as file:
                    html_content = file.read()
                if count_elements_in_toc_tree(html_content) >= 1:
                    headings = extract_headings(html_content)
                    if not headings == None:
                        process_html_file(file_path, create_html_list(headings))
                        print(f"Updated:{file_path}")
                    else:
                        print(f"Heading Unchanged:{file_path}")
                else:
                    print(f"No TOC:{file_path}")
