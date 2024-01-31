#!/usr/bin/env python3

import os
import re
from bs4 import BeautifulSoup

# Get the directory where the script is located
script_dir = os.path.dirname(__file__)

# Define the directory path to start the search
start_dir = os.path.join(script_dir, "build", "html")

# Regular expression pattern to find occurrences
pattern_to_find = r'html">verticapy\.'


def process_html_content(content):
    soup = BeautifulSoup(content, "html.parser")

    # Find all elements with class "next-page" and "prev-page"
    page_links = soup.find_all(class_=["next-page", "prev-page"])

    for link in page_links:
        # Find the element with class "title" inside each page link
        title_element = link.find(class_="title")

        if title_element:
            # Get the content of the "title" element
            title_content = title_element.get_text()

            # Check if "verticapy" is in the content
            if "verticapy" in title_content:
                # Remove everything until the last period "."
                processed_title = title_content.rsplit(".", 1)[-1].strip()

                # Replace the content of the "title" element
                title_element.string = processed_title

    # Find all elements with class "gp" and remove their content [For removing extra text in code blocks]
    for element in soup.find_all(class_="gp"):
        element.clear()

    return str(soup)


# Walk through the directory and subdirectories
for root, _, files in os.walk(start_dir):
    for filename in files:
        if filename.endswith(".html"):
            file_path = os.path.join(root, filename)

            # Read the content of the HTML file
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()

            # Process the content in memory
            modified_content = process_html_content(content)

            # Use regular expression to remove occurrences
            modified_content = re.sub(pattern_to_find, 'html">', modified_content)

            # Write the modified content back to the file
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(modified_content)

            print(f"Modified: {file_path}")
