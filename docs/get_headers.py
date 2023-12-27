from bs4 import BeautifulSoup

def extract_headings_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()

    return extract_headings(html_content)


def extract_headings(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    headings = []

    for section in soup.find_all('section'):
        section_id = section.get('id')
        h2 = section.find('h2')
        if h2:
            heading = {
                'section_id': section_id,
                'heading': h2.text.strip().replace("#", ""),
                'reference': h2.find('a', class_='headerlink').get('href'),
                'subheadings': []
            }

            # Find h3 headings within the current h2 heading
            h3_list = section.find_all('h3')
            for h3 in h3_list:
                subheading = {
                    'heading': h3.text.strip().replace("#", ""),
                    'reference': h3.find('a', class_='headerlink').get('href')
                }
                heading['subheadings'].append(subheading)

            headings.append(heading)
    headings.pop(0)
    return headings

# Example usage
file_path = 'verticapy.machine_learning.vertica.linear_model.LinearRegression.html'
headings = extract_headings_from_file(file_path)

for h2_heading in headings:
    print(f"Section ID: {h2_heading['section_id']}, Heading: {h2_heading['heading']}, Reference: {h2_heading['reference']}")
    for h3_heading in h2_heading['subheadings']:
        print(f"  Subheading: {h3_heading['heading']}, Reference: {h3_heading['reference']}")


# Create HTML structure

html_structure = '<ul>\n'

for h2_heading in headings:
    html_structure += f'<li class = ""><a class="reference internal" href="{h2_heading["reference"]}">{h2_heading["heading"]}</a><ul>\n'

    for h3_heading in h2_heading['subheadings']:
        html_structure += f'<li class = ""><a class="reference internal" href="{h3_heading["reference"]}">{h3_heading["heading"]}</a></li>\n'

    html_structure += '</ul></li>\n'

html_structure += '</ul>'

print(html_structure)