import http.server
import socketserver
import urllib.parse
from typing import Dict, List
import requests
import json
from bs4 import BeautifulSoup
import os


class MyHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            paragraphs = parse_html()
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(paragraphs).encode())
        else:
            self.send_response(404)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Page not found.")


def parse_html() -> List[Dict[str, str]]:
    """
    Parse HTML content from the given URL, and extract paragraphs that are separated by <h2> tags.

    Returns a list of paragraphs, where each paragraph is a dictionary with two keys:
    - "title": the text content of the <h2> tag (or empty string if there is no <h2> tag).
    - "description": the text content of the paragraph.
    """
    url = os.environ.get('BROWSERLESS_URL')

    x = requests.post(
        url,
        json={
            "url": "https://www.feishu.cn/hc/zh-CN/articles/360049067483",
            "userAgent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.122 Safari/537.36 Edg/81.0.416.64",
            "gotoOptions": {"timeout": "20000", "waitUntil": "networkidle2"},
        },
    )

    soup = BeautifulSoup(x.text, "html.parser")

    html = soup.find_all(
        name="div",
        attrs={"class": "heraAdit-articleBody js-heraAdit-richText-body"},
    )
    soup = BeautifulSoup(str(html[0]), "html.parser")

    paragraphs = []
    curr_heading = ""
    curr_content = ""
    for elem in soup.find_all():
        if "class" in elem.attrs and "heading-h3" in elem.attrs["class"]:
            if curr_heading:
                paragraphs.append({"title": curr_heading, "description": curr_content})
            curr_heading = (
                elem.text.replace("\u200b", "").replace("\n", "").replace(" ", "")
            )
            curr_content = ""
        else:
            new_content = (
                elem.text.replace("\u200b", "").replace("\n", "").replace(" ", "")
            )

            if new_content == curr_heading or new_content in curr_content:
                curr_content += ""
            else:
                curr_content += new_content
    if curr_heading:
        paragraphs.append({"title": curr_heading, "description": curr_content})

    return paragraphs


if __name__ == "__main__":
    PORT = 8000
    with socketserver.TCPServer(("", PORT), MyHandler) as httpd:
        print(f"Serving at port {PORT}")
        httpd.serve_forever()
