import logging
import os

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

import crawler.htmlParser


def filter_url(url):
    extensions_to_filter = ['jpg', 'jpeg', 'png', 'gif', 'pdf', 'mp4', 'mp3', 'avi', 'mov', 'xml']
    parsed_url = urlparse(url)
    path = parsed_url.path
    _, file_extension = os.path.splitext(path)
    if not file_extension or file_extension[1:] not in extensions_to_filter:
        return True
    return False


def normalize_url(url):
    agreement = url.split('//')[0]
    parts = url.split('//')[1].split('/')
    part_set = set()
    normalized_parts = []
    for part in parts:
        if part != '' and part not in part_set:
            normalized_parts.append(part)
            part_set.add(part)
    return agreement + '//' + '/'.join(normalized_parts)


def download_all_subpages(url, directory):
    # Normalize the starting URL
    base_url = urlparse(url)
    queue = [normalize_url(url)]
    visited = set([normalize_url(url)])
    # 创建目录以保存页面
    if not os.path.exists(directory):
        os.makedirs(directory)

    while queue:
        current_url = queue.pop(0)
        # logging.info("current_url: " + str(current_url))
        data = {"url": current_url}
        try:
            response = requests.post("http://209.141.37.138:7000/fetch_html", json=data, timeout=60)
        except Exception as e:
            logging.error(f"Error requests: {e}")
            if current_url == normalize_url(url):
                return False, e
            else:
                continue
        if response.status_code == 200:
            html_content = response.json().get("html")
        else:
            error = response.json().get("error")
            logging.error(f"Error fetching HTML: {error}")
            if current_url == normalize_url(url):
                return False, error
            else:
                continue
        try:
            # 保存页面内容到文件
            filename = os.path.join(directory,
                                    f"{current_url.replace('http://', '').replace('https://', '').replace('/', '__')}.html")
            with open(filename, 'w', encoding='utf-8') as file:
                file.write(html_content)
        except Exception as e:
            logging.error(f"Error write file: {e}")
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            for link in soup.find_all('a'):
                href = link.get('href')
                if href is not None:
                    href = str(href).replace('\\', '').replace('\"', '').replace('\'', '').lstrip('/')
                    if not filter_url(href):
                        continue
                    if href:
                        full_url = urljoin(current_url, href)
                        if not full_url.startswith("http"):
                            continue
                        # Normalize the full URL before checking domain
                        normalized_full_url = normalize_url(full_url)

                        # Check if the domain matches the base URL domain
                        parsed_url = urlparse(normalized_full_url)
                        if parsed_url.netloc == base_url.netloc:
                            # 设置爬虫上限
                            if len(queue) > 1000:
                                break
                            if len(visited) > 10000:
                                break
                            if normalized_full_url not in visited:
                                visited.add(normalized_full_url)
                                queue.append(normalized_full_url)
    return True, ""


def main(data, random_uuid):
    url = data["path"]
    is_success, message = download_all_subpages(url, random_uuid)
    if is_success:
        return crawler.htmlParser.parse_html_files(random_uuid, str(random_uuid) + "_output", data, random_uuid)
    else:
        return False, "", message, ""
