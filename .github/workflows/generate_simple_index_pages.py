# SPDX-FileCopyrightText: Copyright 2024 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import re
from urllib.parse import urlparse

import requests
from jinja2 import Environment, select_autoescape

GITHUB_REPO = os.getenv("GITHUB_REPOSITORY")


def is_safe_https_url(url):
    parsed = urlparse(url)
    return parsed.scheme == "https" and bool(parsed.netloc)


def list_python_wheels():
    releases_url = f"https://api.github.com/repos/{GITHUB_REPO}/releases"
    response = requests.get(releases_url, timeout=30)

    if response.status_code != 200:
        raise Exception(
            f"Failed to fetch releases: {response.status_code} {response.text}"
        )

    wheel_files = []

    for release in response.json():
        for asset in release.get("assets", []):
            filename = asset["name"]

            if not filename.endswith(".whl"):
                continue

            pattern = r"^(?P<name>[\w\d_.]+)-(?P<version>[\d.]+)(?P<local>\+[\w\d.]+)?-(?P<python_tag>[\w]+)-(?P<abi_tag>[\w]+)-(?P<platform_tag>[\w]+)\.whl$"
            match = re.match(pattern, filename)

            if not match:
                raise ValueError(f"Invalid wheel filename: {filename}")

            download_url = asset["browser_download_url"]
            if not is_safe_https_url(download_url):
                raise ValueError(f"Invalid download URL: {download_url}")

            local_version = match.group("local")
            local_version = local_version.lstrip("+") if local_version else None

            wheel_files.append(
                {
                    "release_name": release["name"],
                    "wheel_name": filename,
                    "download_url": download_url,
                    "package_name": match.group("name"),
                    "local_version": local_version,
                }
            )

    return wheel_files


def generate_simple_index_htmls(wheels, outdir):
    template_versions_str = """
    <!DOCTYPE html>
    <html>
    <head><title>Python wheels links for {{ repo_name }}</title></head>
    <body>
    <h1>Python wheels for {{ repo_name }}</h1>

    {% for wheel in wheels %}
    <a href="{{ wheel.download_url }}">{{ wheel.wheel_name }}</a><br/>
    {% endfor %}

    </body>
    </html>
    """

    template_packages_str = """
    <!DOCTYPE html>
    <html>
    <body>
    {% for package_name in package_names %}
        <a href="{{ package_name }}/">{{ package_name }}</a><br/>
    {% endfor %}
    </body>
    </html>
    """

    env = Environment(autoescape=select_autoescape(["html", "xml"]))

    template_versions = env.from_string(template_versions_str)
    template_packages = env.from_string(template_packages_str)

    packages = {}
    for wheel in wheels:
        package_name = wheel["package_name"]
        packages.setdefault(package_name, []).append(wheel)

    html_content = template_packages.render(
        package_names=[str(k) for k in packages.keys()]
    )

    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "index.html"), "w", encoding="utf-8") as file:
        file.write(html_content)

    for package_name, package_wheels in packages.items():
        html_page = template_versions.render(
            repo_name=GITHUB_REPO,
            wheels=package_wheels,
        )
        package_dir = os.path.join(outdir, package_name)
        os.makedirs(package_dir, exist_ok=True)

        with open(os.path.join(package_dir, "index.html"), "w", encoding="utf-8") as file:
            file.write(html_page)


def generate_all_pages(args):
    wheels = list_python_wheels()

    if wheels:
        print("Python Wheels found in releases:")
        for wheel in wheels:
            print(
                f"Release: {wheel['release_name']}, "
                f"Wheel: {wheel['wheel_name']}, "
                f"URL: {wheel['download_url']}"
            )
    else:
        print("No Python wheels found in the releases.")

    generate_simple_index_htmls(wheels, outdir=args.outdir)

    wheels_per_local_version = {}
    for wheel in wheels:
        local_version = wheel["local_version"]
        if local_version is not None:
            wheels_per_local_version.setdefault(local_version, []).append(wheel)

    for local_version, local_wheels in wheels_per_local_version.items():
        local_outdir = os.path.join(args.outdir, local_version)
        os.makedirs(local_outdir, exist_ok=True)
        generate_simple_index_htmls(local_wheels, outdir=local_outdir)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Generate Python Wheels Index Pages"
    )
    argparser.add_argument(
        "--outdir",
        help="Output directory for the index pages",
        default=".",
    )
    args = argparser.parse_args()
    generate_all_pages(args)
