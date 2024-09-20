import requests
import os
import argparse
from jinja2 import Template
import re

# Automatically get the repository name in the format "owner/repo" from the GitHub workflow environment
GITHUB_REPO = os.getenv("GITHUB_REPOSITORY")


def list_python_wheels():
    # GitHub API URL for releases
    releases_url = f"https://api.github.com/repos/{GITHUB_REPO}/releases"

    response = requests.get(releases_url)

    if response.status_code != 200:
        raise Exception(f"Failed to fetch releases: {response.status_code} {response.text}")

    releases = response.json()

    wheel_files = []

    # Iterate through releases and assets
    for release in releases:
        assets = release.get("assets", [])
        for asset in assets:
            filename = asset["name"]
            if filename.endswith(".whl"):
                pattern = r'^(?P<name>[\w\d_.]+)-(?P<version>[\d.]+)(?P<local>\+[\w\d.]+)?-(?P<python_tag>[\w]+)-(?P<abi_tag>[\w]+)-(?P<platform_tag>[\w]+)\.whl'
    
                match = re.match(pattern, filename)
    
                if match:
                    local_version = match.group('local')
                    if local_version:
                        local_version = local_version.lstrip('+')  # Return the local version without the '+' sign
                    else:
                        local_version = None
                else:
                    raise ValueError(f"Invalid wheel filename: {filename}")
                wheel_files.append({
                    "release_name": release["name"],
                    "wheel_name": asset["name"],
                    "download_url": asset["browser_download_url"],
                    "package_name":  match.group('name'),
                    "local_version": local_version,
                })

    return wheel_files


def generate_simple_index_htmls(wheels, outdir):
    # Jinja2 template as a string
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
    <html>
    <body>
    {% for package_name in package_names %}
        <a href="{{package_name}}/">{{package_name}}</a><br/>
    {% endfor %}
    </body>
    </html>
    """

    # Create a Jinja2 Template object from the string
    template_versions = Template(template_versions_str)
    template_packages = Template(template_packages_str)

    # group the wheels by package name
    packages = {}
    for wheel in wheels:
        package_name = wheel['package_name']
        if package_name not in packages:
            packages[package_name] = []
        packages[package_name].append(wheel)
    
    # Render the HTML the list the package names
    html_content = template_packages.render(package_names=[str(k) for k in packages.keys()])
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "index.html"), "w") as file:
        file.write(html_content)

    # for each package, render the HTML to list the wheels
    for package_name, wheels in packages.items():
        html_page = template_versions.render(repo_name=GITHUB_REPO, wheels=wheels)
        os.makedirs(os.path.join(outdir, package_name), exist_ok=True)
        with open(os.path.join(outdir, package_name, "index.html"), "w") as file:
            file.write(html_page)

 
def generate_all_pages():
    wheels = list_python_wheels()
    if wheels:
        print("Python Wheels found in releases:")
        for wheel in wheels:
            print(f"Release: {wheel['release_name']}, Wheel: {wheel['wheel_name']}, URL: {wheel['download_url']}")
    else:
        print("No Python wheels found in the releases.")

    # Generate Simple Index HTML pages the wheel with all local versions
    generate_simple_index_htmls(wheels, outdir=args.outdir)

    # group wheels per local version
    wheels_per_local_version = {}
    for wheel in wheels:
        local_version = wheel['local_version']
        if local_version not in wheels_per_local_version:
            wheels_per_local_version[local_version] = []
        wheels_per_local_version[local_version].append(wheel)
    
    # create a subdirectory for each local version
    for local_version, wheels in wheels_per_local_version.items():
        os.makedirs(os.path.join(args.outdir, local_version), exist_ok=True)
        generate_simple_index_htmls(wheels, outdir=os.path.join(args.outdir, local_version))

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Generate Python Wheels Index Pages")
    argparser.add_argument("--outdir", help="Output directory for the index pages", default=".")
    args = argparser.parse_args()
    generate_all_pages()




