"""
Enhanced dataset building script.
Processes website templates from project_templates/ and creates/updates the dataset with advanced analysis.
"""

import os
import json
import re
import logging
from pathlib import Path
from bs4 import BeautifulSoup
from typing import Dict, List, Optional, Tuple, Set
from collections import Counter
import hashlib
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration - get paths relative to project root
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
PROJECT_TEMPLATES_DIR = PROJECT_ROOT / "project_templates"
DATASET_DIR = PROJECT_ROOT / "dataset"
OUTPUT_FILE = DATASET_DIR / "train.jsonl"
LOG_FILE = DATASET_DIR / "build.log"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Configuration constants
MAX_HTML_LENGTH = 1000000  # Max chars for HTML in training example (for full pages)
MAX_CSS_LENGTH = 50000  # Max chars for CSS in training example
MAX_JS_LENGTH = 30000  # Max chars for JS in training example
MAX_TOTAL_LENGTH = 200000  # Max total chars for code sections

# Component-level limits (smaller for individual components)
MAX_COMPONENT_HTML_LENGTH = 30000  # Max chars for component HTML
MAX_COMPONENT_CSS_LENGTH = 20000  # Max chars for component CSS
MAX_COMPONENT_JS_LENGTH = 10000  # Max chars for component JS


def is_obfuscated_or_minified(code: str, code_type: str = "js") -> bool:
    """Detect if code is obfuscated or minified"""
    if not code or len(code.strip()) < 50:
        return False

    code_lower = code.lower()

    # Check for common obfuscation patterns
    obfuscation_patterns = [
        r"_0x[a-f0-9]+",  # Hex obfuscation like _0x1a2b
        r'[a-z]\s*=\s*["\'][a-z]{1,3}["\']',  # Single char variables
        r"eval\s*\(",  # eval() usage
        r"atob\s*\(",  # Base64 decoding
        r"String\.fromCharCode",  # Character code obfuscation
        r"\\x[0-9a-f]{2}",  # Hex escape sequences
    ]

    for pattern in obfuscation_patterns:
        if re.search(pattern, code, re.IGNORECASE):
            return True

    # Check for minified code characteristics
    lines = code.split("\n")
    non_empty_lines = [l for l in lines if l.strip()]

    if not non_empty_lines:
        return False

    # Minified code typically has very long lines and few line breaks
    avg_line_length = sum(len(l) for l in non_empty_lines) / len(non_empty_lines)
    line_break_ratio = len(non_empty_lines) / len(code) if code else 0

    # Check for minified indicators
    if code_type == "js":
        # Minified JS: very long lines (>500 chars), few breaks, no comments
        if avg_line_length > 500 and line_break_ratio < 0.01:
            return True
        # No spaces around operators/braces (common in minified code)
        if re.search(r"[a-z]\{[a-z]|[a-z]\}[a-z]", code) and avg_line_length > 200:
            return True
    elif code_type == "css":
        # Minified CSS: very long lines, no formatting
        if avg_line_length > 300 and line_break_ratio < 0.005:
            return True
        # No spaces after colons or semicolons
        if re.search(r":[a-z]|;[a-z]", code) and avg_line_length > 200:
            return True

    # Check for build tool artifacts
    build_indicators = [
        "webpack",
        "rollup",
        "bundle",
        "chunk",
        "dist/",
        "build/",
        "/*!",
        "/*#",
        "sourceMappingURL",
        "sourcemap",
    ]
    if any(indicator in code_lower for indicator in build_indicators):
        # Only skip if it's clearly a bundle (has minified characteristics)
        if avg_line_length > 200:
            return True

    return False


def replace_image_urls(html_content: str, assets_dir: Path) -> str:
    """Replace all image URLs with picsum.photos placeholders"""
    soup = BeautifulSoup(html_content, "html.parser")

    # Default dimensions for different image contexts
    default_dimensions = {
        "hero": (1920, 1080),
        "banner": (1200, 600),
        "thumbnail": (300, 300),
        "icon": (64, 64),
        "logo": (200, 100),
        "default": (800, 600),
    }

    # Replace img src attributes
    for img in soup.find_all("img"):
        src = img.get("src", "")
        if src and not src.startswith("https://picsum.photos"):
            # Try to get dimensions from attributes
            width = img.get("width")
            height = img.get("height")

            # Try to infer from class names or parent context
            if not width or not height:
                classes = " ".join(img.get("class", [])).lower()
                parent_classes = " ".join(
                    img.parent.get("class", []) if img.parent else []
                ).lower()
                context = classes + " " + parent_classes

                if any(word in context for word in ["hero", "banner", "header"]):
                    width, height = default_dimensions["hero"]
                elif any(word in context for word in ["thumb", "thumbnail"]):
                    width, height = default_dimensions["thumbnail"]
                elif any(word in context for word in ["icon", "logo"]):
                    width, height = default_dimensions["icon"]
                else:
                    width, height = default_dimensions["default"]

            # Ensure dimensions are integers
            try:
                width = int(width) if width else default_dimensions["default"][0]
                height = int(height) if height else default_dimensions["default"][1]
            except (ValueError, TypeError):
                width, height = default_dimensions["default"]

            # Replace with picsum.photos URL
            img["src"] = f"https://picsum.photos/{width}/{height}"
            # Remove data-src, data-lazy-src, etc. to avoid confusion
            for attr in list(img.attrs.keys()):
                if attr.startswith("data-") and "src" in attr.lower():
                    del img[attr]

    # Replace background-image URLs in style attributes
    for element in soup.find_all(style=True):
        style = element.get("style", "")
        if "background-image" in style or "background:" in style:
            # Replace url(...) patterns
            style = re.sub(
                r'url\(["\']?[^"\')]+["\']?\)',
                lambda m: f'url("https://picsum.photos/800/600")',
                style,
            )
            element["style"] = style

    # Replace background-image in CSS (will be handled separately in CSS processing)
    # But also check inline style tags
    for style_tag in soup.find_all("style"):
        if style_tag.string:
            css_content = style_tag.string
            # Replace background-image URLs
            css_content = re.sub(
                r'url\(["\']?[^"\')]+\.(jpg|jpeg|png|gif|webp|svg)["\']?\)',
                lambda m: 'url("https://picsum.photos/800/600")',
                css_content,
                flags=re.IGNORECASE,
            )
            style_tag.string = css_content

    return str(soup)


def extract_url_from_folder_name(folder_name: str) -> str:
    """Extract clean URL from folder name (e.g., 'httpsamano-clinic.jp' -> 'https://amano-clinic.jp')"""
    url = folder_name.replace("https", "https://").replace("http", "http://")
    if not url.startswith("http"):
        url = "https://" + url
    return url


def parse_info_html(info_path: Path) -> Dict[str, any]:
    """Parse info.html to extract comprehensive metadata"""
    metadata = {
        "url": "",
        "title": "",
        "category": [],
        "industry": "",
        "creators": "",
        "description": "",
        "keywords": [],
    }

    try:
        with open(info_path, "r", encoding="utf-8") as f:
            content = f.read()

        soup = BeautifulSoup(content, "html.parser")

        # Extract URL
        url_elem = soup.find("dt", string="URL")
        if url_elem:
            next_dd = url_elem.find_next_sibling("dd")
            if next_dd:
                link = next_dd.find("a")
                if link:
                    metadata["url"] = link.get("href", "").strip()
                    if not metadata["url"]:
                        text = link.get_text(strip=True)
                        metadata["url"] = (
                            text if text.startswith("http") else f"https://{text}"
                        )

        # Extract title
        title_elem = soup.find("title")
        if title_elem:
            metadata["title"] = title_elem.get_text(strip=True)

        # Extract categories
        category_elem = soup.find("dt", string="CATEGORY")
        if category_elem:
            next_dd = category_elem.find_next_sibling("dd")
            if next_dd:
                category_links = next_dd.find_all("a")
                for link in category_links:
                    category_text = link.get_text(strip=True)
                    if category_text:
                        metadata["category"].append(category_text)

        # Enhanced industry detection
        all_text = str(metadata.get("category", [])) + " " + metadata.get("title", "")
        if any(
            keyword in all_text
            for keyword in ["医療", "病院", "hospital", "clinic", "診療", "クリニック"]
        ):
            metadata["industry"] = "Healthcare"
        elif any(
            keyword in all_text
            for keyword in ["企業", "corporate", "コーポレート", "company"]
        ):
            metadata["industry"] = "Corporate"
        elif any(
            keyword in all_text for keyword in ["美容", "beauty", "salon", "サロン"]
        ):
            metadata["industry"] = "Beauty"
        elif any(
            keyword in all_text
            for keyword in ["飲食", "restaurant", "レストラン", "cafe"]
        ):
            metadata["industry"] = "Food & Beverage"
        elif any(
            keyword in all_text
            for keyword in ["教育", "education", "school", "スクール"]
        ):
            metadata["industry"] = "Education"

        # Extract creators
        creators_elem = soup.find("dt", string="制作者一覧")
        if creators_elem:
            next_dd = creators_elem.find_next_sibling("dd")
            if next_dd:
                metadata["creators"] = next_dd.get_text(strip=True)

        # Extract description
        desc_elem = soup.find("meta", attrs={"name": "description"})
        if desc_elem:
            metadata["description"] = desc_elem.get("content", "").strip()

        # Extract from URL if not found
        if not metadata["url"]:
            folder_name = info_path.parent.name
            metadata["url"] = extract_url_from_folder_name(folder_name)

    except Exception as e:
        logger.warning(f"Error parsing {info_path}: {e}")

    return metadata


def find_css_files(html_path: Path, assets_dir: Path) -> List[Tuple[str, str]]:
    """Find and extract CSS files referenced in HTML, filtering out obfuscated/minified code"""
    css_files = []
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read()

        soup = BeautifulSoup(content, "html.parser")

        # Find all link tags with stylesheet
        for link in soup.find_all("link", rel="stylesheet"):
            href = link.get("href", "")
            if href:
                # Skip external URLs and CDN links
                if href.startswith("http"):
                    continue

                # Resolve relative paths
                if href.startswith("./"):
                    css_path = html_path.parent / href[2:]
                elif href.startswith("/"):
                    css_path = assets_dir / href[1:]
                else:
                    css_path = html_path.parent / href

                if css_path.exists():
                    try:
                        with open(css_path, "r", encoding="utf-8") as css_file:
                            css_content = css_file.read()

                            # Skip obfuscated or minified CSS
                            if is_obfuscated_or_minified(css_content, "css"):
                                logger.debug(
                                    f"Skipping obfuscated/minified CSS: {css_path.name}"
                                )
                                continue

                            # Replace image URLs in CSS
                            css_content = re.sub(
                                r'url\(["\']?[^"\')]+\.(jpg|jpeg|png|gif|webp|svg)["\']?\)',
                                lambda m: 'url("https://picsum.photos/800/600")',
                                css_content,
                                flags=re.IGNORECASE,
                            )

                            # Get filename for reference
                            filename = css_path.name
                            css_files.append((filename, css_content[:MAX_CSS_LENGTH]))
                    except Exception as e:
                        logger.debug(f"Could not read CSS file {css_path}: {e}")

        # Also check for inline styles
        for style in soup.find_all("style"):
            css_content = style.string or ""
            if css_content:
                # Skip obfuscated inline CSS
                if is_obfuscated_or_minified(css_content, "css"):
                    logger.debug("Skipping obfuscated inline CSS")
                    continue

                # Replace image URLs
                css_content = re.sub(
                    r'url\(["\']?[^"\')]+\.(jpg|jpeg|png|gif|webp|svg)["\']?\)',
                    lambda m: 'url("https://picsum.photos/800/600")',
                    css_content,
                    flags=re.IGNORECASE,
                )

                css_files.append(("inline", css_content[:MAX_CSS_LENGTH]))

    except Exception as e:
        logger.warning(f"Error finding CSS files: {e}")

    return css_files


def find_js_files(html_path: Path, assets_dir: Path) -> List[Tuple[str, str]]:
    """Find and extract JavaScript files referenced in HTML, filtering out obfuscated/minified code"""
    js_files = []
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read()

        soup = BeautifulSoup(content, "html.parser")

        # Find all script tags with src
        for script in soup.find_all("script", src=True):
            src = script.get("src", "")
            if src:
                # Skip external URLs and CDN links
                if src.startswith("http"):
                    continue

                # Skip common minified/built libraries
                src_lower = src.lower()
                if any(
                    skip in src_lower
                    for skip in [".min.js", "bundle", "chunk", "vendor", "lib"]
                ):
                    logger.debug(f"Skipping built/minified library: {src}")
                    continue

                # Resolve relative paths
                if src.startswith("./"):
                    js_path = html_path.parent / src[2:]
                elif src.startswith("/"):
                    js_path = assets_dir / src[1:]
                else:
                    js_path = html_path.parent / src

                if js_path.exists():
                    try:
                        with open(js_path, "r", encoding="utf-8") as js_file:
                            js_content = js_file.read()

                            # Skip obfuscated or minified JavaScript
                            if is_obfuscated_or_minified(js_content, "js"):
                                logger.debug(
                                    f"Skipping obfuscated/minified JS: {js_path.name}"
                                )
                                continue

                            filename = js_path.name
                            js_files.append((filename, js_content[:MAX_JS_LENGTH]))
                    except Exception as e:
                        logger.debug(f"Could not read JS file {js_path}: {e}")

        # Also check for inline scripts (but be more selective)
        for script in soup.find_all("script"):
            if not script.get("src"):
                js_content = script.string or ""
                if (
                    js_content and len(js_content) > 50
                ):  # Only include substantial scripts
                    # Skip obfuscated inline scripts
                    if is_obfuscated_or_minified(js_content, "js"):
                        logger.debug("Skipping obfuscated inline JavaScript")
                        continue

                    js_files.append(("inline", js_content[:MAX_JS_LENGTH]))

    except Exception as e:
        logger.warning(f"Error finding JS files: {e}")

    return js_files


def extract_html_content(html_path: Path, assets_dir: Path) -> str:
    """Extract and clean HTML content, replacing image URLs with placeholders"""
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Clean up saved page artifacts
        content = re.sub(r"<!-- saved from url=.*?-->", "", content)

        # Replace image URLs with picsum.photos placeholders
        content = replace_image_urls(content, assets_dir)

        return content
    except Exception as e:
        logger.warning(f"Error reading {html_path}: {e}")
        return ""


def extract_text_from_html(html_path: Path, max_length: int = 5000) -> str:
    """Extract meaningful text content from HTML for analysis"""
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read()

        soup = BeautifulSoup(content, "html.parser")

        # Remove script, style, meta, link elements
        for element in soup(["script", "style", "meta", "link", "noscript"]):
            element.decompose()

        # Get text
        text = soup.get_text()

        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = " ".join(chunk for chunk in chunks if chunk)

        # Truncate if too long
        if len(text) > max_length:
            text = text[:max_length] + "..."

        return text
    except Exception as e:
        logger.warning(f"Error extracting text from {html_path}: {e}")
        return ""


def extract_html_components(html_content: str) -> Dict[str, List[str]]:
    """Extract individual UI components from HTML"""
    components = {
        "headers": [],
        "footers": [],
        "sections": [],
        "buttons": [],
        "navigation": [],
        "hero": [],
        "cards": [],
        "forms": [],
        "typography": [],
    }

    try:
        soup = BeautifulSoup(html_content, "html.parser")

        # Extract headers (header tag or elements with header-related classes)
        for header in soup.find_all(["header"]):
            header_html = str(header)
            if len(header_html) > 100:  # Only include substantial headers
                components["headers"].append(header_html)

        # Also look for header-like divs
        for header_div in soup.find_all("div", class_=re.compile(r"header|nav", re.I)):
            header_html = str(header_div)
            if len(header_html) > 100 and header_html not in components["headers"]:
                components["headers"].append(header_html)

        # Extract footers
        for footer in soup.find_all(["footer"]):
            footer_html = str(footer)
            if len(footer_html) > 100:
                components["footers"].append(footer_html)

        # Also look for footer-like divs
        for footer_div in soup.find_all("div", class_=re.compile(r"footer", re.I)):
            footer_html = str(footer_div)
            if len(footer_html) > 100 and footer_html not in components["footers"]:
                components["footers"].append(footer_html)

        # Extract sections (section tags or main content sections)
        for section in soup.find_all(["section"]):
            section_html = str(section)
            if len(section_html) > 200:  # Only substantial sections
                components["sections"].append(section_html)

        # Extract hero sections (usually first large section)
        for hero in soup.find_all(["div", "section"], class_=re.compile(r"hero|banner|mv|main-visual", re.I)):
            hero_html = str(hero)
            if len(hero_html) > 200:
                components["hero"].append(hero_html)

        # Extract buttons
        for button in soup.find_all(["button", "a"], class_=re.compile(r"btn|button", re.I)):
            # Get button and its parent context (for styling context)
            button_html = str(button.parent) if button.parent and len(str(button.parent)) < 500 else str(button)
            if len(button_html) > 50:
                components["buttons"].append(button_html)

        # Extract navigation menus
        for nav in soup.find_all(["nav"]):
            nav_html = str(nav)
            if len(nav_html) > 100:
                components["navigation"].append(nav_html)

        # Extract cards (common card patterns)
        for card in soup.find_all(["div", "article"], class_=re.compile(r"card|item|product|post", re.I)):
            card_html = str(card)
            if 200 < len(card_html) < 2000:  # Reasonable card size
                components["cards"].append(card_html)

        # Extract forms
        for form in soup.find_all(["form"]):
            form_html = str(form)
            if len(form_html) > 100:
                components["forms"].append(form_html)

        # Extract typography examples (headings with context)
        for heading in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
            # Get heading with some context (parent or sibling)
            parent = heading.parent
            if parent:
                typo_html = str(parent) if len(str(parent)) < 500 else str(heading)
                if len(typo_html) > 50:
                    components["typography"].append(typo_html)

    except Exception as e:
        logger.warning(f"Error extracting components: {e}")

    # Remove duplicates and filter by size
    for key in components:
        # Remove duplicates while preserving order
        seen = set()
        unique_components = []
        for comp in components[key]:
            comp_hash = hashlib.md5(comp.encode()).hexdigest()
            if comp_hash not in seen:
                seen.add(comp_hash)
                unique_components.append(comp)
        components[key] = unique_components

    return components


def extract_css_for_component(component_html: str, all_css: List[Tuple[str, str]]) -> str:
    """Extract relevant CSS for a specific component"""
    relevant_css = []
    soup = BeautifulSoup(component_html, "html.parser")

    # Collect all class names and IDs from the component
    classes = set()
    ids = set()

    for element in soup.find_all(True):  # Find all elements
        if element.get("class"):
            classes.update(element.get("class"))
        if element.get("id"):
            ids.add(element.get("id"))

    # Search CSS for matching selectors
    for css_file_name, css_content in all_css:
        css_lines = css_content.split("\n")
        relevant_lines = []

        for line in css_lines:
            # Check if line contains any of our classes or IDs
            line_lower = line.lower()
            if any(f".{cls}" in line_lower or f"#{cls}" in line_lower for cls in classes):
                relevant_lines.append(line)
            elif any(f"#{id_val}" in line_lower for id_val in ids):
                relevant_lines.append(line)
            # Also include common component-related CSS
            elif any(keyword in line_lower for keyword in ["header", "footer", "nav", "button", "card", "section"]):
                if len(relevant_lines) < 50:  # Limit context
                    relevant_lines.append(line)

        if relevant_lines:
            relevant_css.append(f"/* {css_file_name} */\n" + "\n".join(relevant_lines[:50]))

    return "\n\n".join(relevant_css[:MAX_COMPONENT_CSS_LENGTH])


def extract_js_for_component(component_html: str, all_js: List[Tuple[str, str]]) -> str:
    """Extract relevant JavaScript for a specific component"""
    relevant_js = []
    soup = BeautifulSoup(component_html, "html.parser")

    # Collect IDs and data attributes that might be used in JS
    identifiers = set()
    for element in soup.find_all(True):
        if element.get("id"):
            identifiers.add(element.get("id"))
        for attr in element.attrs:
            if attr.startswith("data-"):
                identifiers.add(attr)

    # Search JS for matching identifiers
    for js_file_name, js_content in all_js:
        js_lower = js_content.lower()
        # Check if JS references any identifiers or common component patterns
        if any(identifier.lower() in js_lower for identifier in identifiers):
            # Extract relevant portion (first 1000 chars)
            relevant_js.append(f"// {js_file_name}\n" + js_content[:MAX_COMPONENT_JS_LENGTH])

    return "\n\n".join(relevant_js)


def analyze_color_scheme(
    html_path: Path, css_files: List[Tuple[str, str]]
) -> Dict[str, any]:
    """Analyze color scheme from HTML and CSS"""
    colors = {"primary": [], "background": [], "text": [], "scheme": "Unknown"}

    try:
        # Collect all CSS content
        all_css = " ".join([css[1] for css in css_files])

        # Extract color values
        color_patterns = [
            r"color:\s*([#\w]+)",
            r"background(?:-color)?:\s*([#\w]+)",
            r"#[0-9a-fA-F]{3,6}",
            r"rgb\([^)]+\)",
            r"rgba\([^)]+\)",
        ]

        found_colors = []
        for pattern in color_patterns:
            matches = re.findall(pattern, all_css, re.IGNORECASE)
            found_colors.extend(matches)

        # Count most common colors
        color_counter = Counter([c.lower() for c in found_colors if len(c) > 2])
        colors["primary"] = [color for color, count in color_counter.most_common(3)]

        # Determine color scheme
        if any(
            "dark" in str(c).lower() or "black" in str(c).lower()
            for c in colors["primary"]
        ):
            colors["scheme"] = "Dark"
        elif any(
            "light" in str(c).lower() or "white" in str(c).lower()
            for c in colors["primary"]
        ):
            colors["scheme"] = "Light"
        else:
            colors["scheme"] = "Mixed"

    except Exception as e:
        logger.debug(f"Error analyzing colors: {e}")

    return colors


def analyze_website_characteristics(
    html_path: Path,
    metadata: Dict,
    css_files: List[Tuple[str, str]],
    js_files: List[Tuple[str, str]],
) -> Dict[str, any]:
    """Comprehensive analysis of website characteristics"""
    characteristics = {
        "tone": "Professional",
        "layout": "Standard",
        "photo_usage": "Medium",
        "motion": "None",
        "stack": "HTML + CSS + JS",
        "responsive": False,
        "accessibility": "Basic",
        "color_scheme": "Unknown",
        "typography": "Standard",
        "interactivity": "Low",
    }

    try:
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        all_content = html_content.lower()
        all_css = " ".join([css[1].lower() for css in css_files])
        all_js = " ".join([js[1].lower() for js in js_files])
        combined = all_content + " " + all_css + " " + all_js

        # Motion detection
        if any(lib in combined for lib in ["three.js", "webgl", "webgl2", "babylon"]):
            characteristics["motion"] = "Advanced 3D animations"
        elif any(
            lib in combined for lib in ["gsap", "anime.js", "framer", "lottie", "aos"]
        ):
            characteristics["motion"] = "Subtle animations"
        elif (
            "transition" in all_css or "animation" in all_css or "transform" in all_css
        ):
            characteristics["motion"] = "CSS animations"

        # Layout detection
        if (
            "display: grid" in all_css
            or "display:grid" in all_css
            or "grid-template" in all_css
        ):
            characteristics["layout"] = "Grid-based"
        elif (
            "display: flex" in all_css
            or "display:flex" in all_css
            or "flexbox" in all_css
        ):
            characteristics["layout"] = "Flex-based"
        elif "bootstrap" in combined or "foundation" in combined:
            characteristics["layout"] = "Framework-based"

        # Photo usage
        img_count = (
            all_content.count("<img")
            + all_content.count("background-image")
            + all_content.count("background: url")
        )
        if img_count > 30:
            characteristics["photo_usage"] = "Very High"
        elif img_count > 20:
            characteristics["photo_usage"] = "High"
        elif img_count < 5:
            characteristics["photo_usage"] = "Low"

        # Framework detection
        if any(fw in combined for fw in ["react", "vue", "angular", "svelte"]):
            characteristics["stack"] = "Modern Framework"
        elif "jquery" in combined:
            characteristics["stack"] = "HTML + CSS + JS + jQuery"
        elif "typescript" in combined or ".ts" in combined:
            characteristics["stack"] = "HTML + CSS + TypeScript"

        # Responsive design
        if (
            "viewport" in html_content
            or "@media" in all_css
            or "responsive" in combined
        ):
            characteristics["responsive"] = True

        # Typography
        if any(
            font in all_css
            for font in ["googleapis", "fonts.com", "typekit", "adobe fonts"]
        ):
            characteristics["typography"] = "Custom web fonts"
        elif "font-family" in all_css:
            characteristics["typography"] = "Custom typography"

        # Interactivity
        event_count = (
            all_js.count("addEventListener")
            + all_js.count("onclick")
            + all_js.count("on(")
        )
        if event_count > 10:
            characteristics["interactivity"] = "High"
        elif event_count > 5:
            characteristics["interactivity"] = "Medium"

        # Color scheme
        color_info = analyze_color_scheme(html_path, css_files)
        characteristics["color_scheme"] = color_info.get("scheme", "Unknown")

        # Enhanced tone detection
        if metadata.get("industry") == "Healthcare":
            characteristics["tone"] = "Professional, Trustworthy, Medical"
        elif metadata.get("industry") == "Beauty":
            characteristics["tone"] = "Elegant, Refined, Aesthetic"
        elif metadata.get("industry") == "Food & Beverage":
            characteristics["tone"] = "Warm, Inviting, Appetizing"
        elif (
            "清涼" in str(metadata.get("category", []))
            or "fresh" in str(metadata.get("category", [])).lower()
        ):
            characteristics["tone"] = "Fresh, Modern, Clean"
        elif (
            "ピンク" in str(metadata.get("category", []))
            or "pink" in str(metadata.get("category", [])).lower()
        ):
            characteristics["tone"] = "Warm, Friendly, Approachable"
        elif "minimal" in combined or "minimalist" in combined:
            characteristics["tone"] = "Minimalist, Clean, Simple"
        elif "luxury" in combined or "premium" in combined:
            characteristics["tone"] = "Luxury, Premium, Sophisticated"

        # Accessibility
        if "aria-" in html_content or "role=" in html_content or "alt=" in html_content:
            characteristics["accessibility"] = "Enhanced"

    except Exception as e:
        logger.warning(f"Error analyzing {html_path}: {e}")

    return characteristics


def generate_design_reasoning_with_openai(
    metadata: Dict,
    characteristics: Dict,
    css_files: List[Tuple[str, str]],
    js_files: List[Tuple[str, str]],
) -> Optional[str]:
    """Generate design reasoning using OpenAI API (80% automated draft)"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning(
            "OPENAI_API_KEY not found in environment. Skipping AI reasoning generation."
        )
        return None

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        # Build context for reasoning
        context_parts = []
        if metadata.get("industry"):
            context_parts.append(f"Industry: {metadata['industry']}")
        if metadata.get("category"):
            context_parts.append(f"Categories: {', '.join(metadata['category'][:5])}")
        if characteristics.get("tone"):
            context_parts.append(f"Tone: {characteristics['tone']}")
        if characteristics.get("layout"):
            context_parts.append(f"Layout: {characteristics['layout']}")
        if characteristics.get("photo_usage"):
            context_parts.append(f"Photo usage: {characteristics['photo_usage']}")
        if characteristics.get("motion") and characteristics["motion"] != "None":
            context_parts.append(f"Motion: {characteristics['motion']}")
        if (
            characteristics.get("color_scheme")
            and characteristics["color_scheme"] != "Unknown"
        ):
            context_parts.append(f"Color scheme: {characteristics['color_scheme']}")
        if characteristics.get("stack"):
            context_parts.append(f"Tech stack: {characteristics['stack']}")
        if css_files:
            context_parts.append(f"CSS files: {len(css_files)} file(s)")
        if js_files:
            context_parts.append(f"JavaScript files: {len(js_files)} file(s)")

        context = "\n".join(context_parts)

        prompt = f"""You are a senior creative front-end engineer analyzing a website design.

Given the following website characteristics:
{context}

Explain the design reasoning in 3-4 concise bullet points:
1. Why this layout approach fits the industry/brand
2. Why this motion style (if any) was likely chosen and what emotion it conveys
3. How the visual style (tone, colors, photo usage) supports the brand message
4. Any notable technical choices and their purpose

Be specific, professional, and focus on design intent. Keep each point to 1-2 sentences.
Format as bullet points starting with "- "."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using mini for cost efficiency
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert front-end designer who explains design decisions clearly and concisely.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=300,
        )

        reasoning = response.choices[0].message.content.strip()
        logger.debug(f"Generated reasoning: {reasoning[:100]}...")
        return reasoning

    except ImportError:
        logger.warning("openai package not installed. Install with: pip install openai")
        return None
    except Exception as e:
        logger.warning(f"Error generating reasoning with OpenAI: {e}")
        return None


def create_component_example(
    component_type: str,
    component_html: str,
    metadata: Dict,
    css_files: List[Tuple[str, str]],
    js_files: List[Tuple[str, str]],
    characteristics: Dict,
) -> Optional[Dict]:
    """Create a training example for a single UI component"""
    # Extract relevant CSS and JS for this component
    component_css = extract_css_for_component(component_html, css_files)
    component_js = extract_js_for_component(component_html, js_files)

    # Build component-specific instruction
    instruction_parts = []

    if metadata.get("industry"):
        instruction_parts.append(f"Industry: {metadata['industry']}")
    if characteristics.get("tone"):
        instruction_parts.append(f"Brand tone: {characteristics['tone']}")

    if (
        characteristics.get("color_scheme")
        and characteristics["color_scheme"] != "Unknown"
    ):
        instruction_parts.append(f"Color scheme: {characteristics['color_scheme']}")

    if characteristics.get("responsive"):
        instruction_parts.append("Requirements: Responsive design (mobile-first)")

    # Component-specific task
    component_tasks = {
        "headers": "Create a website header/navigation component with semantic HTML, modern CSS styling, and responsive behavior. Include logo, navigation menu, and any interactive elements.",
        "footers": "Create a website footer component with semantic HTML and clean CSS styling. Include links, copyright information, and social media icons if applicable.",
        "sections": "Create a content section component with semantic HTML structure and modern CSS layout. Focus on proper spacing, typography, and visual hierarchy.",
        "buttons": "Create a button component with semantic HTML, attractive CSS styling, and hover/active states. Ensure accessibility and responsive behavior.",
        "navigation": "Create a navigation menu component with semantic HTML, CSS styling, and responsive mobile menu behavior if applicable.",
        "hero": "Create a hero/banner section component with semantic HTML, eye-catching CSS styling, and responsive layout. Include heading, subheading, and call-to-action if applicable.",
        "cards": "Create a card component with semantic HTML structure and modern CSS styling. Include proper spacing, shadows, and hover effects if applicable.",
        "forms": "Create a form component with semantic HTML, accessible form elements, and clean CSS styling. Include proper labels and validation styling.",
        "typography": "Create a typography example with semantic HTML headings and text elements, styled with modern CSS. Focus on font hierarchy, spacing, and readability.",
    }

    task = component_tasks.get(component_type, "Create a UI component with semantic HTML and modern CSS styling.")
    instruction_parts.append(f"Task: {task}")
    instruction_parts.append("Use placeholder images from https://picsum.photos/ with appropriate dimensions.")

    instruction_text = "\n".join(instruction_parts)

    # Build output
    output_parts = ["Design reasoning:"]
    output_parts.append(
        f"- Component type: {component_type.capitalize()} component"
    )
    output_parts.append(
        f"- Design style: {characteristics.get('tone', 'Professional')} brand tone"
    )
    if characteristics.get("responsive"):
        output_parts.append("- Responsive: Mobile-first approach with flexible layout")

    output_parts.append("\nCode:")
    output_parts.append("```html")
    # Limit component HTML size
    component_html_limited = component_html[:MAX_COMPONENT_HTML_LENGTH]
    if len(component_html) > MAX_COMPONENT_HTML_LENGTH:
        component_html_limited += "\n<!-- ... rest of component ... -->"
    output_parts.append(component_html_limited)
    output_parts.append("```")

    if component_css:
        output_parts.append("\n```css")
        css_limited = component_css[:MAX_COMPONENT_CSS_LENGTH]
        if len(component_css) > MAX_COMPONENT_CSS_LENGTH:
            css_limited += "\n/* ... more CSS ... */"
        output_parts.append(css_limited)
        output_parts.append("```")

    if component_js and len(component_js) < MAX_COMPONENT_JS_LENGTH:
        output_parts.append("\n```javascript")
        output_parts.append(component_js)
        output_parts.append("```")

    output_text = "\n".join(output_parts)

    example = {"instruction": instruction_text, "output": output_text}
    if validate_example(example):
        return example
    return None


def create_grouped_example(
    component_types: List[str],
    component_htmls: List[str],
    metadata: Dict,
    css_files: List[Tuple[str, str]],
    js_files: List[Tuple[str, str]],
    characteristics: Dict,
) -> Optional[Dict]:
    """Create a training example for grouped components (e.g., header + hero)"""
    # Combine HTML
    combined_html = "\n".join(component_htmls)
    if len(combined_html) > MAX_COMPONENT_HTML_LENGTH * 2:
        combined_html = combined_html[:MAX_COMPONENT_HTML_LENGTH * 2] + "\n<!-- ... more components ... -->"

    # Extract relevant CSS/JS
    combined_css = extract_css_for_component(combined_html, css_files)
    combined_js = extract_js_for_component(combined_html, js_files)

    # Build instruction
    instruction_parts = []
    if metadata.get("industry"):
        instruction_parts.append(f"Industry: {metadata['industry']}")
    if characteristics.get("tone"):
        instruction_parts.append(f"Brand tone: {characteristics['tone']}")

    if (
        characteristics.get("color_scheme")
        and characteristics["color_scheme"] != "Unknown"
    ):
        instruction_parts.append(f"Color scheme: {characteristics['color_scheme']}")

    if characteristics.get("responsive"):
        instruction_parts.append("Requirements: Responsive design (mobile-first)")

    component_names = " + ".join([c.capitalize() for c in component_types])
    instruction_parts.append(
        f"Task: Create a combined {component_names} component group with semantic HTML structure, "
        "modern CSS for layout and styling, and clean JavaScript for interactivity. "
        "Ensure components work together harmoniously. "
        "Use placeholder images from https://picsum.photos/ with appropriate dimensions."
    )

    instruction_text = "\n".join(instruction_parts)

    # Build output
    output_parts = ["Design reasoning:"]
    output_parts.append(f"- Component group: {component_names}")
    output_parts.append(f"- Design style: {characteristics.get('tone', 'Professional')} brand tone")
    if characteristics.get("responsive"):
        output_parts.append("- Responsive: Mobile-first approach")

    output_parts.append("\nCode:")
    output_parts.append("```html")
    output_parts.append(combined_html)
    output_parts.append("```")

    if combined_css:
        output_parts.append("\n```css")
        css_limited = combined_css[:MAX_COMPONENT_CSS_LENGTH * 2]
        if len(combined_css) > MAX_COMPONENT_CSS_LENGTH * 2:
            css_limited += "\n/* ... more CSS ... */"
        output_parts.append(css_limited)
        output_parts.append("```")

    if combined_js and len(combined_js) < MAX_COMPONENT_JS_LENGTH * 2:
        output_parts.append("\n```javascript")
        output_parts.append(combined_js)
        output_parts.append("```")

    output_text = "\n".join(output_parts)

    example = {"instruction": instruction_text, "output": output_text}
    if validate_example(example):
        return example
    return None


def create_training_example(
    metadata: Dict,
    html_content: str,
    css_files: List[Tuple[str, str]],
    js_files: List[Tuple[str, str]],
    characteristics: Dict,
    use_ai_reasoning: bool = True,
    is_full_page: bool = True,
) -> Dict:
    """Create a comprehensive training example focused on layout design"""
    # Build instruction focused on layout design
    instruction_parts = []

    # Primary focus: Layout design
    if metadata.get("industry"):
        instruction_parts.append(f"Industry: {metadata['industry']}")
    if characteristics.get("tone"):
        instruction_parts.append(f"Brand tone: {characteristics['tone']}")

    # Layout is the most important aspect
    layout_description = characteristics.get("layout", "Standard")
    instruction_parts.append(f"Layout style: {layout_description}")

    if characteristics.get("responsive"):
        instruction_parts.append("Requirements: Responsive design (mobile-first)")

    if (
        characteristics.get("color_scheme")
        and characteristics["color_scheme"] != "Unknown"
    ):
        instruction_parts.append(f"Color scheme: {characteristics['color_scheme']}")

    if characteristics.get("photo_usage"):
        instruction_parts.append(
            f"Image usage: {characteristics['photo_usage']} (use placeholder images)"
        )

    if characteristics.get("motion") and characteristics["motion"] != "None":
        instruction_parts.append(f"Interactions: {characteristics['motion']}")

    # Main task focused on layout
    instruction_parts.append(
        "Task: Create a complete, production-ready website layout with semantic HTML structure, "
        "modern CSS for responsive design, and clean JavaScript for interactivity. "
        "Focus on layout structure, spacing, typography, and visual hierarchy. "
        "Use placeholder images from https://picsum.photos/ with appropriate dimensions."
    )

    instruction_text = "\n".join(instruction_parts)

    # Generate reasoning (AI-assisted or fallback) - focused on layout
    reasoning_text = None
    if use_ai_reasoning:
        reasoning_text = generate_design_reasoning_with_openai(
            metadata, characteristics, css_files, js_files
        )

    # Build assistant response with layout focus
    assistant_parts = ["Design reasoning:"]

    if reasoning_text:
        # Use AI-generated reasoning
        assistant_parts.append(reasoning_text)
    else:
        # Fallback reasoning focused on layout
        assistant_parts.append(
            f"- Layout structure: {characteristics.get('layout', 'Standard')} layout approach "
            f"for optimal information hierarchy and user experience"
        )
        assistant_parts.append(
            f"- Visual design: {characteristics.get('tone', 'Professional')} brand tone with "
            f"{characteristics.get('color_scheme', 'balanced')} color palette"
        )

        if characteristics.get("responsive"):
            assistant_parts.append(
                "- Responsive design: Mobile-first approach with flexible grid system and breakpoints"
            )

        if characteristics.get("motion") and characteristics["motion"] != "None":
            assistant_parts.append(
                f"- Interactions: {characteristics['motion']} for enhanced user engagement"
            )

    assistant_parts.append("\nCode:")

    # Include HTML (already has image URLs replaced)
    html_preview = html_content[:MAX_HTML_LENGTH]
    if len(html_content) > MAX_HTML_LENGTH:
        html_preview += "\n<!-- ... rest of HTML ... -->"
    assistant_parts.append("```html")
    assistant_parts.append(html_preview)
    assistant_parts.append("```")

    # Include CSS if available (focus on layout CSS)
    if css_files:
        assistant_parts.append("\n```css")
        # Prioritize layout-related CSS
        css_content = "\n\n/* " + css_files[0][0] + " */\n" + css_files[0][1]
        if len(css_files) > 1:
            css_content += f"\n\n/* ... {len(css_files) - 1} more CSS file(s) ... */"
        assistant_parts.append(css_content)
        assistant_parts.append("```")

    # Include JS only if it's layout-related (navigation, responsive behavior) and not too large
    if js_files and len(js_files[0][1]) < 2000:
        # Only include if it seems layout-related (navigation, menu, responsive)
        js_content_lower = js_files[0][1].lower()
        layout_js_keywords = [
            "menu",
            "nav",
            "toggle",
            "responsive",
            "mobile",
            "breakpoint",
            "scroll",
        ]
        if any(keyword in js_content_lower for keyword in layout_js_keywords):
            assistant_parts.append("\n```javascript")
            js_content = "\n\n// " + js_files[0][0] + "\n" + js_files[0][1]
            if len(js_files) > 1:
                js_content += f"\n\n// ... {len(js_files) - 1} more JS file(s) ..."
            assistant_parts.append(js_content)
            assistant_parts.append("```")

    assistant_response = "\n".join(assistant_parts)

    # Format as instruction/output for DeepSeek Coder training
    # The instruction will be wrapped by build_instruction_prompt() in train.py
    return {"instruction": instruction_text, "output": assistant_response}


def validate_example(example: Dict) -> bool:
    """Validate training example quality"""
    if not example or "instruction" not in example or "output" not in example:
        return False

    instruction = example.get("instruction", "")
    output = example.get("output", "")

    # Check that both fields have content
    if len(instruction) < 50 or len(output) < 100:
        return False

    # Check that code blocks are present in output
    if "```html" not in output and "```" not in output:
        return False

    return True


def process_website_template(
    template_dir: Path, use_ai_reasoning: bool = True, include_full_page: bool = True
) -> List[Dict]:
    """Process a single website template directory and extract multiple component examples"""
    index_path = template_dir / "index.html"
    info_path = template_dir / "info.html"
    assets_dir = template_dir / "assets"

    if not index_path.exists():
        logger.warning(f"{index_path} not found, skipping {template_dir.name}")
        return []

    logger.info(f"Processing: {template_dir.name}")

    examples = []

    try:
        # Parse metadata
        metadata = parse_info_html(info_path) if info_path.exists() else {}

        # Extract HTML content (with image URL replacement)
        html_content = extract_html_content(index_path, assets_dir)
        if not html_content:
            logger.warning(f"Could not extract content from {index_path}")
            return []

        # Find and extract CSS/JS files
        css_files = (
            find_css_files(index_path, assets_dir) if assets_dir.exists() else []
        )
        js_files = find_js_files(index_path, assets_dir) if assets_dir.exists() else []

        # Analyze characteristics
        characteristics = analyze_website_characteristics(
            index_path, metadata, css_files, js_files
        )

        # Extract individual components
        components = extract_html_components(html_content)
        logger.debug(f"  Extracted components: {sum(len(v) for v in components.values())} total")

        # Create examples for each component type
        component_counts = {}
        for component_type, component_list in components.items():
            if not component_list:
                continue

            component_counts[component_type] = 0
            for component_html in component_list:
                example = create_component_example(
                    component_type,
                    component_html,
                    metadata,
                    css_files,
                    js_files,
                    characteristics,
                )
                if example:
                    examples.append(example)
                    component_counts[component_type] += 1

        logger.debug(f"  Created {sum(component_counts.values())} component examples")

        # Create grouped examples (header + hero, multiple sections, etc.)
        # Header + Hero
        if components.get("headers") and components.get("hero"):
            for header in components["headers"][:1]:  # Take first header
                for hero in components["hero"][:1]:  # Take first hero
                    example = create_grouped_example(
                        ["header", "hero"],
                        [header, hero],
                        metadata,
                        css_files,
                        js_files,
                        characteristics,
                    )
                    if example:
                        examples.append(example)

        # Header + Navigation
        if components.get("headers") and components.get("navigation"):
            for header in components["headers"][:1]:
                for nav in components["navigation"][:1]:
                    example = create_grouped_example(
                        ["header", "navigation"],
                        [header, nav],
                        metadata,
                        css_files,
                        js_files,
                        characteristics,
                    )
                    if example:
                        examples.append(example)

        # Multiple sections together (2-3 sections)
        if len(components.get("sections", [])) >= 2:
            sections_to_group = components["sections"][:3]  # Take up to 3 sections
            if len(sections_to_group) >= 2:
                example = create_grouped_example(
                    ["section"] * len(sections_to_group),
                    sections_to_group,
                    metadata,
                    css_files,
                    js_files,
                    characteristics,
                )
                if example:
                    examples.append(example)

        # Cards group (2-4 cards)
        if len(components.get("cards", [])) >= 2:
            cards_to_group = components["cards"][:4]
            if len(cards_to_group) >= 2:
                example = create_grouped_example(
                    ["card"] * len(cards_to_group),
                    cards_to_group,
                    metadata,
                    css_files,
                    js_files,
                    characteristics,
                )
                if example:
                    examples.append(example)

        logger.debug(f"  Created {len(examples) - sum(component_counts.values())} grouped examples")

        # Create full page example (optional, but useful for context)
        if include_full_page:
            full_page_example = create_training_example(
                metadata,
                html_content,
                css_files,
                js_files,
                characteristics,
                use_ai_reasoning=use_ai_reasoning,
                is_full_page=True,
            )
            if full_page_example and validate_example(full_page_example):
                examples.append(full_page_example)
                logger.debug(f"  Created 1 full page example")

        logger.info(f"  ✓ Generated {len(examples)} examples from {template_dir.name}")

    except Exception as e:
        logger.error(f"Error processing {template_dir.name}: {e}", exc_info=True)

    return examples


def build_dataset(incremental: bool = False, use_ai: bool = True, include_full_page: bool = True):
    """Main function to build the dataset with enhanced features"""
    logger.info("=" * 80)
    logger.info("Building dataset from project templates")
    logger.info("=" * 80)

    # Check OpenAI API key
    if use_ai:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            logger.info("✓ OpenAI API key found - AI reasoning generation enabled")
        else:
            logger.warning("⚠ OPENAI_API_KEY not found - using fallback reasoning")
            logger.warning("  Set OPENAI_API_KEY in .env file to enable AI reasoning")
    else:
        logger.info("AI reasoning disabled (manual mode)")

    # Ensure dataset directory exists
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    # Check if project_templates exists
    if not PROJECT_TEMPLATES_DIR.exists():
        logger.error(f"{PROJECT_TEMPLATES_DIR} not found!")
        logger.error("Make sure you're running this script from the src/ directory")
        return

    # Get all template directories
    template_dirs = sorted([d for d in PROJECT_TEMPLATES_DIR.iterdir() if d.is_dir()])

    if not template_dirs:
        logger.warning(f"No template directories found in {PROJECT_TEMPLATES_DIR}")
        return

    logger.info(f"Found {len(template_dirs)} website templates")

    # Load existing examples for incremental updates
    existing_hashes = set()
    if incremental and OUTPUT_FILE.exists():
        logger.info("Loading existing dataset for incremental update...")
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        example = json.loads(line)
                        # Create hash from output for deduplication (new format)
                        output = example.get("output", "")
                        if "Website URL:" in output or len(output) > 0:
                            output_hash = hashlib.md5(output.encode()).hexdigest()
                            existing_hashes.add(output_hash)
        except Exception as e:
            logger.warning(f"Could not load existing dataset: {e}")

    # Process all templates
    examples = []
    skipped = 0
    errors = 0
    total_components = 0
    total_grouped = 0
    total_full_pages = 0

    for i, template_dir in enumerate(template_dirs, 1):
        logger.info(f"[{i}/{len(template_dirs)}] Processing {template_dir.name}...")

        template_examples = process_website_template(
            template_dir, use_ai_reasoning=use_ai, include_full_page=include_full_page
        )
        if template_examples:
            examples.extend(template_examples)
            # Count example types (rough estimate based on instruction content)
            for ex in template_examples:
                inst = ex.get("instruction", "")
                if "full page" in inst.lower() or "complete" in inst.lower():
                    total_full_pages += 1
                elif "+" in inst or "group" in inst.lower():
                    total_grouped += 1
                else:
                    total_components += 1
        else:
            skipped += 1

    logger.info(f"\n{'=' * 80}")
    logger.info(f"Processing complete!")
    logger.info(f"  - Total examples: {len(examples)}")
    logger.info(f"    • Component examples: {total_components}")
    logger.info(f"    • Grouped examples: {total_grouped}")
    logger.info(f"    • Full page examples: {total_full_pages}")
    logger.info(f"  - Websites processed: {len(template_dirs) - skipped}")
    logger.info(f"  - Skipped: {skipped} templates")
    logger.info(f"  - Errors: {errors} errors")
    logger.info(f"{'=' * 80}")

    if not examples:
        logger.error("No valid examples generated!")
        return

    # Write to JSONL file
    logger.info(f"Writing to {OUTPUT_FILE}...")
    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")

        logger.info(f"✓ Dataset created successfully!")
        logger.info(f"  - {len(examples)} examples written to {OUTPUT_FILE}")
        logger.info(f"  - Log file: {LOG_FILE}")

        # Print statistics
        total_size = OUTPUT_FILE.stat().st_size
        logger.info(f"  - File size: {total_size / 1024:.2f} KB")

    except Exception as e:
        logger.error(f"Error writing dataset: {e}", exc_info=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build dataset from website templates")
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Update existing dataset incrementally",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "--no-ai",
        action="store_true",
        help="Disable AI reasoning generation (use fallback)",
    )
    parser.add_argument(
        "--no-full-page",
        action="store_true",
        help="Skip full page examples (only generate component and grouped examples)",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Check if BeautifulSoup is available
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        logger.error("beautifulsoup4 is required. Install it with:")
        logger.error("pip install beautifulsoup4")
        exit(1)

    build_dataset(
        incremental=args.incremental,
        use_ai=not args.no_ai,
        include_full_page=not args.no_full_page,
    )
