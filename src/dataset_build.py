"""
Enhanced dataset building script.
Processes website templates from project_templates/ and creates/updates the dataset with advanced analysis.
"""

import os
import json
import re
import logging
import time
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
RESULT_DIR = PROJECT_ROOT / "result"

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
MAX_COMPONENT_HTML_LENGTH = 100000  # Max chars for component HTML
MAX_COMPONENT_CSS_LENGTH = 200000  # Max chars for component CSS (increased to prevent truncation)
MAX_COMPONENT_JS_LENGTH = 100000  # Max chars for component JS (increased to prevent truncation)


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


def format_html(html_content: str) -> str:
    """Format and clean HTML content"""
    try:
        soup = BeautifulSoup(html_content, "html.parser")
        # Use prettify to format HTML with proper indentation
        formatted = soup.prettify()
        
        # Clean up excessive blank lines (more than 2 consecutive)
        lines = formatted.split('\n')
        cleaned_lines = []
        blank_count = 0
        
        for line in lines:
            if line.strip() == '':
                blank_count += 1
                if blank_count <= 1:  # Keep single blank lines
                    cleaned_lines.append('')
            else:
                blank_count = 0
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    except Exception as e:
        logger.debug(f"Error formatting HTML: {e}, returning original")
        return html_content


def format_css(css_content: str) -> str:
    """Format and clean CSS content, preserving function calls like clamp(), calc(), rgba()"""
    try:
        # Don't format CSS - preserve original formatting to avoid breaking functions
        # Only clean up excessive blank lines
        lines = css_content.split('\n')
        cleaned_lines = []
        blank_count = 0
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                blank_count += 1
                if blank_count <= 1:  # Keep single blank lines
                    cleaned_lines.append('')
                continue
            else:
                blank_count = 0
                cleaned_lines.append(line)
        
        result = '\n'.join(cleaned_lines)
        # Remove more than 2 consecutive blank lines
        result = re.sub(r'\n{3,}', '\n\n', result)
        
        return result.strip()
    except Exception as e:
        logger.debug(f"Error formatting CSS: {e}, returning original")
        return css_content


def format_javascript(js_content: str) -> str:
    """Format and clean JavaScript content (basic formatting)"""
    try:
        # Basic formatting: add newlines after semicolons, braces, etc.
        # This is a simple formatter - for complex JS, consider using a proper JS formatter
        
        # Remove excessive whitespace but preserve structure
        js_content = re.sub(r';\s*', ';\n', js_content)
        js_content = re.sub(r'\{\s*', ' {\n', js_content)
        js_content = re.sub(r'\}\s*', '}\n', js_content)
        js_content = re.sub(r',\s*', ', ', js_content)
        
        # Basic indentation
        lines = js_content.split('\n')
        formatted_lines = []
        indent_level = 0
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            
            # Decrease indent before closing braces
            if stripped.startswith('}'):
                indent_level = max(0, indent_level - 1)
            
            # Add proper indentation
            formatted_lines.append('  ' * indent_level + stripped)
            
            # Increase indent after opening braces
            if stripped.endswith('{'):
                indent_level += 1
        
        # Clean up excessive blank lines
        result = '\n'.join(formatted_lines)
        result = re.sub(r'\n{3,}', '\n\n', result)
        
        return result.strip()
    except Exception as e:
        logger.debug(f"Error formatting JavaScript: {e}, returning original")
        return js_content


def replace_urls_with_placeholders(html_content: str) -> str:
    """Replace all real URLs in links with placeholder links"""
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Common placeholder patterns based on link text or context
    def generate_placeholder_url(link_element):
        """Generate a semantic placeholder URL based on link context"""
        href = link_element.get("href", "")
        link_text = link_element.get_text(strip=True).lower()
        
        # Skip if already a placeholder
        if href.startswith("#") or href.startswith("/") and not href.startswith("//"):
            return href
        
        # Skip mailto: and tel: links
        if href.startswith("mailto:") or href.startswith("tel:"):
            return href
        
        # Skip external links that are clearly external (social media, etc.)
        if any(domain in href.lower() for domain in ["facebook.com", "twitter.com", "instagram.com", "linkedin.com", "youtube.com"]):
            return "#"  # Replace social links with #
        
        # Generate semantic placeholder based on link text or href
        if link_text:
            # Common navigation patterns
            if any(word in link_text for word in ["about", "会社", "概要", "私たち"]):
                return "/about"
            elif any(word in link_text for word in ["contact", "問い合わせ", "お問い合わせ", "連絡"]):
                return "/contact"
            elif any(word in link_text for word in ["service", "サービス", "診療", "治療"]):
                return "/services"
            elif any(word in link_text for word in ["product", "製品", "商品", "メニュー"]):
                return "/products"
            elif any(word in link_text for word in ["news", "ニュース", "お知らせ"]):
                return "/news"
            elif any(word in link_text for word in ["blog", "ブログ"]):
                return "/blog"
            elif any(word in link_text for word in ["faq", "よくある", "質問"]):
                return "/faq"
            elif any(word in link_text for word in ["home", "ホーム", "トップ"]):
                return "/"
            else:
                # Extract meaningful part from href or text
                slug = re.sub(r'[^a-z0-9]+', '-', link_text[:30]).strip('-')
                return f"/{slug}" if slug else "#"
        else:
            # Extract from href path
            if "/" in href:
                path = href.split("/")[-1] or href.split("/")[-2]
                if path and not path.startswith("http"):
                    return f"/{path}"
            return "#"
    
    # Replace all anchor href attributes
    for link in soup.find_all("a", href=True):
        original_href = link.get("href", "")
        if original_href and (original_href.startswith("http://") or original_href.startswith("https://")):
            placeholder = generate_placeholder_url(link)
            link["href"] = placeholder
    
    # Replace form action URLs
    for form in soup.find_all("form", action=True):
        action = form.get("action", "")
        if action and (action.startswith("http://") or action.startswith("https://")):
            form["action"] = "#"
    
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
                            
                            # Format CSS for clean output
                            css_content = format_css(css_content)

                            # Get filename for reference
                            filename = css_path.name
                            # Save all CSS - use very high limit (10MB) to prevent truncation
                            css_files.append((filename, css_content[:10000000]))
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
                
                # Format CSS for clean output
                css_content = format_css(css_content)

                css_files.append(("inline", css_content[:10000000]))

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

                            # Format JavaScript for clean output
                            js_content = format_javascript(js_content)

                            filename = js_path.name
                            # Save all JS - use very high limit (10MB) to prevent truncation
                            js_files.append((filename, js_content[:10000000]))
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

                    # Format JavaScript for clean output
                    js_content = format_javascript(js_content)

                    js_files.append(("inline", js_content[:10000000]))

    except Exception as e:
        logger.warning(f"Error finding JS files: {e}")

    return js_files


def extract_html_content(html_path: Path, assets_dir: Path) -> str:
    """Extract and clean HTML content, replacing image URLs and site URLs with placeholders"""
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Clean up saved page artifacts
        content = re.sub(r"<!-- saved from url=.*?-->", "", content)

        # Replace image URLs with picsum.photos placeholders
        content = replace_image_urls(content, assets_dir)
        
        # Replace real site URLs with placeholder links
        content = replace_urls_with_placeholders(content)
        
        # Format HTML for clean output
        content = format_html(content)

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
            header_html = format_html(str(header_div))
            if len(header_html) > 100 and header_html not in components["headers"]:
                components["headers"].append(header_html)

        # Extract footers
        for footer in soup.find_all(["footer"]):
            footer_html = format_html(str(footer))
            if len(footer_html) > 100:
                components["footers"].append(footer_html)

        # Also look for footer-like divs
        for footer_div in soup.find_all("div", class_=re.compile(r"footer", re.I)):
            footer_html = format_html(str(footer_div))
            if len(footer_html) > 100 and footer_html not in components["footers"]:
                components["footers"].append(footer_html)

        # Extract sections (section tags or main content sections)
        for section in soup.find_all(["section"]):
            section_html = format_html(str(section))
            if len(section_html) > 200:  # Only substantial sections
                components["sections"].append(section_html)

        # Extract hero sections (usually first large section)
        for hero in soup.find_all(["div", "section"], class_=re.compile(r"hero|banner|mv|main-visual", re.I)):
            hero_html = format_html(str(hero))
            if len(hero_html) > 200:
                components["hero"].append(hero_html)

        # Extract buttons
        for button in soup.find_all(["button", "a"], class_=re.compile(r"btn|button", re.I)):
            # Get button and its parent context (for styling context)
            button_raw = str(button.parent) if button.parent and len(str(button.parent)) < 500 else str(button)
            button_html = format_html(button_raw)
            if len(button_html) > 50:
                components["buttons"].append(button_html)

        # Extract navigation menus
        for nav in soup.find_all(["nav"]):
            nav_html = format_html(str(nav))
            if len(nav_html) > 100:
                components["navigation"].append(nav_html)

        # Extract cards (common card patterns)
        for card in soup.find_all(["div", "article"], class_=re.compile(r"card|item|product|post", re.I)):
            card_html = format_html(str(card))
            if 200 < len(card_html) < 2000:  # Reasonable card size
                components["cards"].append(card_html)

        # Extract forms
        for form in soup.find_all(["form"]):
            form_html = format_html(str(form))
            if len(form_html) > 100:
                components["forms"].append(form_html)

        # Extract typography examples (headings with context)
        for heading in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
            # Get heading with some context (parent or sibling)
            parent = heading.parent
            if parent:
                typo_raw = str(parent) if len(str(parent)) < 500 else str(heading)
                typo_html = format_html(typo_raw)
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


def extract_complete_css_rules(css_content: str, max_length: int = None) -> str:
    """Return complete CSS content - NO TRUNCATION. All related CSS is saved."""
    # No truncation - return all CSS content
    return css_content



def extract_complete_css_rule_by_selector(css_content: str, selector_pattern: str) -> str:
    """Extract a complete CSS rule that matches a selector pattern"""
    # Find all CSS rules in the content
    rules = []
    i = 0
    while i < len(css_content):
        # Find selector start (look for patterns like .class, #id, tag, etc.)
        selector_start = i
        brace_start = css_content.find('{', i)
        if brace_start == -1:
            break
        
        # Get selector
        selector = css_content[selector_start:brace_start].strip()
        
        # Check if selector matches pattern
        if selector_pattern.lower() in selector.lower():
            # Find matching closing brace
            brace_depth = 1
            brace_end = brace_start + 1
            while brace_end < len(css_content) and brace_depth > 0:
                if css_content[brace_end] == '{':
                    brace_depth += 1
                elif css_content[brace_end] == '}':
                    brace_depth -= 1
                brace_end += 1
            
            if brace_depth == 0:
                # Found complete rule
                full_rule = css_content[selector_start:brace_end].strip()
                rules.append(full_rule)
                i = brace_end
            else:
                i = brace_start + 1
        else:
            # Skip this rule
            brace_depth = 1
            brace_end = brace_start + 1
            while brace_end < len(css_content) and brace_depth > 0:
                if css_content[brace_end] == '{':
                    brace_depth += 1
                elif css_content[brace_end] == '}':
                    brace_depth -= 1
                brace_end += 1
            i = brace_end
    
    return '\n\n'.join(rules)

def extract_css_for_component(component_html: str, all_css: List[Tuple[str, str]]) -> str:
    """Extract relevant CSS for a specific component, preserving complete CSS rules"""
    relevant_css_chunks = []
    soup = BeautifulSoup(component_html, "html.parser")

    # Collect all class names and IDs from the component
    classes = set()
    ids = set()
    tag_names = set()

    for element in soup.find_all(True):  # Find all elements
        tag_names.add(element.name.lower())
        if element.get("class"):
            classes.update(element.get("class"))
        if element.get("id"):
            ids.add(element.get("id"))

    # Search CSS for matching selectors - extract COMPLETE rules only
    for css_file_name, css_content in all_css:
        # Extract complete CSS rules that match component classes/IDs
        matching_rules = []
        root_rules_found = []  # Store :root rules separately
        
        # Find complete CSS rules (selector { properties })
        # First, normalize CSS to handle comments and multi-line selectors
        # Remove comments
        css_normalized = re.sub(r'/\*.*?\*/', '', css_content, flags=re.DOTALL)
        
        i = 0
        while i < len(css_normalized):
            # Find selector start - look backwards from { to find selector start
            brace_start = css_normalized.find('{', i)
            if brace_start == -1:
                break
            
            # Find selector start (go backwards to find start of selector)
            # Look for start of line, @, or previous }
            selector_start = i
            for j in range(brace_start - 1, max(0, brace_start - 200), -1):
                if css_normalized[j] in ['\n', '@', '}']:
                    selector_start = j + 1
                    break
                # Also check if we hit start of file
                if j == 0:
                    selector_start = 0
                    break
            
            # Get selector (everything before {)
            selector = css_normalized[selector_start:brace_start].strip()
            
            # Check if selector matches component
            selector_lower = selector.lower()
            is_match = False
            
            # Check for class matches (exact match required)
            for cls in classes:
                # Match .class or .class. or .class: or .class[ or .class { or .class,
                if re.search(rf'\.{re.escape(cls)}\s*[\.:,\[{{]', selector_lower) or selector_lower.endswith(f'.{cls}'):
                    is_match = True
                    break
            
            # Check for ID matches
            if not is_match:
                for id_val in ids:
                    if f"#{id_val}" in selector_lower or f"#{id_val}." in selector_lower:
                        is_match = True
                        break
            
            # Check for tag matches (only exact tag selectors)
            if not is_match:
                for tag in tag_names:
                    # Match tag as standalone selector (not in class/id)
                    if re.search(rf'^({tag}|[^{{]*\s+{tag})\s*\{{', selector_lower) and '.' not in selector_lower and '#' not in selector_lower:
                        is_match = True
                        break
            
            # Find complete rule (matching braces)
            if is_match or selector.strip() == ':root':
                brace_depth = 1
                brace_end = brace_start + 1
                while brace_end < len(css_content) and brace_depth > 0:
                    if css_content[brace_end] == '{':
                        brace_depth += 1
                    elif css_content[brace_end] == '}':
                        brace_depth -= 1
                    brace_end += 1
                
                if brace_depth == 0:
                    # Found complete rule - get from original CSS (preserve formatting)
                    full_rule = css_content[selector_start:brace_end].strip()
                    if selector.strip() == ':root':
                        # Don't include :root here - we'll add it later if needed
                        # Store it separately for later checking
                        root_rules_found.append(full_rule)
                    else:
                        matching_rules.append(full_rule)
                    i = brace_end
                else:
                    i = brace_start + 1
            else:
                # Skip this rule - find its end
                brace_depth = 1
                brace_end = brace_start + 1
                while brace_end < len(css_content) and brace_depth > 0:
                    if css_content[brace_end] == '{':
                        brace_depth += 1
                    elif css_content[brace_end] == '}':
                        brace_depth -= 1
                    brace_end += 1
                i = brace_end
        
        # Include :root rules only if they contain variables used by component rules
        if matching_rules:
            # Check if any matching rules use CSS variables
            rules_text = '\n'.join(matching_rules)
            if 'var(' in rules_text:
                # Extract all CSS variables used in component rules
                used_vars = set(re.findall(r'var\((--[\w-]+)\)', rules_text))
                
                # Find :root rules that define these variables
                i = 0
                while i < len(css_content):
                    # Check if this position starts :root (at start of line or after whitespace)
                    if i == 0 or css_content[i-1] in ['\n', ' ', '\t']:
                        if css_content[i:].startswith(':root'):
                            brace_start = css_content.find('{', i)
                        if brace_start > 0:
                            brace_depth = 1
                            brace_end = brace_start + 1
                            while brace_end < len(css_content) and brace_depth > 0:
                                if css_content[brace_end] == '{':
                                    brace_depth += 1
                                elif css_content[brace_end] == '}':
                                    brace_depth -= 1
                                brace_end += 1
                            
                            if brace_depth == 0:
                                root_rule = css_content[i:brace_end].strip()
                                # Extract variables defined in this :root
                                defined_vars = set(re.findall(r'--[\w-]+', root_rule))
                                # Only include if this :root defines variables used in component
                                if used_vars.intersection(defined_vars):
                                    if root_rule not in matching_rules:
                                        matching_rules.insert(0, root_rule)  # Add at beginning
                                i = brace_end
                            else:
                                i = brace_start + 1
                        else:
                            i += 1
                    else:
                        i += 1
                        if i >= len(css_content):
                            break
        
        if matching_rules:
            css_chunk = "\n\n".join(matching_rules)
            relevant_css_chunks.append(f"/* {css_file_name} */\n{css_chunk}")

    result = "\n\n".join(relevant_css_chunks)
    # Return all CSS - no truncation
    return result


def extract_js_for_component(component_html: str, all_js: List[Tuple[str, str]]) -> str:
    """Extract relevant JavaScript for a specific component"""
    relevant_js = []
    soup = BeautifulSoup(component_html, "html.parser")

    # Collect IDs, classes, and data attributes that might be used in JS
    identifiers = set()
    classes = set()
    for element in soup.find_all(True):
        if element.get("id"):
            identifiers.add(element.get("id"))
        if element.get("class"):
            classes.update(element.get("class"))
        for attr in element.attrs:
            if attr.startswith("data-"):
                identifiers.add(attr)
            # Also collect common JS-related attributes
            if attr in ["onclick", "onchange", "onsubmit", "onload", "onerror"]:
                identifiers.add(attr)

    # Search JS for matching identifiers, classes, or component-related patterns
    for js_file_name, js_content in all_js:
        js_lower = js_content.lower()
        is_relevant = False
        
        # Check if JS references component IDs (exact match required)
        for id_val in identifiers:
            # Match getElementById, querySelector with ID, or direct ID reference
            if (f'getelementbyid("{id_val}")' in js_lower or 
                f'getelementbyid(\'{id_val}\')' in js_lower or
                f'queryselector("#{id_val}")' in js_lower or
                f'queryselector(\'#{id_val}\')' in js_lower or
                f'#{id_val}' in js_lower):
                is_relevant = True
                break
        
        # Check if JS references component classes (exact match required)
        if not is_relevant:
            for cls in classes:
                # Match querySelector with class, getElementsByClassName, or classList operations
                if (f'queryselector(".{cls}")' in js_lower or 
                    f'queryselector(\'.{cls}\')' in js_lower or
                    f'getelementsbyclassname("{cls}")' in js_lower or
                    f'getelementsbyclassname(\'{cls}\')' in js_lower or
                    (f'.{cls}' in js_lower and 'classlist' in js_lower)):
                    is_relevant = True
                    break
        
        # Check for data attributes used in component
        if not is_relevant:
            for identifier in identifiers:
                if identifier.startswith('data-'):
                    attr_name = identifier.replace('data-', '')
                    if (f'dataset.{attr_name}' in js_lower or 
                        f'getattribute("{identifier}")' in js_lower or
                        f'getattribute(\'{identifier}\')' in js_lower):
                        is_relevant = True
                        break
        
        # Only include if it's actually relevant to THIS component
        if is_relevant:
            relevant_js.append(f"// {js_file_name}\n{js_content}")

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


def extract_design_keywords(css_content: str, html_content: str) -> List[str]:
    """Extract design-specific keywords from CSS and HTML"""
    keywords = []
    
    # Analyze CSS for design patterns
    css_lower = css_content.lower()
    html_lower = html_content.lower()
    combined = css_lower + " " + html_lower
    
    # Layout patterns
    if any(word in combined for word in ["grid", "display: grid", "grid-template"]):
        keywords.append("grid-based layout")
    if any(word in combined for word in ["flex", "display: flex", "flexbox"]):
        keywords.append("flexbox layout")
    if any(word in combined for word in ["center", "text-align: center", "margin: 0 auto"]):
        keywords.append("centered alignment")
    if any(word in combined for word in ["justify-content: space-between", "space-between"]):
        keywords.append("space-between distribution")
    
    # Visual style
    if any(word in combined for word in ["minimal", "minimalist", "clean", "simple"]):
        keywords.append("minimalist design")
    if any(word in combined for word in ["bold", "font-weight: bold", "strong"]):
        keywords.append("bold typography")
    if any(word in combined for word in ["shadow", "box-shadow", "drop-shadow"]):
        keywords.append("shadow effects")
    if any(word in combined for word in ["gradient", "linear-gradient", "radial-gradient"]):
        keywords.append("gradient styling")
    if any(word in combined for word in ["rounded", "border-radius", "rounded corners"]):
        keywords.append("rounded corners")
    if any(word in combined for word in ["transparent", "rgba", "opacity"]):
        keywords.append("transparency effects")
    
    # Typography
    if any(word in combined for word in ["serif", "times", "georgia"]):
        keywords.append("serif typography")
    if any(word in combined for word in ["sans-serif", "arial", "helvetica", "roboto"]):
        keywords.append("sans-serif typography")
    if any(word in combined for word in ["uppercase", "text-transform: uppercase"]):
        keywords.append("uppercase text")
    if any(word in combined for word in ["letter-spacing", "tracking"]):
        keywords.append("letter spacing")
    
    # Spacing
    if any(word in combined for word in ["padding", "margin", "gap"]):
        keywords.append("generous spacing")
    if any(word in combined for word in ["clamp", "min(", "max(", "responsive units"]):
        keywords.append("fluid typography")
    
    # Color patterns
    if any(word in combined for word in ["dark", "#000", "rgb(0", "black"]):
        keywords.append("dark color scheme")
    if any(word in combined for word in ["light", "#fff", "rgb(255", "white"]):
        keywords.append("light color scheme")
    if any(word in combined for word in ["primary", "accent", "brand color"]):
        keywords.append("accent colors")
    
    # Interaction
    if any(word in combined for word in ["hover", ":hover", "transition"]):
        keywords.append("hover interactions")
    if any(word in combined for word in ["animation", "@keyframes", "transform"]):
        keywords.append("animations")
    if any(word in combined for word in ["smooth", "ease", "cubic-bezier"]):
        keywords.append("smooth transitions")
    
    # Layout structure
    if any(word in combined for word in ["container", "wrapper", "max-width"]):
        keywords.append("contained layout")
    if any(word in combined for word in ["full-width", "width: 100%", "fullwidth"]):
        keywords.append("full-width sections")
    if any(word in combined for word in ["sticky", "position: sticky", "fixed"]):
        keywords.append("sticky positioning")
    
    return keywords[:8]  # Limit to top 8 keywords


def generate_enhanced_design_reasoning(
    component_type: str,
    metadata: Dict,
    characteristics: Dict,
    css_files: List[Tuple[str, str]],
    component_html: str = "",
) -> str:
    """Generate enhanced design reasoning with specific design keywords"""
    reasoning_parts = []
    
    # Component-specific reasoning
    component_reasoning = {
        "headers": "Header design establishes brand identity and navigation hierarchy. Fixed positioning ensures constant access to navigation while maintaining visual prominence.",
        "footers": "Footer design provides essential site information and secondary navigation. Clean, organized layout supports user trust and site credibility.",
        "sections": "Section layout creates visual rhythm and content hierarchy. Proper spacing and typography guide user attention through content flow.",
        "buttons": "Button design emphasizes call-to-action with clear visual hierarchy. Hover states and transitions provide interactive feedback.",
        "navigation": "Navigation design prioritizes usability and accessibility. Responsive behavior adapts to different screen sizes while maintaining functionality.",
        "hero": "Hero section creates immediate visual impact and communicates primary message. Large imagery and typography establish brand presence.",
        "cards": "Card design organizes content into digestible units. Shadows and spacing create depth and visual separation.",
        "forms": "Form design prioritizes usability and accessibility. Clear labels and validation states guide user input.",
        "typography": "Typography establishes content hierarchy and readability. Font choices and spacing create visual rhythm.",
    }
    
    base_reasoning = component_reasoning.get(component_type, "Component design follows modern web standards with focus on usability and visual appeal.")
    reasoning_parts.append(f"- {base_reasoning}")
    
    # Add design style
    tone = characteristics.get("tone", "Professional")
    if tone:
        tone_descriptions = {
            "Professional": "Professional tone emphasizes trust, credibility, and clarity",
            "Modern": "Modern design emphasizes clean lines, minimalism, and contemporary aesthetics",
            "Elegant": "Elegant design emphasizes sophistication, refinement, and premium feel",
            "Bold": "Bold design emphasizes strong visual impact, high contrast, and confident styling",
            "Minimalist": "Minimalist design emphasizes simplicity, whitespace, and essential elements",
            "Creative": "Creative design emphasizes unique layouts, artistic elements, and visual interest",
        }
        tone_desc = tone_descriptions.get(tone, f"{tone} design style")
        reasoning_parts.append(f"- Design style: {tone_desc}")
    
    # Extract and add design keywords
    all_css = " ".join([css[1] for css in css_files])
    design_keywords = extract_design_keywords(all_css, component_html)
    if design_keywords:
        keywords_text = ", ".join(design_keywords)
        reasoning_parts.append(f"- Design features: {keywords_text}")
    
    # Layout approach
    layout = characteristics.get("layout", "Standard")
    if layout and layout != "Standard":
        reasoning_parts.append(f"- Layout approach: {layout} layout structure for optimal content organization")
    
    # Responsive design
    if characteristics.get("responsive"):
        reasoning_parts.append("- Responsive design: Mobile-first approach with flexible grid system and breakpoints for all devices")
    
    # Color scheme
    color_scheme = characteristics.get("color_scheme", "")
    if color_scheme and color_scheme != "Unknown":
        reasoning_parts.append(f"- Color palette: {color_scheme} color scheme supporting brand identity and visual hierarchy")
    
    # Photo usage
    photo_usage = characteristics.get("photo_usage", "")
    if photo_usage:
        reasoning_parts.append(f"- Visual content: {photo_usage} imagery to support brand message and user engagement")
    
    return "\n".join(reasoning_parts)


def generate_design_reasoning_with_openai(
    metadata: Dict,
    characteristics: Dict,
    css_files: List[Tuple[str, str]],
    js_files: List[Tuple[str, str]],
    component_type: str = "",
    component_html: str = "",
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

        # Extract design keywords for context
        all_css = " ".join([css[1] for css in css_files])
        design_keywords = extract_design_keywords(all_css, component_html)
        keywords_text = ", ".join(design_keywords) if design_keywords else "standard web design patterns"
        
        prompt = f"""You are a senior creative front-end engineer analyzing a website design.

Given the following website characteristics:
{context}

Design keywords detected: {keywords_text}

Explain the design reasoning in 4-5 detailed bullet points:
1. Component/design purpose: Why this design approach fits the industry/brand and user needs
2. Visual design: How the visual style (tone, colors, typography, spacing) supports the brand message and creates hierarchy
3. Layout structure: Why this layout pattern (grid/flexbox/centered/etc.) was chosen and how it organizes content
4. Interactive elements: How motion, transitions, and interactions enhance user experience and convey brand personality
5. Technical implementation: Notable CSS/design techniques used and their purpose (e.g., clamp for fluid typography, CSS Grid for complex layouts, custom properties for theming)

Be specific, professional, and focus on design intent and user experience. Include design keywords like: {keywords_text}
Keep each point to 2-3 sentences with concrete details.
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
        
        # Sleep to avoid rate limiting (1.5 seconds between API calls)
        time.sleep(1.5)
        
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

    # Extract design keywords
    all_css = " ".join([css[1] for css in css_files])
    design_keywords = extract_design_keywords(all_css, component_html)
    
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
    
    # Add design keywords to instruction
    if design_keywords:
        keywords_text = ", ".join(design_keywords)
        instruction_parts.append(f"Design style: {keywords_text}")

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

    # Generate enhanced design reasoning
    reasoning = generate_enhanced_design_reasoning(
        component_type,
        metadata,
        characteristics,
        css_files,
        component_html,
    )
    
    # Build output
    output_parts = ["Design reasoning:"]
    output_parts.append(reasoning)

    output_parts.append("\nCode:")
    output_parts.append("```html")
    # Get ALL HTML - no truncation
    output_parts.append(component_html)
    output_parts.append("```")

    if component_css:
        output_parts.append("\n```css")
        # Only include complete CSS rules, no truncation markers
        # Get ALL CSS - no truncation
        css_limited = component_css
        output_parts.append(css_limited)
        output_parts.append("```")

    # Include JavaScript if available (no strict length limit, but prefer reasonable size)
    if component_js:
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

    # Extract design keywords
    all_css = " ".join([css[1] for css in css_files])
    design_keywords = extract_design_keywords(all_css, combined_html)
    
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
    
    # Add design keywords
    if design_keywords:
        keywords_text = ", ".join(design_keywords)
        instruction_parts.append(f"Design style: {keywords_text}")

    component_names = " + ".join([c.capitalize() for c in component_types])
    instruction_parts.append(
        f"Task: Create a combined {component_names} component group with semantic HTML structure, "
        "modern CSS for layout and styling, and clean JavaScript for interactivity. "
        "Ensure components work together harmoniously. "
        "Use placeholder images from https://picsum.photos/ with appropriate dimensions."
    )

    instruction_text = "\n".join(instruction_parts)

    # Generate enhanced reasoning for grouped components
    reasoning_parts = []
    reasoning_parts.append(f"- Component group: {component_names} working together harmoniously")
    reasoning_parts.append(f"- Design style: {characteristics.get('tone', 'Professional')} brand tone with consistent visual language")
    if design_keywords:
        keywords_text = ", ".join(design_keywords)
        reasoning_parts.append(f"- Design features: {keywords_text}")
    if characteristics.get("responsive"):
        reasoning_parts.append("- Responsive: Mobile-first approach with flexible layout system")
    if characteristics.get("color_scheme") and characteristics["color_scheme"] != "Unknown":
        reasoning_parts.append(f"- Color palette: {characteristics['color_scheme']} color scheme for visual cohesion")
    
    # Build output
    output_parts = ["Design reasoning:"]
    output_parts.append("\n".join(reasoning_parts))

    output_parts.append("\nCode:")
    output_parts.append("```html")
    output_parts.append(combined_html)
    output_parts.append("```")

    if combined_css:
        output_parts.append("\n```css")
        # Only include complete CSS rules, no truncation markers
        # Get ALL CSS - no truncation
        css_limited = combined_css
        output_parts.append(css_limited)
        output_parts.append("```")

    # Include JavaScript if available
    if combined_js:
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

    # Extract and add design keywords
    all_css = " ".join([css[1] for css in css_files])
    design_keywords = extract_design_keywords(all_css, html_content)
    if design_keywords:
        keywords_text = ", ".join(design_keywords)
        instruction_parts.append(f"Design style: {keywords_text}")

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
            metadata, characteristics, css_files, js_files, component_type="full-page", component_html=html_content
        )

    # Build assistant response with layout focus
    assistant_parts = ["Design reasoning:"]

    if reasoning_text:
        # Use AI-generated reasoning
        assistant_parts.append(reasoning_text)
    else:
        # Enhanced fallback reasoning
        reasoning_parts = []
        reasoning_parts.append(
            f"- Layout structure: {characteristics.get('layout', 'Standard')} layout approach "
            f"for optimal information hierarchy and user experience"
        )
        reasoning_parts.append(
            f"- Visual design: {characteristics.get('tone', 'Professional')} brand tone with "
            f"{characteristics.get('color_scheme', 'balanced')} color palette"
        )
        if design_keywords:
            keywords_text = ", ".join(design_keywords)
            reasoning_parts.append(f"- Design features: {keywords_text}")
        if characteristics.get("responsive"):
            reasoning_parts.append(
                "- Responsive design: Mobile-first approach with flexible grid system and breakpoints"
            )
        if characteristics.get("motion") and characteristics["motion"] != "None":
            reasoning_parts.append(
                f"- Interactions: {characteristics['motion']} for enhanced user engagement"
            )
        if characteristics.get("photo_usage"):
            reasoning_parts.append(
                f"- Visual content: {characteristics['photo_usage']} imagery to support brand message"
            )
        assistant_parts.append("\n".join(reasoning_parts))

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

    # Include JS if it's layout-related (navigation, responsive behavior, interactions)
    if js_files:
        # Check if JS is layout-related (navigation, menu, responsive, interactions)
        js_content_lower = js_files[0][1].lower()
        layout_js_keywords = [
            "menu",
            "nav",
            "toggle",
            "responsive",
            "mobile",
            "breakpoint",
            "scroll",
            "dropdown",
            "modal",
            "animation",
            "transition",
            "eventlistener",
            "queryselector",
            "getelementbyid"
        ]
        if any(keyword in js_content_lower for keyword in layout_js_keywords):
            assistant_parts.append("\n```javascript")
            js_content = "\n\n// " + js_files[0][0] + "\n" + js_files[0][1]
            # Include additional JS files if they're also relevant and not too large
            for js_file_name, js_file_content in js_files[1:]:
                if len(js_file_content) < MAX_JS_LENGTH:
                    js_content += f"\n\n// {js_file_name}\n{js_file_content}"
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


def build_dataset(incremental: bool = False, use_ai: bool = True, include_full_page: bool = True, export_to_folders: bool = True, fresh_start: bool = False):
    """Main function to build the dataset with enhanced features"""
    logger.info("=" * 80)
    logger.info("Building dataset from project templates")
    logger.info("=" * 80)
    
    # Handle fresh start - remove existing dataset and result directory
    if fresh_start:
        logger.warning("FRESH START: Removing existing dataset and results")
        if OUTPUT_FILE.exists():
            OUTPUT_FILE.unlink()
            logger.info(f"✓ Removed existing dataset: {OUTPUT_FILE}")
        
        if RESULT_DIR.exists():
            import shutil
            try:
                shutil.rmtree(RESULT_DIR)
                logger.info(f"✓ Removed existing results: {RESULT_DIR}")
            except Exception as e:
                logger.warning(f"Could not remove {RESULT_DIR}: {e}")
        
        logger.info("Starting fresh - building new dataset from scratch")
        incremental = False  # Force non-incremental when starting fresh

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

    # Process all templates with incremental saving
    examples = []
    skipped = 0
    errors = 0
    total_components = 0
    total_grouped = 0
    total_full_pages = 0
    
    # Open output file for incremental writing (append mode to preserve existing data)
    file_mode = "w" if fresh_start else "a"
    output_file_handle = None
    
    try:
        # Ensure directory exists
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        output_file_handle = open(OUTPUT_FILE, file_mode, encoding="utf-8")
        
        for i, template_dir in enumerate(template_dirs, 1):
            logger.info(f"[{i}/{len(template_dirs)}] Processing {template_dir.name}...")

            try:
                template_examples = process_website_template(
                    template_dir, use_ai_reasoning=use_ai, include_full_page=include_full_page
                )
                if template_examples:
                    # Filter out duplicates if in incremental mode
                    new_examples = []
                    for ex in template_examples:
                        if incremental:
                            # Check for duplicates using output hash
                            output = ex.get("output", "")
                            if output:
                                output_hash = hashlib.md5(output.encode()).hexdigest()
                                if output_hash not in existing_hashes:
                                    existing_hashes.add(output_hash)
                                    new_examples.append(ex)
                                else:
                                    logger.debug(f"  Skipping duplicate example")
                            else:
                                new_examples.append(ex)
                        else:
                            new_examples.append(ex)
                    
                    if new_examples:
                        # Write examples immediately to preserve progress
                        for ex in new_examples:
                            output_file_handle.write(json.dumps(ex, ensure_ascii=False) + "\n")
                            output_file_handle.flush()  # Ensure data is written immediately
                        
                        examples.extend(new_examples)
                        # Count example types (rough estimate based on instruction content)
                        for ex in new_examples:
                            inst = ex.get("instruction", "")
                            if "full page" in inst.lower() or "complete" in inst.lower():
                                total_full_pages += 1
                            elif "+" in inst or "group" in inst.lower():
                                total_grouped += 1
                            else:
                                total_components += 1
                        logger.info(f"  ✓ Saved {len(new_examples)} examples to dataset")
                    else:
                        logger.debug(f"  All examples were duplicates, skipping")
                        skipped += 1
                else:
                    skipped += 1
            except Exception as e:
                # If using AI and error occurs, switch to non-AI mode and retry
                if use_ai:
                    logger.warning(f"⚠ Error occurred while processing with AI: {e}")
                    logger.warning(f"   Switching to non-AI mode from this point forward...")
                    use_ai = False
                    
                    # Retry processing this template without AI
                    try:
                        logger.info(f"   Retrying {template_dir.name} without AI...")
                        template_examples = process_website_template(
                            template_dir, use_ai_reasoning=False, include_full_page=include_full_page
                        )
                        if template_examples:
                            # Filter out duplicates if in incremental mode
                            new_examples = []
                            for ex in template_examples:
                                if incremental:
                                    output = ex.get("output", "")
                                    if output:
                                        output_hash = hashlib.md5(output.encode()).hexdigest()
                                        if output_hash not in existing_hashes:
                                            existing_hashes.add(output_hash)
                                            new_examples.append(ex)
                                    else:
                                        new_examples.append(ex)
                                else:
                                    new_examples.append(ex)
                            
                            if new_examples:
                                # Write examples immediately
                                for ex in new_examples:
                                    output_file_handle.write(json.dumps(ex, ensure_ascii=False) + "\n")
                                    output_file_handle.flush()
                                
                                examples.extend(new_examples)
                                # Count example types
                                for ex in new_examples:
                                    inst = ex.get("instruction", "")
                                    if "full page" in inst.lower() or "complete" in inst.lower():
                                        total_full_pages += 1
                                    elif "+" in inst or "group" in inst.lower():
                                        total_grouped += 1
                                    else:
                                        total_components += 1
                                logger.info(f"  ✓ Saved {len(new_examples)} examples to dataset (non-AI)")
                            else:
                                skipped += 1
                        else:
                            skipped += 1
                    except Exception as retry_error:
                        logger.error(f"Error processing {template_dir.name} (non-AI retry): {retry_error}", exc_info=True)
                        errors += 1
                        skipped += 1
                else:
                    # Already in non-AI mode, just log the error
                    logger.error(f"Error processing {template_dir.name}: {e}", exc_info=True)
                    errors += 1
                    skipped += 1
    finally:
        # Close the file handle
        if output_file_handle:
            output_file_handle.close()

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

    # Dataset has already been written incrementally, just report final stats
    logger.info(f"✓ Dataset saved incrementally to {OUTPUT_FILE}")
    logger.info(f"  - {len(examples)} examples written (saved incrementally during processing)")
    logger.info(f"  - Log file: {LOG_FILE}")

    # Print statistics
    try:
        if OUTPUT_FILE.exists():
            total_size = OUTPUT_FILE.stat().st_size
            logger.info(f"  - File size: {total_size / 1024:.2f} KB")
    except Exception as e:
        logger.warning(f"Could not get file size: {e}")
    
    # Export to readable format in result/ directory
    if export_to_folders:
        export_examples_to_folders(examples)


def fix_incomplete_css(css_content: str) -> str:
    """Attempt to fix common CSS issues: wrap CSS variables in :root, fix incomplete rules, remove truncation markers"""
    if not css_content:
        return css_content
    
    # Remove any truncation markers first
    css_content = re.sub(r'/\*\s*\.\.\.\s*more\s+CSS\s*\.\.\.\s*\*/', '', css_content, flags=re.IGNORECASE)
    css_content = re.sub(r'/\*\s*\.\.\.\s*more\s+.*?\s*\.\.\.\s*\*/', '', css_content, flags=re.IGNORECASE)
    
    lines = css_content.split('\n')
    
    # Check if CSS variables need :root wrapper
    has_vars = any(line.strip().startswith('--') for line in lines)
    has_root = ':root' in css_content
    
    if has_vars and not has_root:
        # Collect CSS variables and other CSS
        var_lines = []
        other_lines = []
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('--') and ':' in stripped:
                var_lines.append(stripped)
            else:
                other_lines.append(line)
        
        # Wrap variables in :root
        if var_lines:
            wrapped_vars = ':root {\n  ' + '\n  '.join(var_lines) + '\n}'
            if other_lines:
                return wrapped_vars + '\n\n' + '\n'.join(other_lines)
            return wrapped_vars
    
    # Try to fix incomplete selectors (selectors without opening braces)
    fixed_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Skip comments (but keep them)
        if stripped.startswith('/*') and not stripped.endswith('*/'):
            # Multi-line comment, include until closing
            comment_lines = [line]
            i += 1
            while i < len(lines) and '*/' not in lines[i]:
                comment_lines.append(lines[i])
                i += 1
            if i < len(lines):
                comment_lines.append(lines[i])
            fixed_lines.extend(comment_lines)
            i += 1
            continue
        
        # Skip truncation markers
        if '...' in stripped and 'more' in stripped.lower():
            i += 1
            continue
        
        # If line looks like a selector but has no opening brace
        if (stripped and 
            not stripped.startswith('--') and 
            ':' not in stripped and 
            '{' not in stripped and 
            '}' not in stripped and
            (stripped.endswith(',') or 
             any(char in stripped for char in ['.', '#', '@']) or
             any(tag in stripped.lower() for tag in ['header', 'footer', 'nav', 'div', 'section', 'button', 'a', 'ul', 'li']))):
            
            # Look ahead for opening brace
            found_brace = False
            for j in range(i + 1, min(i + 5, len(lines))):
                if '{' in lines[j]:
                    found_brace = True
                    break
            
            if not found_brace:
                # Incomplete selector, skip it
                i += 1
                continue
        
        fixed_lines.append(line)
        i += 1
    
    result = '\n'.join(fixed_lines)
    
    # Final cleanup: remove any remaining truncation markers
    result = re.sub(r'/\*\s*\.\.\.\s*.*?\s*\.\.\.\s*\*/', '', result, flags=re.IGNORECASE | re.DOTALL)
    
    return result.strip()


def parse_code_blocks(output: str) -> Dict[str, str]:
    """Parse code blocks from output string"""
    code_blocks = {
        "html": "",
        "css": "",
        "javascript": "",
        "reasoning": "",
    }
    
    # Extract reasoning (everything before "Code:")
    if "Code:" in output:
        reasoning_part = output.split("Code:")[0]
        code_blocks["reasoning"] = reasoning_part.replace("Design reasoning:", "").strip()
    
    # Extract HTML block
    html_match = re.search(r'```html\s*\n(.*?)```', output, re.DOTALL)
    if html_match:
        code_blocks["html"] = html_match.group(1).strip()
    
    # Extract CSS block
    css_match = re.search(r'```css\s*\n(.*?)```', output, re.DOTALL)
    if css_match:
        css_content = css_match.group(1).strip()
        # Try to fix incomplete CSS
        code_blocks["css"] = fix_incomplete_css(css_content)
    
    # Extract JavaScript block
    js_match = re.search(r'```javascript\s*\n(.*?)```', output, re.DOTALL)
    if js_match:
        code_blocks["javascript"] = js_match.group(1).strip()
    
    return code_blocks


def export_examples_to_folders(examples: List[Dict]):
    """Export each example to its own folder with separate HTML, CSS, JS files"""
    logger.info(f"\nExporting examples to {RESULT_DIR}...")
    
    # Clean existing result directory BEFORE creating new one
    if RESULT_DIR.exists():
        import shutil
        logger.info(f"Cleaning existing {RESULT_DIR}...")
        try:
            # Count items before deletion
            items = list(RESULT_DIR.iterdir())
            dir_count = sum(1 for item in items if item.is_dir())
            file_count = sum(1 for item in items if item.is_file())
            
            # Remove all contents
            for item in items:
                try:
                    if item.is_dir():
                        shutil.rmtree(item)
                    elif item.is_file():
                        item.unlink()
                except Exception as e:
                    logger.warning(f"Could not remove {item.name}: {e}")
            
            logger.info(f"  ✓ Cleaned {dir_count} folders and {file_count} files")
        except Exception as e:
            logger.error(f"Error cleaning {RESULT_DIR}: {e}")
            # Try to continue anyway
    
    # Create result directory (fresh)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    
    exported_count = 0
    
    for idx, example in enumerate(examples, 1):
        try:
            # Create folder for this example
            example_folder = RESULT_DIR / f"example_{idx:04d}"
            example_folder.mkdir(parents=True, exist_ok=True)
            
            # Parse code blocks from output
            output = example.get("output", "")
            code_blocks = parse_code_blocks(output)
            
            # Save instruction
            instruction = example.get("instruction", "")
            instruction_file = example_folder / "instruction.txt"
            with open(instruction_file, "w", encoding="utf-8") as f:
                f.write(instruction)
            
            # Save reasoning
            if code_blocks["reasoning"]:
                reasoning_file = example_folder / "reasoning.txt"
                with open(reasoning_file, "w", encoding="utf-8") as f:
                    f.write(code_blocks["reasoning"])
            
            # Save HTML (wrap in basic HTML structure if it's just a fragment)
            if code_blocks["html"]:
                html_file = example_folder / "index.html"
                html_content = code_blocks["html"].strip()
                
                # Check if it's already a complete HTML document
                is_complete_html = (
                    html_content.startswith(("<!DOCTYPE", "<!doctype", "<html", "<HTML")) or
                    ("<head>" in html_content.lower() and "<body>" in html_content.lower())
                )
                
                # If it's just a fragment, wrap it in a basic HTML structure
                if not is_complete_html:
                    # Only add script tag if there's JS
                    script_tag = '\n    <script src="script.js"></script>' if code_blocks["javascript"] else ''
                    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Example {idx}</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
{html_content}{script_tag}
</body>
</html>"""
                
                with open(html_file, "w", encoding="utf-8") as f:
                    f.write(html_content)
            
            # Save CSS
            if code_blocks["css"]:
                css_file = example_folder / "style.css"
                with open(css_file, "w", encoding="utf-8") as f:
                    f.write(code_blocks["css"])
            
            # Save JavaScript
            if code_blocks["javascript"]:
                js_file = example_folder / "script.js"
                with open(js_file, "w", encoding="utf-8") as f:
                    f.write(code_blocks["javascript"])
            
            # Save full output for reference
            output_file = example_folder / "output.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(output)
            
            exported_count += 1
            
        except Exception as e:
            logger.warning(f"Error exporting example {idx}: {e}")
            continue
    
    logger.info(f"✓ Exported {exported_count} examples to {RESULT_DIR}")
    logger.info(f"  - Each example is in its own folder (example_0001, example_0002, etc.)")
    logger.info(f"  - Files: instruction.txt, reasoning.txt, index.html, style.css, script.js, output.txt")


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
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Skip exporting examples to result/ directory (only create JSONL)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Remove existing dataset and results, start from scratch (WARNING: This deletes dataset/train.jsonl and result/ directory)",
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
        export_to_folders=not args.no_export,
        fresh_start=args.fresh,
    )
