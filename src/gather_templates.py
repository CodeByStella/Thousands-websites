"""
Script to gather website templates from muuuuu.org.
Fetches site listings, detail pages, and actual website content.
"""

import os
import json
import re
import time
import logging
import requests
from pathlib import Path
from bs4 import BeautifulSoup
from typing import Dict, List, Optional, Tuple, Set
from urllib.parse import urljoin, urlparse
import hashlib
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Configuration - get paths relative to project root
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
PROJECT_TEMPLATES_DIR = PROJECT_ROOT / "project_templates"
LOG_FILE = PROJECT_ROOT / "gather.log"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# API Configuration
API_ENDPOINT = "https://muuuuu.org/wp-content/themes/muuuuuorg/ajax-item.php"
BASE_URL = "https://muuuuu.org"
REQUEST_INTERVAL = 2  # seconds between requests to avoid rate limiting
BATCH_SIZE = 20  # number of items to fetch per request
MAX_RETRIES = 3
TIMEOUT = 30  # seconds

# Create project_templates directory if it doesn't exist
PROJECT_TEMPLATES_DIR.mkdir(exist_ok=True)


def sanitize_filename(url: str) -> str:
    """Convert URL to a safe filename."""
    # Remove protocol
    url = url.replace("https://", "").replace("http://", "")
    # Replace invalid characters
    url = re.sub(r'[<>:"|?*]', '_', url)
    # Remove trailing slash
    url = url.rstrip('/')
    return url


def get_session() -> requests.Session:
    """Create a requests session with proper headers."""
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        # Do NOT force 'br' here; let requests/urllib3 advertise brotli only if supported
        # 'Accept-Encoding': is intentionally not set to avoid advertising 'br' without a decoder
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    })
    return session


def _get_decoded_bytes(response: requests.Response) -> bytes:
    """
    Return response bytes, explicitly handling Content-Encoding if needed.
    - Handles 'br' (brotli) when brotli/brotlicffi is available
    - Handles 'gzip' as a fallback (requests usually auto-decompresses)
    """
    content_encoding = (response.headers.get('Content-Encoding') or '').lower()
    raw = response.content
    if 'br' in content_encoding:
        try:
            try:
                import brotli  # type: ignore
            except ImportError:
                import brotlicffi as brotli  # type: ignore
            decoded = brotli.decompress(raw)
            logger.info(f"[HTTP] Brotli decoded: {len(raw)} -> {len(decoded)} bytes")
            return decoded
        except Exception as e:
            logger.warning(f"[HTTP] Brotli decode failed, using raw bytes: {e}")
            return raw
    if 'gzip' in content_encoding:
        try:
            import gzip
            decoded = gzip.decompress(raw)
            logger.info(f"[HTTP] Gzip decoded: {len(raw)} -> {len(decoded)} bytes")
            return decoded
        except Exception as e:
            # requests usually auto-decompresses gzip; if this fails, keep raw
            logger.debug(f"[HTTP] Gzip manual decode failed, using raw bytes: {e}")
            return raw
    return raw


def fetch_site_list(session: requests.Session, post_num_now: int, post_num_add: int) -> Optional[str]:
    """Fetch site list from API endpoint."""
    try:
        response = session.post(
            API_ENDPOINT,
            data={
                'post_num_now': post_num_now,
                'post_num_add': post_num_add
            },
            timeout=TIMEOUT
        )
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        logger.error(f"Error fetching site list (post_num_now={post_num_now}): {e}")
        return None


def parse_site_list_items(html: str) -> List[Dict[str, str]]:
    """Parse HTML list items and extract site information."""
    soup = BeautifulSoup(html, 'html.parser')
    items = []
    
    for li in soup.find_all('li', class_='c-post-list__item'):
        try:
            # Extract detail page URL
            detail_link = li.find('a', class_='c-post-list__title-link')
            detail_url = detail_link.get('href') if detail_link else None
            
            # Extract site URL from the image link
            img_link = li.find('a', class_='c-post-list__link')
            site_url = img_link.get('href') if img_link else None
            
            # Extract site title
            title_elem = li.find('h3', class_='c-post-list__ttl')
            title = None
            if title_elem:
                title_link = title_elem.find('a')
                if title_link:
                    title_span = title_link.find('span', class_='c-linelink__txt')
                    if title_span:
                        title = title_span.get_text(strip=True)
            
            # Extract credit information from detail section
            credit_detail = li.find('div', class_='c-post-list__credit-detail')
            credit_info = {}
            if credit_detail:
                # Extract URL
                url_elem = credit_detail.find('dt', string=re.compile('URL'))
                if url_elem:
                    url_dd = url_elem.find_next_sibling('dd')
                    if url_dd:
                        url_span = url_dd.find('span')
                        if url_span:
                            credit_info['url'] = url_span.get_text(strip=True)
                
                # Extract copyright owner
                copyright_elem = credit_detail.find('dt', string=re.compile('著作権所有'))
                if copyright_elem:
                    copyright_dd = copyright_elem.find_next_sibling('dd')
                    if copyright_dd:
                        copyright_span = copyright_dd.find('span')
                        if copyright_span:
                            credit_info['copyright'] = copyright_span.get_text(strip=True)
                
                # Extract creator list
                creator_elem = credit_detail.find('dt', string=re.compile('制作者一覧'))
                if creator_elem:
                    creator_dd = creator_elem.find_next_sibling('dd')
                    if creator_dd:
                        credit_info['creators'] = creator_dd.get_text(strip=True)
                
                # Extract creator source
                creator_source_elem = credit_detail.find('dt', string=re.compile('制作者情報引用元'))
                if creator_source_elem:
                    creator_source_dd = creator_source_elem.find_next_sibling('dd')
                    if creator_source_dd:
                        creator_source_span = creator_source_dd.find('span')
                        if creator_source_span:
                            credit_info['creator_source'] = creator_source_span.get_text(strip=True)
            
            if detail_url and site_url:
                items.append({
                    'detail_url': detail_url if detail_url.startswith('http') else urljoin(BASE_URL, detail_url),
                    'site_url': site_url,
                    'title': title or 'Unknown',
                    'credit_info': credit_info
                })
        except Exception as e:
            logger.warning(f"Error parsing list item: {e}")
            continue
    
    return items


def process_detail_page(session: requests.Session, detail_url: str, project_dir: Path) -> bool:
    """Fetch and process detail page HTML, downloading all assets."""
    for attempt in range(MAX_RETRIES):
        try:
            # Fetch main HTML with proper encoding handling
            try:
                response = session.get(detail_url, timeout=TIMEOUT, stream=False)
                response.raise_for_status()
                logger.info(f"[HTTP] GET detail {detail_url} -> {response.status_code}")
                logger.info(f"[HTTP] Detail Headers: Content-Type={response.headers.get('Content-Type')}, Content-Encoding={response.headers.get('Content-Encoding')}, Transfer-Encoding={response.headers.get('Transfer-Encoding')}")
                logger.info(f"[HTTP] Detail raw content length: {len(response.content)} bytes")
            except requests.exceptions.ConnectionError as e:
                logger.error(f"Connection error fetching detail page {detail_url} (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(REQUEST_INTERVAL * (attempt + 1))  # Exponential backoff
                    continue
                else:
                    return False
            except requests.exceptions.Timeout as e:
                logger.error(f"Timeout fetching detail page {detail_url} (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(REQUEST_INTERVAL * (attempt + 1))
                    continue
                else:
                    return False
            except requests.exceptions.RequestException as e:
                logger.error(f"Request error fetching detail page {detail_url} (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(REQUEST_INTERVAL * (attempt + 1))
                    continue
                else:
                    return False
            
            # Use the same encoding detection logic as fetch_website_content
            html_content = None
            detected_encoding = None
            
            # Try to detect charset from HTML content itself
            content_bytes = _get_decoded_bytes(response)
            logger.info(f"[HTTP] Detail decoded-bytes length (after Content-Encoding handling): {len(content_bytes)} bytes")
            preview_bytes = content_bytes[:8192]
            
            for test_encoding in ['utf-8', 'shift_jis', 'euc-jp', 'cp932', 'iso-2022-jp']:
                try:
                    preview_text = preview_bytes.decode(test_encoding, errors='ignore')
                    charset_match = re.search(
                        r'<meta[^>]*charset\s*=\s*["\']?([^"\'>\s]+)["\']?',
                        preview_text,
                        re.IGNORECASE
                    )
                    if charset_match:
                        detected_encoding = charset_match.group(1).lower()
                        encoding_map = {
                            'utf-8': 'utf-8', 'utf8': 'utf-8',
                            'shift_jis': 'shift_jis', 'shift-jis': 'shift_jis', 'sjis': 'shift_jis',
                            'euc-jp': 'euc-jp', 'eucjp': 'euc-jp',
                            'iso-2022-jp': 'iso-2022-jp',
                            'cp932': 'cp932',
                        }
                        detected_encoding = encoding_map.get(detected_encoding, detected_encoding)
                        break
                except (UnicodeDecodeError, LookupError):
                    continue
            
            encoding = detected_encoding or response.encoding
            if not encoding or encoding.lower() == 'iso-8859-1':
                try:
                    import chardet
                    detected = chardet.detect(content_bytes)
                    if detected and detected.get('encoding') and detected['confidence'] > 0.7:
                        encoding = detected['encoding']
                except ImportError:
                    encoding = 'utf-8'
                    for test_encoding in ['utf-8', 'shift_jis', 'euc-jp', 'iso-2022-jp', 'cp932']:
                        try:
                            test_decode = content_bytes.decode(test_encoding, errors='strict')
                            if '<html' in test_decode.lower() or '<!doctype' in test_decode.lower():
                                encoding = test_encoding
                                break
                        except (UnicodeDecodeError, LookupError):
                            continue
            
            # Decode with proper encoding
            try:
                html_content = content_bytes.decode(encoding, errors='replace')
                if not ('<html' in html_content.lower() or '<!doctype' in html_content.lower()):
                    html_content = content_bytes.decode('utf-8', errors='replace')
            except (UnicodeDecodeError, LookupError):
                html_content = content_bytes.decode('utf-8', errors='replace')
            
            html_content = unicodedata.normalize('NFC', html_content)
            
            # Validate HTML content
            if not html_content or len(html_content.strip()) < 100:
                logger.error(f"Detail page HTML content is too short or empty for {detail_url}")
                return False
            
            html_lower = html_content.lower()
            if not ('<html' in html_lower or '<!doctype' in html_lower or '<body' in html_lower):
                html_start = html_content.find('<html')
                if html_start == -1:
                    html_start = html_content.find('<!DOCTYPE')
                if html_start == -1:
                    html_start = html_content.find('<body')
                if html_start > 0:
                    html_content = html_content[html_start:]
                else:
                    logger.error(f"Could not find HTML start markers in detail page for {detail_url}")
                    return False
            
            # Parse HTML
            try:
                soup = BeautifulSoup(html_content, 'html.parser')
                if not soup.find('html') and not soup.find('body') and len(soup.find_all()) == 0:
                    logger.error(f"BeautifulSoup failed to parse detail page HTML for {detail_url}")
                    return False
            except Exception as e:
                logger.error(f"BeautifulSoup parsing error for detail page {detail_url}: {e}")
                return False
            
            base_url = f"{urlparse(detail_url).scheme}://{urlparse(detail_url).netloc}"
            
            # Create assets directory
            assets_dir = project_dir / "assets"
            assets_dir.mkdir(exist_ok=True)
            
            # Ensure proper HTML structure with charset meta tag
            head = soup.find('head')
            if not head:
                html_tag = soup.find('html')
                if not html_tag:
                    html_tag = soup.new_tag('html')
                    soup.insert(0, html_tag)
                head = soup.new_tag('head')
                html_tag.insert(0, head)
            
            # Ensure charset meta tag
            charset_meta = soup.find('meta', attrs={'charset': True})
            if charset_meta:
                charset_meta['charset'] = 'utf-8'
            else:
                http_equiv_meta = soup.find('meta', attrs={'http-equiv': re.compile(r'content-type', re.I)})
                if http_equiv_meta:
                    http_equiv_meta.decompose()
                charset_meta = soup.new_tag('meta', charset='utf-8')
                head.insert(0, charset_meta)
            
            # Extract and save inline CSS
            inline_css = extract_inline_css(str(soup))
            if inline_css.strip():
                inline_css_path = assets_dir / "inline_styles_info.css"
                inline_css = unicodedata.normalize('NFC', inline_css)
                with open(inline_css_path, 'w', encoding='utf-8', errors='replace') as f:
                    f.write(inline_css)
            
            # Download CSS files
            css_count = 0
            for link in soup.find_all('link', rel='stylesheet'):
                href = link.get('href', '')
                if not href or href.startswith('data:') or href.startswith('#'):
                    continue
                
                css_url = urljoin(base_url, href)
                parsed_css_url = urlparse(css_url)
                
                # Skip external CDN links
                if any(cdn in css_url.lower() for cdn in ['cdnjs', 'cdn.jsdelivr', 'googleapis', 'bootstrapcdn', 'fonts.googleapis', 'cloudflare']):
                    link.decompose()
                    continue
                
                css_path_part = parsed_css_url.path.lstrip('/')
                if css_path_part:
                    css_filename = os.path.basename(css_path_part)
                    if not css_filename or '.' not in css_filename:
                        css_filename = f"style_info_{css_count}.css"
                else:
                    css_filename = f"style_info_{css_count}.css"
                
                if '?' in css_filename:
                    css_filename = css_filename.split('?')[0]
                if not css_filename.endswith('.css'):
                    css_filename += '.css'
                
                original_filename = css_filename
                counter = 1
                while (assets_dir / css_filename).exists():
                    name, ext = os.path.splitext(original_filename)
                    css_filename = f"{name}_{counter}{ext}"
                    counter += 1
                
                css_path = assets_dir / css_filename
                if download_file(session, css_url, css_path, is_text=True):
                    link['href'] = f"assets/{css_filename}"
                    css_count += 1
                else:
                    link.decompose()
            
            # Extract and save inline JavaScript
            inline_js = extract_inline_js(str(soup))
            if inline_js.strip():
                inline_js_path = assets_dir / "inline_scripts_info.js"
                inline_js = unicodedata.normalize('NFC', inline_js)
                with open(inline_js_path, 'w', encoding='utf-8', errors='replace') as f:
                    f.write(inline_js)
            
            # Download JavaScript files
            js_count = 0
            for script in soup.find_all('script', src=True):
                src = script.get('src', '')
                if not src or src.startswith('data:'):
                    continue
                
                js_url = urljoin(base_url, src)
                parsed_js_url = urlparse(js_url)
                
                # Skip external CDN links
                if any(cdn in js_url.lower() for cdn in ['cdnjs', 'cdn.jsdelivr', 'googleapis', 'googletagmanager', 'gtag', 'google-analytics', 'analytics', 'cloudflare', 'ajax.googleapis']):
                    script.decompose()
                    continue
                
                js_path_part = parsed_js_url.path.lstrip('/')
                if js_path_part:
                    js_filename = os.path.basename(js_path_part)
                    if not js_filename or '.' not in js_filename:
                        js_filename = f"script_info_{js_count}.js"
                else:
                    js_filename = f"script_info_{js_count}.js"
                
                if '?' in js_filename:
                    js_filename = js_filename.split('?')[0]
                if not js_filename.endswith('.js'):
                    js_filename += '.js'
                
                original_filename = js_filename
                counter = 1
                while (assets_dir / js_filename).exists():
                    name, ext = os.path.splitext(original_filename)
                    js_filename = f"{name}_{counter}{ext}"
                    counter += 1
                
                js_path = assets_dir / js_filename
                if download_file(session, js_url, js_path, is_text=True):
                    script['src'] = f"assets/{js_filename}"
                    js_count += 1
                else:
                    script.decompose()
            
            # Download images
            images_dir = assets_dir / "images"
            images_dir.mkdir(exist_ok=True)
            img_count = 0
            
            for img in soup.find_all('img', src=True):
                src = img.get('src', '')
                if not src or src.startswith('data:') or src.startswith('#'):
                    continue
                
                img_url = urljoin(base_url, src)
                parsed_img_url = urlparse(img_url)
                
                # Only download images from the same domain (or relative paths)
                if not (parsed_img_url.netloc == urlparse(base_url).netloc or not parsed_img_url.netloc):
                    continue
                
                img_path_part = parsed_img_url.path.lstrip('/')
                if img_path_part:
                    img_filename = os.path.basename(img_path_part)
                    if not img_filename or '.' not in img_filename:
                        img_filename = f"image_info_{img_count}.jpg"
                else:
                    img_filename = f"image_info_{img_count}.jpg"
                
                if '?' in img_filename:
                    img_filename = img_filename.split('?')[0]
                
                if not any(img_filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg', '.ico']):
                    ext_match = re.search(r'\.(jpg|jpeg|png|gif|webp|svg|ico)', parsed_img_url.path.lower())
                    if ext_match:
                        img_filename += ext_match.group(0)
                    else:
                        img_filename += '.jpg'
                
                original_filename = img_filename
                counter = 1
                while (images_dir / img_filename).exists():
                    name, ext = os.path.splitext(original_filename)
                    img_filename = f"{name}_{counter}{ext}"
                    counter += 1
                
                img_path = images_dir / img_filename
                if download_image(session, img_url, img_path):
                    img['src'] = f"assets/images/{img_filename}"
                    img_count += 1
                
                # Handle srcset
                if img.get('srcset'):
                    srcset = img.get('srcset', '')
                    srcset_parts = []
                    for part in srcset.split(','):
                        part = part.strip()
                        if not part:
                            continue
                        url_desc = part.split(None, 1)
                        if len(url_desc) == 2:
                            srcset_url, descriptor = url_desc
                        else:
                            srcset_url = url_desc[0]
                            descriptor = ''
                        
                        srcset_abs_url = urljoin(base_url, srcset_url)
                        parsed_srcset_url = urlparse(srcset_abs_url)
                        
                        if parsed_srcset_url.netloc == urlparse(base_url).netloc or not parsed_srcset_url.netloc:
                            srcset_filename = os.path.basename(parsed_srcset_url.path) or f"image_info_{img_count}.jpg"
                            if '?' in srcset_filename:
                                srcset_filename = srcset_filename.split('?')[0]
                            if not any(srcset_filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg']):
                                srcset_filename += '.jpg'
                            
                            srcset_path = images_dir / srcset_filename
                            if download_image(session, srcset_abs_url, srcset_path):
                                srcset_parts.append(f"assets/images/{srcset_filename} {descriptor}")
                                img_count += 1
                            else:
                                srcset_parts.append(part)
                        else:
                            srcset_parts.append(part)
                    
                    if srcset_parts:
                        img['srcset'] = ', '.join(srcset_parts)
            
            # Handle background images in inline styles
            for element in soup.find_all(style=True):
                style_attr = element.get('style', '')
                if 'background-image' in style_attr or 'background:' in style_attr:
                    url_match = re.search(r'url\(["\']?([^"\')]+)["\']?\)', style_attr)
                    if url_match:
                        bg_url = url_match.group(1)
                        if bg_url.startswith('http') or bg_url.startswith('//'):
                            bg_img_url = urljoin(base_url, bg_url)
                            parsed_bg_url = urlparse(bg_img_url)
                            
                            if parsed_bg_url.netloc == urlparse(base_url).netloc:
                                bg_filename = os.path.basename(parsed_bg_url.path) or f"bg_info_{img_count}.jpg"
                                if '?' in bg_filename:
                                    bg_filename = bg_filename.split('?')[0]
                                if not any(bg_filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg']):
                                    bg_filename += '.jpg'
                                
                                bg_path = images_dir / bg_filename
                                if download_image(session, bg_img_url, bg_path):
                                    new_style = style_attr.replace(bg_url, f"assets/images/{bg_filename}")
                                    element['style'] = new_style
                                    img_count += 1
            
            # Remove comment nodes
            from bs4 import Comment
            for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
                comment_text = str(comment).strip()
                if not comment_text or comment_text in ['', ' ', '\n', '\t']:
                    comment.extract()
            
            # Save processed HTML
            info_path = project_dir / "info.html"
            updated_html = str(soup)
            updated_html = unicodedata.normalize('NFC', updated_html)
            
            if not updated_html.strip().startswith('<!DOCTYPE'):
                updated_html = '<!DOCTYPE html>\n' + updated_html
            
            charset_pattern = r'<meta[^>]*charset\s*=\s*["\']?utf-8["\']?[^>]*>'
            has_charset = re.search(charset_pattern, updated_html, re.IGNORECASE)
            if not has_charset:
                updated_html = re.sub(
                    r'(<head[^>]*>)',
                    r'\1\n    <meta charset="utf-8">',
                    updated_html,
                    count=1,
                    flags=re.IGNORECASE
                )
            
            # Clean up comment artifacts
            updated_html = re.sub(r'<!--\s*-->', '', updated_html)
            updated_html = re.sub(r'<!--\s*<!--[^>]*-->', '', updated_html)
            updated_html = re.sub(r'<!--[^>]*-->\s*<!--', '', updated_html)
            updated_html = re.sub(r'<!--\s+-->', '', updated_html)
            updated_html = re.sub(r'<!--[^>]*$', '', updated_html, flags=re.MULTILINE)
            updated_html = re.sub(r'^[^<]*-->', '', updated_html, flags=re.MULTILINE)
            
            with open(info_path, 'w', encoding='utf-8', errors='replace', newline='') as f:
                f.write(updated_html)
            
            logger.info(f"Downloaded detail page with {css_count} CSS files, {js_count} JS files, and {img_count} images")
            return True
            
        except requests.RequestException as e:
            logger.warning(f"Error fetching detail page (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(REQUEST_INTERVAL)
            else:
                logger.error(f"Failed to fetch detail page after {MAX_RETRIES} attempts: {detail_url}")
                return False
        except Exception as e:
            logger.error(f"Error processing detail page for {detail_url}: {e}", exc_info=True)
            return False
    
    return False


def extract_css_links(html: str, base_url: str) -> List[str]:
    """Extract CSS file URLs from HTML."""
    soup = BeautifulSoup(html, 'html.parser')
    css_links = []
    
    # Find all link tags with rel="stylesheet"
    for link in soup.find_all('link', rel='stylesheet'):
        href = link.get('href')
        if href:
            css_links.append(urljoin(base_url, href))
    
    return css_links


def extract_inline_css(html: str) -> str:
    """Extract inline CSS from style tags."""
    soup = BeautifulSoup(html, 'html.parser')
    inline_css = []
    
    for style in soup.find_all('style'):
        if style.string:
            # Get text and normalize encoding
            css_text = str(style.string)
            inline_css.append(css_text)
    
    result = '\n\n'.join(inline_css)
    # Normalize unicode
    result = unicodedata.normalize('NFC', result)
    return result


def extract_js_links(html: str, base_url: str) -> List[str]:
    """Extract JavaScript file URLs from HTML."""
    soup = BeautifulSoup(html, 'html.parser')
    js_links = []
    
    # Find all script tags with src attribute
    for script in soup.find_all('script', src=True):
        src = script.get('src')
        if src:
            js_links.append(urljoin(base_url, src))
    
    return js_links


def extract_inline_js(html: str) -> str:
    """Extract inline JavaScript from script tags."""
    soup = BeautifulSoup(html, 'html.parser')
    inline_js = []
    
    for script in soup.find_all('script', src=False):
        if script.string:
            # Get text and normalize encoding
            js_text = str(script.string)
            inline_js.append(js_text)
    
    result = '\n\n'.join(inline_js)
    # Normalize unicode
    result = unicodedata.normalize('NFC', result)
    return result


def download_file(session: requests.Session, url: str, save_path: Path, is_text: bool = False) -> bool:
    """Download a file and save it. If is_text=True, handle encoding properly."""
    try:
        # For text files, don't use stream=True so we can access response.content for encoding detection
        response = session.get(url, timeout=TIMEOUT, stream=not is_text)
        response.raise_for_status()
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        if is_text:
            # For text files (CSS, JS), handle encoding properly
            # Read full content - response.content works when stream=False (default)
            logger.debug(f"[HTTP] GET {url} -> {response.status_code} (text)")
            logger.debug(f"[HTTP] Headers: Content-Type={response.headers.get('Content-Type')}, Content-Encoding={response.headers.get('Content-Encoding')}")
            content_bytes = _get_decoded_bytes(response)
            logger.debug(f"[HTTP] Decoded-bytes length: {len(content_bytes)}")
            
            # Try to detect encoding
            encoding = response.encoding
            if not encoding or encoding.lower() == 'iso-8859-1':
                try:
                    import chardet
                    detected = chardet.detect(content_bytes)
                    if detected and detected.get('encoding') and detected['confidence'] > 0.7:
                        encoding = detected['encoding']
                    else:
                        encoding = 'utf-8'
                except ImportError:
                    # Try common encodings for Japanese sites
                    encoding = 'utf-8'
                    for test_encoding in ['utf-8', 'shift_jis', 'euc-jp', 'cp932']:
                        try:
                            test_decode = content_bytes.decode(test_encoding, errors='strict')
                            encoding = test_encoding
                            break
                        except (UnicodeDecodeError, LookupError):
                            continue
            
            # Decode with proper encoding
            try:
                content = content_bytes.decode(encoding, errors='replace')
                # Normalize unicode to handle any encoding artifacts
                content = unicodedata.normalize('NFC', content)
            except (UnicodeDecodeError, LookupError) as e:
                logger.debug(f"Encoding error with {encoding} for {url}, using UTF-8: {e}")
                content = content_bytes.decode('utf-8', errors='replace')
                content = unicodedata.normalize('NFC', content)
            
            # Save as text with UTF-8 encoding
            with open(save_path, 'w', encoding='utf-8', errors='replace', newline='') as f:
                f.write(content)
        else:
            # For binary files (images), save as binary
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        return True
    except requests.RequestException as e:
        logger.warning(f"Error downloading {url}: {e}")
        return False
    except Exception as e:
        logger.warning(f"Error saving {url} to {save_path}: {e}")
        return False


def download_image(session: requests.Session, url: str, save_path: Path) -> bool:
    """Download an image file and save it."""
    try:
        response = session.get(url, timeout=TIMEOUT, stream=True)
        response.raise_for_status()
        
        # Check if it's actually an image
        content_type = response.headers.get('content-type', '').lower()
        if content_type and not content_type.startswith('image/'):
            logger.debug(f"Skipping non-image content: {url} (content-type: {content_type})")
            return False
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # Filter out keep-alive chunks
                    f.write(chunk)
        return True
    except requests.RequestException as e:
        logger.debug(f"Error downloading image {url}: {e}")
        return False
    except Exception as e:
        logger.debug(f"Error saving image {url}: {e}")
        return False


def fetch_website_content(session: requests.Session, site_url: str, project_dir: Path) -> Dict[str, bool]:
    """Fetch website HTML, CSS, and JavaScript files and update HTML paths."""
    results = {
        'html': False,
        'css': False,
        'js': False
    }
    
    try:
        # Fetch main HTML with proper encoding handling
        try:
            response = session.get(site_url, timeout=TIMEOUT, stream=False)
            response.raise_for_status()
            logger.info(f"[HTTP] GET {site_url} -> {response.status_code}")
            logger.info(f"[HTTP] Headers: Content-Type={response.headers.get('Content-Type')}, Content-Encoding={response.headers.get('Content-Encoding')}, Transfer-Encoding={response.headers.get('Transfer-Encoding')}, Vary={response.headers.get('Vary')}")
            logger.info(f"[HTTP] Raw content length: {len(response.content)} bytes")
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error fetching {site_url}: {e}")
            return results
        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout fetching {site_url} (timeout={TIMEOUT}s): {e}")
            return results
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error fetching {site_url}: {e}")
            return results
        
        # First, try to extract charset from HTML meta tags (most reliable)
        # We'll decode a small portion first to check for charset declaration
        html_content = None
        detected_encoding = None
        content_bytes = _get_decoded_bytes(response)
        logger.info(f"[HTTP] Decoded-bytes length (after Content-Encoding handling): {len(content_bytes)} bytes")
        
        # Try to detect charset from HTML content itself
        # Look for charset in the first 8KB of content (where meta tags usually are)
        preview_bytes = content_bytes[:8192]
        
        # Try common encodings to read the charset meta tag
        for test_encoding in ['utf-8', 'shift_jis', 'euc-jp', 'cp932', 'iso-2022-jp']:
            try:
                preview_text = preview_bytes.decode(test_encoding, errors='ignore')
                # Look for charset declaration in meta tags
                charset_match = re.search(
                    r'<meta[^>]*charset\s*=\s*["\']?([^"\'>\s]+)["\']?',
                    preview_text,
                    re.IGNORECASE
                )
                if charset_match:
                    detected_encoding = charset_match.group(1).lower()
                    # Normalize encoding names
                    encoding_map = {
                        'utf-8': 'utf-8',
                        'utf8': 'utf-8',
                        'shift_jis': 'shift_jis',
                        'shift-jis': 'shift_jis',
                        'sjis': 'shift_jis',
                        'euc-jp': 'euc-jp',
                        'eucjp': 'euc-jp',
                        'iso-2022-jp': 'iso-2022-jp',
                        'cp932': 'cp932',
                    }
                    detected_encoding = encoding_map.get(detected_encoding, detected_encoding)
                    logger.debug(f"Found charset in HTML meta tag: {detected_encoding}")
                    break
            except (UnicodeDecodeError, LookupError):
                continue
        
        # If charset found in HTML, use it; otherwise try response encoding
        encoding = detected_encoding or response.encoding
        
        if not encoding or encoding.lower() == 'iso-8859-1':
            # Try to detect from content using chardet
            try:
                import chardet
                detected = chardet.detect(content_bytes)
                if detected and detected.get('encoding') and detected['confidence'] > 0.7:
                    encoding = detected['encoding']
                    logger.debug(f"Detected encoding using chardet: {encoding}")
                else:
                    encoding = 'utf-8'
            except ImportError:
                # Fallback: try common encodings by actually decoding
                encoding = 'utf-8'
                for test_encoding in ['utf-8', 'shift_jis', 'euc-jp', 'iso-2022-jp', 'cp932']:
                    try:
                        test_decode = content_bytes.decode(test_encoding, errors='strict')
                        # Validate it's actually HTML (contains <html or <!DOCTYPE)
                        if '<html' in test_decode.lower() or '<!doctype' in test_decode.lower():
                            encoding = test_encoding
                            logger.debug(f"Validated encoding by decoding: {encoding}")
                            break
                    except (UnicodeDecodeError, LookupError):
                        continue
        
        # Simplified: Just try UTF-8 first (most common), then BeautifulSoup validation
        # This is much more reliable than complex encoding detection
        html_content = None
        tried_encodings = []
        
        # Try UTF-8 first (99% of modern websites use UTF-8)
        logger.info(f"[ENCODING] Attempting UTF-8 decode for {site_url}")
        logger.info(f"[ENCODING] Response content length: {len(content_bytes)} bytes")
        try:
            test_content = content_bytes.decode('utf-8', errors='replace')
            logger.info(f"[ENCODING] UTF-8 decode successful, content length: {len(test_content)} chars")
            logger.info(f"[ENCODING] First 300 chars: {repr(test_content[:300])}")
            
            # Use BeautifulSoup to validate - if it can parse and find tags, it's valid HTML
            try:
                logger.info(f"[PARSING] Attempting BeautifulSoup parse...")
                test_soup = BeautifulSoup(test_content, 'html.parser')
                all_tags = test_soup.find_all()
                logger.info(f"[PARSING] BeautifulSoup found {len(all_tags)} tags")
                
                if len(all_tags) > 0:
                    logger.info(f"[PARSING] First 10 tag names: {[tag.name for tag in all_tags[:10]]}")
                    html_content = test_content
                    encoding = 'utf-8'
                    logger.info(f"[SUCCESS] Successfully decoded with UTF-8 (found {len(all_tags)} tags)")
                else:
                    tried_encodings.append('utf-8 (BeautifulSoup found 0 tags)')
                    logger.warning(f"[PARSING] BeautifulSoup found 0 tags - validation failed")
            except Exception as e:
                # BeautifulSoup failed, but try the content anyway
                logger.warning(f"[PARSING] BeautifulSoup parsing exception: {type(e).__name__}: {e}")
                logger.warning(f"[PARSING] Using content anyway despite parsing error")
                html_content = test_content
                encoding = 'utf-8'
        except (UnicodeDecodeError, LookupError) as e:
            tried_encodings.append(f"utf-8 (decode error: {e})")
            logger.error(f"[ENCODING] UTF-8 decode failed: {type(e).__name__}: {e}")
        
        # If UTF-8 failed, try detected encoding (if different) and Japanese encodings
        if not html_content:
            logger.warning(f"[ENCODING] UTF-8 failed, trying fallback encodings...")
            encodings_to_try = []
            if encoding and encoding.lower() not in ['utf-8', 'iso-8859-1', 'latin1']:
                encodings_to_try.append(encoding)
            encodings_to_try.extend(['shift_jis', 'euc-jp', 'cp932', 'iso-2022-jp'])
            logger.info(f"[ENCODING] Fallback encodings to try: {encodings_to_try}")
            
            for test_encoding in encodings_to_try:
                if html_content:
                    break
                logger.info(f"[ENCODING] Trying {test_encoding}...")
                try:
                    test_content = content_bytes.decode(test_encoding, errors='replace')
                    logger.info(f"[ENCODING] {test_encoding} decode successful, content length: {len(test_content)} chars")
                    
                    # Use BeautifulSoup to validate
                    try:
                        logger.info(f"[PARSING] Attempting BeautifulSoup parse with {test_encoding}...")
                        test_soup = BeautifulSoup(test_content, 'html.parser')
                        all_tags = test_soup.find_all()
                        logger.info(f"[PARSING] BeautifulSoup found {len(all_tags)} tags")
                        
                        if len(all_tags) > 0:
                            html_content = test_content
                            encoding = test_encoding
                            logger.info(f"[SUCCESS] Successfully decoded with {test_encoding} (found {len(all_tags)} tags)")
                            break
                        else:
                            tried_encodings.append(f"{test_encoding} (BeautifulSoup found 0 tags)")
                            logger.warning(f"[PARSING] {test_encoding} found 0 tags - validation failed")
                    except Exception as e:
                        # BeautifulSoup failed, but try the content anyway
                        logger.warning(f"[PARSING] BeautifulSoup parsing exception for {test_encoding}: {type(e).__name__}: {e}")
                        logger.warning(f"[PARSING] Using content anyway despite parsing error")
                        html_content = test_content
                        encoding = test_encoding
                        break
                except (UnicodeDecodeError, LookupError) as e:
                    tried_encodings.append(f"{test_encoding} (decode error: {e})")
                    logger.warning(f"[ENCODING] {test_encoding} decode failed: {type(e).__name__}: {e}")
        
        # Final fallback: use UTF-8 anyway
        if not html_content:
            logger.error(f"[ENCODING] All encoding attempts failed validation for {site_url}")
            logger.error(f"[ENCODING] Tried: {', '.join(tried_encodings)}")
            logger.warning(f"[ENCODING] Using UTF-8 as final fallback...")
            html_content = response.content.decode('utf-8', errors='replace')
            encoding = 'utf-8'
            logger.info(f"[ENCODING] Final fallback content length: {len(html_content)} chars")
        
        logger.info(f"[VALIDATION] Final encoding selected: {encoding}")
        logger.info(f"[VALIDATION] HTML content length before normalization: {len(html_content)} chars")
        
        # Normalize unicode and strip whitespace
        if html_content:
            html_content = unicodedata.normalize('NFC', html_content)
            html_content = html_content.strip()
            logger.info(f"[VALIDATION] HTML content length after normalization: {len(html_content)} chars")
        
        # Validate HTML content length (basic check)
        if not html_content or len(html_content) < 100:
            logger.error(f"[VALIDATION] HTML content is too short or empty for {site_url} (length: {len(html_content) if html_content else 0})")
            return results
        
        logger.info(f"[VALIDATION] HTML content length check passed: {len(html_content)} chars")
        
        # Parse HTML to find CSS and JS files BEFORE downloading
        # Use html.parser which handles encoding better than lxml
        logger.info(f"[PARSING] Starting final BeautifulSoup parse for asset extraction...")
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            logger.info(f"[PARSING] BeautifulSoup parse completed")
            
            # Validate that soup parsed correctly (has some content)
            html_tag = soup.find('html')
            body_tag = soup.find('body')
            all_tags = soup.find_all()
            
            logger.info(f"[PARSING] Parse results:")
            logger.info(f"[PARSING]   - HTML tag found: {html_tag is not None}")
            logger.info(f"[PARSING]   - Body tag found: {body_tag is not None}")
            logger.info(f"[PARSING]   - Total tags found: {len(all_tags)}")
            if len(all_tags) > 0:
                logger.info(f"[PARSING]   - First 15 tag names: {[tag.name for tag in all_tags[:15]]}")
            
            # More lenient validation - accept if we have any tags, even without html/body
            if len(all_tags) == 0:
                logger.error(f"[PARSING] BeautifulSoup failed to parse HTML for {site_url} - no elements found at all")
                logger.error(f"[PARSING] Content preview: {html_content[:500] if html_content else 'None'}")
                return results
            elif not html_tag and not body_tag:
                # If we have tags but no html/body, log warning but continue (might be a fragment)
                logger.warning(f"[PARSING] No <html> or <body> tags found for {site_url}, but found {len(all_tags)} other tags. Continuing anyway.")
                logger.info(f"[PARSING] First few tags: {[tag.name for tag in all_tags[:10]]}")
            
            logger.info(f"[PARSING] Validation passed, proceeding with asset extraction...")
        except Exception as e:
            logger.error(f"[PARSING] BeautifulSoup parsing exception: {type(e).__name__}: {e}")
            logger.error(f"[PARSING] Content preview: {html_content[:500] if html_content else 'None'}")
            return results
        
        # Set encoding explicitly
        if soup.original_encoding:
            soup.original_encoding = 'utf-8'
        base_url = f"{urlparse(site_url).scheme}://{urlparse(site_url).netloc}"
        base_parsed = urlparse(site_url)
        base_domain = base_parsed.netloc.lower()
        # Normalize domain (remove www. prefix for comparison)
        if base_domain.startswith('www.'):
            base_domain_no_www = base_domain[4:]
        else:
            base_domain_no_www = base_domain
        logger.info(f"[DOMAIN] Base domain: {base_domain}, normalized: {base_domain_no_www}")
        
        # Create assets directory
        assets_dir = project_dir / "assets"
        assets_dir.mkdir(exist_ok=True)
        
        # Track downloaded files: {original_url: local_filename}
        downloaded_css = {}
        downloaded_js = {}
        
        # Extract and save inline CSS (extract from soup to ensure proper encoding)
        inline_css = extract_inline_css(str(soup))
        if inline_css.strip():
            inline_css_path = assets_dir / "inline_styles.css"
            # Normalize encoding before saving
            inline_css = unicodedata.normalize('NFC', inline_css)
            with open(inline_css_path, 'w', encoding='utf-8', errors='replace') as f:
                f.write(inline_css)
            results['css'] = True
            logger.debug("Saved inline CSS")
        
        # Download CSS files and track them (with parallel downloads)
        css_count = 0
        css_downloads = []
        for link in soup.find_all('link', rel='stylesheet'):
            href = link.get('href', '')
            if not href:
                continue
            
            # Skip data URIs and inline styles
            if href.startswith('data:') or href.startswith('#'):
                continue
            
            # Convert to absolute URL for downloading
            css_url = urljoin(base_url, href)
            parsed_css_url = urlparse(css_url)
            
            # Skip external CDN links - just remove them (don't comment)
            if any(cdn in css_url.lower() for cdn in ['cdnjs', 'cdn.jsdelivr', 'googleapis', 'bootstrapcdn', 'fonts.googleapis', 'cloudflare']):
                # Remove external CDN links
                link.decompose()
                continue
            
            # Generate local filename from URL path
            css_path_part = parsed_css_url.path.lstrip('/')
            if css_path_part:
                css_filename = os.path.basename(css_path_part)
                # Handle cases where path doesn't have a filename
                if not css_filename or '.' not in css_filename:
                    css_filename = f"style_{css_count}.css"
            else:
                css_filename = f"style_{css_count}.css"
            
            # Remove query parameters from filename
            if '?' in css_filename:
                css_filename = css_filename.split('?')[0]
            
            # Ensure .css extension
            if not css_filename.endswith('.css'):
                css_filename += '.css'
            
            # Avoid filename conflicts
            original_filename = css_filename
            counter = 1
            while (assets_dir / css_filename).exists() and css_url not in downloaded_css:
                name, ext = os.path.splitext(original_filename)
                css_filename = f"{name}_{counter}{ext}"
                counter += 1
            
            css_path = assets_dir / css_filename
            css_downloads.append((css_url, css_path, css_filename, link))
        
        # Download CSS files in parallel (up to 5 concurrent)
        if css_downloads:
            with ThreadPoolExecutor(max_workers=min(5, len(css_downloads))) as executor:
                css_futures = {executor.submit(download_file, session, url, path, True): (url, filename, link_elem) 
                              for url, path, filename, link_elem in css_downloads}
                
                for future in as_completed(css_futures):
                    url, filename, link_elem = css_futures[future]
                    try:
                        if future.result():
                            downloaded_css[url] = filename
                            link_elem['href'] = f"assets/{filename}"
                            css_count += 1
                            results['css'] = True
                        else:
                            link_elem.decompose()
                    except Exception as e:
                        logger.warning(f"Error downloading CSS {url}: {e}")
                        link_elem.decompose()
        
        # Extract and save inline JavaScript (extract from soup to ensure proper encoding)
        inline_js = extract_inline_js(str(soup))
        if inline_js.strip():
            inline_js_path = assets_dir / "inline_scripts.js"
            # Normalize encoding before saving
            inline_js = unicodedata.normalize('NFC', inline_js)
            with open(inline_js_path, 'w', encoding='utf-8', errors='replace') as f:
                f.write(inline_js)
            results['js'] = True
            logger.debug("Saved inline JavaScript")
        
        # Download JavaScript files and track them (with parallel downloads)
        js_count = 0
        js_downloads = []
        for script in soup.find_all('script', src=True):
            src = script.get('src', '')
            if not src:
                continue
            
            # Skip data URIs
            if src.startswith('data:'):
                continue
            
            # Convert to absolute URL for downloading
            js_url = urljoin(base_url, src)
            parsed_js_url = urlparse(js_url)
            
            # Skip external CDN links and tracking scripts - just remove them (don't comment)
            if any(cdn in js_url.lower() for cdn in ['cdnjs', 'cdn.jsdelivr', 'googleapis', 'googletagmanager', 'gtag', 'google-analytics', 'analytics', 'cloudflare', 'ajax.googleapis']):
                # Remove external CDN links
                script.decompose()
                continue
            
            # Generate local filename from URL path
            js_path_part = parsed_js_url.path.lstrip('/')
            if js_path_part:
                js_filename = os.path.basename(js_path_part)
                # Handle cases where path doesn't have a filename
                if not js_filename or '.' not in js_filename:
                    js_filename = f"script_{js_count}.js"
            else:
                js_filename = f"script_{js_count}.js"
            
            # Remove query parameters from filename
            if '?' in js_filename:
                js_filename = js_filename.split('?')[0]
            
            # Ensure .js extension
            if not js_filename.endswith('.js'):
                js_filename += '.js'
            
            # Avoid filename conflicts
            original_filename = js_filename
            counter = 1
            while (assets_dir / js_filename).exists() and js_url not in downloaded_js:
                name, ext = os.path.splitext(original_filename)
                js_filename = f"{name}_{counter}{ext}"
                counter += 1
            
            js_path = assets_dir / js_filename
            js_downloads.append((js_url, js_path, js_filename, script))
        
        # Download JS files in parallel (up to 5 concurrent)
        if js_downloads:
            with ThreadPoolExecutor(max_workers=min(5, len(js_downloads))) as executor:
                js_futures = {executor.submit(download_file, session, url, path, True): (url, filename, script_elem) 
                             for url, path, filename, script_elem in js_downloads}
                
                for future in as_completed(js_futures):
                    url, filename, script_elem = js_futures[future]
                    try:
                        if future.result():
                            downloaded_js[url] = filename
                            script_elem['src'] = f"assets/{filename}"
                            js_count += 1
                            results['js'] = True
                        else:
                            script_elem.decompose()
                    except Exception as e:
                        logger.warning(f"Error downloading JS {url}: {e}")
                        script_elem.decompose()
        
        # Download images and update paths (with parallel downloads)
        images_dir = assets_dir / "images"
        images_dir.mkdir(exist_ok=True)
        downloaded_images = {}
        img_count = 0
        img_downloads = []
        
        for img in soup.find_all('img', src=True):
            src = img.get('src', '')
            if not src:
                continue
            
            # Skip data URIs and placeholder images
            if src.startswith('data:') or src.startswith('#'):
                continue
            
            # Convert to absolute URL for downloading
            img_url = urljoin(base_url, src)
            parsed_img_url = urlparse(img_url)
            img_domain = parsed_img_url.netloc.lower() if parsed_img_url.netloc else ''
            
            # Normalize image domain (remove www. prefix for comparison)
            if img_domain.startswith('www.'):
                img_domain_no_www = img_domain[4:]
            else:
                img_domain_no_www = img_domain
            
            # Only download images from the same domain (or relative paths)
            # Allow both www and non-www versions of the same domain
            is_same_domain = (
                not parsed_img_url.netloc or  # Relative path
                parsed_img_url.netloc.lower() == base_domain or  # Exact match
                img_domain_no_www == base_domain_no_www  # Same domain (www vs non-www)
            )
            
            if not is_same_domain:
                logger.debug(f"[IMAGE] Skipping external image: {img_url} (domain: {img_domain}, base: {base_domain})")
                continue
            
            logger.debug(f"[IMAGE] Processing same-domain image: {img_url}")
            
            # Generate local filename
            img_path_part = parsed_img_url.path.lstrip('/')
            if img_path_part:
                img_filename = os.path.basename(img_path_part)
                # Handle cases where path doesn't have a filename
                if not img_filename or '.' not in img_filename:
                    # Try to get extension from content-type or use default
                    img_filename = f"image_{img_count}.jpg"
            else:
                img_filename = f"image_{img_count}.jpg"
            
            # Remove query parameters from filename
            if '?' in img_filename:
                img_filename = img_filename.split('?')[0]
            
            # Ensure image extension
            if not any(img_filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg', '.ico']):
                # Try to preserve extension from URL or use jpg as default
                ext_match = re.search(r'\.(jpg|jpeg|png|gif|webp|svg|ico)', parsed_img_url.path.lower())
                if ext_match:
                    img_filename += ext_match.group(0)
                else:
                    img_filename += '.jpg'
            
            # Avoid filename conflicts
            original_filename = img_filename
            counter = 1
            while (images_dir / img_filename).exists() and img_url not in downloaded_images:
                name, ext = os.path.splitext(original_filename)
                img_filename = f"{name}_{counter}{ext}"
                counter += 1
            
            img_path = images_dir / img_filename
            img_downloads.append((img_url, img_path, img_filename, img))
        
        # Download images in parallel (up to 10 concurrent for images)
        if img_downloads:
            with ThreadPoolExecutor(max_workers=min(10, len(img_downloads))) as executor:
                img_futures = {executor.submit(download_image, session, url, path): (url, filename, img_elem) 
                              for url, path, filename, img_elem in img_downloads}
                
                for future in as_completed(img_futures):
                    url, filename, img_elem = img_futures[future]
                    try:
                        if future.result():
                            downloaded_images[url] = filename
                            img_elem['src'] = f"assets/images/{filename}"
                            img_count += 1
                    except Exception as e:
                        logger.debug(f"Error downloading image {url}: {e}")
        
        # Process srcset for successfully downloaded images
        for img in soup.find_all('img', src=True):
            # Also handle srcset if present (responsive images)
            if img.get('srcset'):
                    srcset = img.get('srcset', '')
                    # Parse srcset (format: "url1 1x, url2 2x" or "url1 100w, url2 200w")
                    srcset_parts = []
                    for part in srcset.split(','):
                        part = part.strip()
                        if not part:
                            continue
                        # Extract URL and descriptor
                        url_desc = part.split(None, 1)
                        if len(url_desc) == 2:
                            srcset_url, descriptor = url_desc
                        else:
                            srcset_url = url_desc[0]
                            descriptor = ''
                        
                        # Convert to absolute URL
                        srcset_abs_url = urljoin(base_url, srcset_url)
                        parsed_srcset_url = urlparse(srcset_abs_url)
                        srcset_domain = parsed_srcset_url.netloc.lower() if parsed_srcset_url.netloc else ''
                        
                        # Normalize srcset domain (remove www. prefix for comparison)
                        if srcset_domain.startswith('www.'):
                            srcset_domain_no_www = srcset_domain[4:]
                        else:
                            srcset_domain_no_www = srcset_domain
                        
                        # Only download from same domain (allow www vs non-www)
                        is_same_domain = (
                            not parsed_srcset_url.netloc or  # Relative path
                            parsed_srcset_url.netloc.lower() == base_domain or  # Exact match
                            srcset_domain_no_www == base_domain_no_www  # Same domain (www vs non-www)
                        )
                        
                        if is_same_domain:
                            # Check if we already downloaded this image
                            if srcset_abs_url in downloaded_images:
                                # Use already downloaded image
                                srcset_parts.append(f"assets/images/{downloaded_images[srcset_abs_url]} {descriptor}")
                            else:
                                # Try to download
                                srcset_filename = os.path.basename(parsed_srcset_url.path) or f"image_{img_count}.jpg"
                                if '?' in srcset_filename:
                                    srcset_filename = srcset_filename.split('?')[0]
                                if not any(srcset_filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg']):
                                    srcset_filename += '.jpg'
                                
                                srcset_path = images_dir / srcset_filename
                                if download_image(session, srcset_abs_url, srcset_path):
                                    downloaded_images[srcset_abs_url] = srcset_filename
                                    srcset_parts.append(f"assets/images/{srcset_filename} {descriptor}")
                                    img_count += 1
                                else:
                                    # Keep original if download fails
                                    srcset_parts.append(part)
                        else:
                            # External image, keep original
                            srcset_parts.append(part)
                    
                    if srcset_parts:
                        img['srcset'] = ', '.join(srcset_parts)
            else:
                # If download failed, try to keep relative path if it's already relative
                if not src.startswith('http'):
                    # Already relative, keep it
                    pass
                else:
                    # Failed download, remove or keep original
                    logger.debug(f"Could not download image: {img_url}")
        
        # Also handle background images in CSS and inline styles
        # This will be handled when we process CSS files, but we can also update style attributes
        for element in soup.find_all(style=True):
            style_attr = element.get('style', '')
            if 'background-image' in style_attr or 'background:' in style_attr:
                # Extract URL from style attribute
                url_match = re.search(r'url\(["\']?([^"\')]+)["\']?\)', style_attr)
                if url_match:
                    bg_url = url_match.group(1)
                    if bg_url.startswith('http') or bg_url.startswith('//'):
                        bg_img_url = urljoin(base_url, bg_url)
                        parsed_bg_url = urlparse(bg_img_url)
                        bg_domain = parsed_bg_url.netloc.lower() if parsed_bg_url.netloc else ''
                        
                        # Normalize background image domain (remove www. prefix for comparison)
                        if bg_domain.startswith('www.'):
                            bg_domain_no_www = bg_domain[4:]
                        else:
                            bg_domain_no_www = bg_domain
                        
                        # Only download from same domain (allow www vs non-www)
                        is_same_domain = (
                            parsed_bg_url.netloc.lower() == base_domain or  # Exact match
                            bg_domain_no_www == base_domain_no_www  # Same domain (www vs non-www)
                        )
                        
                        if is_same_domain:
                            # Generate filename similar to above
                            bg_filename = os.path.basename(parsed_bg_url.path) or f"bg_{img_count}.jpg"
                            if '?' in bg_filename:
                                bg_filename = bg_filename.split('?')[0]
                            if not any(bg_filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg']):
                                bg_filename += '.jpg'
                            
                            bg_path = images_dir / bg_filename
                            if download_image(session, bg_img_url, bg_path):
                                # Update style attribute
                                new_style = style_attr.replace(bg_url, f"assets/images/{bg_filename}")
                                element['style'] = new_style
                                img_count += 1
        
        # Update other asset paths (favicon, etc.)
        for link in soup.find_all('link', href=True):
            href = link.get('href', '')
            if href.startswith(base_url):
                relative_path = href.replace(base_url, '').lstrip('/')
                link['href'] = relative_path
        
        # Ensure proper HTML structure with charset meta tag
        # Get or create head tag
        head = soup.find('head')
        if not head:
            # If no head tag, create one
            html_tag = soup.find('html')
            if not html_tag:
                # If no html tag, create both
                html_tag = soup.new_tag('html')
                soup.insert(0, html_tag)
            head = soup.new_tag('head')
            html_tag.insert(0, head)
        
        # Ensure charset meta tag exists and is set to UTF-8
        charset_meta = soup.find('meta', attrs={'charset': True})
        if charset_meta:
            # Update existing charset meta tag
            charset_meta['charset'] = 'utf-8'
        else:
            # Check for old-style http-equiv meta tag
            http_equiv_meta = soup.find('meta', attrs={'http-equiv': re.compile(r'content-type', re.I)})
            if http_equiv_meta:
                # Remove old-style meta tag
                http_equiv_meta.decompose()
            
            # Create new charset meta tag as first element in head
            charset_meta = soup.new_tag('meta', charset='utf-8')
            head.insert(0, charset_meta)
        
        # Ensure DOCTYPE exists (check if HTML starts with DOCTYPE)
        # BeautifulSoup stores DOCTYPE separately, so we'll check when converting to string
        # For now, just ensure we have proper HTML5 DOCTYPE
        # The DOCTYPE will be automatically added by BeautifulSoup's str() method if missing
        
        # Save updated HTML with corrected paths
        html_path = project_dir / "index.html"
        
        # Remove any comments that might be causing issues before converting to string
        # Find and remove comment nodes that are empty or problematic
        from bs4 import Comment
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment_text = str(comment).strip()
            # Remove empty comments or comments that are just whitespace
            if not comment_text or comment_text in ['', ' ', '\n', '\t']:
                comment.extract()
        
        # Get HTML string - use decode with explicit UTF-8 encoding declaration
        # BeautifulSoup's str() handles encoding, but we want to ensure proper structure
        updated_html = str(soup)
        
        # Ensure DOCTYPE is present at the start
        if not updated_html.strip().startswith('<!DOCTYPE'):
            updated_html = '<!DOCTYPE html>\n' + updated_html
        
        # Normalize unicode to handle any encoding artifacts
        updated_html = unicodedata.normalize('NFC', updated_html)
        
        # Double-check that charset meta tag is present (in case BeautifulSoup didn't preserve it)
        # This regex looks for charset meta tag in various forms
        charset_pattern = r'<meta[^>]*charset\s*=\s*["\']?utf-8["\']?[^>]*>'
        has_charset = re.search(charset_pattern, updated_html, re.IGNORECASE)
        if not has_charset:
            # Insert charset meta tag right after <head> tag
            updated_html = re.sub(
                r'(<head[^>]*>)',
                r'\1\n    <meta charset="utf-8">',
                updated_html,
                count=1,
                flags=re.IGNORECASE
            )
        
        # Clean up comment artifacts - remove empty or malformed comments more aggressively
        # Remove standalone empty comments (any variation)
        updated_html = re.sub(r'<!--\s*-->', '', updated_html)
        # Remove double/malformed comments
        updated_html = re.sub(r'<!--\s*<!--[^>]*-->', '', updated_html)
        updated_html = re.sub(r'<!--[^>]*-->\s*<!--', '', updated_html)
        # Remove comments that only contain whitespace
        updated_html = re.sub(r'<!--\s+-->', '', updated_html)
        # Clean up any remaining malformed comment tags
        updated_html = re.sub(r'<!--[^>]*$', '', updated_html, flags=re.MULTILINE)
        updated_html = re.sub(r'^[^<]*-->', '', updated_html, flags=re.MULTILINE)
        
        with open(html_path, 'w', encoding='utf-8', errors='replace', newline='') as f:
            f.write(updated_html)
        results['html'] = True
        
        logger.info(f"Downloaded {css_count} CSS files, {js_count} JS files, and {img_count} images for {site_url}")
        
    except requests.RequestException as e:
        logger.error(f"Error fetching website content for {site_url}: {e}")
    except Exception as e:
        logger.error(f"Error processing HTML for {site_url}: {e}", exc_info=True)
    
    return results


def save_metadata(project_dir: Path, metadata: Dict):
    """Save metadata as JSON."""
    metadata_path = project_dir / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def process_site(session: requests.Session, site_info: Dict, force_redownload: bool = False) -> bool:
    """Process a single site: fetch detail page and website content."""
    detail_url = site_info['detail_url']
    site_url = site_info['site_url']
    title = site_info['title']
    
    # Create project directory
    project_name = sanitize_filename(site_url)
    project_dir = PROJECT_TEMPLATES_DIR / project_name
    project_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Processing: {title} ({site_url})")
    
    # Check if already processed (unless forcing redownload)
    if not force_redownload and (project_dir / "info.html").exists() and (project_dir / "index.html").exists():
        logger.info(f"Already processed, skipping: {project_name}")
        return True
    
    # Fetch and process detail page (downloads all assets)
    if process_detail_page(session, detail_url, project_dir):
        logger.info(f"Saved detail page with all assets: {detail_url}")
    else:
        logger.warning(f"Failed to fetch/process detail page: {detail_url}")
    
    # Wait before fetching website (only in sequential mode, parallel workers handle their own timing)
    # Note: When using parallel workers, each worker has its own session and timing
    # The sleep here is minimal since parallel workers naturally space out requests
    if not hasattr(process_site, '_parallel_mode'):
        time.sleep(REQUEST_INTERVAL * 0.5)  # Reduced wait time
    
    # Fetch website content
    results = fetch_website_content(session, site_url, project_dir)
    
    # Save metadata
    metadata = {
        'title': title,
        'detail_url': detail_url,
        'site_url': site_url,
        'credit_info': site_info.get('credit_info', {}),
        'fetched_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'download_results': results
    }
    save_metadata(project_dir, metadata)
    
    if results['html']:
        logger.info(f"Successfully processed: {title}")
        return True
    else:
        logger.warning(f"Failed to fetch website HTML: {site_url}")
        return False


def get_processed_sites() -> Set[str]:
    """Get set of already processed site URLs."""
    processed = set()
    if not PROJECT_TEMPLATES_DIR.exists():
        return processed
    
    for project_dir in PROJECT_TEMPLATES_DIR.iterdir():
        if project_dir.is_dir():
            metadata_path = project_dir / "metadata.json"
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                        if 'site_url' in metadata:
                            processed.add(metadata['site_url'])
                except Exception as e:
                    logger.warning(f"Error reading metadata from {metadata_path}: {e}")
    
    return processed


def gather_templates(start_from: int = 0, max_sites: Optional[int] = None, skip_existing: bool = True, fresh_start: bool = False, workers: int = 1):
    """Main function to gather templates from muuuuu.org.
    
    Args:
        start_from: Starting post number
        max_sites: Maximum number of sites to process
        skip_existing: Skip already processed sites
        fresh_start: Remove all existing templates before starting
        workers: Number of parallel workers for processing sites (default: 1, sequential)
    """
    # Handle fresh start - remove all existing templates
    if fresh_start:
        if PROJECT_TEMPLATES_DIR.exists():
            import shutil
            logger.warning(f"FRESH START: Removing all existing templates in {PROJECT_TEMPLATES_DIR}")
            try:
                shutil.rmtree(PROJECT_TEMPLATES_DIR)
                logger.info(f"✓ Removed {PROJECT_TEMPLATES_DIR}")
            except Exception as e:
                logger.error(f"Error removing {PROJECT_TEMPLATES_DIR}: {e}")
                logger.error("Please manually remove the directory and try again")
                return
        
        # Recreate directory
        PROJECT_TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("Starting fresh - all templates will be downloaded from scratch")
    
    session = get_session()
    post_num_now = start_from
    total_processed = 0
    total_skipped = 0
    empty_responses = 0
    max_empty_responses = 3  # Stop after 3 consecutive empty responses
    
    # Get already processed sites if skipping
    processed_sites = get_processed_sites() if skip_existing else set()
    if processed_sites and not fresh_start:
        logger.info(f"Found {len(processed_sites)} already processed sites. Will skip them.")
    
    logger.info(f"Starting template gathering from post_num_now={start_from}")
    
    while True:
        # Check if we've reached the max
        if max_sites and total_processed >= max_sites:
            logger.info(f"Reached maximum sites limit: {max_sites}")
            break
        
        # Fetch site list
        logger.info(f"Fetching sites {post_num_now} to {post_num_now + BATCH_SIZE}...")
        html = fetch_site_list(session, post_num_now, BATCH_SIZE)
        
        if not html or not html.strip():
            empty_responses += 1
            logger.warning(f"Empty response (count: {empty_responses}/{max_empty_responses})")
            
            if empty_responses >= max_empty_responses:
                logger.info("No more data available. Stopping.")
                break
            
            post_num_now += BATCH_SIZE
            time.sleep(REQUEST_INTERVAL)
            continue
        
        empty_responses = 0  # Reset counter on successful response
        
        # Parse site list
        sites = parse_site_list_items(html)
        logger.info(f"Found {len(sites)} sites in this batch")
        
        if not sites:
            logger.warning("No sites found in response, but response was not empty. Stopping.")
            break
        
        # Process each site (with optional parallelization)
        sites_to_process = []
        for site_info in sites:
            if max_sites and total_processed >= max_sites:
                break
            
            # Skip if already processed
            if skip_existing and site_info['site_url'] in processed_sites:
                total_skipped += 1
                logger.debug(f"Skipping already processed site: {site_info['site_url']}")
                continue
            
            sites_to_process.append(site_info)
        
        # Process sites in parallel if workers > 1
        if workers > 1 and sites_to_process:
            logger.info(f"Processing {len(sites_to_process)} sites with {workers} parallel workers...")
            # Use a lock for thread-safe updates to shared counters
            counter_lock = Lock()
            
            def process_site_wrapper(site_info):
                """Wrapper to process a site with its own session."""
                site_session = get_session()
                try:
                    success = process_site(site_session, site_info, force_redownload=not skip_existing)
                    return success, site_info
                except Exception as e:
                    logger.error(f"Error processing {site_info.get('site_url', 'unknown')}: {e}")
                    return False, site_info
            
            with ThreadPoolExecutor(max_workers=workers) as executor:
                # Submit all tasks
                future_to_site = {executor.submit(process_site_wrapper, site_info): site_info 
                                 for site_info in sites_to_process}
                
                # Process completed tasks
                for future in as_completed(future_to_site):
                    try:
                        success, site_info = future.result()
                        with counter_lock:
                            if success:
                                total_processed += 1
                                if skip_existing:
                                    processed_sites.add(site_info['site_url'])
                    except Exception as e:
                        logger.error(f"Error getting result for site: {e}")
        else:
            # Sequential processing (original behavior)
            for site_info in sites_to_process:
                success = process_site(session, site_info, force_redownload=not skip_existing)
                if success:
                    total_processed += 1
                    if skip_existing:
                        processed_sites.add(site_info['site_url'])
                
                # Wait between sites only in sequential mode
                if workers == 1:
                    time.sleep(REQUEST_INTERVAL)
        
        # Move to next batch
        post_num_now += BATCH_SIZE
        
        # Wait before next batch
        time.sleep(REQUEST_INTERVAL)
    
    logger.info(f"Template gathering complete. Processed {total_processed} new sites, skipped {total_skipped} existing sites.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Gather website templates from muuuuu.org")
    parser.add_argument(
        '--start-from',
        type=int,
        default=0,
        help='Starting post number (default: 0)'
    )
    parser.add_argument(
        '--max-sites',
        type=int,
        default=None,
        help='Maximum number of sites to process (default: no limit)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=20,
        help='Number of sites to fetch per API request (default: 20)'
    )
    parser.add_argument(
        '--interval',
        type=float,
        default=2.0,
        help='Interval between requests in seconds (default: 2.0)'
    )
    parser.add_argument(
        '--no-skip-existing',
        action='store_true',
        help='Re-download sites even if they already exist'
    )
    parser.add_argument(
        '--fresh',
        action='store_true',
        help='Remove all existing templates and start from scratch (WARNING: This deletes all data in project_templates/)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of parallel workers for processing sites (default: 1, sequential). Higher values = faster but more server load. Recommended: 2-4'
    )
    
    args = parser.parse_args()
    
    # Update global config from args
    BATCH_SIZE = args.batch_size
    REQUEST_INTERVAL = args.interval
    
    # Validate workers
    if args.workers < 1:
        logger.warning(f"Invalid workers value {args.workers}, using 1 (sequential)")
        args.workers = 1
    elif args.workers > 10:
        logger.warning(f"High workers value {args.workers} may cause rate limiting. Consider using 2-4.")
    
    gather_templates(
        start_from=args.start_from,
        max_sites=args.max_sites,
        skip_existing=not args.no_skip_existing,
        fresh_start=args.fresh,
        workers=args.workers
    )