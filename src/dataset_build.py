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
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration constants
MAX_HTML_LENGTH = 10000  # Max chars for HTML in training example
MAX_CSS_LENGTH = 5000    # Max chars for CSS in training example
MAX_JS_LENGTH = 3000     # Max chars for JS in training example
MAX_TOTAL_LENGTH = 20000 # Max total chars for code sections

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
        "keywords": []
    }
    
    try:
        with open(info_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        soup = BeautifulSoup(content, 'html.parser')
        
        # Extract URL
        url_elem = soup.find('dt', string='URL')
        if url_elem:
            next_dd = url_elem.find_next_sibling('dd')
            if next_dd:
                link = next_dd.find('a')
                if link:
                    metadata["url"] = link.get('href', '').strip()
                    if not metadata["url"]:
                        text = link.get_text(strip=True)
                        metadata["url"] = text if text.startswith('http') else f"https://{text}"
        
        # Extract title
        title_elem = soup.find('title')
        if title_elem:
            metadata["title"] = title_elem.get_text(strip=True)
        
        # Extract categories
        category_elem = soup.find('dt', string='CATEGORY')
        if category_elem:
            next_dd = category_elem.find_next_sibling('dd')
            if next_dd:
                category_links = next_dd.find_all('a')
                for link in category_links:
                    category_text = link.get_text(strip=True)
                    if category_text:
                        metadata["category"].append(category_text)
        
        # Enhanced industry detection
        all_text = str(metadata.get("category", [])) + " " + metadata.get("title", "")
        if any(keyword in all_text for keyword in ["医療", "病院", "hospital", "clinic", "診療", "クリニック"]):
            metadata["industry"] = "Healthcare"
        elif any(keyword in all_text for keyword in ["企業", "corporate", "コーポレート", "company"]):
            metadata["industry"] = "Corporate"
        elif any(keyword in all_text for keyword in ["美容", "beauty", "salon", "サロン"]):
            metadata["industry"] = "Beauty"
        elif any(keyword in all_text for keyword in ["飲食", "restaurant", "レストラン", "cafe"]):
            metadata["industry"] = "Food & Beverage"
        elif any(keyword in all_text for keyword in ["教育", "education", "school", "スクール"]):
            metadata["industry"] = "Education"
        
        # Extract creators
        creators_elem = soup.find('dt', string='制作者一覧')
        if creators_elem:
            next_dd = creators_elem.find_next_sibling('dd')
            if next_dd:
                metadata["creators"] = next_dd.get_text(strip=True)
        
        # Extract description
        desc_elem = soup.find('meta', attrs={'name': 'description'})
        if desc_elem:
            metadata["description"] = desc_elem.get('content', '').strip()
        
        # Extract from URL if not found
        if not metadata["url"]:
            folder_name = info_path.parent.name
            metadata["url"] = extract_url_from_folder_name(folder_name)
        
    except Exception as e:
        logger.warning(f"Error parsing {info_path}: {e}")
    
    return metadata

def find_css_files(html_path: Path, assets_dir: Path) -> List[Tuple[str, str]]:
    """Find and extract CSS files referenced in HTML"""
    css_files = []
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        soup = BeautifulSoup(content, 'html.parser')
        
        # Find all link tags with stylesheet
        for link in soup.find_all('link', rel='stylesheet'):
            href = link.get('href', '')
            if href:
                # Resolve relative paths
                if href.startswith('./'):
                    css_path = html_path.parent / href[2:]
                elif href.startswith('/'):
                    css_path = assets_dir / href[1:]
                else:
                    css_path = html_path.parent / href
                
                if css_path.exists():
                    try:
                        with open(css_path, 'r', encoding='utf-8') as css_file:
                            css_content = css_file.read()
                            # Get filename for reference
                            filename = css_path.name
                            css_files.append((filename, css_content[:MAX_CSS_LENGTH]))
                    except Exception as e:
                        logger.debug(f"Could not read CSS file {css_path}: {e}")
        
        # Also check for inline styles
        for style in soup.find_all('style'):
            css_content = style.string or ""
            if css_content:
                css_files.append(("inline", css_content[:MAX_CSS_LENGTH]))
    
    except Exception as e:
        logger.warning(f"Error finding CSS files: {e}")
    
    return css_files

def find_js_files(html_path: Path, assets_dir: Path) -> List[Tuple[str, str]]:
    """Find and extract JavaScript files referenced in HTML"""
    js_files = []
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        soup = BeautifulSoup(content, 'html.parser')
        
        # Find all script tags with src
        for script in soup.find_all('script', src=True):
            src = script.get('src', '')
            if src and not src.startswith('http'):  # Skip external URLs
                # Resolve relative paths
                if src.startswith('./'):
                    js_path = html_path.parent / src[2:]
                elif src.startswith('/'):
                    js_path = assets_dir / src[1:]
                else:
                    js_path = html_path.parent / src
                
                if js_path.exists():
                    try:
                        with open(js_path, 'r', encoding='utf-8') as js_file:
                            js_content = js_file.read()
                            filename = js_path.name
                            js_files.append((filename, js_content[:MAX_JS_LENGTH]))
                    except Exception as e:
                        logger.debug(f"Could not read JS file {js_path}: {e}")
        
        # Also check for inline scripts
        for script in soup.find_all('script'):
            if not script.get('src'):
                js_content = script.string or ""
                if js_content and len(js_content) > 50:  # Only include substantial scripts
                    js_files.append(("inline", js_content[:MAX_JS_LENGTH]))
    
    except Exception as e:
        logger.warning(f"Error finding JS files: {e}")
    
    return js_files

def extract_html_content(html_path: Path) -> str:
    """Extract and clean HTML content"""
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Clean up saved page artifacts
        content = re.sub(r'<!-- saved from url=.*?-->', '', content)
        
        return content
    except Exception as e:
        logger.warning(f"Error reading {html_path}: {e}")
        return ""

def extract_text_from_html(html_path: Path, max_length: int = 5000) -> str:
    """Extract meaningful text content from HTML for analysis"""
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        soup = BeautifulSoup(content, 'html.parser')
        
        # Remove script, style, meta, link elements
        for element in soup(["script", "style", "meta", "link", "noscript"]):
            element.decompose()
        
        # Get text
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        # Truncate if too long
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        return text
    except Exception as e:
        logger.warning(f"Error extracting text from {html_path}: {e}")
        return ""

def analyze_color_scheme(html_path: Path, css_files: List[Tuple[str, str]]) -> Dict[str, any]:
    """Analyze color scheme from HTML and CSS"""
    colors = {
        "primary": [],
        "background": [],
        "text": [],
        "scheme": "Unknown"
    }
    
    try:
        # Collect all CSS content
        all_css = " ".join([css[1] for css in css_files])
        
        # Extract color values
        color_patterns = [
            r'color:\s*([#\w]+)',
            r'background(?:-color)?:\s*([#\w]+)',
            r'#[0-9a-fA-F]{3,6}',
            r'rgb\([^)]+\)',
            r'rgba\([^)]+\)',
        ]
        
        found_colors = []
        for pattern in color_patterns:
            matches = re.findall(pattern, all_css, re.IGNORECASE)
            found_colors.extend(matches)
        
        # Count most common colors
        color_counter = Counter([c.lower() for c in found_colors if len(c) > 2])
        colors["primary"] = [color for color, count in color_counter.most_common(3)]
        
        # Determine color scheme
        if any('dark' in str(c).lower() or 'black' in str(c).lower() for c in colors["primary"]):
            colors["scheme"] = "Dark"
        elif any('light' in str(c).lower() or 'white' in str(c).lower() for c in colors["primary"]):
            colors["scheme"] = "Light"
        else:
            colors["scheme"] = "Mixed"
    
    except Exception as e:
        logger.debug(f"Error analyzing colors: {e}")
    
    return colors

def analyze_website_characteristics(html_path: Path, metadata: Dict, css_files: List[Tuple[str, str]], js_files: List[Tuple[str, str]]) -> Dict[str, any]:
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
        "interactivity": "Low"
    }
    
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        all_content = html_content.lower()
        all_css = " ".join([css[1].lower() for css in css_files])
        all_js = " ".join([js[1].lower() for js in js_files])
        combined = all_content + " " + all_css + " " + all_js
        
        # Motion detection
        if any(lib in combined for lib in ['three.js', 'webgl', 'webgl2', 'babylon']):
            characteristics["motion"] = "Advanced 3D animations"
        elif any(lib in combined for lib in ['gsap', 'anime.js', 'framer', 'lottie', 'aos']):
            characteristics["motion"] = "Subtle animations"
        elif 'transition' in all_css or 'animation' in all_css or 'transform' in all_css:
            characteristics["motion"] = "CSS animations"
        
        # Layout detection
        if 'display: grid' in all_css or 'display:grid' in all_css or 'grid-template' in all_css:
            characteristics["layout"] = "Grid-based"
        elif 'display: flex' in all_css or 'display:flex' in all_css or 'flexbox' in all_css:
            characteristics["layout"] = "Flex-based"
        elif 'bootstrap' in combined or 'foundation' in combined:
            characteristics["layout"] = "Framework-based"
        
        # Photo usage
        img_count = all_content.count('<img') + all_content.count('background-image') + all_content.count('background: url')
        if img_count > 30:
            characteristics["photo_usage"] = "Very High"
        elif img_count > 20:
            characteristics["photo_usage"] = "High"
        elif img_count < 5:
            characteristics["photo_usage"] = "Low"
        
        # Framework detection
        if any(fw in combined for fw in ['react', 'vue', 'angular', 'svelte']):
            characteristics["stack"] = "Modern Framework"
        elif 'jquery' in combined:
            characteristics["stack"] = "HTML + CSS + JS + jQuery"
        elif 'typescript' in combined or '.ts' in combined:
            characteristics["stack"] = "HTML + CSS + TypeScript"
        
        # Responsive design
        if 'viewport' in html_content or '@media' in all_css or 'responsive' in combined:
            characteristics["responsive"] = True
        
        # Typography
        if any(font in all_css for font in ['googleapis', 'fonts.com', 'typekit', 'adobe fonts']):
            characteristics["typography"] = "Custom web fonts"
        elif 'font-family' in all_css:
            characteristics["typography"] = "Custom typography"
        
        # Interactivity
        event_count = all_js.count('addEventListener') + all_js.count('onclick') + all_js.count('on(')
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
        elif "清涼" in str(metadata.get("category", [])) or "fresh" in str(metadata.get("category", [])).lower():
            characteristics["tone"] = "Fresh, Modern, Clean"
        elif "ピンク" in str(metadata.get("category", [])) or "pink" in str(metadata.get("category", [])).lower():
            characteristics["tone"] = "Warm, Friendly, Approachable"
        elif "minimal" in combined or "minimalist" in combined:
            characteristics["tone"] = "Minimalist, Clean, Simple"
        elif "luxury" in combined or "premium" in combined:
            characteristics["tone"] = "Luxury, Premium, Sophisticated"
        
        # Accessibility
        if 'aria-' in html_content or 'role=' in html_content or 'alt=' in html_content:
            characteristics["accessibility"] = "Enhanced"
    
    except Exception as e:
        logger.warning(f"Error analyzing {html_path}: {e}")
    
    return characteristics

def generate_design_reasoning_with_openai(metadata: Dict, characteristics: Dict, 
                                         css_files: List[Tuple[str, str]], 
                                         js_files: List[Tuple[str, str]]) -> Optional[str]:
    """Generate design reasoning using OpenAI API (80% automated draft)"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not found in environment. Skipping AI reasoning generation.")
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
        if characteristics.get("color_scheme") and characteristics["color_scheme"] != "Unknown":
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
                {"role": "system", "content": "You are an expert front-end designer who explains design decisions clearly and concisely."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300
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

def create_training_example(metadata: Dict, html_content: str, css_files: List[Tuple[str, str]], 
                           js_files: List[Tuple[str, str]], characteristics: Dict, 
                           use_ai_reasoning: bool = True) -> Dict:
    """Create a comprehensive training example in chat format"""
    # Build user prompt
    user_parts = []
    if metadata.get("industry"):
        user_parts.append(f"Industry: {metadata['industry']}")
    if characteristics.get("tone"):
        user_parts.append(f"Tone: {characteristics['tone']}")
    user_parts.append("Page type: Landing page")
    if characteristics.get("layout"):
        user_parts.append(f"Layout: {characteristics['layout']}")
    if characteristics.get("photo_usage"):
        user_parts.append(f"Photo usage: {characteristics['photo_usage']}")
    if characteristics.get("motion") and characteristics["motion"] != "None":
        user_parts.append(f"Motion: {characteristics['motion']}")
    if characteristics.get("stack"):
        user_parts.append(f"Stack: {characteristics['stack']}")
    if characteristics.get("responsive"):
        user_parts.append("Responsive: Yes")
    if characteristics.get("color_scheme") and characteristics["color_scheme"] != "Unknown":
        user_parts.append(f"Color scheme: {characteristics['color_scheme']}")
    user_parts.append("Task: Design the website based on the provided HTML structure and styling.")
    
    user_prompt = "\n".join(user_parts)
    
    # Generate reasoning (AI-assisted or fallback)
    reasoning_text = None
    if use_ai_reasoning:
        reasoning_text = generate_design_reasoning_with_openai(metadata, characteristics, css_files, js_files)
    
    # Build assistant response
    assistant_parts = ["Design reasoning:"]
    
    if reasoning_text:
        # Use AI-generated reasoning (80% automated)
        assistant_parts.append(reasoning_text)
        assistant_parts.append(f"\nWebsite URL: {metadata.get('url', 'N/A')}")
    else:
        # Fallback to manual reasoning template
        assistant_parts.append(f"- Website URL: {metadata.get('url', 'N/A')}")
        assistant_parts.append(f"- Layout approach: {characteristics.get('layout', 'Standard')} for clear information hierarchy")
        assistant_parts.append(f"- Visual style: {characteristics.get('tone', 'Professional')} tone with {characteristics.get('photo_usage', 'Medium')} photo usage")
        
        if characteristics.get("motion") and characteristics["motion"] != "None":
            assistant_parts.append(f"- Motion: {characteristics['motion']} to enhance user experience")
        
        if characteristics.get("color_scheme") and characteristics["color_scheme"] != "Unknown":
            assistant_parts.append(f"- Color scheme: {characteristics['color_scheme']} palette for visual consistency")
        
        if characteristics.get("responsive"):
            assistant_parts.append("- Responsive design: Mobile-first approach with breakpoints")
    
    assistant_parts.append("\nCode:")
    
    # Include HTML
    html_preview = html_content[:MAX_HTML_LENGTH]
    if len(html_content) > MAX_HTML_LENGTH:
        html_preview += "\n<!-- ... rest of HTML ... -->"
    assistant_parts.append("```html")
    assistant_parts.append(html_preview)
    assistant_parts.append("```")
    
    # Include CSS if available
    if css_files:
        assistant_parts.append("\n```css")
        css_content = "\n\n/* " + css_files[0][0] + " */\n" + css_files[0][1]
        if len(css_files) > 1:
            css_content += f"\n\n/* ... {len(css_files) - 1} more CSS file(s) ... */"
        assistant_parts.append(css_content)
        assistant_parts.append("```")
    
    # Include JS if available and not too large
    if js_files and len(js_files[0][1]) < 2000:
        assistant_parts.append("\n```javascript")
        js_content = "\n\n// " + js_files[0][0] + "\n" + js_files[0][1]
        if len(js_files) > 1:
            js_content += f"\n\n// ... {len(js_files) - 1} more JS file(s) ..."
        assistant_parts.append(js_content)
        assistant_parts.append("```")
    
    assistant_response = "\n".join(assistant_parts)
    
    return {
        "messages": [
            {
                "role": "system",
                "content": "You are a senior creative front-end engineer who designs brand-specific websites."
            },
            {
                "role": "user",
                "content": user_prompt
            },
            {
                "role": "assistant",
                "content": assistant_response
            }
        ]
    }

def validate_example(example: Dict) -> bool:
    """Validate training example quality"""
    if not example or "messages" not in example:
        return False
    
    messages = example["messages"]
    if len(messages) != 3:
        return False
    
    # Check that assistant message has substantial content
    assistant_content = messages[2].get("content", "")
    if len(assistant_content) < 100:
        return False
    
    # Check that code blocks are present
    if "```html" not in assistant_content and "```" not in assistant_content:
        return False
    
    return True

def process_website_template(template_dir: Path, use_ai_reasoning: bool = True) -> Optional[Dict]:
    """Process a single website template directory with enhanced analysis"""
    index_path = template_dir / "index.html"
    info_path = template_dir / "info.html"
    assets_dir = template_dir / "assets"
    
    if not index_path.exists():
        logger.warning(f"{index_path} not found, skipping {template_dir.name}")
        return None
    
    logger.info(f"Processing: {template_dir.name}")
    
    try:
        # Parse metadata
        metadata = parse_info_html(info_path) if info_path.exists() else {}
        
        # Extract HTML content
        html_content = extract_html_content(index_path)
        if not html_content:
            logger.warning(f"Could not extract content from {index_path}")
            return None
        
        # Find and extract CSS/JS files
        css_files = find_css_files(index_path, assets_dir) if assets_dir.exists() else []
        js_files = find_js_files(index_path, assets_dir) if assets_dir.exists() else []
        
        # Analyze characteristics
        characteristics = analyze_website_characteristics(index_path, metadata, css_files, js_files)
        
        # Create training example (with AI reasoning if available)
        example = create_training_example(metadata, html_content, css_files, js_files, 
                                        characteristics, use_ai_reasoning=use_ai_reasoning)
        
        # Validate example
        if not validate_example(example):
            logger.warning(f"Example validation failed for {template_dir.name}")
            return None
        
        return example
    
    except Exception as e:
        logger.error(f"Error processing {template_dir.name}: {e}", exc_info=True)
        return None

def build_dataset(incremental: bool = False, use_ai: bool = True):
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
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        example = json.loads(line)
                        # Create hash from URL for deduplication
                        url = example.get("messages", [{}])[2].get("content", "")
                        if "Website URL:" in url:
                            url_hash = hashlib.md5(url.encode()).hexdigest()
                            existing_hashes.add(url_hash)
        except Exception as e:
            logger.warning(f"Could not load existing dataset: {e}")
    
    # Process all templates
    examples = []
    skipped = 0
    errors = 0
    
    for i, template_dir in enumerate(template_dirs, 1):
        logger.info(f"[{i}/{len(template_dirs)}] Processing {template_dir.name}...")
        
        example = process_website_template(template_dir, use_ai_reasoning=use_ai)
        if example:
            examples.append(example)
        else:
            skipped += 1
    
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Processing complete!")
    logger.info(f"  - Processed: {len(examples)} examples")
    logger.info(f"  - Skipped: {skipped} templates")
    logger.info(f"  - Errors: {errors} errors")
    logger.info(f"{'=' * 80}")
    
    if not examples:
        logger.error("No valid examples generated!")
        return
    
    # Write to JSONL file
    logger.info(f"Writing to {OUTPUT_FILE}...")
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
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
    parser.add_argument("--incremental", action="store_true", 
                       help="Update existing dataset incrementally")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--no-ai", action="store_true",
                       help="Disable AI reasoning generation (use fallback)")
    
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
    
    build_dataset(incremental=args.incremental, use_ai=not args.no_ai)
