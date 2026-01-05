"""
Dataset building script.
Processes website templates from project_templates/ and creates/updates the dataset.
"""

import os
import json
import re
from pathlib import Path
from bs4 import BeautifulSoup
from typing import Dict, List, Optional

# Configuration - get paths relative to project root
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
PROJECT_TEMPLATES_DIR = PROJECT_ROOT / "project_templates"
DATASET_DIR = PROJECT_ROOT / "dataset"
OUTPUT_FILE = DATASET_DIR / "train.jsonl"

def extract_url_from_folder_name(folder_name: str) -> str:
    """Extract clean URL from folder name (e.g., 'httpsamano-clinic.jp' -> 'https://amano-clinic.jp')"""
    # Remove 'https' or 'http' prefix if present
    url = folder_name.replace("https", "https://").replace("http", "http://")
    # Ensure proper format
    if not url.startswith("http"):
        url = "https://" + url
    return url

def parse_info_html(info_path: Path) -> Dict[str, str]:
    """Parse info.html to extract metadata"""
    metadata = {
        "url": "",
        "title": "",
        "category": [],
        "industry": "",
        "creators": ""
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
                
                # Try to extract industry from category
                for cat in metadata["category"]:
                    if "医療" in cat or "病院" in cat or "hospital" in cat.lower():
                        metadata["industry"] = "Healthcare"
                    elif "企業" in cat or "corporate" in cat.lower():
                        metadata["industry"] = "Corporate"
                    elif "コーポレート" in cat:
                        metadata["industry"] = "Corporate"
        
        # Extract creators
        creators_elem = soup.find('dt', string='制作者一覧')
        if creators_elem:
            next_dd = creators_elem.find_next_sibling('dd')
            if next_dd:
                metadata["creators"] = next_dd.get_text(strip=True)
        
        # Extract title from URL if not found
        if not metadata["url"]:
            folder_name = info_path.parent.name
            metadata["url"] = extract_url_from_folder_name(folder_name)
        
    except Exception as e:
        print(f"Warning: Error parsing {info_path}: {e}")
    
    return metadata

def extract_html_content(html_path: Path) -> str:
    """Extract HTML content from file"""
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except Exception as e:
        print(f"Warning: Error reading {html_path}: {e}")
        return ""

def extract_text_from_html(html_path: Path, max_length: int = 5000) -> str:
    """Extract meaningful text content from HTML for analysis"""
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        soup = BeautifulSoup(content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "meta", "link"]):
            script.decompose()
        
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
        print(f"Warning: Error reading {html_path}: {e}")
        return ""

def analyze_website_characteristics(html_path: Path, metadata: Dict) -> Dict[str, str]:
    """Analyze website to determine characteristics"""
    characteristics = {
        "tone": "Professional",
        "layout": "Standard",
        "photo_usage": "Medium",
        "motion": "None",
        "stack": "HTML + CSS + JS"
    }
    
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for animation libraries
        if 'gsap' in content.lower() or 'anime' in content.lower() or 'framer' in content.lower():
            characteristics["motion"] = "Subtle animations"
        if 'three.js' in content.lower() or 'webgl' in content.lower():
            characteristics["motion"] = "Advanced animations"
        
        # Check for grid/flex layouts
        if 'grid' in content.lower() or 'display: grid' in content:
            characteristics["layout"] = "Grid-based"
        elif 'flex' in content.lower() or 'display: flex' in content:
            characteristics["layout"] = "Flex-based"
        
        # Estimate photo usage
        img_count = content.lower().count('<img') + content.lower().count('background-image')
        if img_count > 20:
            characteristics["photo_usage"] = "High"
        elif img_count < 5:
            characteristics["photo_usage"] = "Low"
        
        # Check for frameworks
        if 'react' in content.lower() or 'vue' in content.lower() or 'angular' in content.lower():
            characteristics["stack"] = "Modern Framework"
        
        # Determine tone from metadata
        if metadata.get("industry") == "Healthcare":
            characteristics["tone"] = "Professional, Trustworthy"
        elif "清涼" in str(metadata.get("category", [])) or "fresh" in str(metadata.get("category", [])).lower():
            characteristics["tone"] = "Fresh, Modern"
        elif "ピンク" in str(metadata.get("category", [])) or "pink" in str(metadata.get("category", [])).lower():
            characteristics["tone"] = "Warm, Friendly"
        
    except Exception as e:
        print(f"Warning: Error analyzing {html_path}: {e}")
    
    return characteristics

def create_training_example(metadata: Dict, html_content: str, characteristics: Dict) -> Dict:
    """Create a training example in the chat format"""
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
    user_parts.append("Task: Design the website based on the provided HTML structure and styling.")
    
    user_prompt = "\n".join(user_parts)
    
    # Build assistant response
    assistant_parts = [
        "Design reasoning:",
        f"- Website URL: {metadata.get('url', 'N/A')}",
        f"- Layout approach: {characteristics.get('layout', 'Standard')} for clear information hierarchy",
        f"- Visual style: {characteristics.get('tone', 'Professional')} tone with {characteristics.get('photo_usage', 'Medium')} photo usage",
    ]
    
    if characteristics.get("motion") and characteristics["motion"] != "None":
        assistant_parts.append(f"- Motion: {characteristics['motion']} to enhance user experience")
    
    assistant_parts.append("\nCode:")
    assistant_parts.append("```html")
    # Include a substantial preview of the HTML (first 5000 chars for context)
    html_preview = html_content[:5000] + "\n<!-- ... rest of HTML ... -->" if len(html_content) > 5000 else html_content
    assistant_parts.append(html_preview)
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

def process_website_template(template_dir: Path) -> Optional[Dict]:
    """Process a single website template directory"""
    index_path = template_dir / "index.html"
    info_path = template_dir / "info.html"
    
    if not index_path.exists():
        print(f"Warning: {index_path} not found, skipping {template_dir.name}")
        return None
    
    print(f"Processing: {template_dir.name}")
    
    # Parse metadata
    metadata = parse_info_html(info_path) if info_path.exists() else {}
    
    # Extract full HTML content
    html_content = extract_html_content(index_path)
    if not html_content:
        print(f"Warning: Could not extract content from {index_path}")
        return None
    
    # Extract text for analysis
    text_content = extract_text_from_html(index_path)
    
    # Analyze characteristics
    characteristics = analyze_website_characteristics(index_path, metadata)
    
    # Create training example
    example = create_training_example(metadata, html_content, characteristics)
    
    return example

def build_dataset():
    """Main function to build the dataset"""
    print("Building dataset from project templates...")
    
    # Ensure dataset directory exists
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if project_templates exists
    if not PROJECT_TEMPLATES_DIR.exists():
        print(f"Error: {PROJECT_TEMPLATES_DIR} not found!")
        print("Make sure you're running this script from the src/ directory")
        return
    
    # Get all template directories
    template_dirs = [d for d in PROJECT_TEMPLATES_DIR.iterdir() if d.is_dir()]
    
    if not template_dirs:
        print(f"No template directories found in {PROJECT_TEMPLATES_DIR}")
        return
    
    print(f"Found {len(template_dirs)} website templates")
    
    # Process all templates
    examples = []
    for template_dir in template_dirs:
        example = process_website_template(template_dir)
        if example:
            examples.append(example)
    
    print(f"\nProcessed {len(examples)} examples")
    
    # Write to JSONL file
    print(f"Writing to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"Dataset created successfully! {len(examples)} examples written to {OUTPUT_FILE}")

if __name__ == "__main__":
    # Check if BeautifulSoup is available
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        print("Error: beautifulsoup4 is required. Install it with:")
        print("  pip install beautifulsoup4")
        exit(1)
    
    build_dataset()
