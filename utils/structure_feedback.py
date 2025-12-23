import re

def get_structure_feedback(text):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    lower_text = text.lower()

    feedback = []
    score = 100

    # -------- Page Length --------
    if len(text) / 3000 > 1.3:
        feedback.append("⚠ Resume likely exceeds 1 page")
        score -= 10
    else:
        feedback.append("✅ Resume length is ATS-friendly")

    # -------- Section Detection (Flexible) --------
    SECTION_ALIASES = {
        "Experience": ["experience", "professional experience", "work experience", "employment"],
        "Projects": ["projects", "data science projects", "academic projects", "research", "portfolio"],
        "Skills": ["skills", "technical skills", "core skills", "competencies"],
        "Education": ["education", "academic background", "qualifications"]
    }

    for section, aliases in SECTION_ALIASES.items():
        if not any(
            re.search(rf'^\s*{alias}\b', lower_text, re.MULTILINE)
            for alias in aliases
        ):
            feedback.append(f"⚠ Missing '{section}' section")
            score -= 10
        else:
            feedback.append(f"✅ '{section}' section found")

    # -------- Bullet Detection (PDF-Safe) --------
    bullet_lines = [
        l for l in lines
        if re.match(r'^(\*|-|•|–)', l) or len(l.split()) > 6
    ]

    # -------- Quantification Detection (Improved) --------
    # More comprehensive pattern matching for various quantification formats
    metric_patterns = [
        re.compile(r'\d+(\.\d+)?%'),                    # percentages: 50%, 12.5%
        re.compile(r'\$\d+(\.\d+)?[kKmMbB]?'),          # money: $100, $1.5M, $50k
        re.compile(r'\d+(\.\d+)?[kKmMbB]'),             # numbers with units: 5M, 42k, 1.5B
        re.compile(r'\d{1,3}(,\d{3})+'),                # comma-separated: 42,000, 1,500,000
        re.compile(r'\d+\s*(years?|months?|days?|hours?|weeks?)'),  # time periods
        re.compile(r'\d+\+'),                            # plus notation: 100+, 3+
        re.compile(r'over\s+\d+', re.IGNORECASE),       # over 100
        re.compile(r'more\s+than\s+\d+', re.IGNORECASE), # more than 50
        re.compile(r'(\d+)\s*(to|-)\s*(\d+)'),          # ranges: 10-20, 50 to 100
        re.compile(r'increased\s+by\s+\d+', re.IGNORECASE), # increased by 50%
        re.compile(r'reduced\s+by\s+\d+', re.IGNORECASE), # reduced by 30%
        re.compile(r'\d+x', re.IGNORECASE),              # multipliers: 2x, 3x
        re.compile(r'\d+\s*(people|users|customers|clients|projects|teams)', re.IGNORECASE),  # with context
    ]
    
    # Check all lines, not just bullet lines, for better detection
    all_lines = lines
    quantified_lines = []
    
    for line in all_lines:
        for pattern in metric_patterns:
            if pattern.search(line):
                quantified_lines.append(line)
                break  # Count each line only once
    
    # Also check for standalone numbers that might be metrics (in context)
    standalone_number_pattern = re.compile(r'\b\d{2,}\b')  # 2+ digit numbers
    for line in all_lines:
        if line not in quantified_lines:
            # Check if line has numbers and action words (likely a metric)
            has_number = standalone_number_pattern.search(line)
            action_words = ['increased', 'decreased', 'improved', 'reduced', 'achieved', 
                          'managed', 'led', 'handled', 'processed', 'generated', 'saved',
                          'delivered', 'completed', 'created', 'developed', 'implemented']
            has_action = any(word in line.lower() for word in action_words)
            if has_number and has_action:
                quantified_lines.append(line)
    
    # More lenient quantification detection - check if we have meaningful quantifications
    # Count unique quantified lines and also check bullet lines specifically
    quantified_bullets = [line for line in bullet_lines if line in quantified_lines]
    
    # If we have at least 3 quantified achievements, consider it good
    # Or if 30%+ of bullets have quantifications
    total_bullets = max(len(bullet_lines), 1)
    quant_bullet_pct = (len(quantified_bullets) / total_bullets) * 100 if bullet_lines else 0
    
    # Also check overall - if we have multiple quantifications, that's good
    has_good_quantification = len(quantified_lines) >= 3 or quant_bullet_pct >= 30
    
    if has_good_quantification:
        feedback.append(f"✅ Strong use of quantified achievements ({len(quantified_lines)} found)")
    elif len(quantified_lines) >= 1:
        feedback.append(f"⚠ Good quantification — found {len(quantified_lines)} metric(s), could add more")
        score -= 3  # Reduced penalty
    else:
        feedback.append("⚠ Quantification could be improved — add numbers, percentages, or metrics")
        score -= 5  # Reduced penalty

    # -------- Bullet Length --------
    if any(len(l) > 160 for l in bullet_lines):
        feedback.append("⚠ Some bullets are too long (keep under 160 chars)")
        score -= 7
    else:
        feedback.append("✅ Bullet length is ATS-friendly")

    return max(score, 55), feedback
