"""
ML Challenge 2026: Business Entity Resolution
Data Preprocessing & Normalization Module

Handles text cleaning, abbreviation expansion, tokenization,
and numeric component extraction for business names and addresses.
Preserves raw values while generating standardized representations.
"""

import re
import unicodedata

# Common legal and corporate abbreviations across US, India, and France
NAME_ABBREVIATIONS = {
    r"\bpvt\.?\b": "private",
    r"\bltd\.?\b": "limited",
    r"\binc\.?\b": "incorporated",
    r"\bcorp\.?\b": "corporation",
    r"\bco\.?\b": "company",
    r"\bllc\.?\b": "limited liability company",
    r"\bllp\.?\b": "limited liability partnership",
    r"\btech\.?\b": "technology",
    r"\btechnologies\b": "technology",
    r"\bserv\.?\b": "services",
    r"\bservices\b": "service",
    r"\bsolutions\b": "solution",
    r"\bconsultants\b": "consultant",
    r"\benterprises\b": "enterprise",
    r"\bsarl\.?\b": "sarl",
    r"\bsas\.?\b": "sas",
    r"\bgmbh\.?\b": "gmbh",
    r"\bassoc\.?\b": "associates",
    r"\bintl\.?\b": "international",
    r"\bmfg\.?\b": "manufacturing",
    r"\binds\.?\b": "industries",
    r"&": " and ",
    r"@": " at ",
}

# Common address abbreviations
ADDRESS_ABBREVIATIONS = {
    r"\brd\.?\b": "road",
    r"\bst\.?\b": "street",
    r"\bave\.?\b": "avenue",
    r"\bblvd\.?\b": "boulevard",
    r"\bln\.?\b": "lane",
    r"\bdr\.?\b": "drive",
    r"\bhwy\.?\b": "highway",
    r"\bct\.?\b": "court",
    r"\bpl\.?\b": "place",
    r"\bapt\.?\b": "apartment",
    r"\bste\.?\b": "suite",
    r"\bsec\.?\b": "sector",
    r"\bsect\.?\b": "sector",
    r"\bfl\.?\b": "floor",
    r"\bflr\.?\b": "floor",
    r"\bbldg\.?\b": "building",
    r"\bph\.?\b": "phase",
    r"\bopp\.?\b": "opposite",
    r"\bnr\.?\b": "near",
    r"\bclg\.?\b": "college",
    r"\bhno\.?\b": "house number",
    r"\bplt\.?\b": "plot",
    r"\bdist\.?\b": "district",
    r"\bdept\.?\b": "department",
    r"&": " and ",
}

# Generic corporate stop words to suppress when indexing significant tokens
NAME_STOP_WORDS = {
    "inc", "incorporated", "corp", "corporation", "ltd", "limited", "pvt", "private",
    "co", "company", "llc", "llp", "gmbh", "sa", "sarl", "sas", "the", "and", "of", "in",
    "service", "services", "solution", "solutions", "enterprise", "enterprises",
    "technology", "technologies", "consultant", "consultants", "group", "holdings",
    "international", "industries", "india", "us", "usa", "france", "de", "la", "le",
    "les", "des", "du", "et", "en"
}

# Generic address tokens to suppress when indexing significant address tokens
ADDRESS_STOP_WORDS = {
    "road", "street", "avenue", "boulevard", "lane", "drive", "highway", "court",
    "place", "apartment", "suite", "sector", "floor", "building", "phase", "plot",
    "unit", "near", "beside", "opposite", "city", "district", "state", "township",
    "county", "india", "us", "usa", "france", "and", "the", "in", "of", "at", "to",
    "de", "la", "le", "les", "des", "du", "rue", "avenue", "boulevard", "chemin"
}


def clean_unicode(text):
    """Normalize Unicode characters to standard NFKD ASCII-compatible text."""
    if not isinstance(text, str):
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    return normalized.encode("ascii", "ignore").decode("ascii")


def normalize_name(text):
    """
    Standardize a business name string:
    - Lowercase and Unicode clean
    - Strip domain suffixes (e.g. .com, .org, .net)
    - Expand corporate abbreviations
    - Strip non-alphanumeric punctuation and collapse whitespace
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = clean_unicode(text).lower()

    # Remove URL domain extensions if name is formatted as a website
    text = re.sub(r"\.(com|net|org|in|co|us|fr|io|biz|info)\b", "", text)

    for pattern, repl in NAME_ABBREVIATIONS.items():
        text = re.sub(pattern, repl, text)

    # Replace punctuation with whitespace
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_address(text):
    """
    Standardize a business address string:
    - Lowercase and Unicode clean
    - Expand address abbreviations
    - Strip noisy symbols (e.g., #, ##, *, commas, dots)
    - Collapse whitespace
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = clean_unicode(text).lower()

    for pattern, repl in ADDRESS_ABBREVIATIONS.items():
        text = re.sub(pattern, repl, text)

    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_tokens(text):
    """Extract list of alphanumeric tokens with length > 1."""
    if not text:
        return []
    return [t for t in text.split() if len(t) > 1]


def extract_significant_name_tokens(norm_name):
    """Extract informative, non-stopword tokens from normalized name."""
    tokens = extract_tokens(norm_name)
    return [t for t in tokens if t not in NAME_STOP_WORDS and len(t) > 2]


def extract_significant_address_tokens(norm_address):
    """Extract informative locality/street/landmark tokens from normalized address."""
    tokens = extract_tokens(norm_address)
    return [t for t in tokens if t not in ADDRESS_STOP_WORDS and len(t) > 2]


def extract_numbers(text):
    """Extract all numeric sequences (building numbers, postal codes, etc.)."""
    if not isinstance(text, str) or not text.strip():
        return []
    return re.findall(r"\b\d+\b", text)


def extract_name_prefix(norm_name, length=5):
    """Extract first N alphanumeric characters without spaces."""
    cleaned = norm_name.replace(" ", "")
    return cleaned[:length] if len(cleaned) >= 4 else ""


class CompactRecord:
    """Memory-efficient representation of a preprocessed business record using __slots__."""
    __slots__ = (
        "entity_id", "country", "name_norm", "addr_norm",
        "name_tokens", "name_sig", "name_prefix",
        "addr_tokens", "addr_sig", "addr_numbers"
    )

    def __init__(self, entity_id, country, name_norm, addr_norm, name_tokens, name_sig, name_prefix, addr_tokens, addr_sig, addr_numbers):
        self.entity_id = entity_id
        self.country = country
        self.name_norm = name_norm
        self.addr_norm = addr_norm
        self.name_tokens = name_tokens
        self.name_sig = name_sig
        self.name_prefix = name_prefix
        self.addr_tokens = addr_tokens
        self.addr_sig = addr_sig
        self.addr_numbers = addr_numbers

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)


def preprocess_record(record_dict):
    """
    Generate normalized representations and pre-tokenized features for a record dict.
    Returns a memory-compact CompactRecord with __slots__.
    """
    raw_name = record_dict.get("business_name", "")
    raw_addr = record_dict.get("business_address", "")
    country = str(record_dict.get("country", "")).strip().upper()

    norm_n = normalize_name(raw_name)
    norm_a = normalize_address(raw_addr)

    return CompactRecord(
        entity_id=record_dict["entity_id"],
        country=country,
        name_norm=norm_n,
        addr_norm=norm_a,
        name_tokens=extract_tokens(norm_n),
        name_sig=extract_significant_name_tokens(norm_n),
        name_prefix=extract_name_prefix(norm_n, length=5),
        addr_tokens=extract_tokens(norm_a),
        addr_sig=extract_significant_address_tokens(norm_a),
        addr_numbers=extract_numbers(raw_addr),
    )
