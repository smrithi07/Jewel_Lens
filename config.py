# Central config — thresholds and model names live here, not hardcoded in scripts

DETECTOR_MODEL = "google/owlvit-base-patch32"
EMBEDDER_MODEL = "facebook/dinov2-base"

CATEGORIES = ["earring", "necklace", "bracelet"]

JEWELRY_COLORS = [
    "gold", "rose gold", "silver", "platinum", "white gold", "oxidized",
    "black", "white", "red", "pink", "blue", "green", "purple",
    "yellow", "orange", "multicolour",
]

SEARCH_FACETS = {
    "earring": {
        "subtypes": ["jhumka", "stud", "hoop", "drop dangle", "chandbali",
                     "huggie", "ear cuff", "chandelier"],
        "colors": JEWELRY_COLORS,
    },
    "necklace": {
        "subtypes": ["pendant", "chain", "choker", "layered", "statement",
                     "collar", "beaded", "long chain"],
        "colors": JEWELRY_COLORS,
    },
    "bracelet": {
        "subtypes": ["chain", "cuff", "bangle", "charm", "tennis",
                     "beaded", "link", "kada"],
        "colors": JEWELRY_COLORS,
    },
}

TARGET_PER_SUBTYPE = 160
TARGET_PER_COLOR = 65

RAW_IMAGE_DIR = "data/raw"
CATALOGUE_METADATA_PATH = "data/catalogue_metadata.parquet"

FAISS_INDEX_PATH = "indices/catalogue.faiss"
ID_MAPPING_PATH = "indices/id_mapping.pkl"
CLASSIFIER_PATH = "models/category_classifier.pkl"

REQUESTS_PER_SECOND = 5
DAILY_REQUEST_BUDGET = 5000
REQUEST_BUDGET_BUFFER = 200
SCRAPE_PROGRESS_PATH = "data/scrape_progress.json"
SCRAPE_PARTIAL_PATH = "data/catalogue_metadata_partial.jsonl"
SHOPIFY_DOMAIN_PROGRESS_PATH = "data/shopify_domain_progress.json"

PHASH_THRESHOLD = 5   # Hamming distance; lower = stricter match
DEDUP_LOG_PATH = "data/dedup_log.csv"

SIMILARITY_THRESHOLD = 0.5
MARGIN_THRESHOLD = 0.05

DETECTION_PROMPTS = ["earring", "necklace", "bracelet"]
DETECTION_PROMPT_TEMPLATES = {
    "earring": "a single earring",
    "necklace": "a necklace worn around the neck",
    "bracelet": "a bracelet or bangle worn on the wrist",
}
DETECTION_CONFIDENCE_THRESHOLD = 0.1   # OWL-ViT's own threshold, tune after seeing real results
CROP_PADDING_RATIO = 0.12               # 12% padding around each detected box

# --- Phase 5: classifier + re-rank ---
CLASSIFIER_VAL_SIZE = 0.15
CLASSIFIER_TEST_SIZE = 0.15
RERANK_BOOST_WEIGHT = 0.08   # additive boost to cosine sim for category-matching candidates
RERANK_MIN_CONFIDENCE = 0.4  # below this, classifier's vote isn't trusted enough to matter

HIGH_MATCH_THRESHOLD = 0.65
LOW_MATCH_THRESHOLD = 0.45
MARGIN_THRESHOLD = 0.03   # top1 - top2 gap; below this, top1 isn't a confident winner
# --- Shopify backup source ---
SHOPIFY_STORE_DOMAINS = [
    # "jokerandwitch.com",
    # "missoma.com",
    # "analuisa.com",
    # "mejuri.com",
    # "evryjewels.com",
    "jaypore.com", "caratlane.com", "voylla.com", "zaveribazaar.com", "amrapaligallery.com", "sangeetaboochra.com", "auraa.co.in", "rubans.in", "melorra.com", "giva.co"
]

MAX_PRODUCTS_PER_STORE = 650  # cap per domain — keeps runtime/storage bounded

# Domains with confirmed collection handles (curl-tested) scrape via
# collections/<handle>/products.json instead of the whole catalogue.
# Fill in once you've curl-tested each domain — leave a domain OUT of
# this dict to fall back to whole-catalogue keyword-filtered scanning.
STORE_COLLECTIONS = {
    # "jokerandwitch.com": {
    #     "earring": "earrings",
    #     "necklace": "necklaces",
    #     "bracelet": "bracelets",
    # },
}

CLIP_MODEL = "openai/clip-vit-base-patch32"
CLIP_QA_LOG_PATH = "data/clip_qa_log.csv"
CLIP_AUTO_DROP_THRESHOLD = 0.6   # predicted-wrong-category confidence above this -> auto-drop
CLIP_FLAG_THRESHOLD = 0.4        # above this but below auto-drop -> log for manual review, keep image

CATALOGUE_EMBEDDINGS_PATH = "data/catalogue_embeddings.npy"
CATALOGUE_EMBEDDINGS_META_PATH = "data/catalogue_embeddings_meta.parquet"

# Anchored, word-boundary-matched keywords — deliberately strict so
# rings, nose rings, hip chains, purses, cufflinks etc. never match.
CATEGORY_KEYWORDS = {
    "earring": [
        "earring", "earrings", "jhumka", "jhumki", "chandbali",
        "stud earring", "hoop earring", "drop earring", "dangle earring",
        "huggie earring", "ear cuff", "chandelier earring",
    ],
    "necklace": [
        "necklace", "necklaces",
        "pendant necklace", "choker necklace", "layered necklace",
        "statement necklace", "collar necklace", "beaded necklace",
        "chain necklace", "long necklace",
    ],
    "bracelet": [
        "bracelet", "bracelets", "bangle", "kada",
        "cuff bracelet", "charm bracelet", "tennis bracelet",
        "beaded bracelet", "link bracelet",
    ],
}

SUBTYPE_KEYWORDS = {
    "jhumka": ["jhumka", "jhumki"],
    "stud": ["stud earring"],
    "hoop": ["hoop earring", "hoop"],
    "drop dangle": ["drop earring", "dangle earring"],
    "chandbali": ["chandbali"],
    "huggie": ["huggie"],
    "ear cuff": ["ear cuff"],
    "chandelier": ["chandelier earring"],
    "pendant": ["pendant necklace"],
    "chain": ["chain necklace"],
    "choker": ["choker"],
    "layered": ["layered necklace"],
    "statement": ["statement necklace"],
    "collar": ["collar necklace"],
    "beaded": ["beaded necklace", "beaded bracelet"],
    "bangle": ["bangle"],
    "cuff": ["cuff bracelet"],
    "tennis": ["tennis bracelet"],
    "kada": ["kada"],
    "link": ["link bracelet"],
}

# Products containing any of these are rejected even if a category
# keyword also matches (catches "watch bracelet stack" etc.)
EXCLUDE_KEYWORDS = ["watch", "watches", "dial", "analog", "chronograph"]