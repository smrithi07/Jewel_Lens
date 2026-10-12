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

# --- Prices (v2): Shopify products.json returns each store's own currency. Converted to INR at load
# time in src/pipeline/filter.py; the metadata parquet keeps the original values. Fixed rates — update
# them before a demo. Price <= 0 is treated as unknown.
SHOP_CURRENCY = {"evryjewels.com": "USD", "missoma.com": "GBP"}   # every other store lists in INR
CURRENCY_TO_INR = {"INR": 1.0, "USD": 88.0, "GBP": 117.0}         # approximate, set 2026-10-12
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

# --- Phase 4 (v2): fused category ---
# "legacy" = v1 behaviour (classifier fed the L2-normalised embedding; detector label shown in the UI)
# "fused"  = src/pipeline/category.py (weights in CATEGORY_V2_DIR/fusion.json) used for re-rank and display
CATEGORY_MODE = "fused"
CATEGORY_V2_DIR = "artifacts/category_v2"

# --- Phase 8 (v2): calibrated decision thresholds (src/eval/calibrate.py, results/phase8_calibration.json) ---
# v1 hand-set values were 0.65 / 0.45 / 0.03. Scores are post-re-rank top-1 scores (cosine + category boost).
HIGH_MATCH_THRESHOLD = 0.74   # "Closest matches" tier: P(grade >= 2) >= 0.9 on tune photos (holdout check 1.00)
LOW_MATCH_THRESHOLD = 0.475   # no_match below this; set for a 2.5% false-reject budget on tune positive photos
MARGIN_THRESHOLD = 0.0        # top1 - top2 gap did not improve "match" precision on tune, so it is not used
OOS_THRESHOLD = 0.813         # CLIP out-of-scope probability above this -> no_match (2.5% false-reject budget)
OPENSET_DIR = "artifacts/openset"
# MMR diversity re-rank over the top 20; 1.0 = off. Per endpoint (Phase 8): it helped catalogue-to-catalogue
# retrieval (val nDCG@5 +0.016) but hurt photo search (own-photo holdout nDCG@5 -0.093, tune -0.039).
MMR_LAMBDA_PREDICT = 1.0      # /predict (photo search): off
MMR_LAMBDA_SIMILAR = 0.7      # /similar ("more like this")
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