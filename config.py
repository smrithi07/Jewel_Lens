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
TARGET_PER_COLOR = 65   # lowered — 48 color facets now (16 x 3) vs. 19 before

RAW_IMAGE_DIR = "data/raw"
CATALOGUE_METADATA_PATH = "data/catalogue_metadata.parquet"

FAISS_INDEX_PATH = "indices/catalogue.faiss"
ID_MAPPING_PATH = "indices/id_mapping.pkl"
CLASSIFIER_PATH = "models/category_classifier.pkl"

# Thresholds — placeholders, will tune in Phase 6
SIMILARITY_THRESHOLD = 0.5
MARGIN_THRESHOLD = 0.05