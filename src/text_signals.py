import re


TOKEN_STOPWORDS = {
    "aber",
    "bitte",
    "brauch",
    "brauche",
    "brauchen",
    "chair",
    "classified",
    "content",
    "content_request",
    "eine",
    "einen",
    "einer",
    "einfach",
    "einfache",
    "ergaenzen",
    "equipment",
    "fuer",
    "geraete",
    "geraet",
    "gern",
    "gibt",
    "haette",
    "hause",
    "ich",
    "ihr",
    "koennt",
    "kurz",
    "kurze",
    "lernen",
    "liegen",
    "matte",
    "mehr",
    "mock",
    "moecht",
    "moechte",
    "nach",
    "ohne",
    "plan",
    "position",
    "progression",
    "progressions",
    "request",
    "rules",
    "sitzen",
    "standing",
    "stehen",
    "stehend",
    "stuhl",
    "suche",
    "training",
    "uebung",
    "uebungen",
    "und",
    "variante",
    "varianten",
    "vorbereitung",
    "wall",
    "wand",
    "will",
    "wunsch",
    "zuhause",
    "zuhaus",
}
COMPOUND_SUFFIXES = ("uebungen", "uebung", "voruebungen", "training")
TOPIC_NORMALIZATION = {
    "atmung": "atem",
}


def extract_topic_tokens(text: str) -> set[str]:
    normalized = normalize_text(text)
    tokens = set()
    for token in re.findall(r"[a-z0-9]+", normalized):
        token = normalize_compound_token(token)
        if len(token) < 4 or token in TOKEN_STOPWORDS:
            continue
        tokens.add(token)
    return tokens


def normalize_text(text: str) -> str:
    return (
        text.casefold()
        .replace("ä", "ae")
        .replace("ö", "oe")
        .replace("ü", "ue")
        .replace("ß", "ss")
    )


def normalize_compound_token(token: str) -> str:
    for suffix in COMPOUND_SUFFIXES:
        if token.endswith(suffix) and len(token) > len(suffix) + 3:
            return token[: -len(suffix)]
    if token.endswith("e") and len(token) > 6:
        token = token[:-1]
    return TOPIC_NORMALIZATION.get(token, token)
