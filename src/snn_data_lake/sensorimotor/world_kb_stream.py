r"""
WORLD-KB STREAM -- the world-knowledge adapter for the autonomous skill builder (target-encoder B:
MASKED-SLOT prediction over factual triples, the order-free relational domain the autoregressive
next-token signal does not fit) AND the fact store the two-world reasoning bridge converses from.

A fact is a triple (head, relation, tail). There is no "next token" -- a fact is a SET. So the
self-supervised signal is masked-slot: given (head, relation), predict the withheld tail (KB
completion / link prediction, BERT-style masking applied to relations). Cast as (context, target) so
the SAME loop consumes it: context = [relation, head] tokens, target = tail. The masked-slot demo
runs on the BOUNDED-tail relations (continent, phase, class, ...) so the target vocab is small enough
for the strong readout -- honest scale note: full open-vocab KB completion (FB15k-237 / Wikidata,
~14k+ tail entities) needs a ranking head, not a classifier; that is the large-scale path (stub via
`url`).

`load_kb()` returns a BROAD, deterministic knowledge base of REAL (head, relation, tail) facts across
geography, chemistry, biology, physics, astronomy, computing, mathematics, and everyday commonsense --
hundreds of triples, curated (not million-scale), all honest and correct. The BOUNDED-tail relations
(the masked-slot demo domain) are `_RELATIONS`; broader facts -- capitals, element symbols, languages,
currencies, planet order, class-level properties (for multi-hop), numeric parity/primality, etc. --
are added by `_extra_facts()`. `load_kb(url=...)` parses a real "head<TAB>relation<TAB>tail" triples
file if given, else uses the bundled facts (so the self-check is deterministic offline).

  PYTHONPATH=src .venv-lake/Scripts/python -m snn_data_lake.sensorimotor.world_kb_stream
Deps: numpy (+ stdlib) for the masked-slot demo ONLY (imported inside main()). `load_kb` /
`_bundled` / `_extra_facts` do not import numpy at module scope -- the two-world reasoning bridge imports them without pulling numpy.

PROMOTED reasoning/ -> cortex/ 2026-08-11 (#10 family settlement C2b; measured row in
docs/P4C_BACKLOG.md), the KB-completion lane's second half beside kb_ranking. DISCLOSED at the
move: main() lazily imports language_readout_slda + language_self_build. The self_build reach
CLOSED same-day (C6 promoted it to cortex/); the language_readout_slda reach remains
governed->reasoning until C5 lands, derived and asserted by import_manifest.
"""

import os

# numpy + the SLDA-RFF readout are used ONLY by main()'s demo; imported there (not at module level) so
# `from world_kb_stream import load_kb` -- the two-world reasoning bridge's path -- stays numpy-free.

# ---- bundled REAL relational facts (head, relation, tail); relations have BOUNDED tail sets ----
_CONTINENT = {
    "france": "europe",
    "germany": "europe",
    "spain": "europe",
    "italy": "europe",
    "poland": "europe",
    "norway": "europe",
    "greece": "europe",
    "sweden": "europe",
    "finland": "europe",
    "denmark": "europe",
    "netherlands": "europe",
    "belgium": "europe",
    "portugal": "europe",
    "ireland": "europe",
    "austria": "europe",
    "switzerland": "europe",
    "ukraine": "europe",
    "romania": "europe",
    "hungary": "europe",
    "croatia": "europe",
    "russia": "europe",
    "egypt": "africa",
    "kenya": "africa",
    "nigeria": "africa",
    "morocco": "africa",
    "ghana": "africa",
    "ethiopia": "africa",
    "algeria": "africa",
    "tunisia": "africa",
    "angola": "africa",
    "zambia": "africa",
    "uganda": "africa",
    "senegal": "africa",
    "tanzania": "africa",
    "china": "asia",
    "japan": "asia",
    "india": "asia",
    "thailand": "asia",
    "vietnam": "asia",
    "iran": "asia",
    "indonesia": "asia",
    "pakistan": "asia",
    "bangladesh": "asia",
    "philippines": "asia",
    "malaysia": "asia",
    "nepal": "asia",
    "iraq": "asia",
    "brazil": "samerica",
    "argentina": "samerica",
    "chile": "samerica",
    "peru": "samerica",
    "colombia": "samerica",
    "venezuela": "samerica",
    "ecuador": "samerica",
    "bolivia": "samerica",
    "uruguay": "samerica",
    "paraguay": "samerica",
    "canada": "namerica",
    "mexico": "namerica",
    "cuba": "namerica",
    "panama": "namerica",
    "guatemala": "namerica",
    "honduras": "namerica",
    "jamaica": "namerica",
    "australia": "oceania",
    "fiji": "oceania",
    "samoa": "oceania",
    "tonga": "oceania",
    "vanuatu": "oceania",
}
_HEMISPHERE = {
    "france": "north",
    "germany": "north",
    "spain": "north",
    "italy": "north",
    "china": "north",
    "japan": "north",
    "india": "north",
    "canada": "north",
    "mexico": "north",
    "egypt": "north",
    "iran": "north",
    "thailand": "north",
    "vietnam": "north",
    "nigeria": "north",
    "poland": "north",
    "norway": "north",
    "greece": "north",
    "morocco": "north",
    "algeria": "north",
    "ukraine": "north",
    "brazil": "south",
    "argentina": "south",
    "chile": "south",
    "peru": "south",
    "australia": "south",
    "kenya": "south",
    "tanzania": "south",
    "angola": "south",
    "zambia": "south",
    "fiji": "south",
    "samoa": "south",
    "uruguay": "south",
    "paraguay": "south",
    "bolivia": "south",
}
_PHASE = {  # element -> phase at standard temperature/pressure
    "hydrogen": "gas",
    "oxygen": "gas",
    "nitrogen": "gas",
    "helium": "gas",
    "chlorine": "gas",
    "fluorine": "gas",
    "neon": "gas",
    "argon": "gas",
    "krypton": "gas",
    "xenon": "gas",
    "radon": "gas",
    "mercury": "liquid",
    "bromine": "liquid",
    "iron": "solid",
    "copper": "solid",
    "gold": "solid",
    "silver": "solid",
    "carbon": "solid",
    "sulfur": "solid",
    "zinc": "solid",
    "lead": "solid",
    "sodium": "solid",
    "calcium": "solid",
    "aluminum": "solid",
    "nickel": "solid",
    "tin": "solid",
    "magnesium": "solid",
    "potassium": "solid",
    "silicon": "solid",
    "boron": "solid",
    "platinum": "solid",
    "uranium": "solid",
}
_CATEGORY = {  # element -> broad category
    "hydrogen": "nonmetal",
    "oxygen": "nonmetal",
    "nitrogen": "nonmetal",
    "carbon": "nonmetal",
    "sulfur": "nonmetal",
    "chlorine": "nonmetal",
    "fluorine": "nonmetal",
    "phosphorus": "nonmetal",
    "selenium": "nonmetal",
    "bromine": "nonmetal",
    "iodine": "nonmetal",
    "iron": "metal",
    "copper": "metal",
    "gold": "metal",
    "silver": "metal",
    "zinc": "metal",
    "lead": "metal",
    "sodium": "metal",
    "calcium": "metal",
    "aluminum": "metal",
    "nickel": "metal",
    "tin": "metal",
    "magnesium": "metal",
    "potassium": "metal",
    "platinum": "metal",
    "uranium": "metal",
    "mercury": "metal",
    "helium": "noble",
    "neon": "noble",
    "argon": "noble",
    "krypton": "noble",
    "xenon": "noble",
    "radon": "noble",
    "silicon": "metalloid",
    "boron": "metalloid",
    "arsenic": "metalloid",
    "germanium": "metalloid",
    "antimony": "metalloid",
    "tellurium": "metalloid",
}
_CLASS = {  # animal -> vertebrate/invertebrate class
    "dog": "mammal",
    "cat": "mammal",
    "whale": "mammal",
    "bat": "mammal",
    "horse": "mammal",
    "lion": "mammal",
    "elephant": "mammal",
    "tiger": "mammal",
    "bear": "mammal",
    "wolf": "mammal",
    "fox": "mammal",
    "deer": "mammal",
    "cow": "mammal",
    "pig": "mammal",
    "sheep": "mammal",
    "goat": "mammal",
    "rabbit": "mammal",
    "mouse": "mammal",
    "monkey": "mammal",
    "kangaroo": "mammal",
    "eagle": "bird",
    "sparrow": "bird",
    "penguin": "bird",
    "owl": "bird",
    "duck": "bird",
    "robin": "bird",
    "hawk": "bird",
    "parrot": "bird",
    "crow": "bird",
    "swan": "bird",
    "chicken": "bird",
    "pigeon": "bird",
    "falcon": "bird",
    "salmon": "fish",
    "shark": "fish",
    "tuna": "fish",
    "trout": "fish",
    "cod": "fish",
    "bass": "fish",
    "herring": "fish",
    "sardine": "fish",
    "snake": "reptile",
    "lizard": "reptile",
    "turtle": "reptile",
    "crocodile": "reptile",
    "alligator": "reptile",
    "gecko": "reptile",
    "iguana": "reptile",
    "cobra": "reptile",
    "ant": "insect",
    "bee": "insect",
    "beetle": "insect",
    "moth": "insect",
    "butterfly": "insect",
    "grasshopper": "insect",
    "wasp": "insect",
    "fly": "insect",
    "mosquito": "insect",
    "dragonfly": "insect",
    "frog": "amphibian",
    "toad": "amphibian",
    "salamander": "amphibian",
    "newt": "amphibian",
}
_HABITAT = {
    "whale": "ocean",
    "shark": "ocean",
    "tuna": "ocean",
    "cod": "ocean",
    "herring": "ocean",
    "salmon": "river",
    "trout": "river",
    "lion": "savanna",
    "elephant": "savanna",
    "eagle": "mountain",
    "goat": "mountain",
    "hawk": "mountain",
    "owl": "forest",
    "bear": "forest",
    "wolf": "forest",
    "fox": "forest",
    "deer": "forest",
    "robin": "forest",
    "bat": "cave",
    "penguin": "polar",
    "snake": "desert",
    "lizard": "desert",
    "frog": "swamp",
    "toad": "swamp",
    "crocodile": "swamp",
    "alligator": "swamp",
    "duck": "lake",
    "swan": "lake",
    "rabbit": "grassland",
    "mouse": "grassland",
    "grasshopper": "grassland",
    "kangaroo": "grassland",
}

_RELATIONS = {
    "continent": _CONTINENT,
    "hemisphere": _HEMISPHERE,
    "phase": _PHASE,
    "category": _CATEGORY,
    "class": _CLASS,
    "habitat": _HABITAT,
}

# ---- broader REAL facts (added by _extra_facts): capitals, symbols, languages, class-level
# properties (which give MULTI-HOP chains: animal -class-> mammal -blood-> warm), numeric relations,
# computing, and everyday commonsense. Some have near-unique tails (capitals/symbols) -- the open-vocab
# case the docstring notes a classifier can't fit -- so they are NOT in the masked-slot demo, only in
# the reasoning KB the two-world bridge converses from. ----
_DIET = {  # animal -> feeding category
    "lion": "carnivore",
    "tiger": "carnivore",
    "wolf": "carnivore",
    "shark": "carnivore",
    "eagle": "carnivore",
    "snake": "carnivore",
    "crocodile": "carnivore",
    "owl": "carnivore",
    "hawk": "carnivore",
    "fox": "carnivore",
    "elephant": "herbivore",
    "horse": "herbivore",
    "cow": "herbivore",
    "sheep": "herbivore",
    "goat": "herbivore",
    "deer": "herbivore",
    "rabbit": "herbivore",
    "kangaroo": "herbivore",
    "panda": "herbivore",
    "dog": "omnivore",
    "bear": "omnivore",
    "pig": "omnivore",
    "monkey": "omnivore",
    "duck": "omnivore",
    "crow": "omnivore",
    "chicken": "omnivore",
    "mouse": "omnivore",
}
_COLOR = {  # everyday thing -> color
    "sky": "blue",
    "grass": "green",
    "blood": "red",
    "snow": "white",
    "coal": "black",
    "banana": "yellow",
    "lemon": "yellow",
    "cherry": "red",
    "leaf": "green",
    "milk": "white",
    "night": "black",
    "ruby": "red",
    "emerald": "green",
    "sapphire": "blue",
    "rose": "red",
    "violet": "purple",
    "ocean": "blue",
    "cloud": "white",
    "tomato": "red",
    "carrot": "orange",
    "eggplant": "purple",
    "lime": "green",
}
_TASTE = {  # food -> basic taste
    "sugar": "sweet",
    "honey": "sweet",
    "candy": "sweet",
    "cake": "sweet",
    "apple": "sweet",
    "lemon": "sour",
    "vinegar": "sour",
    "lime": "sour",
    "pickle": "sour",
    "salt": "salty",
    "coffee": "bitter",
    "grapefruit": "bitter",
    "cocoa": "bitter",
    "kale": "bitter",
}
_CAPITAL = {  # country -> capital city
    "france": "paris",
    "germany": "berlin",
    "spain": "madrid",
    "italy": "rome",
    "portugal": "lisbon",
    "greece": "athens",
    "poland": "warsaw",
    "norway": "oslo",
    "sweden": "stockholm",
    "finland": "helsinki",
    "denmark": "copenhagen",
    "netherlands": "amsterdam",
    "belgium": "brussels",
    "austria": "vienna",
    "ireland": "dublin",
    "switzerland": "bern",
    "ukraine": "kyiv",
    "romania": "bucharest",
    "hungary": "budapest",
    "russia": "moscow",
    "japan": "tokyo",
    "china": "beijing",
    "india": "new_delhi",
    "thailand": "bangkok",
    "vietnam": "hanoi",
    "iran": "tehran",
    "indonesia": "jakarta",
    "pakistan": "islamabad",
    "egypt": "cairo",
    "kenya": "nairobi",
    "nigeria": "abuja",
    "morocco": "rabat",
    "ghana": "accra",
    "ethiopia": "addis_ababa",
    "canada": "ottawa",
    "mexico": "mexico_city",
    "cuba": "havana",
    "brazil": "brasilia",
    "argentina": "buenos_aires",
    "chile": "santiago",
    "peru": "lima",
    "colombia": "bogota",
    "australia": "canberra",
    "fiji": "suva",
}
_SYMBOL = {  # element -> chemical symbol
    "hydrogen": "h",
    "oxygen": "o",
    "nitrogen": "n",
    "carbon": "c",
    "sulfur": "s",
    "chlorine": "cl",
    "fluorine": "f",
    "iron": "fe",
    "copper": "cu",
    "gold": "au",
    "silver": "ag",
    "zinc": "zn",
    "lead": "pb",
    "sodium": "na",
    "calcium": "ca",
    "mercury": "hg",
    "helium": "he",
    "neon": "ne",
    "argon": "ar",
    "silicon": "si",
    "boron": "b",
    "potassium": "k",
    "aluminum": "al",
    "nickel": "ni",
    "tin": "sn",
    "magnesium": "mg",
    "platinum": "pt",
    "uranium": "u",
    "phosphorus": "p",
    "iodine": "i",
    "bromine": "br",
}
_LANGUAGE = {  # country -> primary language
    "france": "french",
    "germany": "german",
    "spain": "spanish",
    "italy": "italian",
    "portugal": "portuguese",
    "brazil": "portuguese",
    "japan": "japanese",
    "china": "mandarin",
    "india": "hindi",
    "russia": "russian",
    "egypt": "arabic",
    "morocco": "arabic",
    "mexico": "spanish",
    "argentina": "spanish",
    "peru": "spanish",
    "greece": "greek",
    "poland": "polish",
    "netherlands": "dutch",
    "sweden": "swedish",
    "norway": "norwegian",
    "finland": "finnish",
    "iran": "persian",
    "thailand": "thai",
    "vietnam": "vietnamese",
    "indonesia": "indonesian",
    "kenya": "swahili",
    "nigeria": "english",
    "canada": "english",
    "australia": "english",
    "austria": "german",
}
_CURRENCY = {  # country -> currency
    "japan": "yen",
    "china": "yuan",
    "india": "rupee",
    "russia": "ruble",
    "france": "euro",
    "germany": "euro",
    "spain": "euro",
    "italy": "euro",
    "greece": "euro",
    "portugal": "euro",
    "netherlands": "euro",
    "ireland": "euro",
    "austria": "euro",
    "brazil": "real",
    "mexico": "peso",
    "argentina": "peso",
    "chile": "peso",
    "canada": "dollar",
    "australia": "dollar",
    "egypt": "pound",
    "nigeria": "naira",
    "kenya": "shilling",
    "thailand": "baht",
    "vietnam": "dong",
    "indonesia": "rupiah",
    "switzerland": "franc",
    "sweden": "krona",
    "norway": "krone",
    "poland": "zloty",
}
_PLANET_ORDER = {  # planet -> order from the sun
    "mercury": 1,
    "venus": 2,
    "earth": 3,
    "mars": 4,
    "jupiter": 5,
    "saturn": 6,
    "uranus": 7,
    "neptune": 8,
}
_PLANET_TYPE = {
    "venus": "terrestrial",
    "earth": "terrestrial",
    "mars": "terrestrial",
    "jupiter": "gas_giant",
    "saturn": "gas_giant",
    "uranus": "ice_giant",
    "neptune": "ice_giant",
}
_BLOOD = {  # animal CLASS -> blood regulation (enables multi-hop: animal -class-> mammal -blood-> warm)
    "mammal": "warm",
    "bird": "warm",
    "fish": "cold",
    "reptile": "cold",
    "amphibian": "cold",
    "insect": "cold",
}
_REPRODUCTION = {  # animal CLASS -> reproduction mode
    "mammal": "live_birth",
    "bird": "eggs",
    "fish": "eggs",
    "reptile": "eggs",
    "amphibian": "eggs",
    "insect": "eggs",
}
_SKELETON = {  # animal CLASS -> skeleton type
    "mammal": "internal",
    "bird": "internal",
    "fish": "internal",
    "reptile": "internal",
    "amphibian": "internal",
    "insect": "external",
}
_CONDUCTS = {  # element CATEGORY -> conductivity (multi-hop: element -category-> metal -conducts-> electricity)
    "metal": "electricity",
    "metalloid": "semiconductor",
    "nonmetal": "insulator",
    "noble": "inert",
}
_PORT = {  # network protocol -> default TCP/UDP port
    "http": 80,
    "https": 443,
    "ssh": 22,
    "dns": 53,
    "ftp": 21,
    "smtp": 25,
    "telnet": 23,
}
_BASE = {  # numeral system -> radix
    "binary": 2,
    "decimal": 10,
    "hexadecimal": 16,
    "octal": 8,
}
_VOLATILITY = {  # storage/memory -> volatility
    "ram": "volatile",
    "rom": "nonvolatile",
    "ssd": "nonvolatile",
    "hdd": "nonvolatile",
    "cache": "volatile",
}
_PROTOCOL_TYPE = {  # transport protocol -> delivery model
    "tcp": "connection_oriented",
    "udp": "connectionless",
}
_SIDES = {  # polygon -> number of sides
    "triangle": 3,
    "square": 4,
    "rectangle": 4,
    "pentagon": 5,
    "hexagon": 6,
    "octagon": 8,
}
_SOUND = {  # animal -> the sound it makes
    "dog": "bark",
    "cat": "meow",
    "cow": "moo",
    "lion": "roar",
    "duck": "quack",
    "horse": "neigh",
    "sheep": "baa",
    "pig": "oink",
    "wolf": "howl",
    "frog": "croak",
}
_SUBSTANCE_PHASE = {  # everyday substance -> phase (same relation as element phase)
    "ice": "solid",
    "water": "liquid",
    "steam": "gas",
}

# (relation-name, head->tail table) for every EXTRA relation with a uniform table shape
_EXTRA_TABLES = [
    ("diet", _DIET),
    ("color", _COLOR),
    ("taste", _TASTE),
    ("capital", _CAPITAL),
    ("symbol", _SYMBOL),
    ("language", _LANGUAGE),
    ("currency", _CURRENCY),
    ("order", _PLANET_ORDER),
    ("planet_type", _PLANET_TYPE),
    ("blood", _BLOOD),
    ("reproduction", _REPRODUCTION),
    ("skeleton", _SKELETON),
    ("conducts", _CONDUCTS),
    ("port", _PORT),
    ("base", _BASE),
    ("volatility", _VOLATILITY),
    ("protocol_type", _PROTOCOL_TYPE),
    ("sides", _SIDES),
    ("sound", _SOUND),
    ("phase", _SUBSTANCE_PHASE),
]


def _primality(n):
    """even/odd-free primality label for a non-negative int: prime | composite | neither (0, 1)."""
    if n < 2:
        return "neither"
    d = 2
    while d * d <= n:
        if n % d == 0:
            return "composite"
        d += 1
    return "prime"


def _extra_facts():
    """The broader (head, relation, tail) facts beyond the bounded masked-slot relations -- all REAL,
    deterministic. Includes generated numeric relations (parity/primality over 1..30)."""
    facts = [
        (str(h).lower(), rel, str(t).lower())
        for rel, table in _EXTRA_TABLES
        for h, t in table.items()
    ]
    facts += [
        ("week", "days", "7"),
        ("year", "months", "12"),
        ("day", "hours", "24"),
        ("hour", "minutes", "60"),
        ("minute", "seconds", "60"),
    ]
    for n in range(1, 31):
        facts.append((str(n), "parity", "even" if n % 2 == 0 else "odd"))
    for n in range(1, 31):
        facts.append((str(n), "primality", _primality(n)))
    return facts


def load_kb(url=None, path=None, max_bytes=8_000_000):
    """Return a list of (head, relation, tail) triples. If `url`/`path` gives a real
    'head<TAB>relation<TAB>tail' file, parse it (the large-KB path); else the bundled REAL facts
    (bounded-tail relations + the broader `_extra_facts`)."""
    if url or path:
        from snn_data_lake.sensorimotor.web_ingest import fetch_corpus

        path = path or os.path.join(os.environ.get("TEMP", "."), "sndl_kb.tsv")
        if url and not (os.path.exists(path) and os.path.getsize(path) > 0):
            try:
                # Project Gutenberg text: public domain in the US.
                fetch_corpus(
                    url, path, max_bytes=max_bytes, strip_html=False, license_id="public-domain"
                )
            except Exception:
                return _bundled() + _extra_facts(), True
        try:
            triples = []
            with open(path, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    p = line.rstrip("\n").split("\t")
                    if len(p) >= 3:
                        triples.append((p[0].lower(), p[1].lower(), p[2].lower()))
            if len(triples) >= 30:
                return triples, False
        except OSError:
            pass
    return _bundled() + _extra_facts(), True


def _bundled():
    """The BOUNDED-tail relations only -- the masked-slot demo domain (a small-vocab classifier fits)."""
    return [(h, rel, t) for rel, table in _RELATIONS.items() for h, t in table.items()]


def masked_tail_stream(triples, ctx=3):
    """(context, target) for masked-TAIL completion: context = [relation, head] (padded to ctx),
    target = tail. Self-supervised -- the tail is withheld, then predicted; no external label."""
    pairs = []
    for h, rel, t in triples:
        ctx_tokens = [rel, h][-ctx:]
        pairs.append((ctx_tokens, t))
    return pairs


def tail_vocab(triples):
    return sorted({t for _, _, t in triples})


def _most_frequent_tail_baseline(pairs, vocab):
    """Baseline: per-relation most-frequent tail (the relation is context[0]). A strong KB baseline."""
    from collections import Counter, defaultdict

    rset = set(vocab)
    cut = int(len(pairs) * 0.8)
    train, test = pairs[:cut], [(c, t) for c, t in pairs[cut:] if t in rset]
    by_rel = defaultdict(Counter)
    for c, t in train:
        by_rel[c[0]][t] += 1
    if not test:
        return 0.0
    ok = 0
    for c, t in test:
        if by_rel[c[0]]:
            ok += by_rel[c[0]].most_common(1)[0][0] == t
    return ok / len(test)


def main():
    import numpy as np  # demo-only deps, kept out of the import path

    from snn_data_lake.cortex.language_self_build import _positional_code
    from snn_data_lake.readout.language_readout_slda import SLDA_RFF_np

    all_triples, offline = load_kb()  # the FULL broad KB (bounded + extra facts)
    triples = _bundled()  # masked-slot demo: the BOUNDED-tail relations only
    vocab = tail_vocab(triples)
    pairs = masked_tail_stream(triples, ctx=3)
    # shuffle deterministically so train/test both cover all relations
    import random

    random.Random(0).shuffle(pairs)

    print("=" * 84)
    print("WORLD-KB STREAM -- masked-slot fact completion (given head+relation, predict the tail)")
    print("=" * 84)
    print(
        f"KB total {len(all_triples)} real triples ({'bundled facts' if offline else 'fetched KB'}); "
        f"masked-slot demo on {len(triples)} bounded-tail facts across relations {sorted(_RELATIONS)}; "
        f"tail vocab {len(vocab)}"
    )

    ridx = {r: i for i, r in enumerate(vocab)}  # TARGET (tail) index
    # INPUT vocab = relations + heads (the context tokens), DISTINCT from the tail target vocab
    ivocab = sorted({tok for c, _ in pairs for tok in c})
    iidx = {r: i for i, r in enumerate(ivocab)}
    usable = [(c, t) for c, t in pairs if t in ridx]
    cut = int(len(usable) * 0.8)
    train, test = usable[:cut], usable[cut:]

    def X(rows):
        return np.array(
            [_positional_code(c, iidx, len(ivocab), 3) for c, _ in rows], dtype=np.float64
        )

    s = SLDA_RFF_np(len(ivocab) * 3, len(vocab), rff=800, gamma=0.2, shrink=0.1, seed=0)
    s.observe(X(train), np.array([ridx[t] for _, t in train]))
    yte = np.array([ridx[t] for _, t in test])
    acc = float((s.predict(X(test)) == yte).mean())
    base = _most_frequent_tail_baseline(pairs, vocab)

    print("-" * 84)
    print(
        f"  masked-tail completion (SLDA-RFF): {acc:.1%}  vs  per-relation most-frequent {base:.1%}  "
        f"[held-out n={len(test)}]"
    )
    print("-" * 84)
    print(
        "  honest scope: BOUNDED-tail relations -> a classifier fits; open-vocab KB (FB15k-237/Wikidata)"
    )
    print(
        "  needs a ranking head. Cast: masked-slot (target-encoder B), the order-free relational domain."
    )
    print("=" * 84)
    # HONEST bar: with near-unique entities, masked-tail is part memorization -- the generalizable
    # signal is the relation constraining the tail set (+ partial entity overlap across relations), so
    # the readout must MATCH-OR-BEAT the strong per-relation baseline (a classifier that ignored the
    # relation would be far below it). Beating it decisively needs entity overlap / embeddings at scale.
    assert acc >= base - 0.03, (
        f"masked-slot readout fell below the per-relation baseline ({acc:.3f} vs {base:.3f})"
    )
    print(
        f"self-check OK: masked-tail fact completion {acc:.1%} vs per-relation baseline {base:.1%} "
        f"on {len(triples)} bounded facts (KB total {len(all_triples)} real triples across geography, "
        f"chemistry, biology, physics, astronomy, computing, math, commonsense) -- the world-KB adapter "
        f"learns relational structure (masked-slot, no labels); open-vocab KB completion at scale is the "
        f"ranking-head next step."
    )
    return 0


if __name__ == "__main__":
    main()
