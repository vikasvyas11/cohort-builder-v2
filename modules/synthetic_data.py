"""Synthetic datasets for the built-in demos.

Nothing here is real, downloaded or derived from third-party data: people are
assembled from the short name / place lists below with a seeded random
generator, so the demos work offline, are reproducible and contain no personal
information.

:func:`build_demo_datasets` builds a small UK-style people file (name, date of
birth, city, email, gender, postcode) in which some people appear several times
with typos, plus a damaged sample of it as Dataset B. The voter-registry demo
lives in :mod:`modules.voter_data`.
"""

from __future__ import annotations

import random
from datetime import date, timedelta

import pandas as pd

from modules.corruption import CorruptionSettings, make_noisy_copy

DEFAULT_SEED = 42
HOUSEHOLD_SHARE = 0.06

FEMALE_NAMES = [
    "Alice", "Amelia", "Anna", "Beatrice", "Bethany", "Caroline", "Charlotte", "Chloe",
    "Clara", "Daisy", "Eleanor", "Elizabeth", "Ella", "Emily", "Emma", "Erin", "Evelyn",
    "Fiona", "Florence", "Freya", "Georgia", "Grace", "Hannah", "Harriet", "Hazel", "Heidi",
    "Imogen", "Isabel", "Isla", "Ivy", "Jade", "Jasmine", "Jessica", "Julia", "Katherine",
    "Laura", "Layla", "Leah", "Lily", "Lucy", "Maisie", "Margaret", "Maria", "Megan",
    "Mia", "Molly", "Naomi", "Nina", "Olivia", "Penelope", "Phoebe", "Poppy", "Rachel",
    "Rebecca", "Rosie", "Ruby", "Samantha", "Sarah", "Sophie", "Tessa", "Victoria", "Zoe",
]
MALE_NAMES = [
    "Aaron", "Adam", "Alexander", "Andrew", "Anthony", "Arthur", "Benjamin", "Callum",
    "Charles", "Christopher", "Daniel", "David", "Dylan", "Edward", "Elliot", "Ethan",
    "Felix", "Finn", "Frederick", "Gabriel", "George", "Harry", "Henry", "Hugo", "Isaac",
    "Jack", "Jacob", "James", "Jason", "Joseph", "Joshua", "Kieran", "Kyle", "Leo", "Liam",
    "Lewis", "Louis", "Lucas", "Luke", "Mark", "Matthew", "Michael", "Nathan", "Nicholas",
    "Noah", "Oliver", "Oscar", "Owen", "Patrick", "Paul", "Peter", "Rhys", "Richard",
    "Robert", "Ryan", "Samuel", "Sebastian", "Stephen", "Thomas", "Timothy", "William",
]
SURNAMES = [
    "Adams", "Allen", "Anderson", "Armstrong", "Bailey", "Baker", "Barnes", "Bell",
    "Bennett", "Berry", "Black", "Booth", "Bradley", "Brooks", "Brown", "Butler", "Campbell",
    "Carter", "Chapman", "Clark", "Clarke", "Cole", "Collins", "Cook", "Cooper", "Cox",
    "Davies", "Davis", "Dawson", "Dixon", "Edwards", "Elliott", "Evans", "Fisher", "Fletcher",
    "Ford", "Foster", "Fox", "Francis", "Fraser", "Gibson", "Gordon", "Graham", "Grant",
    "Gray", "Green", "Griffiths", "Hall", "Hamilton", "Harris", "Harrison", "Hart", "Hayes",
    "Henderson", "Hill", "Holmes", "Hughes", "Hunt", "Hussain", "Jackson", "James", "Jenkins",
    "Johnson", "Jones", "Kelly", "Kennedy", "Khan", "King", "Knight", "Lawrence", "Lee",
    "Lewis", "Lloyd", "Marshall", "Martin", "Mason", "Matthews", "Miller", "Mills", "Mitchell",
    "Moore", "Morgan", "Morris", "Murphy", "Murray", "Nelson", "Nicholson", "Owen", "Palmer",
    "Parker", "Patel", "Payne", "Pearce", "Phillips", "Powell", "Price", "Reid", "Reynolds",
    "Richards", "Roberts", "Robinson", "Rogers", "Rose", "Ross", "Russell", "Ryan", "Saunders",
    "Scott", "Shaw", "Simpson", "Singh", "Smith", "Spencer", "Stevens", "Stewart", "Taylor",
    "Thomas", "Thompson", "Turner", "Walker", "Wallace", "Ward", "Watson", "Webb", "Wells",
    "White", "Williams", "Wilson", "Wood", "Wright", "Young",
]
# city -> postcode area prefix (a plausible but synthetic UK-style postcode is built from it)
UK_CITIES = {
    "Aberdeen": "AB", "Bath": "BA", "Belfast": "BT", "Birmingham": "B", "Bristol": "BS",
    "Cambridge": "CB", "Canterbury": "CT", "Cardiff": "CF", "Coventry": "CV", "Derby": "DE",
    "Dundee": "DD", "Durham": "DH", "Edinburgh": "EH", "Exeter": "EX", "Glasgow": "G",
    "Gloucester": "GL", "Hull": "HU", "Inverness": "IV", "Leeds": "LS", "Leicester": "LE",
    "Lincoln": "LN", "Liverpool": "L", "London": "E", "Manchester": "M", "Newcastle": "NE",
    "Norwich": "NR", "Nottingham": "NG", "Oxford": "OX", "Plymouth": "PL", "Portsmouth": "PO",
    "Preston": "PR", "Reading": "RG", "Sheffield": "S", "Southampton": "SO", "Stirling": "FK",
    "Sunderland": "SR", "Newport": "NP", "Wolverhampton": "WV", "Worcester": "WR", "York": "YO",
}
EMAIL_DOMAINS = ["example.com", "example.org", "example.net", "mail.example", "post.example"]

#: Default damage applied to Dataset B in the demo (per populated value).
DEMO_ERROR_RATES = {
    "first_name": 0.10, "surname": 0.08, "city": 0.08,
    "email": 0.12, "gender": 0.04, "postcode": 0.06,
}
DEMO_MISSING_RATES = {"dob": 0.04, "email": 0.05}
DEMO_FIELD_TYPES = {
    "first_name": "first_name", "surname": "surname", "email": "email",
    "city": "location", "postcode": "postcode", "gender": "gender", "dob": "dob",
}


# =============================================================================
# Demo people file
# =============================================================================

def _random_postcode(area: str, rng: random.Random) -> str:
    letters = "ABDEFGHJLNPQRSTUWXYZ"
    return f"{area}{rng.randint(1, 28)} {rng.randint(1, 9)}{rng.choice(letters)}{rng.choice(letters)}"


def _random_dob(rng: random.Random) -> str:
    start, end = date(1938, 1, 1), date(2006, 12, 31)
    return (start + timedelta(days=rng.randrange((end - start).days))).isoformat()


def _random_person(entity: int, rng: random.Random, household: dict | None = None) -> dict:
    """A new person; if ``household`` is given they live with (and share a surname with) that person."""
    gender = rng.choice(["F", "M"])
    first = rng.choice(FEMALE_NAMES if gender == "F" else MALE_NAMES)
    surname = household["surname"] if household else rng.choice(SURNAMES)
    city = household["city"] if household else rng.choice(list(UK_CITIES))
    email = f"{first}.{surname}{rng.randint(1, 99)}@{rng.choice(EMAIL_DOMAINS)}".lower()
    return {
        "cluster": f"E{entity:05d}", "first_name": first, "surname": surname,
        "dob": _random_dob(rng), "city": city, "email": email, "gender": gender,
        "postcode": household["postcode"] if household else _random_postcode(UK_CITIES[city], rng),
    }


def _copies_for(person: dict, rng: random.Random) -> int:
    """How many records describe this person: most people once, some up to four times."""
    return rng.choices([1, 2, 3, 4], weights=[0.58, 0.24, 0.12, 0.06])[0]


def _lightly_damaged(record: dict, rng: random.Random) -> dict:
    """A repeat sighting of the same person, with the drift real registers show:
    typos, blanks, a transposed birth date, a new email address, a house move."""
    out = dict(record)
    if rng.random() < 0.20:
        out["first_name"] = _swap_letter(out["first_name"], rng)
    if rng.random() < 0.15:
        out["surname"] = _swap_letter(out["surname"], rng)

    roll = rng.random()
    if roll < 0.10:
        out["dob"] = None
    elif roll < 0.17:
        year, month, day = out["dob"].split("-")
        if int(day) <= 12 and day != month:
            out["dob"] = f"{year}-{day}-{month}"        # day and month transposed

    roll = rng.random()
    if roll < 0.15:
        out["email"] = None
    elif roll < 0.35:
        out["email"] = (f"{out['first_name']}.{out['surname']}{rng.randint(1, 99)}"
                        f"@{rng.choice(EMAIL_DOMAINS)}").lower()

    roll = rng.random()
    if roll < 0.10:                                       # moved to another city
        out["city"] = rng.choice(list(UK_CITIES))
        out["postcode"] = _random_postcode(UK_CITIES[out["city"]], rng)
    elif roll < 0.25:                                     # moved within the city
        out["postcode"] = _random_postcode(UK_CITIES[out["city"]], rng)
    return out


def _swap_letter(word: str, rng: random.Random) -> str:
    pos = rng.randrange(len(word))
    return word[:pos] + rng.choice("abcdefghijklmnopqrstuvwxyz") + word[pos + 1:]


def generate_people(n_records: int = 1000, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """About ``n_records`` person records, some of which describe the same person.

    Columns: ``unique_id, first_name, surname, dob, city, email, gender,
    postcode, cluster`` - ``cluster`` is the hidden ground truth (records with
    the same value are the same person).
    """
    rng = random.Random(seed)
    rows: list[dict] = []
    entity = 0
    people: list[dict] = []
    while len(rows) < n_records:
        # ~6% of new people share a household (surname, city, postcode) with an earlier person -
        # relatives who look alike on several fields but are not the same person.
        household = rng.choice(people) if people and rng.random() < HOUSEHOLD_SHARE else None
        person = _random_person(entity, rng, household)
        people.append(person)
        rows.append(person)
        for _ in range(_copies_for(person, rng) - 1):
            rows.append(_lightly_damaged(person, rng))
        entity += 1

    df = pd.DataFrame(rows[:n_records]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df.insert(0, "unique_id", df.index.astype(str))
    return df


def build_demo_datasets(
    n_records: int = 1000,
    seed: int = DEFAULT_SEED,
    b_fraction: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return ``(base, dataset_a, dataset_b)`` for the Standard-mode demo.

    ``dataset_a`` is the base people file labelled ``source_dataset = "A"``;
    ``dataset_b`` is a ``b_fraction`` sample of it damaged at the
    :data:`DEMO_ERROR_RATES` / :data:`DEMO_MISSING_RATES` rates.
    """
    base = generate_people(n_records, seed)
    dataset_a = base.assign(source_dataset="A")
    dataset_b = make_noisy_copy(
        dataset_a, DEMO_ERROR_RATES,
        sample_frac=b_fraction, seed=seed, field_types=DEMO_FIELD_TYPES,
        missing_rates=DEMO_MISSING_RATES, settings=CorruptionSettings(text_edits=1),
    )
    return base, dataset_a, dataset_b
