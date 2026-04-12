import re
import unicodedata
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

european_soccer_league_to_teams = {
    "Premier League (England)": [
        "Arsenal F.C.",
        "Aston Villa F.C.",
        "AFC Bournemouth",
        "Brentford F.C.",
        "Brighton & Hove Albion F.C.",
        "Burnley F.C.",
        "Chelsea F.C.",
        "Crystal Palace F.C.",
        "Everton F.C.",
        "Fulham F.C.",
        "Leeds United F.C.",
        "Liverpool F.C.",
        "Manchester City F.C.",
        "Manchester United F.C.",
        "Newcastle United F.C.",
        "Nottingham Forest F.C.",
        "Sunderland A.F.C.",
        "Tottenham Hotspur F.C.",
        "West Ham United F.C.",
        "Wolverhampton Wanderers F.C."
    ], "La Liga (Spain)": [
    "Athletic Club",
    "Atlético Madrid",
    "CA Osasuna",
    "Deportivo Alavés",
    "Elche CF",
    "FC Barcelona",
    "Getafe CF",
    "Girona FC",
    "Levante UD",
    "Rayo Vallecano",
    "RC Celta de Vigo",
    "RCD Espanyol",
    "RCD Mallorca",
    "Real Betis",
    "Real Madrid CF",
    "Real Sociedad",
    "Real Valladolid",
    "Sevilla FC",
    "UD Las Palmas",
    "Villarreal CF"
], "Serie A (Italy)": [
    "Atalanta BC",
    "Bologna FC 1909",
    "Cagliari Calcio",
    "Como 1907",
    "US Cremonese",
    "ACF Fiorentina",
    "Genoa CFC",
    "Hellas Verona FC",
    "Inter Milan",
    "Juventus FC",
    "SS Lazio",
    "US Lecce",
    "AC Milan",
    "SSC Napoli",
    "Parma Calcio 1913",
    "Pisa SC",
    "AS Roma",
    "US Sassuolo Calcio",
    "Torino FC",
    "Udinese Calcio"
], "Bundesliga (Germany)": [
    "FC Bayern Munich",
    "Borussia Dortmund",
    "RB Leipzig",
    "Bayer 04 Leverkusen",
    "Eintracht Frankfurt",
    "VfB Stuttgart",
    "VfL Wolfsburg",
    "Borussia Mönchengladbach",
    "TSG 1899 Hoffenheim",
    "1. FC Union Berlin",
    "SC Freiburg",
    "FC Augsburg",
    "1. FSV Mainz 05",
    "1. FC Köln",
    "VfL Bochum",
    "SV Werder Bremen",
    "1. FC Heidenheim",
    "FC St. Pauli"
], "Ligue 1 (France)": [
    "Paris Saint-Germain",
    "Olympique de Marseille",
    "AS Monaco FC",
    "LOSC Lille",
    "Olympique Lyonnais",
    "Stade Rennais FC",
    "OGC Nice",
    "RC Lens",
    "RC Strasbourg Alsace",
    "Stade Brestois 29",
    "Toulouse FC",
    "FC Lorient",
    "Angers SCO",
    "Le Havre AC",
    "Paris FC",
    "AJ Auxerre",
    "FC Nantes",
    "Montpellier HSC"
]
}


americas_soccer_league_to_teams = {
    "Brasileirão Série A (Brazil)": [
        "Athletico Paranaense",
        "Atlético Goianiense",
        "Atlético Mineiro",
        "Bahia",
        "Botafogo",
        "Corinthians",
        "Criciúma",
        "Cruzeiro",
        "Cuiabá",
        "Flamengo",
        "Fluminense",
        "Fortaleza",
        "Grêmio",
        "Internacional",
        "Juventude",
        "Palmeiras",
        "Red Bull Bragantino",
        "São Paulo FC",
        "Vasco da Gama",
        "Vitória"
    ],

    "Liga Profesional de Fútbol (Argentina)": [
        "Argentinos Juniors",
        "Atlético Tucumán",
        "Banfield",
        "Barracas Central",
        "Belgrano",
        "Boca Juniors",
        "Central Córdoba (Santiago del Estero)",
        "Defensa y Justicia",
        "Deportivo Riestra",
        "Estudiantes de La Plata",
        "Gimnasia y Esgrima La Plata",
        "Godoy Cruz",
        "Huracán",
        "Independiente",
        "Independiente Rivadavia",
        "Instituto",
        "Lanús",
        "Newell's Old Boys",
        "Platense",
        "Racing Club",
        "River Plate",
        "Rosario Central",
        "San Lorenzo",
        "Sarmiento",
        "Talleres de Córdoba",
        "Tigre",
        "Unión de Santa Fe",
        "Vélez Sarsfield",
        "Aldosivi",
        "San Martín de San Juan"
    ],
    "Liga MX (Mexico)": [
        "Club América",
        "Atlas F.C.",
        "Atlético San Luis",
        "Club Tijuana",
        "C.F. Monterrey",
        "Club León",
        "Cruz Azul",
        "C.D. Guadalajara",
        "FC Juárez",
        "Mazatlán F.C.",
        "Club Necaxa",
        "CF Pachuca",
        "Club Puebla",
        "Querétaro F.C.",
        "Santos Laguna",
        "Tigres UANL",
        "Deportivo Toluca F.C.",
        "UNAM Pumas"
    ],
    "Major League Soccer (USA/Canada)": [
        "Atlanta United FC",
        "Austin FC",
        "CF Montréal",
        "Charlotte FC",
        "Chicago Fire FC",
        "Colorado Rapids",
        "Columbus Crew",
        "D.C. United",
        "FC Cincinnati",
        "FC Dallas",
        "Houston Dynamo FC",
        "Inter Miami CF",
        "LA Galaxy",
        "Los Angeles FC",
        "Minnesota United FC",
        "Nashville SC",
        "New England Revolution",
        "New York City FC",
        "New York Red Bulls",
        "Orlando City SC",
        "Philadelphia Union",
        "Portland Timbers",
        "Real Salt Lake",
        "San Jose Earthquakes",
        "Seattle Sounders FC",
        "Sporting Kansas City",
        "St. Louis City SC",
        "Toronto FC",
        "Vancouver Whitecaps FC",
        "San Diego FC"
    ],
    "Categoría Primera A (Colombia)": [
        "América de Cali",
        "Atlético Bucaramanga",
        "Atlético Nacional",
        "Boyacá Chicó",
        "Deportes Tolima",
        "Deportivo Cali",
        "Deportivo Pasto",
        "Deportivo Pereira",
        "Envigado FC",
        "Fortaleza CEIF",
        "Independiente Medellín",
        "Independiente Santa Fe",
        "Jaguares de Córdoba",
        "Junior FC",
        "La Equidad",
        "Millonarios FC",
        "Once Caldas",
        "Patriotas Boyacá",
        "Águilas Doradas",
        "Alianza FC"
    ]
}

basketball_teams = {
    "NBA (USA/Canada)": [
    "Atlanta Hawks",
    "Boston Celtics",
    "Brooklyn Nets",
    "Charlotte Hornets",
    "Chicago Bulls",
    "Cleveland Cavaliers",
    "Dallas Mavericks",
    "Denver Nuggets",
    "Detroit Pistons",
    "Golden State Warriors",
    "Houston Rockets",
    "Indiana Pacers",
    "Los Angeles Clippers",
    "Los Angeles Lakers",
    "Memphis Grizzlies",
    "Miami Heat",
    "Milwaukee Bucks",
    "Minnesota Timberwolves",
    "New Orleans Pelicans",
    "New York Knicks",
    "Oklahoma City Thunder",
    "Orlando Magic",
    "Philadelphia 76ers",
    "Phoenix Suns",
    "Portland Trail Blazers",
    "Sacramento Kings",
    "San Antonio Spurs",
    "Toronto Raptors",
    "Utah Jazz",
    "Washington Wizards"
], "EuroLeague (Europe)": [
    "Anadolu Efes S.K.",
    "AS Monaco Basket",
    "KK Crvena zvezda",
    "Dubai Basketball",
    "Olimpia Milano",
    "FC Barcelona Bàsquet",
    "FC Bayern Munich (basketball)",
    "Fenerbahçe S.K. (basketball)",
    "Hapoel Tel Aviv B.C.",
    "Saski Baskonia",
    "ASVEL Basket",
    "Maccabi Tel Aviv B.C.",
    "Olympiacos B.C.",
    "Panathinaikos B.C.",
    "Paris Basketball",
    "KK Partizan"
], "WNBA (USA/Canada)": [
    "Atlanta Dream",
    "Chicago Sky",
    "Connecticut Sun",
    "Dallas Wings",
    "Golden State Valkyries",
    "Indiana Fever",
    "Las Vegas Aces",
    "Los Angeles Sparks",
    "Minnesota Lynx",
    "New York Liberty",
    "Phoenix Mercury",
    "Portland Fire",
    "Seattle Storm",
    "Toronto Tempo",
    "Washington Mystics"
]
}

football = {"NFL (USA)": [
    "Arizona Cardinals",
    "Atlanta Falcons",
    "Baltimore Ravens",
    "Buffalo Bills",
    "Carolina Panthers",
    "Chicago Bears",
    "Cincinnati Bengals",
    "Cleveland Browns",
    "Dallas Cowboys",
    "Denver Broncos",
    "Detroit Lions",
    "Green Bay Packers",
    "Houston Texans",
    "Indianapolis Colts",
    "Jacksonville Jaguars",
    "Kansas City Chiefs",
    "Las Vegas Raiders",
    "Los Angeles Chargers",
    "Los Angeles Rams",
    "Miami Dolphins",
    "Minnesota Vikings",
    "New England Patriots",
    "New Orleans Saints",
    "New York Giants",
    "New York Jets",
    "Philadelphia Eagles",
    "Pittsburgh Steelers",
    "San Francisco 49ers",
    "Seattle Seahawks",
    "Tampa Bay Buccaneers",
    "Tennessee Titans",
    "Washington Commanders"
]}

baseball = {"MLB (USA/Canada)": [
    "Arizona Diamondbacks",
    "Atlanta Braves",
    "Baltimore Orioles",
    "Boston Red Sox",
    "Chicago Cubs",
    "Chicago White Sox",
    "Cincinnati Reds",
    "Cleveland Guardians",
    "Colorado Rockies",
    "Detroit Tigers",
    "Houston Astros",
    "Kansas City Royals",
    "Los Angeles Angels",
    "Los Angeles Dodgers",
    "Miami Marlins",
    "Milwaukee Brewers",
    "Minnesota Twins",
    "New York Mets",
    "New York Yankees",
    "Oakland Athletics",
    "Philadelphia Phillies",
    "Pittsburgh Pirates",
    "San Diego Padres",
    "San Francisco Giants",
    "Seattle Mariners",
    "St. Louis Cardinals",
    "Tampa Bay Rays",
    "Texas Rangers",
    "Toronto Blue Jays",
    "Washington Nationals"
]}

hockey = {"NHL (USA/Canada)": [
    "Anaheim Ducks",
    "Arizona Coyotes",
    "Boston Bruins",
    "Buffalo Sabres",
    "Calgary Flames",
    "Carolina Hurricanes",
    "Chicago Blackhawks",
    "Colorado Avalanche",
    "Columbus Blue Jackets",
    "Dallas Stars",
    "Detroit Red Wings",
    "Edmonton Oilers",
    "Florida Panthers",
    "Los Angeles Kings",
    "Minnesota Wild",
    "Montreal Canadiens",
    "Nashville Predators",
    "New Jersey Devils",
    "New York Islanders",
    "New York Rangers",
    "Ottawa Senators",
    "Philadelphia Flyers",
    "Pittsburgh Penguins",
    "San Jose Sharks",
    "Seattle Kraken",
    "St. Louis Blues",
    "Tampa Bay Lightning",
    "Toronto Maple Leafs",
    "Utah Hockey Club",
    "Vancouver Canucks",
    "Vegas Golden Knights",
    "Washington Capitals",
    "Winnipeg Jets"
]}


REGEX = re.compile(r"[a-z0-9_]+")

SAFEWORDS = {
    "united", "city", "real", "national", "new", "first",
    "fire", "top", "eleven",
}
STOPWORDS = set(ENGLISH_STOP_WORDS - SAFEWORDS)

def normalize_text(text):
    """
    normalizes the text
    """
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    return text.lower()

def build_multiWord_team_or_league_to_single_token(league_dicts):
    """Maps multi-word team or league names to single tokens."""
    phrases = {}
    for league_dict in league_dicts:
        for teams in league_dict.values():
            for team in teams:
                normalized_text = normalize_text(team)
                words = REGEX.findall(normalized_text)
                if len(words) >= 2:
                    phrase = " ".join(words)
                    token = "_".join(words)
                    phrases[phrase] = token
                    for i in range(len(words) - 1, 1, -1):
                        sub = words[:i]
                        sub_phrase = " ".join(sub)
                        sub_token = "_".join(sub)
                        if sub_phrase not in phrases:
                            phrases[sub_phrase] = sub_token
    singleTokens = dict(sorted(phrases.items(), key=lambda x: -len(x[0])))
    return singleTokens

MULTIWORDS = build_multiWord_team_or_league_to_single_token([
    european_soccer_league_to_teams,
    americas_soccer_league_to_teams,
    basketball_teams,
    football,
    baseball,
    hockey,
])


def tokenize(text):
    """
    tokenizes the text
    """
    cleaned = normalize_text(text or "")
    for phrase, token in MULTIWORDS.items():
        cleaned = cleaned.replace(phrase, token)
    tokens = REGEX.findall(cleaned)
    return [t for t in tokens if t not in STOPWORDS]

def normalize_leagues(league_dicts):
    alias_map = {}
    for league_dict in league_dicts:
        for league_name in league_dict.keys():
            clean = league_name.lower()
            base = clean.split("(")[0].strip()
            key = base.replace(" ", "").replace("-", "")
            alias_map[base] = key
            alias_map[clean] = key
            alias_map[base.replace(" ", "")] = key
            alias_map[base.replace(" ", " ")] = key
            if "premier league" in base:
                alias_map["epl"] = key
                alias_map["english premier league"] = key
            if "major league soccer" in base:
                alias_map["mls"] = key

    return alias_map


LEAGUE_ALIASES = dict(sorted(
    normalize_leagues([
        european_soccer_league_to_teams,
        americas_soccer_league_to_teams,
        basketball_teams,
        football,
        baseball,
        hockey,
    ]).items(),
    key=lambda x: -len(x[0])
))


def normalize_query(text):
    text = text.lower()
    for phrase, replacement in LEAGUE_ALIASES.items():
        if phrase in text:
            text = text.replace(phrase, replacement)
    return text