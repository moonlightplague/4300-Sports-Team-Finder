import json
import os
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
], "Eredivisie (Netherlands)": [
    "AFC Ajax",
    "PSV Eindhoven",
    "Feyenoord",
    "AZ Alkmaar",
    "FC Utrecht",
    "SC Heerenveen",
    "FC Twente",
    "Sparta Rotterdam",
    "NEC Nijmegen",
    "PEC Zwolle",
    "Heracles Almelo",
    "FC Groningen",
],
"Primeira Liga (Portugal)": [
    "FC Porto",
    "S.L. Benfica",
    "Sporting CP",
    "SC Braga",
    "Vitória SC",
    "Boavista FC",
    "Rio Ave FC",
    "Gil Vicente FC",
    "FC Famalicão",
    "Portimonense SC",
    "Marítimo",
    "Santa Clara",
],
"Scottish Premiership (Scotland)": [
    "Celtic F.C.",
    "Rangers F.C.",
    "Aberdeen F.C.",
    "Heart of Midlothian F.C.",
    "Hibernian F.C.",
    "Dundee United F.C.",
    "Motherwell F.C.",
    "Kilmarnock F.C.",
    "St Mirren F.C.",
    "St Johnstone F.C.",
],
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
    "Boca Juniors",
    "River Plate",
    "Racing Club (Avellaneda)",
    "Independiente",
    "San Lorenzo de Almagro",
    "Estudiantes de La Plata",
    "Newell's Old Boys",
    "Rosario Central",
    "Vélez Sársfield",
    "Talleres de Córdoba",
    "Huracán",
    "Godoy Cruz Antonio Tomba",
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
], "NCAA Men's Basketball (USA)": [
    "Kentucky Wildcats men's basketball",
    "North Carolina Tar Heels men's basketball",
    "Kansas Jayhawks men's basketball",
    "Duke Blue Devils men's basketball",
    "UCLA Bruins men's basketball",
    "Indiana Hoosiers men's basketball",
    "Louisville Cardinals men's basketball",
    "Syracuse Orange men's basketball",
    "Michigan State Spartans men's basketball",
    "Arizona Wildcats men's basketball",
    "Connecticut Huskies men's basketball",
    "Villanova Wildcats men's basketball",
    "Michigan Wolverines men's basketball",
    "Ohio State Buckeyes men's basketball",
    "Florida Gators men's basketball",
    "Georgetown Hoyas men's basketball",
    "Gonzaga Bulldogs men's basketball",
    "Illinois Fighting Illini men's basketball",
    "Wisconsin Badgers men's basketball",
    "Maryland Terrapins men's basketball",
    "Purdue Boilermakers men's basketball",
    "Houston Cougars men's basketball",
    "Virginia Cavaliers men's basketball",
    "NC State Wolfpack men's basketball",
    "Arkansas Razorbacks men's basketball",
], "NBA G League (USA)": [
    "Austin Spurs",
    "Birmingham Squadron",
    "Capital City Go-Go",
    "Cleveland Charge",
    "College Park Skyhawks",
    "Delaware Blue Coats",
    "Greensboro Swarm",
    "Grand Rapids Gold",
    "Indiana Mad Ants",
    "Iowa Wolves",
    "Long Island Nets",
    "Maine Celtics",
    "Memphis Hustle",
    "Mexico City Capitanes",
    "Motor City Cruise",
    "Oklahoma City Blue",
    "Osceola Magic",
    "Raptors 905",
    "Rio Grande Valley Vipers",
    "Rip City Remix",
    "Salt Lake City Stars",
    "San Diego Clippers",
    "Santa Cruz Warriors",
    "Sioux Falls Skyforce",
    "South Bay Lakers",
    "Stockton Kings",
    "Texas Legends",
    "Valley Suns",
    "Westchester Knicks",
    "Wisconsin Herd",
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
], "NCAA FBS Football (USA)": [
    "Alabama Crimson Tide football",
    "Georgia Bulldogs football",
    "Ohio State Buckeyes football",
    "Michigan Wolverines football",
    "Notre Dame Fighting Irish football",
    "USC Trojans football",
    "Texas Longhorns football",
    "Oklahoma Sooners football",
    "LSU Tigers football",
    "Clemson Tigers football",
    "Florida Gators football",
    "Auburn Tigers football",
    "Penn State Nittany Lions football",
    "Tennessee Volunteers football",
    "Florida State Seminoles football",
    "Miami Hurricanes football",
    "Nebraska Cornhuskers football",
    "Oregon Ducks football",
    "Wisconsin Badgers football",
    "Iowa Hawkeyes football",
    "Michigan State Spartans football",
    "Texas A&M Aggies football",
    "Arkansas Razorbacks football",
    "Washington Huskies football",
    "Ole Miss Rebels football",
], "CFL (Canada)": [
    "BC Lions",
    "Edmonton Elks",
    "Calgary Stampeders",
    "Saskatchewan Roughriders",
    "Winnipeg Blue Bombers",
    "Hamilton Tiger-Cats",
    "Toronto Argonauts",
    "Ottawa Redblacks",
    "Montreal Alouettes",
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
], "NPB (Japan)": [
    "Yomiuri Giants",
    "Hanshin Tigers",
    "Chunichi Dragons",
    "Tokyo Yakult Swallows",
    "Yokohama DeNA BayStars",
    "Hiroshima Toyo Carp",
    "Fukuoka SoftBank Hawks",
    "Hokkaido Nippon-Ham Fighters",
    "Chiba Lotte Marines",
    "Tohoku Rakuten Golden Eagles",
    "Saitama Seibu Lions",
    "Orix Buffaloes",
],
"KBO League (South Korea)": [
    "LG Twins",
    "Kia Tigers",
    "Samsung Lions",
    "Doosan Bears",
    "SSG Landers",
    "Lotte Giants",
    "Hanwha Eagles",
    "NC Dinos",
    "Kiwoom Heroes",
    "KT Wiz",
],}

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
], "KHL (Russia/Eurasia)": [
    "SKA Saint Petersburg",
    "CSKA Moscow",
    "Ak Bars Kazan",
    "Metallurg Magnitogorsk",
    "Avangard Omsk",
    "Lokomotiv Yaroslavl",
    "Dynamo Moscow",
    "Spartak Moscow",
    "Traktor Chelyabinsk",
    "Salavat Yulaev Ufa",
    "Neftekhimik Nizhnekamsk",
    "Torpedo Nizhny Novgorod",
    "Severstal Cherepovets",
    "Barys Astana",
    "Kunlun Red Star",
    "HC Sochi",
    "Amur Khabarovsk",
    "Sibir Novosibirsk",
    "Admiral Vladivostok",
    "Lada Togliatti",
    "Metallurg Novokuznetsk",
    "HC Yugra",
    "Dinamo Riga",
],
"SHL (Sweden)": [
    "Djurgårdens IF Hockey",
    "Frölunda HC",
    "HV71",
    "Luleå HF",
    "Malmö Redhawks",
    "Brynäs IF",
    "Rögle BK",
    "Skellefteå AIK",
    "Linköping HC",
    "MODO Hockey",
    "Färjestad BK",
    "IFK Gothenburg",
    "Örebro HK",
    "IF Björklöven",
],
"Liiga (Finland)": [
    "Jokerit",
    "HIFK",
    "TPS",
    "Tappara",
    "Ilves",
    "Kärpät",
    "Ässät",
    "JYP",
    "KalPa",
    "HPK",
    "Pelicans",
    "SaiPa",
    "Sport Vaasa",
    "Lukko",
    "KooKoo",
],}


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


TEAM_TO_SPORT = {}
for _team in [t for teams in european_soccer_league_to_teams.values() for t in teams]:
    TEAM_TO_SPORT[_team] = "soccer"
for _team in [t for teams in americas_soccer_league_to_teams.values() for t in teams]:
    TEAM_TO_SPORT[_team] = "soccer"
for _team in [t for teams in basketball_teams.values() for t in teams]:
    TEAM_TO_SPORT[_team] = "basketball"
for _team in [t for teams in football.values() for t in teams]:
    TEAM_TO_SPORT[_team] = "football"
for _team in [t for teams in baseball.values() for t in teams]:
    TEAM_TO_SPORT[_team] = "baseball"
for _team in [t for teams in hockey.values() for t in teams]:
    TEAM_TO_SPORT[_team] = "hockey"


TEAM_TO_LEAGUE = {}
for league, teams in european_soccer_league_to_teams.items():
    for team in teams:
        TEAM_TO_LEAGUE[team] = league
for league, teams in americas_soccer_league_to_teams.items():
    for team in teams:
        TEAM_TO_LEAGUE[team] = league
for league, teams in basketball_teams.items():
    for team in teams:
        TEAM_TO_LEAGUE[team] = league
for league, teams in football.items():
    for team in teams:
        TEAM_TO_LEAGUE[team] = league
for league, teams in baseball.items():
    for team in teams:
        TEAM_TO_LEAGUE[team] = league
for league, teams in hockey.items():
    for team in teams:
        TEAM_TO_LEAGUE[team] = league

SUMMARIES_PATH = os.path.join(os.path.dirname(__file__), "data", "team_summaries.json")
TEAM_TO_SUMMARY = {}
if os.path.exists(SUMMARIES_PATH):
    with open(SUMMARIES_PATH, "r", encoding="utf-8") as f:
        summaries_data = json.load(f)
    TEAM_TO_SUMMARY = {team: entry.get("summary", "") for team, entry in summaries_data.items()}


def tokenize(text):
    """
    tokenizes the text
    """
    cleaned = normalize_text(text or "")
    cleaned = re.sub(r'https?://\S+', '', cleaned)
    cleaned = re.sub(r'www\.\S+', '', cleaned)
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

CITY_ALIASES = {
    "ny": "new york",
    "la": "los angeles",
    "sf": "san francisco",
    "dc": "washington",
}


def normalize_query(text):
    text = text.lower()
    for short_name, expanded_name in CITY_ALIASES.items():
        text = re.sub(rf"\b{re.escape(short_name)}\b", expanded_name, text)
    for phrase, replacement in LEAGUE_ALIASES.items():
        if phrase in text:
            text = text.replace(phrase, replacement)
    return text
