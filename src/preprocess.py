import pandas as pd

def clean_and_merge(fixtures, odds):
    odds_lookup = {}
    for entry in odds:
        if not entry.get('bookmakers'):
            continue
        team = entry.get('home_team') or entry.get('teams', {}).get('home')
        if team:
            odds_lookup[entry['home_team']] = entry

    rows = []
    for f in fixtures:
        if f['fixture']['status']['short'] != 'FT':
            continue

        home = f['teams']['home']['name']
        away = f['teams']['away']['name']
        home_goals = f['goals']['home']
        away_goals = f['goals']['away']

        # Buscar odds correspondientes
        odds_entry = next((o for o in odds if o['home_team'] == home and o['away_team'] == away), None)
        if not odds_entry or not odds_entry.get('bookmakers'):
            continue

        try:
            bookmaker = odds_entry['bookmakers'][0]
            outcomes = bookmaker['markets'][0]['outcomes']
            home_odds = next(o['price'] for o in outcomes if o['name'] == home)
            draw_odds = next(o['price'] for o in outcomes if o['name'].lower() == 'draw')
            away_odds = next(o['price'] for o in outcomes if o['name'] == away)
        except Exception:
            continue

        result = 1 if home_goals > away_goals else 0

        rows.append({
            'home_team': home,
            'away_team': away,
            'home_goals': home_goals,
            'away_goals': away_goals,
            'result_home_win': result,
            'home_odds': home_odds,
            'draw_odds': draw_odds,
            'away_odds': away_odds
        })

    df = pd.DataFrame(rows)
    return df.dropna()
