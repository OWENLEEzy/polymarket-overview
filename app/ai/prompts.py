SEARCH_PLANNING_PROMPT = """Return 3 to 6 Polymarket search queries for the topic.
Use related aliases and event phrasing. Return JSON: {"queries": ["..."]}.
"""

SYNTHESIZE_OVERVIEW_PROMPT = """You are a prediction market analyst. Distill the market data into one or two sentences. Use conditional language when appropriate ("if X happens…"). Flag high uncertainty when anomalies are present. Give the conclusion directly — do not explain the method. Answer in the same language as the topic.
"""
