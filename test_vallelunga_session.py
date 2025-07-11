import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'watch_finder_v2', 'backend'))
from models.simple_sgd_engine import SimpleSgdEngine

# IDs for Autodromo Vallelunga
vallelunga_ids = {'watch_13', 'watch_187', 'watch_262'}

engine = SimpleSgdEngine()
session_id = 'test_vallelunga_session'

print(f"Starting test session: {session_id}")
seen_vallelunga = []

# Simulate 10 recommendation rounds
for round in range(10):
    recs = engine.get_recommendations(session_id)
    print(f"\nRound {round+1}: {len(recs)} recommendations")
    for rec in recs:
        wid = rec['watch_id']
        brand = rec['brand']
        model = rec['model']
        print(f"  - {brand} {model} (ID: {wid})")
        if wid in vallelunga_ids:
            seen_vallelunga.append(wid)
            # Simulate feedback (like)
            engine.update(session_id, wid, 1.0)
            print(f"    -> Gave feedback for {wid}")
            break  # Only give feedback for one per round
    else:
        # No Vallelunga found in this round
        # Simulate feedback for the first recommendation
        if recs:
            engine.update(session_id, recs[0]['watch_id'], 0.0)

print(f"\nSeen Autodromo Vallelunga watches in session: {seen_vallelunga}")
if len(seen_vallelunga) > 1:
    print("❌ ERROR: Multiple Vallelunga variants were recommended in the same session!")
elif len(seen_vallelunga) == 1:
    print("✅ SUCCESS: Only one Vallelunga variant was recommended in the session.")
else:
    print("ℹ️ No Vallelunga variants were recommended in this session.") 