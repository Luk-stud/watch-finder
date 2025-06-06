#!/usr/bin/env python3
"""
Test script to debug feedback issue locally
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from models.fast_linucb_engine import FastLinUCBEngine
from models.production_session_manager import ProductionSessionManager

def test_feedback():
    print("🧪 Testing feedback system locally...")
    
    try:
        # Test the engine directly
        print("📦 Loading FastLinUCBEngine...")
        engine = FastLinUCBEngine(data_dir='data')
        print('✅ Engine loaded successfully')

        # Test session manager
        print("📦 Loading ProductionSessionManager...")
        session_manager = ProductionSessionManager(
            data_dir='data',
            linucb_engine=engine
        )
        print('✅ Session manager loaded successfully')

        # Test creating a session
        print("🔄 Creating session...")
        session_id, recs = session_manager.create_session()
        print(f'✅ Session created: {session_id}, got {len(recs)} recommendations')

        # Test feedback
        if recs:
            watch_id = recs[0]['watch_id']
            print(f'🔄 Testing feedback for watch {watch_id}')
            try:
                result = session_manager.update_feedback(session_id, watch_id, True)
                print(f'✅ Feedback result: {result}')
                
                # Test dislike feedback too
                if len(recs) > 1:
                    watch_id_2 = recs[1]['watch_id']
                    print(f'🔄 Testing dislike feedback for watch {watch_id_2}')
                    result2 = session_manager.update_feedback(session_id, watch_id_2, False)
                    print(f'✅ Dislike feedback result: {result2}')
                    
            except Exception as e:
                print(f'❌ Feedback error: {e}')
                import traceback
                traceback.print_exc()
                return False
        else:
            print("❌ No recommendations to test feedback with")
            return False
            
        print("🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_feedback()
    sys.exit(0 if success else 1) 