#!/usr/bin/env python3
"""
Test script to demonstrate variant filtering effectiveness and recommendation modes.
Shows before/after statistics, examples of variant detection, and current system modes.
"""

import requests
import json
from collections import defaultdict

def test_variant_filtering():
    """Test and demonstrate variant filtering capabilities and recommendation modes."""
    
    base_url = "http://localhost:5001"
    
    print("🧪 Testing Variant Filtering & Recommendation Modes")
    print("=" * 60)
    
    # 1. Get variant filtering stats
    try:
        response = requests.get(f"{base_url}/api/variant-filtering-stats")
        if response.status_code == 200:
            stats = response.json()['variant_filtering_stats']
            
            print(f"📊 Dataset Statistics:")
            print(f"   Total watches: {stats['total_watches']:,}")
            print(f"   Unique signatures: {stats['unique_signatures']:,}")
            print(f"   Variant reduction: {stats['variant_reduction_percentage']:.1f}%")
            print(f"   Unique brands: {stats['unique_brands']}")
            print(f"   Avg variants per signature: {stats['avg_variants_per_signature']:.1f}")
            
            print(f"\n🔍 Most Common Variants (showing duplicate models):")
            for i, variant in enumerate(stats['most_common_variants'][:10], 1):
                print(f"   {i:2d}. {variant['signature']}: {variant['count']} variants")
            
            print(f"\n📈 Brand Distribution (top 10):")
            brand_items = list(stats['brand_distribution'].items())
            brand_items.sort(key=lambda x: x[1], reverse=True)
            for i, (brand, count) in enumerate(brand_items[:10], 1):
                print(f"   {i:2d}. {brand}: {count} watches")
                
        else:
            print(f"❌ Failed to get variant stats: {response.status_code}")
            return
            
    except Exception as e:
        print(f"❌ Error getting variant stats: {e}")
        return
    
    # 2. Test session creation and recommendation modes
    print(f"\n🚀 Testing Recommendation Modes & Strategy Selection")
    print("-" * 50)
    
    try:
        # Start a new session
        session_response = requests.post(f"{base_url}/api/start-session", 
                                       json={"num_seeds": 7})
        if session_response.status_code == 200:
            session_data = session_response.json()
            session_id = session_data['session_id']
            seeds = session_data['seeds']
            
            print(f"✅ Session created: {session_id}")
            print(f"🌱 Seed watches (showing diversity):")
            brands_seen = set()
            for i, seed in enumerate(seeds, 1):
                brand = seed.get('brand', 'Unknown')
                model = seed.get('model', 'Unknown')
                brands_seen.add(brand)
                print(f"   {i}. {brand} - {model}")
            
            print(f"   → {len(brands_seen)} unique brands in {len(seeds)} seeds ({len(brands_seen)/len(seeds)*100:.1f}% brand diversity)")
            
            # 🆕 Test multiple recommendation rounds to see mode transitions
            print(f"\n🎯 Testing Mode Transitions:")
            print("-" * 30)
            
            liked_indices = []
            
            for round_num in range(1, 4):
                print(f"\n🔄 Round {round_num}:")
                
                # Test recommendations
                rec_response = requests.post(f"{base_url}/api/get-recommendations",
                                           json={
                                               "session_id": session_id,
                                               "liked_indices": liked_indices,
                                               "disliked_indices": [],
                                               "current_candidates": [],
                                               "num_recommendations": 5,
                                               "step": round_num
                                           })
                
                if rec_response.status_code == 200:
                    rec_data = rec_response.json()
                    recommendations = rec_data.get('recommendations', [])
                    
                    # 🆕 SHOW ENHANCED MODE INFORMATION
                    algorithm_used = rec_data.get('algorithm_used', 'unknown')
                    strategy_explanation = rec_data.get('strategy_explanation', 'No explanation available')
                    system_state = rec_data.get('system_state', {})
                    clustering_status = rec_data.get('clustering_status', {})
                    exploration_status = rec_data.get('exploration_status', {})
                    mode_indicators = rec_data.get('mode_indicators', {})
                    
                    print(f"   🧠 Strategy: {algorithm_used.upper()}")
                    print(f"   📝 Explanation: {strategy_explanation}")
                    print(f"   🎯 Current Mode: {system_state.get('current_mode', 'unknown')}")
                    print(f"   🔄 Is Clustering: {'✅' if system_state.get('is_clustering', False) else '❌'}")
                    print(f"   🗺️  Is Exploring: {'✅' if system_state.get('is_exploring', False) else '❌'}")
                    print(f"   💪 Preference Strength: {system_state.get('preference_strength', 'unknown')}")
                    print(f"   🎲 Recommendation Confidence: {system_state.get('recommendation_confidence', 'unknown')}")
                    
                    if clustering_status.get('is_using_clusters', False):
                        print(f"   🎯 Clusters: {clustering_status['cluster_count']} (strength: {clustering_status['total_cluster_strength']:.2f})")
                        for cluster in clustering_status.get('cluster_details', []):
                            print(f"      - Cluster {cluster['cluster_id']}: strength {cluster['strength']:.2f}, {cluster['preference_count']} preferences")
                    
                    print(f"   📊 Exploration: {exploration_status['exploration_percentage']:.1f}% ({exploration_status['seen_watches']}/{exploration_status['total_watches']})")
                    print(f"   💬 Engagement: {exploration_status['engagement_level']}")
                    
                    # Show mode transition readiness
                    print(f"   🚀 Ready for clustering: {'✅' if mode_indicators.get('ready_for_clustering', False) else '❌'}")
                    print(f"   🎯 Strong preferences: {'✅' if mode_indicators.get('strong_preferences_detected', False) else '❌'}")
                    print(f"   🗺️  Exploration recommended: {'✅' if mode_indicators.get('exploration_recommended', False) else '❌'}")
                    
                    print(f"   🎁 Recommendations:")
                    rec_brands = []
                    rec_signatures = []
                    for i, rec in enumerate(recommendations, 1):
                        brand = rec.get('brand', 'Unknown')
                        model = rec.get('model', 'Unknown')
                        signature = f"{brand.lower()} - {model.lower()}"
                        rec_brands.append(brand)
                        rec_signatures.append(signature)
                        print(f"      {i}. {brand} - {model} (Index: {rec.get('index', 'N/A')})")
                    
                    if recommendations:
                        unique_brands = len(set(rec_brands))
                        unique_signatures = len(set(rec_signatures))
                        print(f"      → {unique_brands} unique brands, {unique_signatures} unique signatures")
                        
                        if unique_signatures == len(recommendations):
                            print("      ✅ Perfect variant filtering: No duplicate brand-model combinations!")
                        else:
                            print("      ⚠️ Some duplicate brand-model combinations found")
                    
                    # 🆕 Simulate liking some watches for mode progression
                    if recommendations:
                        # Like the first 2 recommendations
                        new_likes = [rec['index'] for rec in recommendations[:2]]
                        liked_indices.extend(new_likes)
                        print(f"   👍 Simulated likes: {new_likes}")
                        
                        # Add feedback to the system
                        for watch_idx in new_likes:
                            requests.post(f"{base_url}/api/add-feedback", json={
                                "session_id": session_id,
                                "watch_index": watch_idx,
                                "feedback_type": "like",
                                "confidence": 1.0
                            })
                    
                else:
                    print(f"   ❌ Failed to get recommendations: {rec_response.status_code}")
                    break
                    
            print(f"\n📈 Final Session Analysis:")
            print("-" * 25)
            print(f"   Total likes given: {len(liked_indices)}")
            print(f"   Session ID: {session_id}")
            
        else:
            print(f"❌ Failed to create session: {session_response.status_code}")
            
    except Exception as e:
        print(f"❌ Error testing recommendations: {e}")
    
    print(f"\n✅ Enhanced variant filtering and mode analysis completed!")

if __name__ == "__main__":
    test_variant_filtering() 