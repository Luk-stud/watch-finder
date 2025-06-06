#!/usr/bin/env python3
"""
Comprehensive Engine Test Suite
===============================
A thorough validation of the LinUCB recommendation engine covering:
- Learning convergence and adaptation
- Temporal consistency 
- Recommendation diversity vs relevance
- Edge case robustness
- Real user journey simulation
- Cross-validation performance
"""

import os
import sys
import pickle
import numpy as np
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import time
import random

# Add backend directory to path
sys.path.append(os.path.dirname(__file__))

from models.optimized_linucb_engine import OptimizedLinUCBEngine

class ComprehensiveTestSuite:
    def __init__(self):
        self.engine = OptimizedLinUCBEngine(dim=50, alpha=0.3, batch_size=5, max_experts=4)
        self.watch_categories = self._load_watch_categories()
        
    def _load_watch_categories(self) -> Dict[str, List[int]]:
        """Load and categorize watches for testing."""
        try:
            with open('data/watch_text_metadata.pkl', 'rb') as f:
                metadata_list = pickle.load(f)
        except:
            return {'general': list(range(50))}  # Fallback
        
        categories = defaultdict(list)
        for idx, watch in enumerate(metadata_list[:100]):
            model_lower = watch.get('model', '').lower()
            brand_lower = watch.get('brand', '').lower()
            
            # Categorize by type
            if any(term in model_lower for term in ['diver', 'dive', 'compressor', 'ocean', 'sea']):
                categories['dive'].append(idx)
            elif any(term in model_lower for term in ['field', 'military', 'pilot', 'tactical']):
                categories['field'].append(idx)
            elif any(term in model_lower for term in ['chrono', 'chronograph', 'timer']):
                categories['chronograph'].append(idx)
            elif any(term in model_lower for term in ['dress', 'classic', 'formal', 'elegant']):
                categories['dress'].append(idx)
            else:
                categories['general'].append(idx)
        
        return dict(categories)

    def test_learning_convergence(self) -> bool:
        """Test: System should learn and improve recommendations over time."""
        print(f"\n🧠 Test: Learning Convergence")
        
        session_id = "convergence_test"
        self.engine.create_session(session_id)
        
        # Define a user preference (like dive watches)
        if 'dive' in self.watch_categories and len(self.watch_categories['dive']) >= 3:
            preferred_category = 'dive'
            preferred_watches = self.watch_categories['dive'][:3]
        else:
            preferred_category = 'general'
            preferred_watches = self.watch_categories['general'][:3]
        
        print(f"   🎯 Simulating user who likes {preferred_category} watches")
        
        # Track recommendation quality over time
        iteration_scores = []
        
        for iteration in range(5):
            # Get recommendations
            recommendations = self.engine.get_recommendations(session_id, np.array([0.5, 0.5]))
            
            # Calculate relevance score (how many recommended watches match user preference)
            relevant_count = 0
            for rec in recommendations:
                watch_id = rec.get('watch_id')
                if watch_id in self.watch_categories[preferred_category]:
                    relevant_count += 1
            
            relevance_score = relevant_count / len(recommendations)
            iteration_scores.append(relevance_score)
            
            # Simulate user feedback (like watches from preferred category)
            feedback_given = 0
            for rec in recommendations:
                watch_id = rec.get('watch_id')
                if watch_id in preferred_watches and feedback_given < 2:
                    self.engine.update(session_id, watch_id, 1.0, np.array([0.5, 0.5]))
                    feedback_given += 1
            
            print(f"     Iteration {iteration + 1}: {relevant_count}/5 relevant recommendations ({relevance_score:.1%})")
        
        # Check if learning improved
        early_avg = np.mean(iteration_scores[:2])
        late_avg = np.mean(iteration_scores[-2:])
        improvement = late_avg - early_avg
        
        print(f"   📈 Learning analysis: {early_avg:.1%} → {late_avg:.1%} (improvement: {improvement:+.1%})")
        
        success = improvement > 0.1  # Expect at least 10% improvement
        print(f"   📊 Result: {'✅ PASS' if success else '❌ FAIL'} - {'Good' if success else 'Poor'} learning convergence")
        return success

    def test_temporal_consistency(self) -> bool:
        """Test: Recommendations should be stable when no new feedback is given."""
        print(f"\n⏰ Test: Temporal Consistency")
        
        session_id = "consistency_test"
        self.engine.create_session(session_id)
        
        # Train the system
        for watch_id in [0, 1, 2]:
            self.engine.update(session_id, watch_id, 1.0, np.array([0.5, 0.5]))
        
        print(f"   🎓 Trained system, now testing consistency...")
        
        # Get multiple recommendation sets without feedback
        recommendation_sets = []
        for i in range(3):
            recs = self.engine.get_recommendations(session_id, np.array([0.5, 0.5]))
            watch_ids = [rec.get('watch_id') for rec in recs]
            recommendation_sets.append(set(watch_ids))
            time.sleep(0.1)  # Small delay
        
        # Calculate overlap between sets
        overlaps = []
        for i in range(len(recommendation_sets)):
            for j in range(i + 1, len(recommendation_sets)):
                overlap = len(recommendation_sets[i] & recommendation_sets[j]) / len(recommendation_sets[i])
                overlaps.append(overlap)
        
        avg_overlap = np.mean(overlaps)
        print(f"   📊 Average recommendation overlap: {avg_overlap:.1%}")
        
        success = avg_overlap > 0.6  # Expect at least 60% consistency
        print(f"   📊 Result: {'✅ PASS' if success else '❌ FAIL'} - {'Good' if success else 'Poor'} temporal consistency")
        return success

    def test_recommendation_diversity(self) -> bool:
        """Test: Recommendations should be diverse while still relevant."""
        print(f"\n🎨 Test: Recommendation Diversity")
        
        session_id = "diversity_test"
        self.engine.create_session(session_id)
        
        # Get recommendations
        recommendations = self.engine.get_recommendations(session_id, np.array([0.5, 0.5]))
        
        # Check brand diversity
        brands = set()
        categories = set()
        watch_ids = set()
        
        for rec in recommendations:
            watch_id = rec.get('watch_id')
            if watch_id is not None:
                brands.add(rec.get('brand', 'Unknown'))
                watch_ids.add(watch_id)
                
                # Determine category
                for cat, watches in self.watch_categories.items():
                    if watch_id in watches:
                        categories.add(cat)
                        break
        
        brand_diversity = len(brands) / len(recommendations)
        category_diversity = len(categories) / min(len(self.watch_categories), len(recommendations))
        
        print(f"   🏷️  Brand diversity: {len(brands)} unique brands ({brand_diversity:.1%})")
        print(f"   📂 Category diversity: {len(categories)} categories ({category_diversity:.1%})")
        print(f"   🔢 Watch ID diversity: {len(watch_ids)} unique watches")
        
        success = brand_diversity > 0.6 and len(watch_ids) == len(recommendations)
        print(f"   📊 Result: {'✅ PASS' if success else '❌ FAIL'} - {'Good' if success else 'Poor'} diversity")
        return success

    def test_edge_case_robustness(self) -> bool:
        """Test: System should handle edge cases gracefully."""
        print(f"\n🛡️  Test: Edge Case Robustness")
        
        edge_cases_passed = 0
        total_edge_cases = 0
        
        # Test 1: Empty session
        try:
            total_edge_cases += 1
            session_id = "empty_test"
            self.engine.create_session(session_id)
            recs = self.engine.get_recommendations(session_id, np.array([0.5, 0.5]))
            if len(recs) == 5:  # Should still return recommendations
                edge_cases_passed += 1
                print(f"   ✅ Empty session handling: PASS")
            else:
                print(f"   ❌ Empty session handling: FAIL")
        except Exception as e:
            print(f"   ❌ Empty session handling: ERROR - {e}")
        
        # Test 2: Extreme context values
        try:
            total_edge_cases += 1
            session_id = "extreme_test"
            self.engine.create_session(session_id)
            recs = self.engine.get_recommendations(session_id, np.array([999.0, -999.0]))
            if len(recs) == 5:
                edge_cases_passed += 1
                print(f"   ✅ Extreme context values: PASS")
            else:
                print(f"   ❌ Extreme context values: FAIL")
        except Exception as e:
            print(f"   ❌ Extreme context values: ERROR - {e}")
        
        # Test 3: Rapid feedback updates
        try:
            total_edge_cases += 1
            session_id = "rapid_test"
            self.engine.create_session(session_id)
            
            # Rapid fire updates
            for i in range(10):
                self.engine.update(session_id, i % 5, 1.0, np.array([0.5, 0.5]))
            
            recs = self.engine.get_recommendations(session_id, np.array([0.5, 0.5]))
            if len(recs) == 5:
                edge_cases_passed += 1
                print(f"   ✅ Rapid feedback updates: PASS")
            else:
                print(f"   ❌ Rapid feedback updates: FAIL")
        except Exception as e:
            print(f"   ❌ Rapid feedback updates: ERROR - {e}")
        
        # Test 4: Conflicting feedback
        try:
            total_edge_cases += 1
            session_id = "conflict_test"
            self.engine.create_session(session_id)
            
            # Give conflicting feedback on same watch
            self.engine.update(session_id, 0, 1.0, np.array([0.5, 0.5]))
            self.engine.update(session_id, 0, 0.0, np.array([0.5, 0.5]))
            self.engine.update(session_id, 0, 1.0, np.array([0.5, 0.5]))
            
            recs = self.engine.get_recommendations(session_id, np.array([0.5, 0.5]))
            if len(recs) == 5:
                edge_cases_passed += 1
                print(f"   ✅ Conflicting feedback: PASS")
            else:
                print(f"   ❌ Conflicting feedback: FAIL")
        except Exception as e:
            print(f"   ❌ Conflicting feedback: ERROR - {e}")
        
        success = edge_cases_passed >= total_edge_cases * 0.75  # 75% pass rate
        print(f"   📊 Result: {edge_cases_passed}/{total_edge_cases} edge cases passed - {'✅ PASS' if success else '❌ FAIL'}")
        return success

    def test_cross_validation_performance(self) -> bool:
        """Test: Cross-validation style test with train/test split."""
        print(f"\n🔄 Test: Cross-Validation Performance")
        
        # Use dive watches as ground truth if available
        if 'dive' in self.watch_categories and len(self.watch_categories['dive']) >= 6:
            target_watches = self.watch_categories['dive']
            category_name = 'dive'
        else:
            target_watches = list(range(min(20, len(self.engine.watch_data))))
            category_name = 'general'
        
        if len(target_watches) < 6:
            print(f"   ⚠️  Not enough watches for cross-validation")
            return False
        
        # Split into train/test
        train_watches = target_watches[:len(target_watches)//2]
        test_watches = target_watches[len(target_watches)//2:]
        
        print(f"   📚 Training on {len(train_watches)} {category_name} watches")
        print(f"   🧪 Testing on {len(test_watches)} {category_name} watches")
        
        session_id = "cv_test"
        self.engine.create_session(session_id)
        
        # Train on first half
        for watch_id in train_watches:
            self.engine.update(session_id, watch_id, 1.0, np.array([0.5, 0.5]))
        
        # Test: How many test watches appear in recommendations?
        recommendations = self.engine.get_recommendations(session_id, np.array([0.5, 0.5]))
        
        test_watches_recommended = 0
        for rec in recommendations:
            if rec.get('watch_id') in test_watches:
                test_watches_recommended += 1
        
        recall = test_watches_recommended / min(len(test_watches), len(recommendations))
        print(f"   📊 Test watch recall: {test_watches_recommended}/{len(recommendations)} ({recall:.1%})")
        
        success = recall > 0.2  # Expect at least 20% recall
        print(f"   📊 Result: {'✅ PASS' if success else '❌ FAIL'} - {'Good' if success else 'Poor'} generalization")
        return success

    def test_real_user_journey_simulation(self) -> bool:
        """Test: Simulate realistic user interaction patterns."""
        print(f"\n👤 Test: Real User Journey Simulation")
        
        session_id = "journey_test"
        self.engine.create_session(session_id)
        
        # Simulate browsing behavior: mostly looking, occasional likes
        interactions = 0
        likes = 0
        
        print(f"   🛒 Simulating user browsing session...")
        
        for round_num in range(4):  # 4 rounds of browsing
            recommendations = self.engine.get_recommendations(session_id, np.array([0.5, 0.5]))
            interactions += len(recommendations)
            
            # Simulate realistic behavior: like 0-2 watches per round
            likes_this_round = min(random.randint(0, 2), len(recommendations))
            
            for i in range(likes_this_round):
                rec = recommendations[i]
                watch_id = rec.get('watch_id')
                if watch_id is not None:
                    self.engine.update(session_id, watch_id, 1.0, np.array([0.5, 0.5]))
                    likes += 1
            
            print(f"     Round {round_num + 1}: Viewed 5 watches, liked {likes_this_round}")
        
        # Final recommendations should be personalized
        final_recs = self.engine.get_recommendations(session_id, np.array([0.5, 0.5]))
        
        engagement_rate = likes / interactions if interactions > 0 else 0
        print(f"   📊 Journey summary: {interactions} views, {likes} likes ({engagement_rate:.1%} engagement)")
        
        # Check that final recommendations are different from initial
        initial_session = "initial_test"
        self.engine.create_session(initial_session)
        initial_recs = self.engine.get_recommendations(initial_session, np.array([0.5, 0.5]))
        
        initial_ids = set(rec.get('watch_id') for rec in initial_recs)
        final_ids = set(rec.get('watch_id') for rec in final_recs)
        
        personalization = 1 - (len(initial_ids & final_ids) / len(initial_ids))
        print(f"   🎯 Personalization: {personalization:.1%} change from initial recommendations")
        
        success = likes > 0 and personalization > 0.2  # Some engagement and 20% personalization
        print(f"   📊 Result: {'✅ PASS' if success else '❌ FAIL'} - {'Realistic' if success else 'Unrealistic'} user journey")
        return success

    def run_comprehensive_tests(self) -> Dict[str, bool]:
        """Run all comprehensive tests and return results."""
        print("🚀 Running Comprehensive Engine Test Suite")
        print("=" * 60)
        
        tests = [
            ('Learning Convergence', self.test_learning_convergence),
            ('Temporal Consistency', self.test_temporal_consistency),
            ('Recommendation Diversity', self.test_recommendation_diversity),
            ('Edge Case Robustness', self.test_edge_case_robustness),
            ('Cross-Validation Performance', self.test_cross_validation_performance),
            ('Real User Journey Simulation', self.test_real_user_journey_simulation)
        ]
        
        results = {}
        for test_name, test_func in tests:
            try:
                start_time = time.time()
                success = test_func()
                duration = time.time() - start_time
                results[test_name] = success
                print(f"   ⏱️  Test completed in {duration:.1f}s")
            except Exception as e:
                print(f"   ❌ Test failed with error: {e}")
                results[test_name] = False
        
        # Summary
        print(f"\n" + "=" * 60)
        print(f"📋 Comprehensive Test Results:")
        passed = sum(1 for success in results.values() if success)
        total = len(results)
        
        for test_name, success in results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"  {test_name:35} | {status}")
        
        print(f"\n🎯 Overall Score: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        
        # Provide assessment
        if passed == total:
            print(f"🌟 EXCELLENT: The system passes all comprehensive tests!")
        elif passed >= total * 0.8:
            print(f"✅ GOOD: The system is working well with minor areas for improvement.")
        elif passed >= total * 0.6:
            print(f"⚠️  FAIR: The system works but has significant areas for improvement.")
        else:
            print(f"❌ POOR: The system needs major improvements before production use.")
        
        return results

def run_comprehensive_tests():
    """Main entry point for comprehensive testing."""
    test_suite = ComprehensiveTestSuite()
    return test_suite.run_comprehensive_tests()

if __name__ == "__main__":
    run_comprehensive_tests() 