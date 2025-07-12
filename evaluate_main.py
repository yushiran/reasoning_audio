#!/usr/bin/env python3
"""
Standalone evaluation script for audio event classification results.
Usage: python evaluate_standalone.py --output_dir outputs/07_08_classification
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.evaluate import evaluate_results
import logging

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate audio event classification results")
    parser.add_argument("--output_dir", default='outputs/07_08_classification',type=str,
                        help="Directory containing output JSON files to evaluate")
    parser.add_argument("--save_results", action="store_true", default=True,
                        help="Save evaluation results to file")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run evaluation
    try:
        results = evaluate_results(args.output_dir, args.save_results)
        print("\n✅ Evaluation completed successfully!")
        
        if 'overall_metrics' in results:
            overall = results['overall_metrics']
            print(f"🎯 Key Results:")
            print(f"   F1 Score (Micro): {overall['avg_f1_micro']:.3f}")
            print(f"   F1 Score (Macro): {overall['avg_f1_macro']:.3f}")
            print(f"   Exact Match Rate: {overall['exact_match_rate']:.3f}")
            print(f"   Jaccard Similarity: {overall['avg_jaccard']:.3f}")
            print(f"   Total Samples: {overall['total_samples']}")
            
    except Exception as e:
        print(f"❌ Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
