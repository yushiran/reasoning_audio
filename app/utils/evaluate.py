import json
import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve, auc, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
import glob
import os
import logging

# Configure logger
logger = logging.getLogger(__name__)

class AudioEventEvaluator:
    """
    Evaluates LLM audio event classification against ground truth segments.
    Supports both event-level and temporal-level evaluation.
    """
    
    def __init__(self, predefined_categories: List[str]):
        """
        Initialize evaluator with predefined sound categories.
        
        Args:
            predefined_categories: List of valid sound event categories
        """
        self.categories = predefined_categories
        self.mlb = MultiLabelBinarizer(classes=predefined_categories)
        self.mlb.fit([predefined_categories])
    
    def evaluate_event_classification(self, 
                                    predicted_events: List[str], 
                                    ground_truth_segments: List[Dict]) -> Dict[str, float]:
        """
        Evaluate event-level classification performance.
        
        Args:
            predicted_events: List of predicted event categories
            ground_truth_segments: List of ground truth segments with labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Extract ground truth events from segments
        gt_events = list(set([segment['label'] for segment in ground_truth_segments]))
        # logger.info(f"Ground truth events: {gt_events}")
        # Filter valid categories only
        logger.info(f"Predicted events: {predicted_events}, Ground truth events: {gt_events}")
        predicted_events = [e for e in predicted_events if e in self.categories]
        gt_events = [e for e in gt_events if e in self.categories]
        logger.info(f"Predicted events: {predicted_events}, Ground truth events: {gt_events}")
        # Convert to binary format
        pred_binary = self.mlb.transform([predicted_events])
        gt_binary = self.mlb.transform([gt_events])
        logger.info(f"Predicted binary: {pred_binary}, Ground truth binary: {gt_binary}")
        # Calculate metrics
        f1_micro = f1_score(gt_binary, pred_binary, average='micro')
        f1_macro = f1_score(gt_binary, pred_binary, average='macro')
        f1_weighted = f1_score(gt_binary, pred_binary, average='weighted')
        
        # Event-level accuracy (exact match)
        exact_match = int(set(predicted_events) == set(gt_events))
        
        # Jaccard similarity (IoU for sets)
        jaccard = len(set(predicted_events) & set(gt_events)) / len(set(predicted_events) | set(gt_events)) if (predicted_events or gt_events) else 0
        
        return {
            'f1_micro': f1_micro,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'exact_match': exact_match,
            'jaccard_similarity': jaccard,
            'predicted_count': len(predicted_events),
            'ground_truth_count': len(gt_events)
        }
    
    def evaluate_temporal_alignment(self, 
                                  predicted_events: List[str], 
                                  ground_truth_segments: List[Dict],
                                  audio_duration: float = None) -> Dict[str, float]:
        """
        Evaluate temporal alignment between predictions and ground truth.
        
        Args:
            predicted_events: List of predicted event categories
            ground_truth_segments: List of ground truth segments with timestamps
            audio_duration: Total duration of audio file
            
        Returns:
            Dictionary containing temporal evaluation metrics
        """
        if not ground_truth_segments:
            return {'temporal_coverage': 0.0, 'event_density_match': 0.0}
        
        # Calculate temporal coverage
        total_gt_duration = sum(segment['end'] - segment['start'] for segment in ground_truth_segments)
        if audio_duration:
            temporal_coverage = total_gt_duration / audio_duration
        else:
            temporal_coverage = 1.0  # Assume full coverage if duration unknown
        
        # Event density matching
        gt_event_types = set(segment['label'] for segment in ground_truth_segments)
        pred_event_types = set(predicted_events)
        
        event_density_match = len(pred_event_types & gt_event_types) / len(gt_event_types) if gt_event_types else 0
        
        return {
            'temporal_coverage': temporal_coverage,
            'event_density_match': event_density_match,
            'gt_segment_count': len(ground_truth_segments),
            'gt_unique_events': len(gt_event_types)
        }
    
    def calculate_category_wise_metrics(self, 
                                      results: List[Dict]) -> Dict[str, Dict[str, float]]:
        """
        Calculate per-category performance metrics across multiple samples.
        
        Args:
            results: List of evaluation results from multiple audio files
            
        Returns:
            Dictionary with per-category metrics
        """
        category_metrics = {}
        
        for category in self.categories:
            tp = fp = tn = fn = 0
            
            for result in results:
                predicted = set(result.get('predicted_events', []))
                ground_truth = set(result.get('ground_truth_events', []))
                
                if category in ground_truth and category in predicted:
                    tp += 1
                elif category not in ground_truth and category in predicted:
                    fp += 1
                elif category in ground_truth and category not in predicted:
                    fn += 1
                else:
                    tn += 1
            
            # Calculate metrics for this category
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            category_metrics[category] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'support': tp + fn  # Number of true instances
            }
        
        return category_metrics

    def parse_llm_response(self, response: str) -> List[str]:
        """
        Parse LLM response to extract identified events.
        Handles various response formats including malformed JSON and mixed languages.
        
        Args:
            response: Raw LLM response string
            
        Returns:
            List of identified event categories
        """
        try:
            response = response.strip()
            
            # Handle Python dict format (single quotes) - most common case
            if response.startswith("{'identified_events'"):
                # Use regex to extract the list content safely
                import re
                match = re.search(r"'identified_events':\s*\[(.*?)\]", response, re.DOTALL)
                if match:
                    events_str = match.group(1)
                    # Extract all quoted strings (both single and double quotes)
                    events = re.findall(r"['\"]([^'\"]*)['\"]", events_str)
                    # Filter out empty strings and normalize
                    events = [event.strip() for event in events if event.strip()]
                    # logger.info(f"Parsed {events} events from Python dict format")
                    return events
            
            # Handle proper JSON format
            elif response.startswith('{"identified_events"'):
                data = json.loads(response)
                events = data.get('identified_events', [])
                # logger.info(f"Parsed {events} events from JSON format")
                return events
                
            # Handle cases where response is truncated or malformed
            elif 'identified_events' in response:
                import re
                # Try to find any list-like structure
                match = re.search(r"identified_events['\"]?\s*:\s*\[(.*?)\]", response, re.DOTALL)
                if match:
                    events_str = match.group(1)
                    events = re.findall(r"['\"]([^'\"]*)['\"]", events_str)
                    events = [event.strip() for event in events if event.strip()]
                    # logger.info(f"Parsed {events} events from malformed response")
                    return events
                    
        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {response[:100]}... Error: {e}")
            
        return []

def evaluate_audio_reasoning_pipeline(output_dir: str, 
                                    predefined_categories: List[str]) -> Dict[str, Any]:
    """
    Evaluate the entire audio reasoning pipeline output.
    
    Args:
        output_dir: Directory containing JSON output files
        predefined_categories: List of valid sound categories
        
    Returns:
        Comprehensive evaluation results
    """
    evaluator = AudioEventEvaluator(predefined_categories)
    results = []
    
    # Process all output files
    json_files = glob.glob(os.path.join(output_dir, "*.json"))
    logger.info(f"Found {len(json_files)} JSON files to evaluate")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract data (handle malformed JSON responses)
            predicted_events = data.get('response', '{}')
            if isinstance(predicted_events, str):
                predicted_events = evaluator.parse_llm_response(predicted_events)
            elif isinstance(predicted_events, dict):
                predicted_events = predicted_events.get('identified_events', [])
            
            ground_truth_segments = data.get('segments', [])
            
            # Evaluate this sample
            event_metrics = evaluator.evaluate_event_classification(
                predicted_events, ground_truth_segments
            )
            
            temporal_metrics = evaluator.evaluate_temporal_alignment(
                predicted_events, ground_truth_segments
            )
            
            # Store results
            sample_result = {
                'file': os.path.basename(json_file),
                'predicted_events': predicted_events,
                'ground_truth_events': [s['label'] for s in ground_truth_segments],
                **event_metrics,
                **temporal_metrics
            }
            results.append(sample_result)
            
        except Exception as e:
            logger.error(f"Error processing {json_file}: {e}")
    
    # Calculate overall metrics
    if results:
        overall_metrics = {
            'avg_f1_micro': np.mean([r['f1_micro'] for r in results]),
            'avg_f1_macro': np.mean([r['f1_macro'] for r in results]),
            'avg_f1_weighted': np.mean([r['f1_weighted'] for r in results]),
            'avg_jaccard': np.mean([r['jaccard_similarity'] for r in results]),
            'exact_match_rate': np.mean([r['exact_match'] for r in results]),
            'avg_temporal_coverage': np.mean([r['temporal_coverage'] for r in results]),
            'avg_event_density_match': np.mean([r['event_density_match'] for r in results]),
            'total_samples': len(results)
        }
        
        category_metrics = evaluator.calculate_category_wise_metrics(results)
        
        return {
            'overall_metrics': overall_metrics,
            'category_metrics': category_metrics,
            'sample_results': results
        }
    
    return {'error': 'No valid results found'}

def print_evaluation_summary(results: Dict[str, Any]):
    """
    Print a formatted summary of evaluation results.
    
    Args:
        results: Evaluation results dictionary
    """
    if 'error' in results:
        logger.error(f"Evaluation failed: {results['error']}")
        return
    
    overall = results['overall_metrics']
    category = results['category_metrics']
    
    print("\n" + "="*60)
    print("AUDIO EVENT CLASSIFICATION EVALUATION SUMMARY")
    print("="*60)
    
    print(f"\nOverall Performance:")
    print(f"  Total Samples: {overall['total_samples']}")
    print(f"  F1 Score (Micro): {overall['avg_f1_micro']:.3f}")
    print(f"  F1 Score (Macro): {overall['avg_f1_macro']:.3f}")
    print(f"  F1 Score (Weighted): {overall['avg_f1_weighted']:.3f}")
    print(f"  Jaccard Similarity: {overall['avg_jaccard']:.3f}")
    print(f"  Exact Match Rate: {overall['exact_match_rate']:.3f}")
    print(f"  Temporal Coverage: {overall['avg_temporal_coverage']:.3f}")
    print(f"  Event Density Match: {overall['avg_event_density_match']:.3f}")
    
    print(f"\nTop 10 Best Performing Categories:")
    # Sort categories by F1 score
    sorted_categories = sorted(category.items(), key=lambda x: x[1]['f1_score'], reverse=True)
    for i, (cat, metrics) in enumerate(sorted_categories[:10]):
        if metrics['support'] > 0:  # Only show categories with actual ground truth instances
            print(f"  {i+1:2d}. {cat:<20} | F1: {metrics['f1_score']:.3f} | P: {metrics['precision']:.3f} | R: {metrics['recall']:.3f} | Support: {metrics['support']}")
    
    print(f"\nBottom 10 Performing Categories:")
    for i, (cat, metrics) in enumerate(sorted_categories[-10:]):
        if metrics['support'] > 0:
            print(f"  {i+1:2d}. {cat:<20} | F1: {metrics['f1_score']:.3f} | P: {metrics['precision']:.3f} | R: {metrics['recall']:.3f} | Support: {metrics['support']}")
    
    print("="*60)

def evaluate_results(output_dir: str, save_results: bool = True):
    """
    Evaluate the audio reasoning pipeline results.
    
    Args:
        output_dir: Directory containing output JSON files
        save_results: Whether to save evaluation results to file
    """
    logger.info(f"Starting evaluation of results in {output_dir}")
    
    # Define predefined categories according to your prompt template
    predefined_categories = [
        "Speech", "Children playing", "Laughter", "Shout or scream", "Footsteps",
        "Instrumental music", "Singing", "Amplified music", "Bird sounds", "Insects",
        "Crickets", "Dog", "Cat", "Poultry", "Horse", "Cows mooing", "Car", "Truck",
        "Bus", "Train", "Tram", "Metro", "Plane", "Helicopter", "Boat", "Bell",
        "Siren", "Alarm", "Vehicle horn", "Reverse beeper", "Explosion", "Gunshot",
        "Jackhammer", "Drill", "Crane/Bulldozer", "Flowing water", "Falling water", "Waves"
    ]
    
    # Run evaluation
    results = evaluate_audio_reasoning_pipeline(output_dir, predefined_categories)
    
    # Print summary
    print_evaluation_summary(results)
    
    # Save evaluation results
    if save_results and 'error' not in results:
        eval_output_path = os.path.join(output_dir, "evaluation_results.json")
        with open(eval_output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Evaluation results saved to {eval_output_path}")
        
        # Save summary statistics
        summary_path = os.path.join(output_dir, "evaluation_summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("Audio Event Classification Evaluation Summary\n")
            f.write("="*50 + "\n\n")
            
            overall = results['overall_metrics']
            f.write(f"Overall Performance:\n")
            f.write(f"  Total Samples: {overall['total_samples']}\n")
            f.write(f"  F1 Score (Micro): {overall['avg_f1_micro']:.3f}\n")
            f.write(f"  F1 Score (Macro): {overall['avg_f1_macro']:.3f}\n")
            f.write(f"  F1 Score (Weighted): {overall['avg_f1_weighted']:.3f}\n")
            f.write(f"  Jaccard Similarity: {overall['avg_jaccard']:.3f}\n")
            f.write(f"  Exact Match Rate: {overall['exact_match_rate']:.3f}\n")
            f.write(f"  Temporal Coverage: {overall['avg_temporal_coverage']:.3f}\n")
            f.write(f"  Event Density Match: {overall['avg_event_density_match']:.3f}\n")
        
        logger.info(f"Evaluation summary saved to {summary_path}")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate audio event classification results")
    parser.add_argument("--output_dir", type=str, required=True,
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
    evaluate_results(args.output_dir, args.save_results)