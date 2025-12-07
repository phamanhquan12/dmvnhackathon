# app/services/sop/interaction.py
"""
Interaction Engine for SOP Assessment
Detects hand-object interactions (grabbing, holding, releasing)
"""
import math
from typing import List, Dict, Any
from .config import GRAB_DISTANCE_THRESHOLD


class InteractionEngine:
    """
    Detects interactions between hands and objects.
    Uses pinch point (between thumb and index finger) to detect grabbing.
    """
    
    def __init__(self):
        pass
    
    def calculate_distance(self, p1: tuple, p2: tuple) -> float:
        """Calculate Euclidean distance between two points."""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def process(self, objects: List[Dict], hands: List[Dict]) -> List[Dict[str, Any]]:
        """
        Detect interactions between hands and objects.
        
        Args:
            objects: List of detected objects with bbox and label
            hands: List of detected hands with landmarks
            
        Returns:
            List of current interactions (hand holding object)
        """
        current_interactions = []
        
        if not hands or not objects:
            return current_interactions
        
        for hand in hands:
            # Get fingertip positions
            index_tip = hand['index_tip']
            thumb_tip = hand['thumb_tip']
            
            # Calculate pinch point (midpoint between thumb and index finger)
            px = int((index_tip[0] + thumb_tip[0]) / 2)
            py = int((index_tip[1] + thumb_tip[1]) / 2)
            
            for obj in objects:
                # Get object bounding box
                x1, y1, x2, y2 = obj['bbox']
                
                # Check if pinch point is inside object bbox (with padding)
                padding = 20
                is_inside_x = (x1 - padding) < px < (x2 + padding)
                is_inside_y = (y1 - padding) < py < (y2 + padding)
                
                if is_inside_x and is_inside_y:
                    current_interactions.append({
                        "hand_id": hand['id'],
                        "hand_type": hand['type'],
                        "item_label": obj['label'],
                        "item_id": obj['id'],
                        "status": "HOLDING",
                        "pinch_point": (px, py),
                        "distance": 0
                    })
        
        return current_interactions
    
    def get_pinch_point(self, hand: Dict) -> tuple:
        """Get the pinch point (interaction point) for a hand."""
        index_tip = hand['index_tip']
        thumb_tip = hand['thumb_tip']
        return (
            int((index_tip[0] + thumb_tip[0]) / 2),
            int((index_tip[1] + thumb_tip[1]) / 2)
        )
    
    def is_pinching(self, hand: Dict, threshold: float = 50) -> bool:
        """Check if hand is in pinching position."""
        index_tip = hand['index_tip']
        thumb_tip = hand['thumb_tip']
        distance = self.calculate_distance(index_tip, thumb_tip)
        return distance < threshold
