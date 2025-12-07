# app/services/sop/sop_engine.py
"""
SOP Engine - State Machine for Procedure Verification
Tracks worker progress through SOP steps and validates completion
"""
import json
import time
import os
import logging as log
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

from .config import (
    REQUIRED_TOUCH_FRAMES,
    MIN_WORK_DURATION,
    HAND_OFF_TIMEOUT,
    REPORTS_FOLDER
)

log.basicConfig(
    level=log.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class SOPEngine:
    """
    State machine engine for SOP (Standard Operating Procedure) verification.
    
    States:
    - WAITING_HAND: Waiting for hand to touch the target object
    - WORKING: Hand is actively working on the step
    - COMPLETED: All steps completed
    
    The engine tracks:
    - Current step progress
    - Time spent on each step
    - Completion status
    - Generates reports
    """
    
    def __init__(self, rules_path: str = None, rules_data: Dict = None):
        """
        Initialize SOP Engine.
        
        Args:
            rules_path: Path to SOP rules JSON file
            rules_data: Direct rules dictionary (alternative to file)
        """
        if rules_data:
            self.steps = rules_data.get('steps', [])
            self.process_name = rules_data.get('process_name', 'Unknown Process')
        elif rules_path:
            with open(rules_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.steps = data.get('steps', [])
                self.process_name = data.get('process_name', 'Unknown Process')
        else:
            raise ValueError("Either rules_path or rules_data must be provided")
        
        # Progress tracking
        self.current_step_index = 0
        self.is_completed = False
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_logs = []
        
        # State machine variables
        self.state = "WAITING_HAND"
        self.consecutive_touch_frames = 0
        self.start_work_time = 0
        self.last_touch_time = 0
        self.step_start_clock = time.time()
        
        log.info(f"SOP Engine initialized: {self.process_name} with {len(self.steps)} steps")
    
    def update(self, current_interactions: List[Dict]) -> Dict[str, Any]:
        """
        Update SOP state based on current interactions.
        
        Args:
            current_interactions: List of hand-object interactions from InteractionEngine
            
        Returns:
            Dictionary with current step status for UI display
        """
        if self.is_completed:
            return {
                "step_id": "DONE",
                "step_index": len(self.steps),
                "total_steps": len(self.steps),
                "description": "HOÀN THÀNH",
                "status": "COMPLETED",
                "timer_ratio": 0.0,
                "is_completed": True
            }
        
        if self.current_step_index >= len(self.steps):
            self.is_completed = True
            return self.update(current_interactions)
        
        current_step = self.steps[self.current_step_index]
        target_label = current_step['target_object']
        
        # Check if any hand is touching the target object
        is_being_touched = any(
            inter['item_label'] == target_label 
            for inter in current_interactions
        )
        
        status_msg = f"Chờ thao tác: {current_step.get('description', target_label)}"
        timer_ratio = 0.0
        now = time.time()
        
        # --- STATE MACHINE LOGIC ---
        
        # STATE 1: WAITING FOR HAND
        if self.state == "WAITING_HAND":
            if is_being_touched:
                self.consecutive_touch_frames += 1
                status_msg = f"Phát hiện... ({self.consecutive_touch_frames}/{REQUIRED_TOUCH_FRAMES})"
                timer_ratio = self.consecutive_touch_frames / REQUIRED_TOUCH_FRAMES
                
                # Start working after enough continuous frames
                if self.consecutive_touch_frames >= REQUIRED_TOUCH_FRAMES:
                    self.state = "WORKING"
                    self.start_work_time = now - (REQUIRED_TOUCH_FRAMES * 0.03)
                    self.last_touch_time = now
                    log.info(f"[SOP] START Step {current_step['step_id']} ({target_label})")
            else:
                self.consecutive_touch_frames = 0
        
        # STATE 2: WORKING
        elif self.state == "WORKING":
            work_duration = self.last_touch_time - self.start_work_time
            
            if is_being_touched:
                # Hand still touching - update active time
                self.last_touch_time = now
                status_msg = f"Đang thực hiện... ({work_duration:.1f}s)"
                timer_ratio = 1.0
            else:
                # Hand released - start cooldown
                idle_time = now - self.last_touch_time
                
                if idle_time > HAND_OFF_TIMEOUT:
                    # Cooldown finished - check if step completed
                    if work_duration >= MIN_WORK_DURATION:
                        self._complete_step(current_step, work_duration)
                        status_msg = "PASSED ✓"
                        timer_ratio = 0.0
                    else:
                        # Too quick - reset
                        log.info(f"[SOP] Too short ({work_duration:.1f}s). Reset.")
                        self.state = "WAITING_HAND"
                        self.consecutive_touch_frames = 0
                        status_msg = "Thao tác quá nhanh - Thử lại"
                else:
                    # Still in cooldown
                    countdown = HAND_OFF_TIMEOUT - idle_time
                    status_msg = f"Xác nhận... ({countdown:.1f}s)"
                    timer_ratio = countdown / HAND_OFF_TIMEOUT
        
        return {
            "step_id": current_step['step_id'],
            "step_index": self.current_step_index,
            "total_steps": len(self.steps),
            "description": current_step.get('description', ''),
            "target_object": target_label,
            "status": status_msg,
            "timer_ratio": timer_ratio,
            "state": self.state,
            "is_completed": self.is_completed
        }
    
    def _complete_step(self, step_info: Dict, duration: float):
        """Record step completion and advance to next step."""
        log.info(f"[SUCCESS] Step {step_info['step_id']} Done. Duration: {duration:.2f}s")
        
        self.report_logs.append({
            "step_id": step_info['step_id'],
            "description": step_info.get('description', ''),
            "target_object": step_info['target_object'],
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "duration": round(duration, 2),
            "status": "PASSED"
        })
        
        self.current_step_index += 1
        self.state = "WAITING_HAND"
        self.consecutive_touch_frames = 0
        
        if self.current_step_index >= len(self.steps):
            self.is_completed = True
            log.info("[SOP] All steps completed!")
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current progress summary."""
        return {
            "session_id": self.session_id,
            "process_name": self.process_name,
            "current_step": self.current_step_index + 1,
            "total_steps": len(self.steps),
            "progress_percentage": (self.current_step_index / len(self.steps)) * 100 if self.steps else 0,
            "is_completed": self.is_completed,
            "completed_steps": self.report_logs
        }
    
    def get_report(self) -> Dict[str, Any]:
        """Generate full assessment report."""
        total_time = sum(log['duration'] for log in self.report_logs)
        
        return {
            "session_id": self.session_id,
            "process_name": self.process_name,
            "total_steps": len(self.steps),
            "completed_steps": len(self.report_logs),
            "is_passed": self.is_completed,
            "total_duration": round(total_time, 2),
            "started_at": self.report_logs[0]['timestamp'] if self.report_logs else None,
            "completed_at": self.report_logs[-1]['timestamp'] if self.report_logs else None,
            "step_details": self.report_logs
        }
    
    def save_report(self, folder: str = None) -> str:
        """
        Save assessment report to JSON file.
        
        Returns:
            Path to saved report file
        """
        save_folder = Path(folder) if folder else REPORTS_FOLDER
        save_folder.mkdir(parents=True, exist_ok=True)
        
        filename = f"report_{self.session_id}.json"
        full_path = save_folder / filename
        
        report_data = self.get_report()
        
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=4, ensure_ascii=False)
        
        log.info(f"[SYSTEM] Report saved: {full_path}")
        return str(full_path)
    
    def reset(self):
        """Reset engine for new assessment."""
        self.current_step_index = 0
        self.is_completed = False
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_logs = []
        self.state = "WAITING_HAND"
        self.consecutive_touch_frames = 0
        self.start_work_time = 0
        self.last_touch_time = 0
        self.step_start_clock = time.time()
        log.info("[SOP] Engine reset for new assessment")
