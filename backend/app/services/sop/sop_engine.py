# app/services/sop/sop_engine.py
"""
SOP Engine - State Machine for Procedure Verification
Tracks worker progress through SOP steps and validates completion
Includes wrong order detection and timeout handling
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
    REPORTS_FOLDER,
    MAX_STEP_IDLE_TIME,
    FATAL_ERROR_WAIT_TIME,
    WRONG_STEP_TOLERANCE,
    ERROR_TYPE_WRONG_ORDER,
    ERROR_TYPE_TIMEOUT,
    ERROR_TYPE_MISSED_STEP
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
    - FATAL_ERROR: A fatal error occurred (wrong order, timeout)
    - COMPLETED: All steps completed
    
    The engine tracks:
    - Current step progress
    - Time spent on each step
    - Wrong order detection with tolerance
    - Timeout detection
    - Error logs and generates reports
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
            self.max_idle_time = rules_data.get('max_step_idle_time', MAX_STEP_IDLE_TIME)
        elif rules_path:
            with open(rules_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.steps = data.get('steps', [])
                self.process_name = data.get('process_name', 'Unknown Process')
                self.max_idle_time = data.get('max_step_idle_time', MAX_STEP_IDLE_TIME)
        else:
            raise ValueError("Either rules_path or rules_data must be provided")
        
        # Build step lookup for fast access
        self._build_step_lookup()
        
        # Progress tracking
        self.current_step_index = 0
        self.is_completed = False
        self.is_failed = False
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_logs = []
        self.error_logs = []
        
        # State machine variables
        self.state = "WAITING_HAND"
        self.consecutive_touch_frames = 0
        self.start_work_time = 0
        self.last_touch_time = 0
        self.step_start_clock = time.time()
        self.error_timer = 0  # Timer for TOO_SHORT warning display
        
        # Wrong order detection (not used in strict mode - immediately triggers fatal)
        self.wrong_step_counter = 0
        self.last_wrong_step = None
        
        # Failed steps tracking
        self.failed_steps = []  # List of steps that failed (wrong order, timeout, etc.)
        
        # Fatal error handling (matching GitHub SOPvn implementation)
        self.fatal_error_mode = False
        self.fatal_error_time = 0
        self.fatal_error_type = None
        self.fatal_error_details = None
        self.fatal_instruction = []
        
        log.info(f"SOP Engine initialized: {self.process_name} with {len(self.steps)} steps")
    
    def _build_step_lookup(self):
        """Build lookup dictionaries for fast step access."""
        self.step_by_target = {}
        self.step_by_index = {}
        for idx, step in enumerate(self.steps):
            target = step.get('target_object', '')
            self.step_by_target[target] = {
                'index': idx,
                'step': step
            }
            self.step_by_index[idx] = step
    
    def update(self, current_interactions: List[Dict]) -> Dict[str, Any]:
        """
        Update SOP state based on current interactions.
        
        Args:
            current_interactions: List of hand-object interactions from InteractionEngine
            
        Returns:
            Dictionary with current step status for UI display
        """
        now = time.time()
        
        # STATE: FATAL ERROR MODE (matching GitHub SOPvn implementation)
        if self.fatal_error_mode:
            elapsed = now - self.fatal_error_time
            remaining = FATAL_ERROR_WAIT_TIME - elapsed
            
            if remaining <= 0:
                # Signal to exit/terminate the session
                return {
                    "signal": "EXIT_APP",
                    "status": "SESSION TERMINATED",
                    "error": "FATAL",
                    "step_id": self.steps[self.current_step_index]['step_id'] if self.current_step_index < len(self.steps) else "ERROR",
                    "step_index": self.current_step_index,
                    "total_steps": len(self.steps),
                    "description": f"LỖI NGHIÊM TRỌNG: {self.fatal_error_details}",
                    "timer_ratio": 0.0,
                    "state": "FATAL_ERROR",
                    "error_type": self.fatal_error_type,
                    "is_completed": False,
                    "is_failed": True
                }
            
            # Show fatal error modal with countdown
            return {
                "signal": "SHOW_FATAL_ERROR",
                "error_details": self.fatal_error_details,
                "instructions": self.fatal_instruction,
                "countdown": remaining,
                "step_id": self.steps[self.current_step_index]['step_id'] if self.current_step_index < len(self.steps) else "ERROR",
                "step_index": self.current_step_index,
                "total_steps": len(self.steps),
                "description": f"LỖI: {self.fatal_error_details}",
                "status": f"Hệ thống sẽ thoát sau: {remaining:.1f}s",
                "timer_ratio": remaining / FATAL_ERROR_WAIT_TIME,
                "state": "FATAL_ERROR",
                "error_type": self.fatal_error_type,
                "is_completed": False,
                "is_failed": True
            }
        
        # STATE: COMPLETED
        if self.is_completed:
            return {
                "signal": "DONE",
                "step_id": "DONE",
                "step_index": len(self.steps),
                "total_steps": len(self.steps),
                "description": "HOÀN THÀNH",
                "status": "COMPLETED",
                "timer_ratio": 0.0,
                "is_completed": True,
                "is_failed": False
            }
        
        if self.current_step_index >= len(self.steps):
            self.is_completed = True
            return self.update(current_interactions)
        
        current_step = self.steps[self.current_step_index]
        target_label = current_step['target_object']
        
        # Check for timeout
        idle_time = now - self.step_start_clock
        if self.state == "WAITING_HAND" and idle_time > self.max_idle_time:
            self._trigger_fatal_error(
                error_type=ERROR_TYPE_TIMEOUT,
                details=f"Quá thời gian chờ bước {current_step['step_id']} ({self.max_idle_time}s)",
                step_info=current_step
            )
            return self.update(current_interactions)
        
        # --- LOGIC BÌNH THƯỜNG (Following GitHub SOPdone) ---
        current_step = self.steps[self.current_step_index]
        target_label = current_step['target_object']
        
        # 1. CHECK SAI VẬT THỂ (WRONG OBJECT) -> GÂY LỖI NGHIÊM TRỌNG (IMMEDIATELY)
        # Any interaction with wrong object triggers fatal error right away
        wrong_interactions = [inter['item_label'] for inter in current_interactions if inter['item_label'] != target_label]
        if wrong_interactions:
            wrong_obj = wrong_interactions[0]
            # Find the wrong step info
            wrong_step_info = self.step_by_target.get(wrong_obj, {}).get('step')
            self._trigger_fatal_error(
                error_type=ERROR_TYPE_WRONG_ORDER,
                details=f"Cầm nhầm: {wrong_obj} (Đang cần: {target_label})",
                step_info=current_step,
                wrong_step_info=wrong_step_info
            )
            return self.update(current_interactions)  # Recursive call to return fatal error state
        
        # 2. CHECK QUÁ NHANH (TOO FAST) -> Warning nhẹ
        if (now - self.error_timer) < 2.0:
            return {
                "signal": "WARNING",
                "step_id": current_step['step_id'],
                "step_index": self.current_step_index,
                "total_steps": len(self.steps),
                "description": current_step.get('description', ''),
                "target_object": target_label,
                "status": "LỖI: THAO TÁC QUÁ NHANH!",
                "timer_ratio": 0.0,
                "state": "TOO_SHORT",
                "error": "TOO_SHORT",
                "is_completed": False,
                "is_failed": False
            }
        
        is_target_touched = any(inter['item_label'] == target_label for inter in current_interactions)
        status_msg = f"CHỜ: {current_step.get('description', target_label)}"
        timer_ratio = 0.0
        
        # --- STATE MACHINE LOGIC ---
        
        # STATE 1: WAITING FOR HAND
        if self.state == "WAITING_HAND":
            idle_duration = now - self.step_start_clock
            
            if is_target_touched:
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
                # CHECK TIMEOUT -> GÂY LỖI NGHIÊM TRỌNG
                if idle_duration > self.max_idle_time:
                    self._trigger_fatal_error(
                        error_type=ERROR_TYPE_TIMEOUT,
                        details=f"Quá thời gian quy định ({self.max_idle_time}s)",
                        step_info=current_step
                    )
                    return self.update(current_interactions)
                else:
                    countdown = self.max_idle_time - idle_duration
                    status_msg = f"CHỜ... ({countdown:.0f}s còn lại)"
        
        # STATE 2: WORKING
        elif self.state == "WORKING":
            work_duration = self.last_touch_time - self.start_work_time
            
            if is_target_touched:
                # Hand still touching - update active time
                self.last_touch_time = now
                status_msg = f"ĐANG LÀM... ({work_duration:.1f}s)"
                timer_ratio = min(work_duration / MIN_WORK_DURATION, 1.0)
            else:
                # Hand released - check idle time
                idle_time = now - self.last_touch_time
                
                if idle_time > HAND_OFF_TIMEOUT:
                    # Cooldown finished - check if step completed
                    if work_duration >= MIN_WORK_DURATION:
                        self._complete_step(current_step, work_duration)
                        status_msg = "PASSED ✓"
                        timer_ratio = 0.0
                    else:
                        # Too quick - Warning (not fatal, just reset)
                        log.info(f"[SOP] Too short ({work_duration:.1f}s). Reset with warning.")
                        self._log_error(
                            error_type="TOO_SHORT",
                            details=f"Thao tác quá nhanh ({work_duration:.1f}s < {MIN_WORK_DURATION}s)",
                            step_info=current_step
                        )
                        self.state = "WAITING_HAND"
                        self.consecutive_touch_frames = 0
                        self.error_timer = now  # Set error timer for warning display
                        self.step_start_clock = now  # Reset idle timer
                        return {
                            "signal": "WARNING",
                            "step_id": current_step['step_id'],
                            "step_index": self.current_step_index,
                            "total_steps": len(self.steps),
                            "description": current_step.get('description', ''),
                            "target_object": target_label,
                            "status": "LỖI: THAO TÁC QUÁ NHANH!",
                            "timer_ratio": 0.0,
                            "state": "TOO_SHORT",
                            "error": "TOO_SHORT",
                            "is_completed": False,
                            "is_failed": False
                        }
                else:
                    # Still in verification countdown
                    countdown = HAND_OFF_TIMEOUT - idle_time
                    status_msg = f"XÁC NHẬN... ({countdown:.1f}s)"
                    timer_ratio = countdown / HAND_OFF_TIMEOUT
        
        return {
            "signal": "NORMAL",
            "step_id": current_step['step_id'],
            "step_index": self.current_step_index,
            "total_steps": len(self.steps),
            "description": current_step.get('description', ''),
            "target_object": target_label,
            "status": status_msg,
            "timer_ratio": timer_ratio,
            "state": self.state,
            "error": None,
            "is_completed": self.is_completed,
            "is_failed": self.is_failed
        }
    
    def _log_error(self, error_type: str, details: str, step_info: Dict):
        """Log an error without triggering fatal error mode."""
        error_log = {
            "error_type": error_type,
            "details": details,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "step_id": step_info['step_id'],
            "step_description": step_info.get('description', ''),
            "step_index": self.current_step_index
        }
        self.error_logs.append(error_log)
        log.warning(f"[SOP] {error_type}: {details}")
    
    def _handle_wrong_step(self, wrong_step_info: Dict, expected_step: Dict):
        """Handle detection of wrong step being touched."""
        wrong_index = wrong_step_info['index']
        wrong_step = wrong_step_info['step']
        
        if self.last_wrong_step == wrong_index:
            self.wrong_step_counter += 1
        else:
            self.wrong_step_counter = 1
            self.last_wrong_step = wrong_index
        
        log.warning(f"[SOP] Wrong step detected: touching {wrong_step['target_object']} "
                   f"(step {wrong_step['step_id']}) instead of {expected_step['target_object']} "
                   f"(step {expected_step['step_id']}). Count: {self.wrong_step_counter}/{WRONG_STEP_TOLERANCE}")
        
        if self.wrong_step_counter >= WRONG_STEP_TOLERANCE:
            self._trigger_fatal_error(
                error_type=ERROR_TYPE_WRONG_ORDER,
                details=f"Sai thứ tự: Đang cầm {wrong_step.get('description', wrong_step['target_object'])} "
                       f"thay vì {expected_step.get('description', expected_step['target_object'])}",
                step_info=expected_step,
                wrong_step_info=wrong_step
            )
    
    def _trigger_fatal_error(self, error_type: str, details: str, step_info: Dict, wrong_step_info: Dict = None):
        """Trigger a fatal error state (matching GitHub SOPvn implementation)."""
        self.state = "FATAL_ERROR"
        self.fatal_error_mode = True  # Set fatal error mode flag
        self.fatal_error_time = time.time()
        self.fatal_error_type = error_type
        self.fatal_error_details = details
        self.is_failed = True
        
        # Get instructions for the step (if available in rules)
        self.fatal_instruction = step_info.get('instructions', [
            f"Bước hiện tại: {step_info.get('description', step_info['step_id'])}",
            f"Đối tượng mục tiêu: {step_info.get('target_object', 'N/A')}",
            "Hãy thực hiện đúng thứ tự các bước."
        ])
        
        error_log = {
            "error_type": error_type,
            "details": details,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "expected_step_id": step_info['step_id'],
            "expected_step_description": step_info.get('description', ''),
            "step_index": self.current_step_index
        }
        
        if wrong_step_info:
            error_log["wrong_step_id"] = wrong_step_info['step_id']
            error_log["wrong_step_description"] = wrong_step_info.get('description', '')
        
        self.error_logs.append(error_log)
        
        # Track this as a failed step
        failed_step = {
            "step_id": step_info['step_id'],
            "step_index": self.current_step_index,
            "description": step_info.get('description', ''),
            "target_object": step_info.get('target_object', ''),
            "error_type": error_type,
            "error_details": details,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "instructions": step_info.get('instructions', []),
            "common_errors": step_info.get('common_errors', [])
        }
        if wrong_step_info:
            failed_step["wrong_step_id"] = wrong_step_info['step_id']
            failed_step["wrong_step_description"] = wrong_step_info.get('description', '')
        
        self.failed_steps.append(failed_step)
        
        log.error(f"[FATAL ERROR] {error_type}: {details}")
    
    def get_step_instructions(self, step_index: int = None) -> List[str]:
        """Get instructions for a specific step or current step."""
        idx = step_index if step_index is not None else self.current_step_index
        if idx < len(self.steps):
            return self.steps[idx].get('instructions', [])
        return []
    
    def get_step_common_errors(self, step_index: int = None) -> List[str]:
        """Get common errors for a specific step or current step."""
        idx = step_index if step_index is not None else self.current_step_index
        if idx < len(self.steps):
            return self.steps[idx].get('common_errors', [])
        return []
    
    def _complete_step(self, step_info: Dict, duration: float):
        """Record step completion and advance to next step."""
        log.info(f"[SUCCESS] Step {step_info['step_id']} Done. Duration: {duration:.2f}s")
        
        self.report_logs.append({
            "step_id": step_info['step_id'],
            "description": step_info.get('description', ''),
            "target_object": step_info['target_object'],
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "duration": round(duration, 2),
            "status": "PASSED",
            "instructions": step_info.get('instructions', []),
            "common_errors": step_info.get('common_errors', [])
        })
        
        self.current_step_index += 1
        self.state = "WAITING_HAND"
        self.consecutive_touch_frames = 0
        self.step_start_clock = time.time()  # Reset step timer for next step
        self.wrong_step_counter = 0
        self.last_wrong_step = None
        
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
            "is_failed": self.is_failed,
            "completed_steps": self.report_logs,
            "error_logs": self.error_logs
        }
    
    def get_report(self) -> Dict[str, Any]:
        """Generate full assessment report."""
        total_time = sum(log['duration'] for log in self.report_logs)
        
        # Determine overall result
        if self.is_completed and not self.error_logs:
            result = "PASSED"
        elif self.is_completed and self.error_logs:
            result = "PASSED_WITH_ERRORS"
        else:
            result = "FAILED"
        
        return {
            "session_id": self.session_id,
            "process_name": self.process_name,
            "total_steps": len(self.steps),
            "completed_steps": len(self.report_logs),
            "result": result,
            "is_passed": self.is_completed,
            "is_failed": self.is_failed,
            "total_duration": round(total_time, 2),
            "started_at": self.report_logs[0]['timestamp'] if self.report_logs else None,
            "completed_at": self.report_logs[-1]['timestamp'] if self.report_logs else None,
            "step_details": self.report_logs,
            "error_logs": self.error_logs,
            "error_count": len(self.error_logs),
            "failed_steps": self.failed_steps,  # Detailed info about failed steps
            "failed_step_ids": [s['step_id'] for s in self.failed_steps]  # Quick list of failed step IDs
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
        self.is_failed = False
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_logs = []
        self.error_logs = []
        self.failed_steps = []  # Reset failed steps tracking
        self.state = "WAITING_HAND"
        self.consecutive_touch_frames = 0
        self.start_work_time = 0
        self.last_touch_time = 0
        self.step_start_clock = time.time()
        self.wrong_step_counter = 0
        self.last_wrong_step = None
        self.fatal_error_mode = False
        self.fatal_error_time = 0
        self.fatal_error_type = None
        self.fatal_error_details = None
        self.fatal_instruction = []
        log.info("[SOP] Engine reset for new assessment")
