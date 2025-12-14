# app/services/sop/assessment_service.py
"""
Assessment Service for SOP Video-Based Assessments
Handles saving and retrieving assessment results from database
Integrates AI-powered feedback generation via RAG
"""
import json
import logging as log
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4
from datetime import datetime
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.session import PracticalSession, PracticalStepResult
from app.models.user import User
from app.services.sop.feedback_service import FeedbackService

log.basicConfig(level=log.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class AssessmentService:
    """Service for managing SOP assessment sessions and results."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.feedback_service = FeedbackService(session)
    
    async def save_assessment(
        self,
        employee_id: str,
        report_data: Dict[str, Any],
        sop_rules: Dict[str, Any] = None,
        video_filename: str = None,
        generate_feedback: bool = True
    ) -> Dict[str, Any]:
        """
        Save assessment result to database with AI-generated feedback.
        
        Args:
            employee_id: User's employee ID
            report_data: Full report from SOPEngine.get_report()
            sop_rules: The SOP rules JSON used for assessment
            video_filename: Original video filename
            generate_feedback: Whether to generate AI feedback (default True)
            
        Returns:
            Saved assessment data with ID and feedback
        """
        # Find user by employee_id
        user_result = await self.session.execute(
            select(User).where(User.employee_id == employee_id)
        )
        user = user_result.scalar_one_or_none()
        
        if not user:
            raise ValueError(f"User with employee_id {employee_id} not found")
        
        # Calculate score
        total_steps = report_data.get('total_steps', 0)
        completed_steps = report_data.get('completed_steps', 0)
        score = (completed_steps / total_steps * 100) if total_steps > 0 else 0
        
        # Use existing AI feedback from report_data if available, otherwise generate new
        feedback_data = report_data.get('ai_feedback')
        
        if not feedback_data and generate_feedback:
            try:
                log.info("[ASSESSMENT] No existing feedback found, generating AI feedback...")
                feedback_data = await self.feedback_service.generate_completion_summary(
                    report=report_data,
                    sop_rules=sop_rules
                )
                log.info("[ASSESSMENT] AI feedback generated successfully")
            except Exception as e:
                log.error(f"[ASSESSMENT] Failed to generate AI feedback: {e}")
                feedback_data = None
        else:
            log.info("[ASSESSMENT] Using existing AI feedback from report_data")
        
        # Create unique session code (timestamp + short UUID to avoid duplicates)
        base_session_id = report_data.get('session_id', datetime.now().strftime("%Y%m%d_%H%M%S"))
        unique_id = str(uuid4())[:8]  # First 8 chars of UUID
        session_code = f"{base_session_id}_{unique_id}"
        
        # Serialize feedback to JSON string for storage
        feedback_json = json.dumps(feedback_data, ensure_ascii=False) if feedback_data else None
        
        # Create PracticalSession with feedback
        practical_session = PracticalSession(
            id=uuid4(),
            session_code=session_code,
            user_id=user.id,
            process_name=report_data.get('process_name', 'Unknown Process'),
            sop_rules_json=sop_rules,
            total_steps=total_steps,
            completed_steps=completed_steps,
            score=score,  # Use original assessment score (not AI score)
            status="PASSED" if report_data.get('is_passed', False) else "FAILED",
            total_duration=report_data.get('total_duration', 0),
            video_filename=video_filename,
            report_data=report_data,  # This includes failed_steps
            feedback=feedback_json,  # AI-generated feedback
            started_at=datetime.strptime(report_data['started_at'], "%H:%M:%S").replace(
                year=datetime.now().year,
                month=datetime.now().month,
                day=datetime.now().day
            ) if report_data.get('started_at') else None,
            completed_at=datetime.strptime(report_data['completed_at'], "%H:%M:%S").replace(
                year=datetime.now().year,
                month=datetime.now().month,
                day=datetime.now().day
            ) if report_data.get('completed_at') else None
        )
        
        self.session.add(practical_session)
        
        # Create step results for completed steps
        for idx, step_detail in enumerate(report_data.get('step_details', [])):
            step_result = PracticalStepResult(
                session_id=practical_session.id,
                step_id=str(step_detail.get('step_id', f'step_{idx+1}')),
                step_index=idx,
                description=str(step_detail.get('description', '')),
                target_object=str(step_detail.get('target_object', '')),
                status=str(step_detail.get('status', 'UNKNOWN')),
                duration=float(step_detail.get('duration', 0)),
                timestamp=str(step_detail.get('timestamp', ''))
            )
            self.session.add(step_result)
        
        # Also create step results for failed steps (with FAILED status)
        for failed_step in report_data.get('failed_steps', []):
            step_result = PracticalStepResult(
                session_id=practical_session.id,
                step_id=str(failed_step.get('step_id', 'unknown')),
                step_index=failed_step.get('step_index', -1),
                description=str(failed_step.get('description', '')),
                target_object=str(failed_step.get('target_object', '')),
                status="FAILED",
                duration=0.0,
                timestamp=str(failed_step.get('timestamp', '')),
                detection_data={
                    "error_type": failed_step.get('error_type'),
                    "error_details": failed_step.get('error_details'),
                    "wrong_step_id": failed_step.get('wrong_step_id'),
                    "wrong_step_description": failed_step.get('wrong_step_description')
                }
            )
            self.session.add(step_result)
        
        await self.session.commit()
        await self.session.refresh(practical_session)
        
        log.info(f"[ASSESSMENT] Saved assessment {session_code} for user {employee_id}")
        
        return {
            "id": str(practical_session.id),
            "session_code": practical_session.session_code,
            "status": practical_session.status,
            "score": practical_session.score,
            "message": "Assessment saved successfully",
            "feedback": feedback_data,  # Return feedback to UI
            "failed_steps": report_data.get('failed_steps', [])
        }
    
    async def get_user_assessments(
        self,
        employee_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get assessment history for a user.
        
        Args:
            employee_id: User's employee ID
            limit: Maximum number of results
            
        Returns:
            List of assessment summaries
        """
        # Find user
        user_result = await self.session.execute(
            select(User).where(User.employee_id == employee_id)
        )
        user = user_result.scalar_one_or_none()
        
        if not user:
            return []
        
        # Get sessions with step results
        result = await self.session.execute(
            select(PracticalSession)
            .where(PracticalSession.user_id == user.id)
            .options(selectinload(PracticalSession.step_results))
            .order_by(desc(PracticalSession.created_at))
            .limit(limit)
        )
        sessions = result.scalars().all()
        
        assessments = []
        for session in sessions:
            assessments.append({
                "id": str(session.id),
                "session_code": session.session_code,
                "process_name": session.process_name,
                "total_steps": session.total_steps,
                "completed_steps": session.completed_steps,
                "score": session.score,
                "status": session.status,
                "total_duration": session.total_duration,
                "video_filename": session.video_filename,
                "created_at": session.created_at.isoformat() if session.created_at else None,
                "step_count": len(session.step_results)
            })
        
        return assessments
    
    async def get_assessment_detail(
        self,
        session_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get detailed assessment result by session ID.
        
        Args:
            session_id: UUID string of the session
            
        Returns:
            Full assessment details including step results
        """
        try:
            session_uuid = UUID(session_id)
        except ValueError:
            return None
        
        result = await self.session.execute(
            select(PracticalSession)
            .where(PracticalSession.id == session_uuid)
            .options(selectinload(PracticalSession.step_results))
        )
        session = result.scalar_one_or_none()
        
        if not session:
            return None
        
        # Build step details
        step_details = []
        for step in sorted(session.step_results, key=lambda x: x.step_index):
            step_details.append({
                "step_id": step.step_id,
                "step_index": step.step_index,
                "description": step.description,
                "target_object": step.target_object,
                "status": step.status,
                "duration": step.duration,
                "timestamp": step.timestamp
            })
        
        return {
            "id": str(session.id),
            "session_code": session.session_code,
            "process_name": session.process_name,
            "total_steps": session.total_steps,
            "completed_steps": session.completed_steps,
            "score": session.score,
            "status": session.status,
            "total_duration": session.total_duration,
            "video_filename": session.video_filename,
            "sop_rules": session.sop_rules_json,
            "report_data": session.report_data,
            "feedback": session.feedback,
            "started_at": session.started_at.isoformat() if session.started_at else None,
            "completed_at": session.completed_at.isoformat() if session.completed_at else None,
            "created_at": session.created_at.isoformat() if session.created_at else None,
            "step_details": step_details
        }


# Async helper functions for use in Streamlit
async def save_assessment_result(
    employee_id: str,
    report_data: Dict[str, Any],
    sop_rules: Dict[str, Any] = None,
    video_filename: str = None
) -> Dict[str, Any]:
    """Helper function to save assessment from Streamlit."""
    from app.core.database import get_async_session
    
    session_factory = get_async_session()
    async with session_factory() as session:
        service = AssessmentService(session)
        return await service.save_assessment(
            employee_id=employee_id,
            report_data=report_data,
            sop_rules=sop_rules,
            video_filename=video_filename
        )


async def get_user_assessment_history(
    employee_id: str,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """Helper function to get assessment history from Streamlit."""
    from app.core.database import get_async_session
    
    session_factory = get_async_session()
    async with session_factory() as session:
        service = AssessmentService(session)
        return await service.get_user_assessments(employee_id, limit)


async def get_assessment_details(session_id: str) -> Optional[Dict[str, Any]]:
    """Helper function to get assessment details from Streamlit."""
    from app.core.database import get_async_session
    
    session_factory = get_async_session()
    async with session_factory() as session:
        service = AssessmentService(session)
        return await service.get_assessment_detail(session_id)


async def generate_ai_feedback(
    report_data: Dict[str, Any],
    sop_rules: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Generate AI feedback for assessment without saving to database.
    This is called immediately after assessment completes.
    
    Args:
        report_data: Full report from SOPEngine.get_report()
        sop_rules: The SOP rules JSON used for assessment
        
    Returns:
        AI-generated feedback dictionary
    """
    from app.core.database import get_async_session
    
    session_factory = get_async_session()
    async with session_factory() as session:
        feedback_service = FeedbackService(session)
        try:
            log.info("[FEEDBACK] Generating AI feedback for assessment...")
            feedback = await feedback_service.generate_completion_summary(
                report=report_data,
                sop_rules=sop_rules
            )
            log.info("[FEEDBACK] AI feedback generated successfully")
            return feedback
        except Exception as e:
            log.error(f"[FEEDBACK] Failed to generate AI feedback: {e}")
            # Return a basic fallback feedback
            return {
                "overall_rating": "average" if report_data.get('is_passed', False) else "needs_improvement",
                "score": 70 if report_data.get('is_passed', False) else 40,
                "summary": f"Hoàn thành {report_data.get('completed_steps', 0)}/{report_data.get('total_steps', 0)} bước trong {report_data.get('total_duration', 0):.1f}s",
                "strengths": ["Đã hoàn thành bài đánh giá"] if report_data.get('is_passed', False) else [],
                "areas_for_improvement": ["Cần hoàn thành đầy đủ các bước"] if not report_data.get('is_passed', False) else [],
                "specific_recommendations": [],
                "next_steps": ["Xem lại các bước chưa hoàn thành", "Luyện tập thêm"],
                "encouragement": "Tiếp tục cố gắng! Mỗi lần luyện tập là một cơ hội để tiến bộ.",
                "error": str(e)
            }


async def get_all_assessments(limit: int = 100) -> List[Dict[str, Any]]:
    """Get all assessments across all users."""
    from app.core.database import get_async_session
    from sqlalchemy import desc
    
    session_factory = get_async_session()
    async with session_factory() as session:
        result = await session.execute(
            select(PracticalSession, User)
            .join(User, PracticalSession.user_id == User.id)
            .options(selectinload(PracticalSession.step_results))
            .order_by(desc(PracticalSession.created_at))
            .limit(limit)
        )
        rows = result.fetchall()
        
        assessments = []
        for ps, user in rows:
            assessments.append({
                "id": str(ps.id),
                "session_code": ps.session_code,
                "process_name": ps.process_name,
                "total_steps": ps.total_steps,
                "completed_steps": ps.completed_steps,
                "score": ps.score,
                "status": ps.status,
                "total_duration": ps.total_duration,
                "video_filename": ps.video_filename,
                "is_passed": ps.status == "PASSED",
                "created_at": ps.created_at.isoformat() if ps.created_at else None,
                "completed_at": ps.completed_at.isoformat() if ps.completed_at else None,
                "user_name": user.full_name,
                "employee_id": user.employee_id,
                "user_id": str(user.id),
                "feedback": ps.feedback
            })
        
        return assessments


async def delete_user_assessments(employee_id: str) -> Dict[str, Any]:
    """Delete all assessments for a specific user."""
    from app.core.database import get_async_session
    from sqlalchemy import delete
    
    session_factory = get_async_session()
    async with session_factory() as session:
        # Find user
        user_result = await session.execute(
            select(User).where(User.employee_id == employee_id)
        )
        user = user_result.scalar_one_or_none()
        
        if not user:
            return {"success": False, "message": f"User {employee_id} not found", "deleted_count": 0}
        
        # Get count of sessions to delete
        count_result = await session.execute(
            select(PracticalSession).where(PracticalSession.user_id == user.id)
        )
        sessions = count_result.scalars().all()
        count = len(sessions)
        
        # Delete step results first (foreign key constraint)
        for ps in sessions:
            await session.execute(
                delete(PracticalStepResult).where(PracticalStepResult.session_id == ps.id)
            )
        
        # Delete sessions
        await session.execute(
            delete(PracticalSession).where(PracticalSession.user_id == user.id)
        )
        
        await session.commit()
        log.info(f"[ASSESSMENT] Deleted {count} assessments for user {employee_id}")
        
        return {"success": True, "message": f"Deleted {count} assessments for {employee_id}", "deleted_count": count}


async def delete_all_assessments() -> Dict[str, Any]:
    """Delete all assessments for all users."""
    from app.core.database import get_async_session
    from sqlalchemy import delete
    
    session_factory = get_async_session()
    async with session_factory() as session:
        # Get count
        count_result = await session.execute(select(PracticalSession))
        count = len(count_result.scalars().all())
        
        # Delete all step results first
        await session.execute(delete(PracticalStepResult))
        
        # Delete all sessions
        await session.execute(delete(PracticalSession))
        
        await session.commit()
        log.info(f"[ASSESSMENT] Deleted ALL {count} assessments")
        
        return {"success": True, "message": f"Deleted all {count} assessments", "deleted_count": count}
