"""add_practical_assessment_fields

Revision ID: add_practical_assessment
Revises: add_learning_content
Create Date: 2025-01-15

Adds new fields for Stage B SOP video assessment
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'add_practical_assessment'
down_revision: Union[str, None] = 'add_learning_content'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add new columns to practical_sessions table
    op.add_column('practical_sessions', sa.Column('session_code', sa.String(50), unique=True, nullable=True))
    op.add_column('practical_sessions', sa.Column('process_name', sa.String(255), nullable=True))
    op.add_column('practical_sessions', sa.Column('sop_rules_json', sa.JSON(), nullable=True))
    op.add_column('practical_sessions', sa.Column('total_steps', sa.Integer(), nullable=True, default=0))
    op.add_column('practical_sessions', sa.Column('completed_steps', sa.Integer(), nullable=True, default=0))
    op.add_column('practical_sessions', sa.Column('score', sa.Float(), nullable=True))
    op.add_column('practical_sessions', sa.Column('total_duration', sa.Float(), nullable=True))
    op.add_column('practical_sessions', sa.Column('video_filename', sa.String(500), nullable=True))
    op.add_column('practical_sessions', sa.Column('video_path', sa.Text(), nullable=True))
    op.add_column('practical_sessions', sa.Column('report_data', sa.JSON(), nullable=True))
    op.add_column('practical_sessions', sa.Column('started_at', sa.DateTime(timezone=True), nullable=True))
    op.add_column('practical_sessions', sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True))
    op.add_column('practical_sessions', sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True))
    
    # Create index on session_code
    op.create_index('ix_practical_sessions_session_code', 'practical_sessions', ['session_code'], unique=True)
    
    # Create practical_step_results table
    op.create_table(
        'practical_step_results',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('session_id', sa.UUID(as_uuid=True), sa.ForeignKey('practical_sessions.id'), nullable=False),
        sa.Column('step_id', sa.String(50), nullable=False),
        sa.Column('step_index', sa.Integer(), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('target_object', sa.String(100), nullable=True),
        sa.Column('status', sa.String(50), nullable=False),
        sa.Column('duration', sa.Float(), nullable=True),
        sa.Column('timestamp', sa.String(20), nullable=True),
        sa.Column('detection_data', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index('ix_practical_step_results_id', 'practical_step_results', ['id'])


def downgrade() -> None:
    # Drop practical_step_results table
    op.drop_index('ix_practical_step_results_id', 'practical_step_results')
    op.drop_table('practical_step_results')
    
    # Remove new columns from practical_sessions
    op.drop_index('ix_practical_sessions_session_code', 'practical_sessions')
    op.drop_column('practical_sessions', 'updated_at')
    op.drop_column('practical_sessions', 'completed_at')
    op.drop_column('practical_sessions', 'started_at')
    op.drop_column('practical_sessions', 'report_data')
    op.drop_column('practical_sessions', 'video_path')
    op.drop_column('practical_sessions', 'video_filename')
    op.drop_column('practical_sessions', 'total_duration')
    op.drop_column('practical_sessions', 'score')
    op.drop_column('practical_sessions', 'completed_steps')
    op.drop_column('practical_sessions', 'total_steps')
    op.drop_column('practical_sessions', 'sop_rules_json')
    op.drop_column('practical_sessions', 'process_name')
    op.drop_column('practical_sessions', 'session_code')
