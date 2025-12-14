"""Add learning progress tracking tables

Revision ID: add_progress_tracking
Revises: fc7095eb3879
Create Date: 2024-01-15

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'add_progress_tracking'
down_revision: Union[str, None] = 'fc7095eb3879'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add video-specific columns to documents table
    op.add_column('documents', sa.Column('duration_seconds', sa.Integer(), nullable=True))
    op.add_column('documents', sa.Column('transcript_path', sa.String(), nullable=True))
    
    # Create chunk_interactions table
    op.create_table('chunk_interactions',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('chunk_id', sa.String(), nullable=False),
        sa.Column('interaction_type', sa.String(), nullable=False),
        sa.Column('was_successful', sa.Boolean(), default=False),
        sa.Column('interaction_count', sa.Integer(), default=1),
        sa.Column('first_interaction', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('last_interaction', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.ForeignKeyConstraint(['chunk_id'], ['document_chunks.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id', 'chunk_id', 'interaction_type', name='uix_user_chunk_interaction')
    )
    op.create_index(op.f('ix_chunk_interactions_id'), 'chunk_interactions', ['id'], unique=False)
    
    # Create document_progress table
    op.create_table('document_progress',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('document_id', sa.Integer(), nullable=False),
        sa.Column('total_chunks', sa.Integer(), default=0),
        sa.Column('chunks_studied', sa.Integer(), default=0),
        sa.Column('chunks_quizzed', sa.Integer(), default=0),
        sa.Column('chunks_flashcarded', sa.Integer(), default=0),
        sa.Column('chunks_mastered', sa.Integer(), default=0),
        sa.Column('overall_progress', sa.Float(), default=0.0),
        sa.Column('last_activity', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id', 'document_id', name='uix_user_document_progress')
    )
    op.create_index(op.f('ix_document_progress_id'), 'document_progress', ['id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_document_progress_id'), table_name='document_progress')
    op.drop_table('document_progress')
    op.drop_index(op.f('ix_chunk_interactions_id'), table_name='chunk_interactions')
    op.drop_table('chunk_interactions')
    op.drop_column('documents', 'transcript_path')
    op.drop_column('documents', 'duration_seconds')
