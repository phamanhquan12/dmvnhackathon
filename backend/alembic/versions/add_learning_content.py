"""add pre-generated learning content tables

Revision ID: add_learning_content
Revises: add_progress_tracking
Create Date: 2025-12-03

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'add_learning_content'
down_revision: Union[str, None] = 'add_progress_tracking'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add new columns to documents table
    op.add_column('documents', sa.Column('flashcards_generated', sa.Boolean(), nullable=True, server_default='false'))
    op.add_column('documents', sa.Column('quizzes_generated', sa.Boolean(), nullable=True, server_default='false'))
    
    # Create flashcards table
    op.create_table(
        'flashcards',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('document_id', sa.Integer(), nullable=False),
        sa.Column('front', sa.Text(), nullable=False),
        sa.Column('back', sa.Text(), nullable=False),
        sa.Column('chunk_ids', postgresql.JSON(astext_type=sa.Text()), nullable=True, server_default='[]'),
        sa.Column('order_index', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_flashcards_id'), 'flashcards', ['id'], unique=False)
    op.create_index(op.f('ix_flashcards_document_id'), 'flashcards', ['document_id'], unique=False)
    
    # Create quiz_sets table
    op.create_table(
        'quiz_sets',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('document_id', sa.Integer(), nullable=False),
        sa.Column('set_number', sa.Integer(), nullable=False),
        sa.Column('title', sa.String(), nullable=True),
        sa.Column('chunk_ids', postgresql.JSON(astext_type=sa.Text()), nullable=True, server_default='[]'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_quiz_sets_id'), 'quiz_sets', ['id'], unique=False)
    op.create_index(op.f('ix_quiz_sets_document_id'), 'quiz_sets', ['document_id'], unique=False)
    
    # Create quiz_questions table
    op.create_table(
        'quiz_questions',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('quiz_set_id', sa.UUID(), nullable=False),
        sa.Column('question', sa.Text(), nullable=False),
        sa.Column('options', postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column('correct_answer', sa.String(), nullable=False),
        sa.Column('explanation', sa.Text(), nullable=True),
        sa.Column('chunk_ids', postgresql.JSON(astext_type=sa.Text()), nullable=True, server_default='[]'),
        sa.Column('order_index', sa.Integer(), nullable=True, server_default='0'),
        sa.ForeignKeyConstraint(['quiz_set_id'], ['quiz_sets.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_quiz_questions_id'), 'quiz_questions', ['id'], unique=False)
    
    # Create user_flashcard_progress table
    op.create_table(
        'user_flashcard_progress',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('user_id', sa.UUID(), nullable=False),
        sa.Column('flashcard_id', sa.UUID(), nullable=False),
        sa.Column('is_completed', sa.Boolean(), nullable=True, server_default='false'),
        sa.Column('review_count', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('ease_factor', sa.Integer(), nullable=True, server_default='2'),
        sa.Column('next_review', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_reviewed', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['flashcard_id'], ['flashcards.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_user_flashcard_progress_id'), 'user_flashcard_progress', ['id'], unique=False)
    
    # Create user_quiz_attempts table
    op.create_table(
        'user_quiz_attempts',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('user_id', sa.UUID(), nullable=False),
        sa.Column('quiz_set_id', sa.UUID(), nullable=False),
        sa.Column('score', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('total_questions', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('is_passed', sa.Boolean(), nullable=True, server_default='false'),
        sa.Column('answers', postgresql.JSON(astext_type=sa.Text()), nullable=True, server_default='{}'),
        sa.Column('attempted_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['quiz_set_id'], ['quiz_sets.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_user_quiz_attempts_id'), 'user_quiz_attempts', ['id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_user_quiz_attempts_id'), table_name='user_quiz_attempts')
    op.drop_table('user_quiz_attempts')
    
    op.drop_index(op.f('ix_user_flashcard_progress_id'), table_name='user_flashcard_progress')
    op.drop_table('user_flashcard_progress')
    
    op.drop_index(op.f('ix_quiz_questions_id'), table_name='quiz_questions')
    op.drop_table('quiz_questions')
    
    op.drop_index(op.f('ix_quiz_sets_document_id'), table_name='quiz_sets')
    op.drop_index(op.f('ix_quiz_sets_id'), table_name='quiz_sets')
    op.drop_table('quiz_sets')
    
    op.drop_index(op.f('ix_flashcards_document_id'), table_name='flashcards')
    op.drop_index(op.f('ix_flashcards_id'), table_name='flashcards')
    op.drop_table('flashcards')
    
    op.drop_column('documents', 'quizzes_generated')
    op.drop_column('documents', 'flashcards_generated')
