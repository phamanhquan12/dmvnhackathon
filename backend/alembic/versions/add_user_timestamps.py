"""Add created_at and updated_at to users table

Revision ID: add_user_timestamps
Revises: add_user_role_password
Create Date: 2025-12-14

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'add_user_timestamps'
down_revision: Union[str, None] = 'add_user_role_password'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add created_at column with default value
    op.add_column('users', sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True))
    # Add updated_at column
    op.add_column('users', sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column('users', 'updated_at')
    op.drop_column('users', 'created_at')
