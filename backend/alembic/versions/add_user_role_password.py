"""Add user role and password hash columns

Revision ID: add_user_role_password
Revises: add_practical_assessment
Create Date: 2025-01-21 10:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'add_user_role_password'
down_revision: Union[str, None] = 'add_practical_assessment'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create enum type for user role
    user_role_enum = sa.Enum('user', 'admin', name='userrole')
    user_role_enum.create(op.get_bind(), checkfirst=True)
    
    # Add role column with default 'user'
    op.add_column('users', sa.Column('role', 
        sa.Enum('user', 'admin', name='userrole'),
        nullable=False,
        server_default='user'
    ))
    
    # Add password_hash column (nullable for backward compatibility)
    op.add_column('users', sa.Column('password_hash', sa.String(), nullable=True))


def downgrade() -> None:
    # Remove columns
    op.drop_column('users', 'password_hash')
    op.drop_column('users', 'role')
    
    # Drop enum type
    user_role_enum = sa.Enum('user', 'admin', name='userrole')
    user_role_enum.drop(op.get_bind(), checkfirst=True)
