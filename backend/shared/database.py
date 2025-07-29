"""
数据库配置和连接管理
"""
import os
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func
from contextlib import contextmanager

# 数据库配置
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "sqlite:///./train_platform.db"  # 默认使用SQLite
)

# 创建数据库引擎
engine = create_engine(
    DATABASE_URL,
    echo=False,  # 设置为True可以看到SQL语句
    pool_pre_ping=True,
    pool_recycle=300
)

# 创建会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 创建基础模型类
Base = declarative_base()


def get_db() -> Session:
    """获取数据库会话"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_context():
    """数据库会话上下文管理器"""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db():
    """初始化数据库"""
    try:
        # 导入所有模型以确保它们被注册
        from auth_service.models.user import User
        
        # 创建所有表
        Base.metadata.create_all(bind=engine)
        print("✅ 数据库表创建成功")
        
        # 创建默认管理员用户
        try:
            create_default_admin()
        except Exception as e:
            print(f"⚠️ 创建默认管理员失败: {e}")
        
    except Exception as e:
        print(f"❌ 数据库初始化失败: {e}")
        raise


def check_db_connection() -> bool:
    """检查数据库连接"""
    try:
        with get_db_context() as db:
            from sqlalchemy import text
            db.execute(text("SELECT 1"))
        return True
    except Exception as e:
        print(f"❌ 数据库连接失败: {e}")
        return False


def create_default_admin():
    """创建默认管理员用户"""
    from auth_service.models.user import User
    from auth_service.utils.auth import get_password_hash
    
    try:
        with get_db_context() as db:
            # 检查是否已存在管理员用户
            admin = db.query(User).filter(User.username == "admin").first()
            if not admin:
                # 创建默认管理员
                admin_user = User(
                    username="admin",
                    email="admin@example.com",
                    full_name="系统管理员",
                    hashed_password=get_password_hash("admin123"),
                    is_admin=True,
                    is_active=True
                )
                db.add(admin_user)
                db.commit()
                print("✅ 默认管理员用户创建成功 (用户名: admin, 密码: admin123)")
            else:
                print("ℹ️ 管理员用户已存在")
    except Exception as e:
        print(f"❌ 创建默认管理员失败: {e}")


# 数据库模型基类
class TimestampMixin:
    """时间戳混入类"""
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class SoftDeleteMixin:
    """软删除混入类"""
    is_deleted = Column(Boolean, default=False)
    deleted_at = Column(DateTime(timezone=True), nullable=True) 