from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    """
    Model User với phân quyền sinh viên và giáo viên
    """
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    full_name = db.Column(db.String(100), nullable=False)
    
    # Phân quyền: chỉ 'teacher' (admin cũng là teacher với username='admin')
    role = db.Column(db.String(20), nullable=False, default='teacher')
    
    # Thông tin bổ sung cho giáo viên
    teacher_id = db.Column(db.String(20), unique=True, nullable=True)  # Mã giáo viên
    department = db.Column(db.String(100), nullable=True)  # Khoa/Bộ môn
    position = db.Column(db.String(50), nullable=True)  # Chức vụ
    
    # Thông tin hệ thống
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    approval_status = db.Column(db.String(20), default='approved', nullable=False)  # 'pending', 'approved', 'rejected'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, nullable=True)
    
    def set_password(self, password):
        """Mã hóa và lưu password"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Kiểm tra password"""
        return check_password_hash(self.password_hash, password)
    

    
    def is_teacher(self):
        """Kiểm tra có phải giáo viên không"""
        return self.role == 'teacher'
    
    def is_admin(self):
        """Kiểm tra có phải admin không (dựa vào username và role)"""
        # Admin có thể có role 'teacher' nhưng username là 'admin'
        return self.username == 'admin' or self.role == 'admin'
    
    def is_pending_approval(self):
        """Kiểm tra có đang chờ phê duyệt không"""
        return self.approval_status == 'pending'
    
    def is_approved(self):
        """Kiểm tra đã được phê duyệt chưa"""
        return self.approval_status == 'approved'
    
    def is_rejected(self):
        """Kiểm tra đã bị từ chối chưa"""
        return self.approval_status == 'rejected'
    
    def get_display_name(self):
        """Lấy tên hiển thị"""
        if self.role == 'teacher' and self.teacher_id:
            return f"{self.full_name} ({self.teacher_id})"
        return self.full_name
    
    def to_dict(self):
        """Chuyển đổi thành dictionary để sử dụng trong API"""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'full_name': self.full_name,
            'role': self.role,
            'teacher_id': self.teacher_id,
            'department': self.department,
            'position': self.position,
            'is_active': self.is_active,
            'approval_status': self.approval_status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None
        }
    
    def __repr__(self):
        return f'<User {self.username} ({self.role})>'


class SearchHistory(db.Model):
    """
    Model lưu lịch sử tìm kiếm của người dùng
    """
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    query = db.Column(db.Text, nullable=False)
    source = db.Column(db.String(20), nullable=False, default='default')  # 'default' hoặc 'pdf'
    results_count = db.Column(db.Integer, nullable=False, default=0)
    search_time = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Quan hệ với User
    user = db.relationship('User', backref=db.backref('search_history', lazy=True))
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'query': self.query,
            'source': self.source,
            'results_count': self.results_count,
            'search_time': self.search_time.isoformat() if self.search_time else None
        }
    
    def __repr__(self):
        return f'<SearchHistory {self.query[:50]}... by User {self.user_id}>'


class Document(db.Model):
    """
    Model lưu trữ documents trong database
    """
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    
    # Metadata
    source_type = db.Column(db.String(50), nullable=False, default='manual')  # 'manual', 'pdf', 'data.txt'
    source_file = db.Column(db.String(255), nullable=True)  # Tên file nguồn nếu có
    source_info = db.Column(db.Text, nullable=True)  # JSON string chứa thông tin thêm
    
    # Thông tin quản lý
    added_by = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    added_date = db.Column(db.DateTime, default=datetime.utcnow)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    
    # Thông tin kỹ thuật
    char_count = db.Column(db.Integer, nullable=False)
    line_number = db.Column(db.Integer, nullable=True)  # Số dòng trong file gốc
    page_number = db.Column(db.Integer, nullable=True)  # Số trang trong PDF
    
    # Quan hệ với User
    user = db.relationship('User', backref=db.backref('added_documents', lazy=True))
    
    def to_dict(self):
        """Chuyển đổi thành dictionary"""
        import json
        source_info_dict = {}
        if self.source_info:
            try:
                source_info_dict = json.loads(self.source_info)
            except:
                source_info_dict = {}
                
        return {
            'id': self.id,
            'content': self.content,
            'source_type': self.source_type,
            'source_file': self.source_file,
            'source_info': source_info_dict,
            'added_by': self.added_by,
            'added_date': self.added_date.isoformat() if self.added_date else None,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None,
            'is_active': self.is_active,
            'char_count': self.char_count,
            'line_number': self.line_number,
            'page_number': self.page_number
        }
    
    def get_preview(self, max_length=100):
        """Lấy preview của content"""
        if len(self.content) <= max_length:
            return self.content
        return self.content[:max_length] + '...'
    
    @classmethod
    def create_from_text(cls, content, source_type='manual', source_file=None, 
                        source_info=None, added_by=None, line_number=None, page_number=None):
        """Tạo document mới từ text"""
        import json
        
        doc = cls(
            content=content.strip(),
            source_type=source_type,
            source_file=source_file,
            source_info=json.dumps(source_info) if source_info else None,
            added_by=added_by,
            char_count=len(content.strip()),
            line_number=line_number,
            page_number=page_number
        )
        return doc
    
    def __repr__(self):
        return f'<Document {self.id}: {self.get_preview(50)}>'
