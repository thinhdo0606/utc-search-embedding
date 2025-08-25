#!/usr/bin/env python3

# Chạy file này để xóa user là Sinh viên đi, chỉ giữ lại giáo viên
"""
Script migration: Chuyển hệ thống sang chỉ có teacher/admin
- Xóa tất cả student users
- Cập nhật database schema
"""

from app import app
from models import db, User

def migrate_to_teacher_only():
    #Migration sang teacher-only system
    print("🔄 Bắt đầu migration sang teacher-only system...")
    
    with app.app_context():
        # 1. Đếm và hiển thị thông tin hiện tại
        total_users = User.query.count()
        student_users = User.query.filter_by(role='student').count()
        teacher_users = User.query.filter_by(role='teacher').count()
        
        print(f"Trạng thái hiện tại:")
        print(f"  - Tổng users: {total_users}")
        print(f"  - Students: {student_users}")
        print(f"  - Teachers: {teacher_users}")
        
        # 2. Xóa tất cả student users
        if student_users > 0:
            print(f"Xóa {student_users} student users...")
            deleted = User.query.filter_by(role='student').delete()
            db.session.commit()
            print(f"Đã xóa {deleted} student users")
        else:
            print("Không có student users để xóa")
        
        # 3. Kiểm tra lại
        remaining_users = User.query.count()
        remaining_teachers = User.query.filter_by(role='teacher').count()
        
        print(f"\nSau migration:")
        print(f"  - Tổng users: {remaining_users}")
        print(f"  - Teachers: {remaining_teachers}")
        
        # 4. Hiển thị danh sách teachers còn lại
        teachers = User.query.filter_by(role='teacher').all()
        print(f"\nTeachers còn lại:")
        for teacher in teachers:
            status = "ADMIN" if teacher.is_admin() else "Teacher"
            approval = f"({teacher.approval_status})"
            print(f"  - {teacher.username}: {teacher.full_name} [{status}] {approval}")
        
        print(f"\nMigration hoàn thành")

if __name__ == '__main__':
    migrate_to_teacher_only()
