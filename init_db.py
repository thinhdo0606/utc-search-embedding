#!/usr/bin/env python3
"""
Script khởi tạo database cho hệ thống tìm kiếm thông tin trường
"""

from app import app
from models import db, User

def init_database():
    """Khởi tạo database và tạo dữ liệu mẫu"""
    with app.app_context():
        # Xóa tất cả bảng cũ
        db.drop_all()
        print("Đã xóa tất cả bảng cũ")
        
        # Tạo lại tất cả bảng
        db.create_all()
        print("Đã tạo lại tất cả bảng")
        
        # Tạo tài khoản admin
        admin = User(
            username='admin',
            email='admin@university.edu.vn',
            full_name='Quản trị viên hệ thống',
            role='teacher',
            teacher_id='ADMIN001',
            department='Khoa Công nghệ thông tin',
            position='giao_su'
        )
        admin.set_password('admin123')
        
        # Tạo tài khoản giáo viên mẫu
        teacher1 = User(
            username='teacher1',
            email='teacher1@university.edu.vn',
            full_name='Nguyễn Văn A',
            role='teacher',
            teacher_id='GV001',
            department='Khoa Kỹ thuật Giao thông',
            position='pho_giao_su'
        )
        teacher1.set_password('teacher123')
        
        teacher2 = User(
            username='teacher2',
            email='teacher2@university.edu.vn',
            full_name='Trần Thị B',
            role='teacher',
            teacher_id='GV002',
            department='Khoa Công nghệ thông tin',
            position='giang_vien'
        )
        teacher2.set_password('teacher123')
        
        # Tạo tài khoản sinh viên mẫu
        student1 = User(
            username='student1',
            email='student1@student.university.edu.vn',
            full_name='Lê Văn C',
            role='student',
            student_id='2021001',
            major='Công nghệ thông tin',
            year=2021
        )
        student1.set_password('student123')
        
        student2 = User(
            username='student2',
            email='student2@student.university.edu.vn',
            full_name='Phạm Thị D',
            role='student',
            student_id='2022001',
            major='Kỹ thuật Giao thông',
            year=2022
        )
        student2.set_password('student123')
        
        student3 = User(
            username='student3',
            email='student3@student.university.edu.vn',
            full_name='Hoàng Văn E',
            role='student',
            student_id='2023001',
            major='Kinh tế Vận tải',
            year=2023
        )
        student3.set_password('student123')
        
        # Thêm tất cả user vào database
        users = [admin, teacher1, teacher2, student1, student2, student3]
        for user in users:
            db.session.add(user)
        
        # Commit tất cả thay đổi
        db.session.commit()
        
        print("\n=== KHỞI TẠO DATABASE THÀNH CÔNG ===")
        print("\nTài khoản đã được tạo:")
        print("-" * 50)
        print("ADMIN:")
        print("  Username: admin")
        print("  Password: admin123")
        print("  Role: Giáo viên (Quản trị viên)")
        print()
        print("GIÁO VIÊN:")
        print("  Username: teacher1 | Password: teacher123")
        print("  Username: teacher2 | Password: teacher123")
        print()
        print("SINH VIÊN:")
        print("  Username: student1 | Password: student123")
        print("  Username: student2 | Password: student123")
        print("  Username: student3 | Password: student123")
        print("-" * 50)
        print("Vui lòng đổi mật khẩu sau khi đăng nhập!")
        print("Database: university_search.db")

if __name__ == '__main__':
    init_database()
