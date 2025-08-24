from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SelectField, IntegerField, BooleanField, SubmitField
from wtforms.validators import DataRequired, Email, Length, EqualTo, ValidationError, Optional
from models import User

class LoginForm(FlaskForm):
    """Form đăng nhập"""
    username = StringField('Tên đăng nhập', validators=[
        DataRequired(message='Vui lòng nhập tên đăng nhập'),
        Length(min=3, max=80, message='Tên đăng nhập phải từ 3-80 ký tự')
    ])
    password = PasswordField('Mật khẩu', validators=[
        DataRequired(message='Vui lòng nhập mật khẩu')
    ])
    remember_me = BooleanField('Ghi nhớ đăng nhập')
    submit = SubmitField('Đăng nhập')

class StudentRegisterForm(FlaskForm):
    """Form đăng ký cho sinh viên"""
    username = StringField('Tên đăng nhập', validators=[
        DataRequired(message='Vui lòng nhập tên đăng nhập'),
        Length(min=3, max=80, message='Tên đăng nhập phải từ 3-80 ký tự')
    ])
    email = StringField('Email', validators=[
        DataRequired(message='Vui lòng nhập email'),
        Email(message='Email không hợp lệ')
    ])
    full_name = StringField('Họ và tên', validators=[
        DataRequired(message='Vui lòng nhập họ và tên'),
        Length(min=2, max=100, message='Họ tên phải từ 2-100 ký tự')
    ])
    student_id = StringField('Mã sinh viên', validators=[
        DataRequired(message='Vui lòng nhập mã sinh viên'),
        Length(min=5, max=20, message='Mã sinh viên phải từ 5-20 ký tự')
    ])
    major = StringField('Ngành học', validators=[
        DataRequired(message='Vui lòng nhập ngành học'),
        Length(max=100, message='Tên ngành không được quá 100 ký tự')
    ])
    year = IntegerField('Năm học', validators=[
        DataRequired(message='Vui lòng nhập năm học')
    ])
    password = PasswordField('Mật khẩu', validators=[
        DataRequired(message='Vui lòng nhập mật khẩu'),
        Length(min=6, message='Mật khẩu phải có ít nhất 6 ký tự')
    ])
    password2 = PasswordField('Xác nhận mật khẩu', validators=[
        DataRequired(message='Vui lòng xác nhận mật khẩu'),
        EqualTo('password', message='Mật khẩu xác nhận không khớp')
    ])
    submit = SubmitField('Đăng ký')
    
    def validate_username(self, username):
        """Kiểm tra username đã tồn tại chưa"""
        user = User.query.filter_by(username=username.data).first()
        if user:
            raise ValidationError('Tên đăng nhập đã tồn tại. Vui lòng chọn tên khác.')
    
    def validate_email(self, email):
        """Kiểm tra email đã tồn tại chưa"""
        user = User.query.filter_by(email=email.data).first()
        if user:
            raise ValidationError('Email đã được sử dụng. Vui lòng sử dụng email khác.')
    
    def validate_student_id(self, student_id):
        """Kiểm tra mã sinh viên đã tồn tại chưa"""
        user = User.query.filter_by(student_id=student_id.data).first()
        if user:
            raise ValidationError('Mã sinh viên đã tồn tại. Vui lòng kiểm tra lại.')

class TeacherRegisterForm(FlaskForm):
    """Form đăng ký cho giáo viên"""
    username = StringField('Tên đăng nhập', validators=[
        DataRequired(message='Vui lòng nhập tên đăng nhập'),
        Length(min=3, max=80, message='Tên đăng nhập phải từ 3-80 ký tự')
    ])
    email = StringField('Email', validators=[
        DataRequired(message='Vui lòng nhập email'),
        Email(message='Email không hợp lệ')
    ])
    full_name = StringField('Họ và tên', validators=[
        DataRequired(message='Vui lòng nhập họ và tên'),
        Length(min=2, max=100, message='Họ tên phải từ 2-100 ký tự')
    ])
    teacher_id = StringField('Mã giáo viên', validators=[
        DataRequired(message='Vui lòng nhập mã giáo viên'),
        Length(min=3, max=20, message='Mã giáo viên phải từ 3-20 ký tự')
    ])
    department = StringField('Khoa/Bộ môn', validators=[
        DataRequired(message='Vui lòng nhập khoa/bộ môn'),
        Length(max=100, message='Tên khoa/bộ môn không được quá 100 ký tự')
    ])
    position = SelectField('Chức vụ', choices=[
        ('giang_vien', 'Giảng viên'),
        ('pho_giao_su', 'Phó Giáo sư'),
        ('giao_su', 'Giáo sư'),
        ('truong_khoa', 'Trưởng khoa'),
        ('pho_truong_khoa', 'Phó Trưởng khoa'),
        ('truong_bo_mon', 'Trưởng bộ môn'),
        ('khac', 'Khác')
    ], validators=[DataRequired(message='Vui lòng chọn chức vụ')])
    password = PasswordField('Mật khẩu', validators=[
        DataRequired(message='Vui lòng nhập mật khẩu'),
        Length(min=6, message='Mật khẩu phải có ít nhất 6 ký tự')
    ])
    password2 = PasswordField('Xác nhận mật khẩu', validators=[
        DataRequired(message='Vui lòng xác nhận mật khẩu'),
        EqualTo('password', message='Mật khẩu xác nhận không khớp')
    ])
    submit = SubmitField('Đăng ký')
    
    def validate_username(self, username):
        """Kiểm tra username đã tồn tại chưa"""
        user = User.query.filter_by(username=username.data).first()
        if user:
            raise ValidationError('Tên đăng nhập đã tồn tại. Vui lòng chọn tên khác.')
    
    def validate_email(self, email):
        """Kiểm tra email đã tồn tại chưa"""
        user = User.query.filter_by(email=email.data).first()
        if user:
            raise ValidationError('Email đã được sử dụng. Vui lòng sử dụng email khác.')
    
    def validate_teacher_id(self, teacher_id):
        """Kiểm tra mã giáo viên đã tồn tại chưa"""
        user = User.query.filter_by(teacher_id=teacher_id.data).first()
        if user:
            raise ValidationError('Mã giáo viên đã tồn tại. Vui lòng kiểm tra lại.')

class ProfileForm(FlaskForm):
    """Form cập nhật thông tin cá nhân"""
    full_name = StringField('Họ và tên', validators=[
        DataRequired(message='Vui lòng nhập họ và tên'),
        Length(min=2, max=100, message='Họ tên phải từ 2-100 ký tự')
    ])
    email = StringField('Email', validators=[
        DataRequired(message='Vui lòng nhập email'),
        Email(message='Email không hợp lệ')
    ])
    
    # Trường cho sinh viên
    major = StringField('Ngành học', validators=[Optional(), Length(max=100)])
    year = IntegerField('Năm học', validators=[Optional()])
    
    # Trường cho giáo viên  
    department = StringField('Khoa/Bộ môn', validators=[Optional(), Length(max=100)])
    position = SelectField('Chức vụ', choices=[
        ('', '-- Chọn chức vụ --'),
        ('giang_vien', 'Giảng viên'),
        ('pho_giao_su', 'Phó Giáo sư'),
        ('giao_su', 'Giáo sư'),
        ('truong_khoa', 'Trưởng khoa'),
        ('pho_truong_khoa', 'Phó Trưởng khoa'),
        ('truong_bo_mon', 'Trưởng bộ môn'),
        ('khac', 'Khác')
    ], validators=[Optional()])
    
    submit = SubmitField('Cập nhật thông tin')

class ChangePasswordForm(FlaskForm):
    """Form đổi mật khẩu"""
    current_password = PasswordField('Mật khẩu hiện tại', validators=[
        DataRequired(message='Vui lòng nhập mật khẩu hiện tại')
    ])
    new_password = PasswordField('Mật khẩu mới', validators=[
        DataRequired(message='Vui lòng nhập mật khẩu mới'),
        Length(min=6, message='Mật khẩu phải có ít nhất 6 ký tự')
    ])
    new_password2 = PasswordField('Xác nhận mật khẩu mới', validators=[
        DataRequired(message='Vui lòng xác nhận mật khẩu mới'),
        EqualTo('new_password', message='Mật khẩu xác nhận không khớp')
    ])
    submit = SubmitField('Đổi mật khẩu')
