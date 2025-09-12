from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session, send_file, abort
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from sentence_transformers import SentenceTransformer, util
import torch
import faiss
import numpy as np
from datetime import datetime
import os
import re
import pdfplumber
import PyPDF2
from werkzeug.utils import secure_filename
from models import db, User, SearchHistory, Document
from forms import LoginForm, StudentRegisterForm, TeacherRegisterForm, ProfileForm, ChangePasswordForm
from document_manager import DocumentManager

app = Flask(__name__)

# Cấu hình cho Flask app
app.config['SECRET_KEY'] = 'thinhdo23080606'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///university_search.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'pdf'}

# Khởi tạo extensions
db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Vui lòng đăng nhập để truy cập trang này.'
login_manager.login_message_category = 'info'


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


# Biến global để lưu trữ dữ liệu PDF hiện tại
current_pdf_data = {
    'filename': None,
    'content': [],
    'embeddings': None,
    'faiss_index': None
}

# Khởi tạo model embedding
print("Đang tải model embedding...")
model = SentenceTransformer('distiluse-base-multilingual-cased')
print("Model đã được tải thành công!")

# DocumentManager sẽ được khởi tạo ngay khi start app
doc_manager = None

def get_doc_manager():
    global doc_manager
    if doc_manager is None:
        print("DocumentManager chưa được khởi tạo! Đang khởi tạo...")
        doc_manager = DocumentManager(model)
    return doc_manager

def init_doc_manager():
    """Khởi tạo DocumentManager ngay khi start app"""
    global doc_manager
    if doc_manager is None:
        print("Khởi tạo DocumentManager...")
        doc_manager = DocumentManager(model)
        print(f"DocumentManager đã sẵn sàng với {len(doc_manager.documents)} documents")
    return doc_manager


# Khởi tạo dữ liệu documents (sẽ được load từ DocumentManager khi cần)
university_documents = []

def ensure_documents_loaded():
    """Đảm bảo documents đã được load từ DocumentManager"""
    global university_documents
    if not university_documents:
        dm = get_doc_manager()
        university_documents = dm.documents.copy()
        print(f"📄 Đã sync {len(university_documents)} documents từ DocumentManager")

# Embeddings và FAISS index được khởi tạo
document_embeddings = None
faiss_index = None

def ensure_embeddings_loaded():
    """Đảm bảo embeddings và FAISS index đã được load (sử dụng DocumentManager)"""
    global document_embeddings, faiss_index
    
    # Sử dụng embeddings từ DocumentManager thay vì tạo lại
    dm = get_doc_manager()
    
    if dm.embeddings is not None and dm.faiss_index is not None:
        # Sync từ DocumentManager
        document_embeddings = dm.embeddings
        faiss_index = dm.faiss_index
        ensure_documents_loaded()  # Sync documents list
        print(f"Đã sync embeddings và FAISS index từ DocumentManager")
    else:
        # Fallback: tạo mới nếu DocumentManager chưa có
        ensure_documents_loaded()
        if university_documents:
            print("Tạo embeddings cho documents...")
            tensor = model.encode(university_documents, convert_to_tensor=True)
            embs = tensor.cpu().numpy()
            
            # Chuẩn hóa embeddings cho cosine similarity
            norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
            document_embeddings = embs / norms

            # Tạo FAISS index với Inner Product cho cosine similarity
            embedding_dim = document_embeddings.shape[1]
            faiss_index = faiss.IndexFlatIP(embedding_dim)
            faiss_index.add(document_embeddings)
            print("FAISS index (cosine similarity) đã được tạo thành công!")
        else:
            print("Không có dữ liệu để tạo embeddings")


# Hàm kiểm tra file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def _is_section_delimiter(text):
    """Kiểm tra xem dòng text có phải là delimiter của section không"""
    text_stripped = text.strip()
    text_upper = text_stripped.upper()
    
    # Kiểm tra các pattern delimiter
    patterns = [
        r'^PHẦN\s+\d+',           # PHẦN 1, PHẦN 2, etc.
        r'^CHƯƠNG\s+[IVX\d]+',   # CHƯƠNG I, CHƯƠNG 1, etc.
        r'^ĐIỀU\s+\d+',          # ĐIỀU 1, ĐIỀU 2, etc.
        r'^[IVX]+\.',            # I., II., III., IV., V., etc.
        r'^[IVX]+\s',            # I , II , III , etc.
    ]
    
    for pattern in patterns:
        if re.match(pattern, text_upper):
            return True
    
    return False


def search_documents(query, k=5, similarity_threshold=0.5, source='default'):
    """Tìm kiếm cải tiến với SentenceTransformer + FAISS + Query expansion + Re-ranking"""
    
    # Chọn nguồn dữ liệu để tìm kiếm
    if source == 'pdf' and current_pdf_data['faiss_index'] is not None:
        # Tìm kiếm PDF tạm thời
        search_index = current_pdf_data['faiss_index']
        search_embeddings = current_pdf_data['embeddings']
        search_content = current_pdf_data['content']
        
        # Chuẩn hóa query giống như document để đảm bảo consistency
        dm = get_doc_manager()
        processed_query = dm.preprocess_document(query)
        
        # Chuẩn hóa query embedding
        query_embedding = model.encode([processed_query], convert_to_tensor=True)
        query_np = query_embedding.cpu().numpy()
        norms = np.linalg.norm(query_np, axis=1, keepdims=True) + 1e-12
        query_np = query_np / norms
        
        # Tìm kiếm với FAISS
        distances, indices = search_index.search(query_np, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= len(search_content):
                continue
            
            # Với normalized embeddings và IndexFlatIP, distances chính là cosine similarity
            similarity = float(distances[0][i])
            
            if similarity >= similarity_threshold:
                content_item = search_content[idx]
                results.append({
                    'index': int(idx),
                    'content': content_item['content'],
                    'page': content_item.get('page', 1),
                    'score': similarity,
                    'similarity': similarity,
                    'rank': len(results) + 1,
                    'source': 'pdf',
                    'filename': current_pdf_data['filename']
                })
        
        return results
    else:
        # Tìm kiếm trong dữ liệu chính - sử dụng DocumentManager với cải tiến
        dm = get_doc_manager()
        if dm.faiss_index is None:
            return []
        
        # Sử dụng search method đã cải tiến của DocumentManager
        results = dm.search(query, k, similarity_threshold)
        
        # Chuyển đổi format để tương thích với frontend
        formatted_results = []
        for result in results:
            # Lấy nội dung gốc từ metadata nếu có, không thì dùng processed content
            main_content = result['content']
            original_main_content = main_content
            
            # Kiểm tra xem có metadata với original_content không
            if result['index'] < len(dm.metadata) and 'original_content' in dm.metadata[result['index']]:
                original_main_content = dm.metadata[result['index']]['original_content']
            
            extended_content = [original_main_content]
            
            # Thêm 4 documents tiếp theo
            doc_idx = result['index']
            for next_idx in range(doc_idx + 1, min(doc_idx + 5, len(dm.documents))):
                if next_idx < len(dm.documents):
                    # Lấy original content nếu có
                    next_content = dm.documents[next_idx]
                    if next_idx < len(dm.metadata) and 'original_content' in dm.metadata[next_idx]:
                        next_content = dm.metadata[next_idx]['original_content']
                    extended_content.append(next_content)
            
            # Nối tất cả thành một chuỗi cho hiển thị
            full_content = " | ".join(extended_content)
            
            # Tạo nội dung mở rộng cho modal (4 documents + mở rộng đến delimiter)
            modal_content_list = extended_content.copy()  # Bắt đầu với 4 documents
            
            # Tiếp tục thêm documents từ vị trí thứ 9 đến khi gặp delimiter
            for next_idx in range(doc_idx + 9, min(doc_idx + 50, len(dm.documents))):
                if next_idx < len(dm.documents):
                    next_doc = dm.documents[next_idx]
                    next_original = next_doc
                    
                    # Lấy original content nếu có
                    if next_idx < len(dm.metadata) and 'original_content' in dm.metadata[next_idx]:
                        next_original = dm.metadata[next_idx]['original_content']
                    
                    # Kiểm tra delimiter với processed content (để logic không thay đổi)
                    if _is_section_delimiter(next_doc):
                        break
                    
                    modal_content_list.append(next_original)
            
            # Tạo text cho modal
            modal_content_text = "\n".join(modal_content_list)
            
            formatted_results.append({
                'index': result['index'],
                'content': full_content,
                'main_content': original_main_content,
                'extended_content': extended_content,  # 4 documents cho hiển thị bên ngoài
                'modal_content': modal_content_text,  # Nội dung đầy đủ cho modal (8 docs + mở rộng)
                'modal_content_list': modal_content_list,  # Danh sách documents cho modal
                'score': result['score'],
                'similarity': result['similarity'],
                'rank': result['rank'],
                'source': 'default'
            })
        
        return formatted_results


# Hàm trích xuất văn bản từ PDF
def extract_text_from_pdf(pdf_path):
    text_content = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text and text.strip():
                    # Xử lý văn bản để tạo thành các đoạn văn hoàn chỉnh
                    page_paragraphs = process_page_text(text, page_num)
                    text_content.extend(page_paragraphs)

        print(f"Đã trích xuất {len(text_content)} đoạn văn bản từ PDF")
        return text_content

    except Exception as e:
        print(f"Lỗi khi trích xuất PDF: {e}")
        return []


def process_page_text(text, page_num):
    # Xử lý văn bản trang để tạo thành các đoạn văn hoàn chỉnh
    paragraphs = []

    # Chia văn bản thành các dòng
    lines = [line.strip() for line in text.split('\n') if line.strip()]

    if not lines:
        return paragraphs

    current_paragraph = ""

    for i, line in enumerate(lines):
        if len(line) < 10:
            continue

        is_new_paragraph = False

        # Các dấu hiệu của đoạn mới:
        # 1. Dòng bắt đầu bằng số (1., 2., 1.1, etc.)
        if re.match(r'^\d+\.(\d+\.)*\s', line):
            is_new_paragraph = True

        # 2. Dòng bắt đầu bằng bullet point
        elif re.match(r'^[-•\*\+]\s', line):
            is_new_paragraph = True

        # 3. Dòng có indent lớn hoặc format đặc biệt
        elif line[0].isupper() and len(current_paragraph) > 0:
            # Kiểm tra xem có phải là câu tiếp theo không
            if not current_paragraph.endswith(('.', '!', '?', ':')):
                # Nếu đoạn trước chưa kết thúc thì nối tiếp
                is_new_paragraph = False
            else:
                is_new_paragraph = True

        # 4. Dòng trước kết thúc bằng dấu chấm và dòng này bắt đầu bằng chữ hoa
        elif (current_paragraph.endswith('.') and
              len(current_paragraph) > 50 and
              line[0].isupper()):
            is_new_paragraph = True

        # Nếu là đoạn mới và đoạn hiện tại đủ dài
        if is_new_paragraph and len(current_paragraph.strip()) > 30:
            paragraphs.append({
                'content': current_paragraph.strip(),
                'page': page_num,
                'type': 'paragraph'
            })
            current_paragraph = line
        else:
            # Nối dòng vào đoạn hiện tại
            if current_paragraph:
                # Thêm khoảng trắng nếu cần
                if not current_paragraph.endswith(' '):
                    current_paragraph += ' '
                current_paragraph += line
            else:
                current_paragraph = line

    # Thêm đoạn cuối cùng nếu đủ dài
    if len(current_paragraph.strip()) > 30:
        paragraphs.append({
            'content': current_paragraph.strip(),
            'page': page_num,
            'type': 'paragraph'
        })

    # Nếu không có đoạn nào được tạo, tạo một đoạn từ toàn bộ text
    if not paragraphs and len(' '.join(lines)) > 30:
        paragraphs.append({
            'content': ' '.join(lines),
            'page': page_num,
            'type': 'paragraph'
        })

    return paragraphs


# Hàm tạo embeddings cho PDF
def create_pdf_embeddings(content_list):
    if not content_list:
        return None, None

    # Lấy nội dung text từ content_list và áp dụng preprocessing
    dm = get_doc_manager()
    texts = [dm.preprocess_document(item['content']) for item in content_list]

    # Tạo embeddings
    tensor = model.encode(texts, convert_to_tensor=True)
    embs = tensor.cpu().numpy()
    
    # Chuẩn hóa embeddings cho cosine similarity
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
    embeddings = embs / norms

    # Tạo FAISS index với Inner Product cho cosine similarity
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embeddings)

    return embeddings, index


# AUTHENTICATION ROUTES

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Trang đăng nhập"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()

        if user and user.check_password(form.password.data):
            if not user.is_active:
                flash('Tài khoản của bạn đã bị vô hiệu hóa. Vui lòng liên hệ quản trị viên.', 'error')
                return render_template('auth/login.html', form=form)

            # Kiểm tra trạng thái phê duyệt cho giáo viên
            if user.is_teacher() and user.is_pending_approval():
                flash('Tài khoản của bạn đang chờ phê duyệt từ quản trị viên. Vui lòng kiên nhẫn chờ đợi.', 'warning')
                return render_template('auth/login.html', form=form)

            if user.is_teacher() and user.is_rejected():
                flash('Tài khoản của bạn đã bị từ chối. Vui lòng liên hệ quản trị viên để biết thêm chi tiết.', 'error')
                return render_template('auth/login.html', form=form)

            login_user(user, remember=form.remember_me.data)

            # Cập nhật thời gian đăng nhập cuối
            user.last_login = datetime.utcnow()
            db.session.commit()

            flash(f'Chào mừng {user.get_display_name()}!', 'success')

            # Chuyển hướng đến trang được yêu cầu hoặc trang chủ
            next_page = request.args.get('next')
            if next_page:
                return redirect(next_page)
            return redirect(url_for('index'))
        else:
            flash('Tên đăng nhập hoặc mật khẩu không đúng.', 'error')

    return render_template('auth/login.html', form=form)


@app.route('/register', methods=['GET', 'POST'])
def register():
    """Đăng ký cho giáo viên"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    form = TeacherRegisterForm()
    if form.validate_on_submit():
        user = User(
            username=form.username.data,
            email=form.email.data,
            full_name=form.full_name.data,
            role='teacher',
            teacher_id=form.teacher_id.data,
            department=form.department.data,
            position=form.position.data,
            approval_status='pending'  # Tài khoản giáo viên cần được phê duyệt
        )
        user.set_password(form.password.data)

        try:
            db.session.add(user)
            db.session.commit()
            flash(
                'Đăng ký thành công! Tài khoản của bạn đang chờ phê duyệt từ quản trị viên. Bạn sẽ được thông báo qua email khi tài khoản được kích hoạt.',
                'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            flash('Có lỗi xảy ra khi đăng ký. Vui lòng thử lại.', 'error')
            print(f"Error registering teacher: {e}")

    return render_template('auth/register_teacher.html', form=form)


@app.route('/logout')
@login_required
def logout():
    """Đăng xuất"""
    logout_user()
    flash('Bạn đã đăng xuất thành công.', 'info')
    return redirect(url_for('login'))


@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    """Trang thông tin cá nhân"""
    form = ProfileForm()

    # Điền thông tin hiện tại vào form
    if request.method == 'GET':
        form.full_name.data = current_user.full_name
        form.email.data = current_user.email
        form.department.data = current_user.department
        form.position.data = current_user.position

    if form.validate_on_submit():
        # Kiểm tra email không trùng với user khác
        existing_user = User.query.filter(
            User.email == form.email.data,
            User.id != current_user.id
        ).first()

        if existing_user:
            flash('Email đã được sử dụng bởi tài khoản khác.', 'error')
        else:
            current_user.full_name = form.full_name.data
            current_user.email = form.email.data
            current_user.department = form.department.data
            current_user.position = form.position.data

            try:
                db.session.commit()
                flash('Thông tin đã được cập nhật thành công!', 'success')
                return redirect(url_for('profile'))
            except Exception as e:
                db.session.rollback()
                flash('Có lỗi xảy ra khi cập nhật thông tin.', 'error')
                print(f"Error updating profile: {e}")

    return render_template('auth/profile.html', form=form)


@app.route('/change_password', methods=['GET', 'POST'])
@login_required
def change_password():
    """Đổi mật khẩu"""
    form = ChangePasswordForm()

    if form.validate_on_submit():
        if not current_user.check_password(form.current_password.data):
            flash('Mật khẩu hiện tại không đúng.', 'error')
        else:
            current_user.set_password(form.new_password.data)
            try:
                db.session.commit()
                flash('Mật khẩu đã được thay đổi thành công!', 'success')
                return redirect(url_for('profile'))
            except Exception as e:
                db.session.rollback()
                flash('Có lỗi xảy ra khi đổi mật khẩu.', 'error')
                print(f"Error changing password: {e}")

    return render_template('auth/change_password.html', form=form)


# MAIN APPLICATION ROUTES

@app.route('/')
def index():
    # Trang chủ public
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    """API upload file PDF tạm thời cho tìm kiếm (không lưu vào database)"""
    if 'file' not in request.files:
        return jsonify({'error': 'Không có file được chọn'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'Không có file được chọn'}), 400

    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            file.save(filepath)

            # Trích xuất nội dung
            content = extract_text_from_pdf(filepath)

            if not content:
                os.remove(filepath)
                return jsonify({'error': 'Không thể trích xuất nội dung từ file PDF'}), 400

            # Tạo embeddings
            embeddings, index = create_pdf_embeddings(content)

            if embeddings is None:
                os.remove(filepath)
                return jsonify({'error': 'Không thể tạo embeddings cho nội dung PDF'}), 400

            # Lưu vào biến global cho tìm kiếm tạm thời
            current_pdf_data['filename'] = filename
            current_pdf_data['content'] = content
            current_pdf_data['embeddings'] = embeddings
            current_pdf_data['faiss_index'] = index

            return jsonify({
                'message': 'Upload file thành công! Bạn có thể tìm kiếm trong file này ngay bây giờ.',
                'filename': filename,
                'content_count': len(content),
                'timestamp': datetime.now().isoformat(),
                'note': 'File chỉ được lưu tạm thời cho phiên làm việc này.'
            })

        except Exception as e:
            return jsonify({'error': f'Lỗi khi xử lý file: {str(e)}'}), 500

    return jsonify({'error': 'File không hợp lệ. Chỉ chấp nhận file PDF.'}), 400


@app.route('/search', methods=['POST'])
def search():
    # API tìm kiếm
    data = request.get_json()
    query = data.get('query', '')
    k = data.get('k', 5)
    threshold = data.get('threshold', 0.3)
    source = data.get('source', 'default')

    if not query:
        return jsonify({'error': 'Query không được để trống'}), 400

    # Kiểm tra nếu tìm kiếm PDF nhưng chưa upload
    if source == 'pdf' and current_pdf_data['faiss_index'] is None:
        return jsonify({'error': 'Chưa có file PDF nào được upload. Vui lòng upload file PDF trước.'}), 400

    # Thực hiện tìm kiếm với ngưỡng
    results = search_documents(query, k, threshold, source)

    # Lưu lịch sử tìm kiếm (chỉ khi có user đăng nhập)
    if current_user.is_authenticated:
        try:
            search_history = SearchHistory(
                user_id=current_user.id,
                query=query,
                source=source,
                results_count=len(results)
            )
            db.session.add(search_history)
            db.session.commit()
        except Exception as e:
            print(f"Error saving search history: {e}")

    # Kiểm tra nếu không có kết quả
    if not results:
        message = 'Không tìm thấy thông tin liên quan đến truy vấn của bạn.'
        if source == 'pdf':
            message += f' Hãy thử tìm kiếm với từ khóa khác trong file "{current_pdf_data["filename"]}".'
        else:
            message += ' Hãy thử tìm kiếm với từ khóa liên quan đến trường ĐHGTVT.'

        return jsonify({
            'query': query,
            'results': [],
            'no_results': True,
            'source': source,
            'timestamp': datetime.now().isoformat()
        })

    return jsonify({
        'query': query,
        'results': results,
        'no_results': False,
        'source': source,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/pdf_info')
def pdf_info():
    # API lấy thông tin PDF hiện tại
    if current_pdf_data['filename']:
        return jsonify({
            'uploaded': True,
            'filename': current_pdf_data['filename'],
            'content_count': len(current_pdf_data['content']) if current_pdf_data['content'] else 0,
            'has_embeddings': current_pdf_data['embeddings'] is not None,
            'has_faiss_index': current_pdf_data['faiss_index'] is not None,
            'sample_content': current_pdf_data['content'][0]['content'][:200] + '...' if current_pdf_data[
                'content'] else None
        })
    else:
        return jsonify({
            'uploaded': False,
            'filename': None,
            'content_count': 0,
            'has_embeddings': False,
            'has_faiss_index': False,
            'sample_content': None
        })


@app.route('/debug_search', methods=['POST'])
def debug_search():
    # API debug tìm kiếm
    data = request.get_json()
    query = data.get('query', '')
    source = data.get('source', 'default')

    debug_info = {
        'query': query,
        'source': source,
        'current_pdf_data_status': {
            'filename': current_pdf_data['filename'],
            'content_count': len(current_pdf_data['content']) if current_pdf_data['content'] else 0,
            'has_embeddings': current_pdf_data['embeddings'] is not None,
            'has_faiss_index': current_pdf_data['faiss_index'] is not None
        }
    }

    if source == 'pdf' and current_pdf_data['content']:
        debug_info['sample_pdf_content'] = [
            {
                'index': i,
                'content': item['content'][:100] + '...',
                'page': item.get('page', 1)
            }
            for i, item in enumerate(current_pdf_data['content'][:3])
        ]

    return jsonify(debug_info)


@app.route('/documents')
def get_documents():
    # API lấy tất cả documents
    ensure_documents_loaded()
    docs = []
    for i, doc in enumerate(university_documents):
        docs.append({
            'index': i,
            'content': doc
        })
    return jsonify({'documents': docs})


@app.route('/context', methods=['POST'])
def get_context():
    # API lấy ngữ cảnh mở rộng cho một kết quả tìm kiếm
    data = request.get_json()
    source = data.get('source', 'default')
    index = data.get('index')
    context_size = data.get('context_size', 3)

    if index is None:
        return jsonify({'error': 'Index không được để trống'}), 400

    try:
        if source == 'pdf' and current_pdf_data['content']:
            # Lấy ngữ cảnh từ PDF
            content_list = current_pdf_data['content']
            if index >= len(content_list):
                return jsonify({'error': 'Index vượt quá phạm vi'}), 400

            # Lấy ngữ cảnh từ cùng trang và các trang lân cận
            current_item = content_list[index]
            current_page = current_item['page']

            # Tìm tất cả đoạn trong cùng trang và trang lân cận
            context_items = []

            # Trước tiên, lấy tất cả đoạn trong cùng trang
            same_page_items = []
            for i, item in enumerate(content_list):
                if item['page'] == current_page:
                    same_page_items.append({
                        'index': i,
                        'content': item['content'],
                        'page': item['page'],
                        'is_target': i == index
                    })

            # Sắp xếp theo thứ tự index
            same_page_items.sort(key=lambda x: x['index'])

            # Tìm vị trí của target trong cùng trang
            target_pos_in_page = next((i for i, item in enumerate(same_page_items) if item['is_target']), 0)

            # Lấy context trong cùng trang trước
            start_pos = max(0, target_pos_in_page - context_size // 2)
            end_pos = min(len(same_page_items), target_pos_in_page + context_size // 2 + 1)

            context_items = same_page_items[start_pos:end_pos]

            # Nếu không đủ context, bổ sung từ các trang lân cận
            remaining_context = context_size * 2 + 1 - len(context_items)
            if remaining_context > 0:
                adjacent_items = []
                for i, item in enumerate(content_list):
                    page_diff = abs(item['page'] - current_page)
                    if page_diff == 1:  # Chỉ lấy từ trang liền kề
                        adjacent_items.append({
                            'index': i,
                            'content': item['content'],
                            'page': item['page'],
                            'is_target': False
                        })

                # Sắp xếp theo trang và index
                adjacent_items.sort(key=lambda x: (x['page'], x['index']))

                # Thêm một số đoạn từ trang lân cận
                context_items.extend(adjacent_items[:remaining_context])

                # Sắp xếp lại theo trang và index
                context_items.sort(key=lambda x: (x['page'], x['index']))

            final_context = context_items

            return jsonify({
                'context': final_context,
                'target_index': index,
                'total_items': len(final_context),
                'source': 'pdf'
            })

        else:
            # Lấy ngữ cảnh từ dữ liệu mặc định
            ensure_documents_loaded()
            if index >= len(university_documents):
                return jsonify({'error': 'Index vượt quá phạm vi'}), 400

            # Lấy các câu xung quanh
            start_idx = max(0, index - context_size)
            end_idx = min(len(university_documents), index + context_size + 1)

            context_items = []
            for i in range(start_idx, end_idx):
                context_items.append({
                    'index': i,
                    'content': university_documents[i],
                    'is_target': i == index
                })

            return jsonify({
                'context': context_items,
                'target_index': index,
                'total_items': len(context_items),
                'source': 'default'
            })

    except Exception as e:
        return jsonify({'error': f'Lỗi khi lấy ngữ cảnh: {str(e)}'}), 500


@app.route('/search_history')
@login_required
def search_history():
    """Xem lịch sử tìm kiếm của người dùng"""
    page = request.args.get('page', 1, type=int)
    per_page = 20  # số lượng kết quả mỗi trang

    history = (db.session.query(SearchHistory)
               .filter(SearchHistory.user_id == current_user.id)
               .order_by(SearchHistory.search_time.desc())
               .paginate(page=page, per_page=per_page, error_out=False))

    return render_template('search_history.html', history=history)


@app.route('/admin/users')
@login_required
def admin_users():
    """Trang quản lý người dùng - chỉ dành cho admin"""
    if not current_user.is_admin():
        flash('Bạn không có quyền truy cập trang này.', 'error')
        return redirect(url_for('index'))

    page = request.args.get('page', 1, type=int)
    per_page = 20

    users = User.query.order_by(User.created_at.desc()) \
        .paginate(page=page, per_page=per_page, error_out=False)

    return render_template('admin/users.html', users=users)


@app.route('/admin/user/<int:user_id>')
@login_required
def get_user_detail(user_id):
    """API lấy thông tin chi tiết người dùng - chỉ dành cho admin"""
    if not current_user.is_admin():
        return jsonify({'error': 'Không có quyền truy cập'}), 403

    user = db.session.get(User, user_id) or abort(404)

    # Lấy thống kê lịch sử tìm kiếm
    search_count = db.session.query(SearchHistory).filter(SearchHistory.user_id == user_id).count()
    recent_searches = (db.session.query(SearchHistory)
                       .filter(SearchHistory.user_id == user_id)
                       .order_by(SearchHistory.search_time.desc())
                       .limit(5)
                       .all())

    user_data = user.to_dict()
    user_data['search_statistics'] = {
        'total_searches': search_count,
        'recent_searches': [search.to_dict() for search in recent_searches]
    }

    return jsonify(user_data)


@app.route('/admin/toggle_user_status/<int:user_id>', methods=['POST'])
@login_required
def toggle_user_status(user_id):
    """Bật/tắt trạng thái người dùng - chỉ dành cho admin"""
    if not current_user.is_admin():
        return jsonify({'error': 'Không có quyền truy cập'}), 403

    user = db.session.get(User, user_id) or abort(404)

    # Không cho phép admin tự vô hiệu hóa tài khoản của mình
    if user.id == current_user.id:
        return jsonify({'error': 'Không thể thay đổi trạng thái tài khoản của chính mình'}), 400

    try:
        # Toggle trạng thái
        user.is_active = not user.is_active
        db.session.commit()

        action = 'kích hoạt' if user.is_active else 'vô hiệu hóa'
        return jsonify({
            'success': True,
            'message': f'Đã {action} tài khoản {user.full_name}',
            'user': user.to_dict()
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Lỗi khi thay đổi trạng thái: {str(e)}'}), 500


@app.route('/admin/teacher_approvals')
@login_required
def admin_teacher_approvals():
    """Trang phê duyệt tài khoản giáo viên - chỉ dành cho admin"""
    if not current_user.is_admin():
        flash('Bạn không có quyền truy cập trang này.', 'error')
        return redirect(url_for('index'))

    page = request.args.get('page', 1, type=int)
    status_filter = request.args.get('status', 'pending')
    per_page = 20

    # Lấy danh sách giáo viên theo trạng thái
    query = User.query.filter(User.role == 'teacher')

    if status_filter == 'pending':
        query = query.filter(User.approval_status == 'pending')
    elif status_filter == 'approved':
        query = query.filter(User.approval_status == 'approved')
    elif status_filter == 'rejected':
        query = query.filter(User.approval_status == 'rejected')

    teachers = query.order_by(User.created_at.desc()) \
        .paginate(page=page, per_page=per_page, error_out=False)

    # Đếm số lượng theo từng trạng thái
    pending_count = User.query.filter(User.role == 'teacher', User.approval_status == 'pending').count()
    approved_count = User.query.filter(User.role == 'teacher', User.approval_status == 'approved').count()
    rejected_count = User.query.filter(User.role == 'teacher', User.approval_status == 'rejected').count()

    counts = {
        'pending': pending_count,
        'approved': approved_count,
        'rejected': rejected_count
    }

    return render_template('admin/teacher_approvals.html',
                           teachers=teachers,
                           status_filter=status_filter,
                           counts=counts)


@app.route('/admin/approve_teacher/<int:user_id>', methods=['POST'])
@login_required
def approve_teacher(user_id):
    """Phê duyệt tài khoản giáo viên"""
    if not current_user.is_admin():
        return jsonify({'error': 'Không có quyền truy cập'}), 403

    user = db.session.get(User, user_id) or abort(404)

    if user.role != 'teacher':
        return jsonify({'error': 'Chỉ có thể phê duyệt tài khoản giáo viên'}), 400

    if user.approval_status != 'pending':
        return jsonify({'error': 'Tài khoản này không ở trạng thái chờ phê duyệt'}), 400

    try:
        user.approval_status = 'approved'
        db.session.commit()

        return jsonify({
            'success': True,
            'message': f'Đã phê duyệt tài khoản giáo viên {user.full_name}',
            'user': user.to_dict()
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Lỗi khi phê duyệt: {str(e)}'}), 500


@app.route('/admin/reject_teacher/<int:user_id>', methods=['POST'])
@login_required
def reject_teacher(user_id):
    """Từ chối tài khoản giáo viên"""
    if not current_user.is_admin():
        return jsonify({'error': 'Không có quyền truy cập'}), 403

    user = db.session.get(User, user_id) or abort(404)

    if user.role != 'teacher':
        return jsonify({'error': 'Chỉ có thể từ chối tài khoản giáo viên'}), 400

    if user.approval_status != 'pending':
        return jsonify({'error': 'Tài khoản này không ở trạng thái chờ phê duyệt'}), 400

    try:
        user.approval_status = 'rejected'
        db.session.commit()

        return jsonify({
            'success': True,
            'message': f'Đã từ chối tài khoản giáo viên {user.full_name}',
            'user': user.to_dict()
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Lỗi khi từ chối: {str(e)}'}), 500


@app.route('/admin/reset_teacher_status/<int:user_id>', methods=['POST'])
@login_required
def reset_teacher_status(user_id):
    """Reset trạng thái tài khoản giáo viên về pending"""
    if not current_user.is_admin():
        return jsonify({'error': 'Không có quyền truy cập'}), 403

    user = db.session.get(User, user_id) or abort(404)

    if user.role != 'teacher':
        return jsonify({'error': 'Chỉ có thể reset tài khoản giáo viên'}), 400

    try:
        user.approval_status = 'pending'
        db.session.commit()

        return jsonify({
            'success': True,
            'message': f'Đã reset trạng thái tài khoản giáo viên {user.full_name} về chờ phê duyệt',
            'user': user.to_dict()
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Lỗi khi reset: {str(e)}'}), 500


@app.route('/admin/documents')
@login_required
def admin_documents():
    """Trang quản lý documents"""
    if not current_user.is_teacher():
        flash('Bạn không có quyền truy cập trang này.', 'error')
        return redirect(url_for('index'))

    # Lấy tham số page để phân trang
    page = request.args.get('page', 1, type=int)
    per_page = 50  # Hiển thị 50 documents mỗi trang

    all_documents = get_doc_manager().get_documents_info()

    # Tính toán phân trang
    total = len(all_documents)
    start = (page - 1) * per_page
    end = start + per_page
    documents = all_documents[start:end]

    # Tính toán thông tin phân trang
    has_prev = page > 1
    has_next = end < total
    prev_num = page - 1 if has_prev else None
    next_num = page + 1 if has_next else None

    pagination_info = {
        'page': page,
        'per_page': per_page,
        'total': total,
        'pages': (total + per_page - 1) // per_page,
        'has_prev': has_prev,
        'has_next': has_next,
        'prev_num': prev_num,
        'next_num': next_num
    }

    return render_template('admin/documents.html',
                           documents=documents,
                           pagination=pagination_info,
                           total_documents=total)


@app.route('/admin/add_document', methods=['POST'])
@login_required
def add_document():
    """Thêm document mới"""
    if not current_user.is_teacher():
        return jsonify({'error': 'Không có quyền'}), 403

    data = request.get_json()
    content = data.get('content', '').strip()

    if not content:
        return jsonify({'error': 'Nội dung không được để trống'}), 400

    if len(content) < 20:
        return jsonify({'error': 'Nội dung quá ngắn (tối thiểu 20 ký tự)'}), 400

    success, message = get_doc_manager().add_text_document(
        content,
        {'source': 'web_interface'},
        current_user.id
    )

    if success:
        # Cập nhật biến global để tương thích
        global university_documents
        university_documents = get_doc_manager().documents

        return jsonify({'message': message})
    else:
        return jsonify({'error': message}), 500


@app.route('/admin/upload_pdf_file', methods=['POST'])
@login_required
def upload_pdf_file():
    """Upload file PDF và lưu vĩnh viễn vào database"""
    if not current_user.is_teacher():
        return jsonify({'error': 'Không có quyền'}), 403

    if 'file' not in request.files:
        return jsonify({'error': 'Không có file được chọn'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Không có file được chọn'}), 400

    # Kiểm tra loại file
    if not (file.filename.lower().endswith('.pdf')):
        return jsonify({'error': 'Chỉ chấp nhận file .pdf'}), 400

    try:
        # Lưu file tạm thời
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join('temp', filename)

        # Tạo thư mục temp nếu chưa có
        os.makedirs('temp', exist_ok=True)

        file.save(filepath)

        # Thêm documents từ file PDF
        success, message = get_doc_manager().add_documents_from_pdf_file(filepath, current_user.id)

        # Xóa file tạm
        os.remove(filepath)

        if success:
            # Cập nhật biến global
            global university_documents
            university_documents = get_doc_manager().documents

            return jsonify({'message': message})
        else:
            return jsonify({'error': message}), 500

    except Exception as e:
        return jsonify({'error': f'Lỗi xử lý file PDF: {str(e)}'}), 500


@app.route('/admin/delete_document/<doc_id>', methods=['DELETE'])
@login_required
def delete_document(doc_id):
    """Xóa document"""
    if not current_user.is_teacher():
        return jsonify({'error': 'Không có quyền'}), 403

    success, message = get_doc_manager().delete_document(doc_id)

    if success:
        # Cập nhật biến global
        global university_documents
        university_documents = get_doc_manager().documents

        return jsonify({'message': message})
    else:
        return jsonify({'error': message}), 404


@app.route('/admin/export_documents')
@login_required
def export_documents():
    """Export tất cả documents ra file"""
    if not current_user.is_teacher():
        flash('Bạn không có quyền truy cập trang này.', 'error')
        return redirect(url_for('index'))

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"documents_export_{timestamp}.txt"
        filepath = os.path.join('exports', filename)

        # Tạo thư mục exports nếu chưa có
        os.makedirs('exports', exist_ok=True)

        success, message = get_doc_manager().export_to_file(filepath)

        if success:
            # Trả về file để download
            return send_file(filepath, as_attachment=True, download_name=filename)
        else:
            flash(f'Lỗi export: {message}', 'error')
            return redirect(url_for('admin_documents'))

    except Exception as e:
        flash(f'Lỗi export: {str(e)}', 'error')
        return redirect(url_for('admin_documents'))


@app.route('/admin/document_detail/<doc_id>')
@login_required
def get_document_detail(doc_id):
    """API lấy chi tiết document"""
    if not current_user.is_teacher():
        return jsonify({'error': 'Không có quyền truy cập'}), 403

    try:
        # Tìm document trong DocumentManager
        dm = get_doc_manager()
        
        # Tìm document theo ID
        doc_found = None
        doc_index = None
        
        for i, metadata in enumerate(dm.metadata):
            if metadata.get('id') == doc_id:
                doc_found = metadata
                doc_index = i
                break
        
        if not doc_found:
            return jsonify({'error': 'Không tìm thấy document'}), 404
        
        # Lấy nội dung document
        if doc_index < len(dm.documents):
            # Lấy nội dung đã xử lý
            processed_content = dm.documents[doc_index]
            # Lấy nội dung gốc nếu có
            original_content = doc_found.get('original_content', processed_content)
        else:
            return jsonify({'error': 'Document index không hợp lệ'}), 404
        
        # Tạo source badge HTML
        source = doc_found.get('source', 'unknown')
        source_file = doc_found.get('source_file')
        
        if source == 'data.txt':
            source_badge = '<span class="badge bg-primary"><i class="fas fa-file"></i> Dữ liệu gốc</span>'
        elif source == 'manual':
            source_badge = '<span class="badge bg-success"><i class="fas fa-keyboard"></i> Tự nhập</span>'
        elif source == 'pdf':
            source_badge = '<span class="badge bg-warning text-dark"><i class="fas fa-file-pdf"></i> PDF Upload</span>'
        elif source_file:
            source_badge = f'<span class="badge bg-warning text-dark"><i class="fas fa-upload"></i> {source_file}</span>'
        else:
            source_badge = f'<span class="badge bg-secondary"><i class="fas fa-question"></i> {source}</span>'
        
        # Thống kê nội dung
        word_count = len(original_content.split())
        sentence_count = len([s for s in original_content.split('.') if s.strip()])
        
        # Thông tin dòng/trang
        line_number = doc_found.get('line_number')
        page_number = doc_found.get('page_number')
        
        if line_number and page_number:
            line_page_info = f"Dòng {line_number}, Trang {page_number}"
        elif line_number:
            line_page_info = f"Dòng {line_number}"
        elif page_number:
            line_page_info = f"Trang {page_number}"
        else:
            line_page_info = "N/A"
        
        # Format ngày thêm
        added_date = doc_found.get('added_date')
        if added_date:
            try:
                # Parse ISO format và format lại
                from datetime import datetime
                dt = datetime.fromisoformat(added_date.replace('Z', '+00:00'))
                formatted_date = dt.strftime('%d/%m/%Y %H:%M:%S')
            except:
                formatted_date = added_date
        else:
            formatted_date = 'N/A'
        
        # Format người thêm
        added_by = doc_found.get('added_by')
        if added_by:
            if str(added_by).isdigit():
                # Tìm user theo ID
                user = db.session.get(User, int(added_by))
                if user:
                    added_by_display = f"{user.full_name} (ID: {added_by})"
                else:
                    added_by_display = f"User ID: {added_by}"
            else:
                added_by_display = str(added_by)
        else:
            added_by_display = 'Hệ thống'
        
        # Tạo metadata info để hiển thị
        metadata_info = None
        if doc_found:
            import json
            # Loại bỏ các field đã hiển thị ở trên
            display_metadata = {k: v for k, v in doc_found.items() 
                              if k not in ['id', 'original_content', 'added_date', 'added_by', 'source', 'source_file']}
            if display_metadata:
                metadata_info = json.dumps(display_metadata, ensure_ascii=False, indent=2)
        
        return jsonify({
            'id': doc_id,
            'length': len(original_content),
            'full_content': original_content,
            'source_badge': source_badge,
            'source_file': source_file,
            'added_by': added_by_display,
            'added_date': formatted_date,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'line_page_info': line_page_info,
            'metadata_info': metadata_info
        })
        
    except Exception as e:
        return jsonify({'error': f'Lỗi server: {str(e)}'}), 500


# DATABASE INITIALIZATION

def init_database():
    """Khởi tạo database và tạo bảng"""
    with app.app_context():
        db.create_all()
        print("Database đã được khởi tạo thành công!")

        # Tạo tài khoản admin mặc định nếu chưa có
        admin = User.query.filter_by(username='admin').first()
        if not admin:
            admin = User(
                username='admin',
                email='admin@university.edu.vn',
                full_name='Quản trị viên',
                role='teacher',
                teacher_id='ADMIN001',
                department='Khoa Công nghệ thông tin',
                position='giao_su',
                approval_status='approved'  # Admin luôn được phê duyệt
            )
            admin.set_password('admin123')  # Đổi password này trong production

            db.session.add(admin)
            db.session.commit()
            print("Tài khoản admin mặc định đã được tạo!")
            print("Username: admin")
            print("Password: admin123")
            print("Vui lòng đổi mật khẩu sau khi đăng nhập!")
        
        # Kiểm tra số lượng documents trong database
        document_count = Document.query.filter_by(is_active=True).count()
        print(f"📄 Hiện có {document_count} documents trong database")
        
        # Khởi tạo DocumentManager ngay để sẵn sàng cho search
        print("Khởi tạo DocumentManager và embeddings...")
        init_doc_manager()
        print("Hệ thống đã sẵn sàng cho tìm kiếm!")


if __name__ == '__main__':
    print("Khởi động ứng dụng Flask...")

    # Khởi tạo database
    init_database()

    app.run()
