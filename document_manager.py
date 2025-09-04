
import os
import json
import numpy as np
import faiss
import pickle
from datetime import datetime
from models import db, User, Document
import re

class DocumentManager:
    def __init__(self, model, data_dir='data', index_file='search_index.pkl'):
        self.model = model
        self.data_dir = data_dir
        self.index_file = index_file
        self.documents = []
        self.embeddings = None
        self.faiss_index = None
        self.metadata = []
        
        # Tạo thư mục data nếu chưa có
        os.makedirs(data_dir, exist_ok=True)
        
        # Load dữ liệu hiện có
        self.load_existing_data()
    
    def load_existing_data(self):
        """Load dữ liệu từ database"""
        print("Đang load dữ liệu từ database...")
        
        # Xóa dữ liệu cũ
        self.documents = []
        self.metadata = []
        self.embeddings = None
        self.faiss_index = None
        
        try:
            # Load từ database
            db_documents = Document.query.filter_by(is_active=True).order_by(Document.id).all()
            
            if db_documents:
                print(f"🔄 Đang load {len(db_documents)} documents từ database...")
                for doc in db_documents:
                    self.documents.append(doc.content)
                    self.metadata.append({
                        'id': f'db_{doc.id}',
                        'db_id': doc.id,
                        'source': doc.source_type,
                        'source_file': doc.source_file,
                        'added_date': doc.added_date.isoformat() if doc.added_date else None,
                        'added_by': doc.added_by,
                        'type': 'database',
                        'line_number': doc.line_number,
                        'page_number': doc.page_number
                    })
                print(f"✅ Đã load {len(self.documents)} documents từ database")
            else:
                print("Database chưa có documents nào")
                
        except Exception as e:
            print(f"❌ Lỗi khi load từ database: {e}")
        
        # Rebuild index từ dữ liệu đã load
        if self.documents:
            print("🔧 Building index từ documents...")
            self.rebuild_index()
        else:
            print("Không có documents để build index")
        
        print(f"DocumentManager đã sẵn sàng với {len(self.documents)} documents")
    

    


    
    def add_text_document(self, text, metadata=None, user_id=None):
        """Thêm document text mới (lưu vào database)"""
        if not text or not text.strip():
            return False, "Nội dung rỗng"
        
        try:
            # Tạo source_info từ metadata
            source_info = metadata if metadata else {}
            
            # Tạo document trong database
            doc = Document.create_from_text(
                content=text.strip(),
                source_type='manual',
                source_file=None,
                source_info=source_info,
                added_by=user_id
            )
            
            # Lưu vào database
            db.session.add(doc)
            db.session.commit()
            
            # Thêm vào memory để có thể search ngay
            self.documents.append(text.strip())
            doc_metadata = {
                'id': f'db_{doc.id}',
                'db_id': doc.id,
                'source': 'manual',
                'added_date': doc.added_date.isoformat(),
                'added_by': user_id,
                'type': 'database',
                'length': len(text.strip())
            }
            self.metadata.append(doc_metadata)
            
            # Cập nhật index
            success = self._add_to_index(text.strip())
            
            if success:
                return True, f"Đã thêm document ID: {doc.id}"
            else:
                # Rollback nếu lỗi index
                self.documents.pop()
                self.metadata.pop()
                return False, "Lỗi khi cập nhật search index"
                
        except Exception as e:
            db.session.rollback()
            return False, f"Lỗi lưu database: {str(e)}"
    
    def add_documents_from_pdf_file(self, file_path, user_id=None):
        """Thêm documents từ file PDF (lưu vào database)"""
        if not os.path.exists(file_path):
            return False, "File không tồn tại"
        
        try:
            if file_path.endswith('.pdf'):
                documents = self._extract_from_pdf(file_path)
            else:
                return False, "Chỉ hỗ trợ file .pdf"
            
            if not documents:
                return False, "Không tìm thấy nội dung trong file PDF"
            
            # Thêm từng document vào database
            added_count = 0
            source_file = os.path.basename(file_path)
            
            for i, doc_text in enumerate(documents):
                if len(doc_text.strip()) > 50:  # Chỉ thêm document đủ dài
                    try:
                        # Tạo document trong database
                        doc = Document.create_from_text(
                            content=doc_text.strip(),
                            source_type='pdf',
                            source_file=source_file,
                            source_info={'pdf_extract_index': i},
                            added_by=user_id,
                            page_number=i + 1  # Giả định mỗi document từ 1 page
                        )
                        
                        db.session.add(doc)
                        db.session.commit()
                        
                        # Thêm vào memory
                        self.documents.append(doc_text.strip())
                        self.metadata.append({
                            'id': f'db_{doc.id}',
                            'db_id': doc.id,
                            'source': 'pdf',
                            'source_file': source_file,
                            'added_date': doc.added_date.isoformat(),
                            'added_by': user_id,
                            'type': 'database',
                            'page_number': i + 1
                        })
                        
                        # Cập nhật index
                        self._add_to_index(doc_text.strip())
                        added_count += 1
                        
                    except Exception as doc_error:
                        print(f"Lỗi thêm document {i}: {doc_error}")
                        db.session.rollback()
                        continue
            
            return True, f"Đã thêm {added_count}/{len(documents)} documents từ file PDF"
            
        except Exception as e:
            return False, f"Lỗi xử lý file PDF: {str(e)}"
    

    
    def _extract_from_txt(self, file_path):
        """Trích xuất từ file txt"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Chia thành các paragraph
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        # Nếu không có paragraph, chia theo dòng
        if len(paragraphs) <= 1:
            paragraphs = [line.strip() for line in content.split('\n') if line.strip()]
        
        return paragraphs
    
    def _extract_from_pdf(self, file_path):
        """Trích xuất từ file PDF (sử dụng PyPDF2 như trong app.py)"""
        try:
            import PyPDF2
            
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text_content = []
                
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        # Chia thành các paragraph
                        paragraphs = page_text.split('\n\n')
                        for para in paragraphs:
                            para = para.strip()
                            if len(para) > 50:  # Chỉ lấy paragraph đủ dài
                                text_content.append(para)
                
                return text_content
        except Exception as e:
            print(f"Lỗi đọc file PDF: {e}")
            return []


    
    def _add_to_index(self, text):
        """Thêm một document vào index hiện có"""
        try:
            # Tạo embedding cho text mới
            new_embedding = self.model.encode([text], convert_to_tensor=True)
            new_embedding_np = new_embedding.cpu().numpy()
            
            # Chuẩn hóa vector về độ dài 1 (để sử dụng cosine similarity)
            norms = np.linalg.norm(new_embedding_np, axis=1, keepdims=True) + 1e-12
            new_embedding_np = new_embedding_np / norms
            
            if self.faiss_index is None:
                # Tạo index mới nếu chưa có - sử dụng Inner Product cho cosine similarity
                embedding_dim = new_embedding_np.shape[1]
                self.faiss_index = faiss.IndexFlatIP(embedding_dim)
                self.embeddings = new_embedding_np
            else:
                # Thêm vào index hiện có
                self.embeddings = np.vstack([self.embeddings, new_embedding_np])
            
            # Thêm vào FAISS index
            self.faiss_index.add(new_embedding_np)
            
            return True
        except Exception as e:
            print(f"Lỗi thêm vào index: {e}")
            return False
    
    def rebuild_index(self):
        """Rebuild toàn bộ index từ documents hiện có"""
        if not self.documents:
            print("❌ Không có documents để rebuild")
            return
        
        print(f"🔧 Rebuilding index cho {len(self.documents)} documents...")
        
        try:
            # Reset trước khi rebuild
            self.embeddings = None
            self.faiss_index = None
            
            # Tạo embeddings cho tất cả documents
            print("🔄 Tạo embeddings...")
            embeddings = self.model.encode(self.documents, convert_to_tensor=True)
            embeddings_np = embeddings.cpu().numpy()
            
            # Chuẩn hóa để sử dụng cosine similarity
            norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True) + 1e-12
            self.embeddings = embeddings_np / norms
            print(f"✅ Embeddings shape: {self.embeddings.shape}")
            
            # Tạo FAISS index mới - sử dụng Inner Product cho cosine similarity
            print("🔄 Tạo FAISS index (cosine similarity)...")
            embedding_dim = self.embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(embedding_dim)
            self.faiss_index.add(self.embeddings)
            print(f"✅ FAISS index với {self.faiss_index.ntotal} vectors")
            
            print(f"✅ Rebuild hoàn thành: {len(self.documents)} documents")
            
            # Lưu index
            self.save_index()
            
        except Exception as e:
            print(f"❌ Lỗi rebuild index: {e}")
            # Reset về trạng thái an toàn
            self.embeddings = None
            self.faiss_index = None
    

    
    def search(self, query, k=5, similarity_threshold=0.5):
        """Tìm kiếm documents với cải tiến độ chính xác"""
        if self.faiss_index is None or len(self.documents) == 0:
            return []
        
        # Kiểm tra đồng bộ giữa documents và embeddings
        if hasattr(self, 'embeddings') and self.embeddings is not None and len(self.documents) != self.embeddings.shape[0]:
            print(f"Warning: Documents ({len(self.documents)}) và embeddings ({self.embeddings.shape[0]}) không đồng bộ. Rebuilding index...")
            self.rebuild_index()
        
        try:
            # Tạo embedding cho query gốc (không expand quá rộng)
            original_embedding = self.model.encode([query], convert_to_tensor=True)
            original_np = original_embedding.cpu().numpy()
            
            # Chuẩn hóa query embedding
            norms = np.linalg.norm(original_np, axis=1, keepdims=True) + 1e-12
            query_np = original_np / norms
            
            # Tìm kiếm với FAISS (lấy nhiều hơn để có thể filter và re-rank)
            search_k = min(k * 5, len(self.documents))
            distances, indices = self.faiss_index.search(query_np, search_k)
        except Exception as e:
            print(f"❌ Lỗi trong quá trình search: {e}")
            return []
        
        candidates = []
        for i, idx in enumerate(indices[0]):
            if idx >= len(self.documents):
                continue
            
            # Với normalized embeddings và IndexFlatIP, distances chính là cosine similarity
            similarity = float(distances[0][i])
            
            # Áp dụng ngưỡng similarity cao hơn
            if similarity >= similarity_threshold:
                doc_content = self.documents[idx]
                
                # Tính lexical overlap score (quan trọng hơn)
                overlap_score = self._calculate_keyword_overlap(query, doc_content)
                
                # Kiểm tra relevance contextual
                context_score = self._calculate_context_relevance(query, doc_content)
                
                # Penalty cho những document có nhiều từ khóa không liên quan
                noise_penalty = self._calculate_noise_penalty(query, doc_content)
                
                # Kết hợp điểm số với trọng số mới:
                # - 50% semantic similarity
                # - 30% keyword overlap  
                # - 20% context relevance
                # - Trừ noise penalty
                combined_score = (0.5 * similarity + 
                                0.3 * overlap_score + 
                                0.2 * context_score - 
                                0.1 * noise_penalty)
                
                # Chỉ lấy những kết quả có điểm tổng hợp cao
                if combined_score >= 0.4:  # Ngưỡng tổng hợp cao hơn
                    candidates.append({
                        'index': int(idx),
                        'content': doc_content,
                        'metadata': self.metadata[idx] if idx < len(self.metadata) else {},
                        'score': float(combined_score),
                        'similarity': similarity,
                        'overlap_score': overlap_score,
                        'context_score': context_score,
                        'noise_penalty': noise_penalty,
                        'rank': 0,  # Will be set after sorting
                        'source': 'default'
                    })
        
        # Re-rank theo combined score
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # Lọc thêm bằng cách kiểm tra sự liên quan thực sự
        filtered_candidates = []
        for candidate in candidates:
            if self._is_truly_relevant(query, candidate['content']):
                filtered_candidates.append(candidate)
        
        # Chỉ lấy top k và gán rank
        results = []
        for rank, candidate in enumerate(filtered_candidates[:k], 1):
            candidate['rank'] = rank
            results.append(candidate)
        
        return results
    
    def _expand_query_vietnamese(self, query):
        """Mở rộng query với các từ đồng nghĩa tiếng Việt"""
        query_lower = query.lower()
        expansions = [query]  # Luôn bao gồm query gốc
        
        # Từ điển đồng nghĩa cho các thuật ngữ phổ biến
        synonyms_map = {
            'rèn luyện': ['điểm rèn luyện', 'đánh giá rèn luyện', 'DRL', 'điểm DRL', 'rèn luyện sinh viên', 'ĐÁNH GIÁ RÈN LUYỆN SINH VIÊN'],
            'cố vấn học tập': ['cố vấn', 'tư vấn học tập', 'đánh giá cố vấn', 'cố vấn hoc tap', 'ĐÁNH GIÁ CỐ VẤN HỌC TẬP'],
            'học phí': ['mức học phí', 'thu học phí', 'miễn giảm học phí', 'hoc phi'],
            'học bổng': ['hoc bong', 'học bổng khuyến khích', 'học bổng khuyến học'],
            'sinh viên': ['sinh vien', 'học sinh', 'hoc sinh'],
            'giảng viên': ['giang vien', 'thầy cô', 'giáo viên'],
            'tuyển sinh': ['tuyen sinh', 'xét tuyển', 'thi tuyển'],
            'đào tạo': ['dao tao', 'chương trình đào tạo', 'chuong trinh dao tao'],
            'thư viện': ['thu vien', 'library', 'kho sách'],
            'ký túc xá': ['ki tuc xa', 'ktx', 'dormitory'],
            'hoạt động': ['hoat dong', 'sinh hoạt', 'tổ chức']
        }
        
        # Tìm các từ khóa phù hợp và thêm đồng nghĩa
        for key, synonyms in synonyms_map.items():
            if key in query_lower or any(syn in query_lower for syn in synonyms):
                for synonym in synonyms:
                    if synonym not in expansions:
                        expansions.append(synonym)
        
        return expansions[:5]  # Giới hạn số lượng để tránh quá tải
    
    def _calculate_keyword_overlap(self, query, document):
        """Tính điểm overlap từ khóa giữa query và document"""
        # Chuẩn hóa text
        query_clean = re.sub(r'[^\w\s]', ' ', query.lower())
        doc_clean = re.sub(r'[^\w\s]', ' ', document.lower())
        
        # Tách từ và loại bỏ từ ngắn
        query_words = set([w for w in query_clean.split() if len(w) > 2])
        doc_words = set([w for w in doc_clean.split() if len(w) > 2])
        
        if not query_words:
            return 0.0
        
        # Tính tỷ lệ từ trùng khớp
        intersection = len(query_words & doc_words)
        return intersection / len(query_words)
    
    def _calculate_context_relevance(self, query, document):
        """Tính điểm liên quan theo ngữ cảnh"""
        query_lower = query.lower()
        doc_lower = document.lower()
        
        # Định nghĩa các chủ đề chính và từ khóa liên quan
        topic_keywords = {
            'học_tập': ['học tập', 'học phí', 'môn học', 'tín chỉ', 'điểm', 'thi', 'kiểm tra', 'bài tập', 'giảng dạy', 'chương trình'],
            'sinh_viên': ['sinh viên', 'học sinh', 'tân sinh viên', 'cựu sinh viên', 'lớp', 'khóa học'],
            'giảng_viên': ['giảng viên', 'giáo viên', 'thầy', 'cô', 'giáo sư', 'phó giáo sư', 'tiến sĩ'],
            'hành_chính': ['đăng ký', 'thủ tục', 'giấy tờ', 'chứng nhận', 'xác nhận', 'phòng ban', 'văn phòng'],
            'cơ_sở_vật_chất': ['thư viện', 'phòng học', 'giảng đường', 'phòng thí nghiệm', 'ký túc xá', 'ktx'],
            'hoạt_động': ['hoạt động', 'sự kiện', 'hội thảo', 'seminar', 'nghiên cứu', 'khoa học'],
            'quy_định': ['quy định', 'quy chế', 'luật', 'điều', 'khoản', 'nghị định', 'thông tư']
        }
        
        # Xác định chủ đề của query
        query_topics = []
        for topic, keywords in topic_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                query_topics.append(topic)
        
        if not query_topics:
            return 0.5  # Điểm trung bình nếu không xác định được chủ đề
        
        # Tính điểm liên quan theo chủ đề
        relevance_score = 0.0
        for topic in query_topics:
            topic_keywords_in_doc = sum(1 for keyword in topic_keywords[topic] if keyword in doc_lower)
            if topic_keywords_in_doc > 0:
                relevance_score += topic_keywords_in_doc / len(topic_keywords[topic])
        
        return min(relevance_score / len(query_topics), 1.0)
    
    def _calculate_noise_penalty(self, query, document):
        """Tính penalty cho document có nhiều từ khóa không liên quan"""
        doc_lower = document.lower()
        query_lower = query.lower()
        
        # Danh sách từ khóa "nhiễu" thường gây nhầm lẫn
        noise_keywords = {
            'thể_thao': ['đá cầu', 'bóng bàn', 'bóng đá', 'bóng chuyền', 'cầu lông', 'tennis', 'bóng rổ', 'võ thuật', 'thể dục', 'thể thao'],
            'câu_lạc_bộ': ['clb', 'câu lạc bộ', 'club', 'nhóm', 'đội'],
            'địa_điểm_xa': ['hà nội', 'hồ chí minh', 'đà nẵng', 'cần thơ'] if not any(city in query_lower for city in ['hà nội', 'hồ chí minh', 'đà nẵng', 'cần thơ']) else [],
            'thông_tin_cá_nhân': ['số điện thoại', 'email cá nhân', 'địa chỉ nhà'] if not any(info in query_lower for info in ['liên hệ', 'thông tin']) else []
        }
        
        penalty = 0.0
        total_noise_words = 0
        
        for category, keywords in noise_keywords.items():
            noise_count = sum(1 for keyword in keywords if keyword in doc_lower)
            total_noise_words += len(keywords)
            
            # Nếu query không liên quan đến category này nhưng document có nhiều từ khóa category
            if noise_count > 0:
                category_in_query = any(keyword in query_lower for keyword in keywords)
                if not category_in_query:
                    penalty += noise_count / len(keywords)
        
        return min(penalty, 1.0)
    
    def _is_truly_relevant(self, query, document):
        """Kiểm tra document có thực sự liên quan đến query không"""
        query_lower = query.lower()
        doc_lower = document.lower()
        
        # Nếu query về học tập mà document chỉ nói về thể thao -> không liên quan
        academic_keywords = ['học', 'thi', 'kiểm tra', 'điểm', 'môn', 'tín chỉ', 'chương trình', 'đào tạo']
        sports_keywords = ['đá cầu', 'bóng bàn', 'bóng đá', 'thể thao', 'câu lạc bộ thể thao']
        
        query_is_academic = any(keyword in query_lower for keyword in academic_keywords)
        doc_is_mainly_sports = (
            sum(1 for keyword in sports_keywords if keyword in doc_lower) >= 2 and
            sum(1 for keyword in academic_keywords if keyword in doc_lower) == 0
        )
        
        if query_is_academic and doc_is_mainly_sports:
            return False
        
        # Nếu query về quy định mà document chỉ nói về hoạt động giải trí
        regulation_keywords = ['quy định', 'quy chế', 'luật', 'điều', 'khoản', 'nghị định']
        entertainment_keywords = ['giải trí', 'vui chơi', 'sinh hoạt', 'party', 'lễ hội']
        
        query_is_regulation = any(keyword in query_lower for keyword in regulation_keywords)
        doc_is_entertainment = sum(1 for keyword in entertainment_keywords if keyword in doc_lower) >= 2
        
        if query_is_regulation and doc_is_entertainment:
            return False
        
        # Kiểm tra độ dài tối thiểu của overlap
        query_words = set(re.findall(r'\b\w{3,}\b', query_lower))
        doc_words = set(re.findall(r'\b\w{3,}\b', doc_lower))
        overlap = len(query_words & doc_words)
        
        # Cần có ít nhất 1 từ trùng khớp hoặc semantic similarity cao
        return overlap >= 1 or len(query_words) == 0
    
    def delete_document(self, doc_id):
        """Xóa document theo ID (từ database và memory)"""
        try:
            # Tìm trong database trước
            if doc_id.startswith('db_'):
                db_id = int(doc_id.replace('db_', ''))
                doc = db.session.get(Document, db_id)
                if doc:
                    # Xóa trong database (soft delete)
                    doc.is_active = False
                    db.session.commit()
                    
                    # Xóa khỏi memory
                    for i, meta in enumerate(self.metadata):
                        if meta.get('db_id') == db_id:
                            self.documents.pop(i)
                            self.metadata.pop(i)
                            break
                    
                    # Rebuild index
                    self.rebuild_index()
                    return True, f"Đã xóa document {doc_id}"
            
            # Fallback: tìm theo ID trong metadata
            for i, meta in enumerate(self.metadata):
                if meta.get('id') == doc_id:
                    # Nếu có db_id thì xóa trong database
                    if 'db_id' in meta:
                        doc = db.session.get(Document, meta['db_id'])
                        if doc:
                            doc.is_active = False
                            db.session.commit()
                    
                    # Xóa khỏi memory
                    self.documents.pop(i)
                    self.metadata.pop(i)
                    
                    # Rebuild index
                    self.rebuild_index()
                    return True, f"Đã xóa document {doc_id}"
            
            return False, "Không tìm thấy document"
            
        except Exception as e:
            db.session.rollback()
            return False, f"Lỗi xóa document: {str(e)}"
    
    def get_documents_info(self):
        """Lấy thông tin tất cả documents"""
        info = []
        for i, (doc, meta) in enumerate(zip(self.documents, self.metadata)):
            info.append({
                'index': i,
                'id': meta.get('id', f'doc_{i}'),
                'preview': doc[:100] + '...' if len(doc) > 100 else doc,
                'length': len(doc),
                'metadata': meta
            })
        return info
    
    def save_index(self):
        """Lưu index và metadata"""
        try:
            data = {
                'documents': self.documents,
                'metadata': self.metadata,
                'embeddings': self.embeddings,
                'index_data': faiss.serialize_index(self.faiss_index) if self.faiss_index else None,
                'saved_at': datetime.now().isoformat()
            }
            
            with open(self.index_file, 'wb') as f:
                pickle.dump(data, f)
            
            print(f"Đã lưu index với {len(self.documents)} documents")
            return True
        except Exception as e:
            print(f"Lỗi lưu index: {e}")
            return False
    
    def load_index(self):
        """Load index đã lưu"""
        try:
            with open(self.index_file, 'rb') as f:
                data = pickle.load(f)
            
            self.documents = data['documents']
            self.metadata = data['metadata']
            
            # Xử lý embeddings an toàn
            if 'embeddings' in data and data['embeddings'] is not None:
                embeddings_data = data['embeddings']
                if hasattr(embeddings_data, 'shape'):
                    self.embeddings = embeddings_data
                else:
                    # Convert to numpy array if needed
                    self.embeddings = np.array(embeddings_data)
            else:
                self.embeddings = None
            
            # Load FAISS index
            if 'index_data' in data and data['index_data'] is not None:
                try:
                    self.faiss_index = faiss.deserialize_index(data['index_data'])
                except Exception as faiss_error:
                    print(f"Lỗi load FAISS index: {faiss_error}")
                    self.faiss_index = None
            else:
                self.faiss_index = None
            
            # Kiểm tra tính nhất quán
            if self.embeddings is not None and len(self.documents) != self.embeddings.shape[0]:
                print(f"Warning: Mismatch documents ({len(self.documents)}) vs embeddings ({self.embeddings.shape[0]})")
                self.embeddings = None
                self.faiss_index = None
            
            print(f"Đã load index với {len(self.documents)} documents")
            print(f"Embeddings: {self.embeddings.shape if self.embeddings is not None else 'None'}")
            print(f"FAISS index: {'OK' if self.faiss_index is not None else 'None'}")
            
            return True
        except Exception as e:
            print(f"Lỗi load index: {e}")
            self.embeddings = None
            self.faiss_index = None
            return False
    
    def export_to_file(self, file_path):
        """Export tất cả documents ra file"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                for i, doc in enumerate(self.documents):
                    f.write(f"=== Document {i+1} ===\n")
                    f.write(f"ID: {self.metadata[i].get('id', 'N/A')}\n")
                    f.write(f"Added: {self.metadata[i].get('added_date', 'N/A')}\n")
                    f.write(f"Source: {self.metadata[i].get('source', 'N/A')}\n")
                    f.write("Content:\n")
                    f.write(doc)
                    f.write("\n\n")
            return True, f"Đã export {len(self.documents)} documents"
        except Exception as e:
            return False, f"Lỗi export: {str(e)}"
    
    def debug_status(self):
        """Debug thông tin hiện tại của DocumentManager"""
        status = {
            'documents_count': len(self.documents),
            'has_embeddings': self.embeddings is not None,
            'embeddings_shape': self.embeddings.shape if self.embeddings is not None else None,
            'has_faiss_index': self.faiss_index is not None,
            'faiss_index_size': self.faiss_index.ntotal if self.faiss_index is not None else None,
            'metadata_count': len(self.metadata),
            'model_available': self.model is not None
        }
        print("DocumentManager Debug Status")
        for key, value in status.items():
            print(f"{key}: {value}")
        return status
