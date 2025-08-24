"""
Document Manager - Quản lý động documents và embeddings
"""

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
                print("📄 Database chưa có documents nào")
                
        except Exception as e:
            print(f"❌ Lỗi khi load từ database: {e}")
        
        # Rebuild index từ dữ liệu đã load
        if self.documents:
            print("🔧 Building index từ documents...")
            self.rebuild_index()
        else:
            print("ℹ️ Không có documents để build index")
        
        print(f"✅ DocumentManager đã sẵn sàng với {len(self.documents)} documents")
    

    


    
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
            
            if self.faiss_index is None:
                # Tạo index mới nếu chưa có
                embedding_dim = new_embedding_np.shape[1]
                self.faiss_index = faiss.IndexFlatL2(embedding_dim)
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
            self.embeddings = embeddings.cpu().numpy()
            print(f"✅ Embeddings shape: {self.embeddings.shape}")
            
            # Tạo FAISS index mới
            print("🔄 Tạo FAISS index...")
            embedding_dim = self.embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatL2(embedding_dim)
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
    

    
    def search(self, query, k=5, similarity_threshold=0.3):
        """Tìm kiếm documents"""
        if self.faiss_index is None or len(self.documents) == 0:
            return []
        
        # Kiểm tra đồng bộ giữa documents và embeddings
        if hasattr(self, 'embeddings') and self.embeddings is not None and len(self.documents) != self.embeddings.shape[0]:
            print(f"Warning: Documents ({len(self.documents)}) và embeddings ({self.embeddings.shape[0]}) không đồng bộ. Rebuilding index...")
            self.rebuild_index()
        
        try:
            # Tạo embedding cho query
            query_embedding = self.model.encode([query], convert_to_tensor=True)
            query_np = query_embedding.cpu().numpy()
            
            # Tìm kiếm với FAISS
            distances, indices = self.faiss_index.search(query_np, min(k, len(self.documents)))
        except Exception as e:
            print(f"❌ Lỗi trong quá trình search: {e}")
            return []
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= len(self.documents):
                continue
            
            # Tính cosine similarity
            doc_embedding = self.embeddings[idx:idx+1]
            
            # Kiểm tra embedding hợp lệ
            if len(doc_embedding) == 0 or len(doc_embedding[0]) == 0:
                continue
                
            doc_norm = np.linalg.norm(doc_embedding[0])
            query_norm = np.linalg.norm(query_np[0])
            
            if doc_norm == 0 or query_norm == 0:
                continue
                
            similarity = np.dot(query_np[0], doc_embedding[0]) / (query_norm * doc_norm)
            
            if similarity >= similarity_threshold:
                results.append({
                    'index': int(idx),
                    'content': self.documents[idx],
                    'metadata': self.metadata[idx] if idx < len(self.metadata) else {},
                    'score': float(distances[0][i]),
                    'similarity': float(similarity),
                    'rank': len(results) + 1,
                    'source': 'default'
                })
        
        return results
    
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
        print("=== DocumentManager Debug Status ===")
        for key, value in status.items():
            print(f"{key}: {value}")
        print("=====================================")
        return status
