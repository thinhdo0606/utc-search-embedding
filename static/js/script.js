
document.addEventListener('DOMContentLoaded', function() {
    const searchForm = document.getElementById('searchForm');
    const queryInput = document.getElementById('queryInput');
    const kValue = document.getElementById('kValue');
    const loading = document.getElementById('loading');
    const resultsSection = document.getElementById('resultsSection');
    const searchResults = document.getElementById('searchResults');
    const resultCount = document.getElementById('resultCount');

    const sampleQueryButtons = document.querySelectorAll('.sample-query');
    
    // PDF upload elements
    const uploadForm = document.getElementById('uploadForm');
    const pdfFile = document.getElementById('pdfFile');
    const uploadBtn = document.getElementById('uploadBtn');
    const pdfInfo = document.getElementById('pdfInfo');
    const currentPdfName = document.getElementById('currentPdfName');
    const currentPdfContent = document.getElementById('currentPdfContent');
    
    // Search source elements
    const sourceDefault = document.getElementById('sourceDefault');
    const sourcePdf = document.getElementById('sourcePdf');

    // Handle form submission
    searchForm.addEventListener('submit', function(e) {
        e.preventDefault();
        performSearch();
    });

    // Handle sample query buttons
    sampleQueryButtons.forEach(button => {
        button.addEventListener('click', function() {
            const query = this.getAttribute('data-query');
            queryInput.value = query;
            performSearch();
        });
    });

    // Handle PDF upload
    uploadBtn.addEventListener('click', function() {
        uploadPDF();
    });

    // Handle source toggle
    sourcePdf.addEventListener('change', function() {
        if (this.checked) {
            checkPdfStatus();
        }
    });

    // Initialize: Check PDF status
    checkPdfStatus();



    // Perform search function
    async function performSearch() {
        const query = queryInput.value.trim();
        const k = parseInt(kValue.value);
        const source = document.querySelector('input[name="searchSource"]:checked').value;

        console.log('=== SEARCH DEBUG ===');
        console.log('Query:', query);
        console.log('Source:', source);
        console.log('K:', k);

        if (!query) {
            showAlert('Vui lòng nhập câu hỏi tìm kiếm!', 'danger');
            return;
        }

        // Kiểm tra nếu chọn PDF nhưng chưa upload
        if (source === 'pdf') {
            const pdfStatus = await checkPdfStatus();
            console.log('PDF Status:', pdfStatus);
            if (!pdfStatus.uploaded) {
                showAlert('Vui lòng upload file PDF trước khi tìm kiếm!', 'warning');
                return;
            }
        }

        // Show loading
        loading.style.display = 'block';
        resultsSection.style.display = 'none';

        try {
            const response = await fetch('/search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: query,
                    k: k,
                    source: source
                })
            });

            const data = await response.json();
            console.log('Search response:', data);

            if (response.ok) {
                displayResults(data);
            } else {
                showAlert(data.error || 'Có lỗi xảy ra khi tìm kiếm!', 'danger');
            }
        } catch (error) {
            console.error('Search error:', error);
            showAlert('Có lỗi xảy ra khi kết nối với server!', 'danger');
        } finally {
            loading.style.display = 'none';
        }
    }

    // Display search results
    function displayResults(data) {
        searchResults.innerHTML = '';
        
        // Cập nhật số lượng kết quả
        const resultLength = data.results ? data.results.length : 0;
        resultCount.textContent = `${resultLength} kết quả`;
        
        // Kiểm tra nếu không có kết quả
        if (data.no_results || !data.results || data.results.length === 0) {
            const noResultsDiv = document.createElement('div');
            noResultsDiv.className = 'text-center py-5';
            noResultsDiv.innerHTML = `
                <div class="mb-4">
                    <i class="fas fa-search fa-3x text-muted mb-3"></i>
                    <h5 class="text-muted">Không tìm thấy kết quả liên quan</h5>
                    <p class="text-muted">
                        Truy vấn "<strong>${data.query}</strong>" không có thông tin liên quan đến trường ĐHGTVT.
                    </p>
                    <div class="alert alert-info mt-3">
                        <i class="fas fa-lightbulb me-2"></i>
                        <strong>Gợi ý:</strong> Hãy thử tìm kiếm với các từ khóa như: 
                        <em>học phí, ký túc xá, thư viện, ngành đào tạo, hoạt động sinh viên</em>
                    </div>
                </div>
            `;
            searchResults.appendChild(noResultsDiv);
            resultCount.textContent = '0 kết quả';
        } else {
            // Hiển thị kết quả bình thường
            data.results.forEach((result, index) => {
                const resultElement = createResultElement(result, index);
                searchResults.appendChild(resultElement);
            });
            
            // Add animation cho kết quả
            setTimeout(() => {
                document.querySelectorAll('.search-result-item').forEach((item, index) => {
                    setTimeout(() => {
                        item.classList.add('fade-in-up');
                    }, index * 100);
                });
            }, 100);
        }

        resultsSection.style.display = 'block';
    }

    // Create result element
    function createResultElement(result, index) {
        const div = document.createElement('div');
        div.className = 'search-result-item d-flex align-items-start';
        div.setAttribute('data-result-index', index);
        
        // Tạo nội dung metadata nếu có
        let metadataHtml = '';
        if (result.metadata) {
            const metadata = result.metadata;
            const metadataParts = [];
            
            if (metadata.dates && metadata.dates.length > 0) {
                metadataParts.push(`<span class="badge bg-primary me-1"><i class="fas fa-calendar"></i> ${metadata.dates.join(', ')}</span>`);
            }
            
            if (metadata.numbers && metadata.numbers.length > 0) {
                metadataParts.push(`<span class="badge bg-success me-1"><i class="fas fa-hashtag"></i> ${metadata.numbers.join(', ')}</span>`);
            }
            
            if (metadata.money && metadata.money.length > 0) {
                metadataParts.push(`<span class="badge bg-warning me-1"><i class="fas fa-money-bill"></i> ${metadata.money.join(', ')}</span>`);
            }
            
            if (metadataParts.length > 0) {
                metadataHtml = `<div class="mt-2">${metadataParts.join('')}</div>`;
            }
        }
        
        // Thông tin nguồn
        let sourceInfo = '';
        if (result.source === 'pdf') {
            sourceInfo = `
                <div class="result-source-info mt-2">
                    <span class="badge bg-danger me-2">
                        <i class="fas fa-file-pdf"></i> PDF
                    </span>
                    <small class="text-muted">
                        <strong>${result.filename}</strong> - Trang ${result.page}
                        | Độ liên quan: ${(result.similarity * 100).toFixed(1)}%
                    </small>
                </div>
            `;
        } else {
            sourceInfo = `
                <div class="result-source-info mt-2">
                    <span class="badge bg-primary me-2">
                        <i class="fas fa-database"></i> Dữ liệu mặc định
                    </span>
                    <small class="text-muted">ID: ${result.index}</small>
                </div>
            `;
        }

        // Xử lý content mở rộng
        let contentPreview = '';
        let hasMoreContent = false;
        
        if (result.source === 'default' && result.extended_content) {
            // Hiển thị nội dung chính (document tìm được)
            const mainContent = result.main_content || result.extended_content[0];
            contentPreview = `<div class="main-result">
                <strong>📌 Kết quả chính:</strong><br>
                ${mainContent}
            </div>`;
            
            // Thêm 8 documents tiếp theo nếu có
            if (result.extended_content.length > 1) {
                contentPreview += `<div class="extended-results mt-2">
                    <small class="text-muted"><strong>Nội dung liên quan:</strong></small>`;
                
                for (let i = 1; i < result.extended_content.length; i++) {
                    const additionalContent = result.extended_content[i];
                    contentPreview += `<div class="additional-content mt-1">
                        <small class="text-muted"></small>${additionalContent}
                    </div>`;
                }
                contentPreview += `</div>`;
            }
            hasMoreContent = result.modal_content && result.modal_content.length > result.content.length;
        } else {
            // Logic cũ cho PDF hoặc content thông thường
            const sentences = result.content.split(/[.!?]+/).filter(s => s.trim().length > 0);
            const maxSentences = 3;
            
            if (sentences.length <= maxSentences) {
                contentPreview = result.content;
            } else {
                contentPreview = sentences.slice(0, maxSentences).join('. ').trim() + '...';
            }
            hasMoreContent = sentences.length > maxSentences;
        }

        div.innerHTML = `
            <div class="result-rank">
                <span class="rank-number">${result.rank}</span>
            </div>
            <div class="result-content flex-grow-1">
                <div class="result-text mb-2">${contentPreview}</div>
                ${hasMoreContent ? '<div class="expand-hint"><i class="fas fa-expand-alt me-1"></i> Click để xem chi tiết</div>' : ''}
                ${metadataHtml}
                ${sourceInfo}
            </div>
        `;
        
        // Thêm event listener để mở modal
        div.addEventListener('click', () => {
            showResultDetail(result, index);
        });
        
        return div;
    }

    // Show alert function
    function showAlert(message, type = 'info') {
        // Remove existing alerts
        const existingAlerts = document.querySelectorAll('.alert-floating');
        existingAlerts.forEach(alert => alert.remove());

        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-floating position-fixed`;
        alertDiv.style.cssText = `
            top: 20px;
            right: 20px;
            z-index: 9999;
            min-width: 300px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        `;
        
        alertDiv.innerHTML = `
            <div class="d-flex justify-content-between align-items-center">
                <span>${message}</span>
                <button type="button" class="btn-close" aria-label="Close"></button>
            </div>
        `;

        document.body.appendChild(alertDiv);

        // Auto remove after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, 5000);

        // Manual close
        alertDiv.querySelector('.btn-close').addEventListener('click', () => {
            alertDiv.remove();
        });
    }

    // Scroll animations
    function handleScrollAnimations() {
        const animateElements = document.querySelectorAll('.scroll-animate');
        
        animateElements.forEach(element => {
            const elementTop = element.getBoundingClientRect().top;
            const elementVisible = 150;
            
            if (elementTop < window.innerHeight - elementVisible) {
                element.classList.add('show');
            }
        });
    }

    // Initialize scroll animations
    window.addEventListener('scroll', handleScrollAnimations);
    handleScrollAnimations(); // Check on load

    // Auto-focus on search input
    queryInput.focus();

    // Enable enter key for sample queries
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && e.target.classList.contains('sample-query')) {
            e.target.click();
        }
    });

    // Add smooth scrolling for internal links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Add loading state to buttons
    function addLoadingState(button, originalText) {
        button.disabled = true;
        button.innerHTML = `
            <span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
            Đang tải...
        `;
        
        return function removeLoadingState() {
            button.disabled = false;
            button.innerHTML = originalText;
        };
    }

    // Enhanced form validation
    queryInput.addEventListener('input', function() {
        const query = this.value.trim();
        const submitBtn = searchForm.querySelector('button[type="submit"]');
        
        if (query.length > 0) {
            submitBtn.disabled = false;
            this.classList.remove('is-invalid');
        } else {
            submitBtn.disabled = true;
        }
    });

    // Upload PDF function
    async function uploadPDF() {
        const file = pdfFile.files[0];
        
        if (!file) {
            showAlert('Vui lòng chọn file PDF!', 'warning');
            return;
        }

        if (file.type !== 'application/pdf') {
            showAlert('Chỉ chấp nhận file PDF!', 'danger');
            return;
        }

        if (file.size > 16 * 1024 * 1024) { // 16MB
            showAlert('File quá lớn! Vui lòng chọn file nhỏ hơn 16MB.', 'danger');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        // Show loading state
        uploadBtn.disabled = true;
        uploadBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Đang xử lý...';

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                showAlert(data.message, 'success');
                updatePdfInfo(data);
                sourcePdf.disabled = false;
                uploadForm.reset();
            } else {
                showAlert(data.error || 'Có lỗi xảy ra khi upload file!', 'danger');
            }
        } catch (error) {
            console.error('Upload error:', error);
            showAlert('Có lỗi xảy ra khi kết nối với server!', 'danger');
        } finally {
            uploadBtn.disabled = false;
            uploadBtn.innerHTML = '<i class="fas fa-upload"></i> Upload PDF';
        }
    }

    // Check PDF status
    async function checkPdfStatus() {
        try {
            const response = await fetch('/pdf_info');
            const data = await response.json();
            
            if (data.uploaded) {
                updatePdfInfo(data);
                sourcePdf.disabled = false;
                return data;
            } else {
                pdfInfo.style.display = 'none';
                sourcePdf.disabled = true;
                if (sourcePdf.checked) {
                    sourceDefault.checked = true;
                }
                return data;
            }
        } catch (error) {
            console.error('PDF status check error:', error);
            sourcePdf.disabled = true;
            return { uploaded: false };
        }
    }

    // Update PDF info display
    function updatePdfInfo(data) {
        if (data.uploaded) {
            currentPdfName.textContent = data.filename;
            currentPdfContent.textContent = `${data.content_count} đoạn văn bản được trích xuất`;
            pdfInfo.style.display = 'block';
        } else {
            pdfInfo.style.display = 'none';
        }
        
        // Debug: Log PDF info
        console.log('PDF Info updated:', data);
    }

    // Show result detail in modal
    async function showResultDetail(result, index) {
        const modalContent = document.getElementById('modalContent');
        const modal = new bootstrap.Modal(document.getElementById('resultDetailModal'));
        
        // Cập nhật title modal
        document.getElementById('resultDetailModalLabel').innerHTML = `
            <i class="fas fa-info-circle"></i> Chi tiết kết quả #${result.rank}
        `;
        
        // Tạo nội dung chi tiết đơn giản
        let detailContent = `
            <div class="result-detail-content">
                <div class="result-detail-header">
                    <div class="result-detail-rank">${result.rank}</div>
                    <div class="result-detail-score">
                        <i class="fas fa-star"></i> Score: ${result.score ? result.score.toFixed(3) : 'N/A'}
                        ${result.similarity ? ` | Similarity: ${(result.similarity * 100).toFixed(1)}%` : ''}
                    </div>
                </div>
                
                <div class="content-section">
                    <div class="content-title">
                        <i class="fas fa-file-text"></i> Chi tiết nội dung
                    </div>
                    <div class="result-detail-text">
                        ${result.source === 'default' && result.modal_content ? 
                            generateModalContent(result) 
                            : `<div class="modal-content-text">${result.content}</div>`
                        }
                    </div>
                </div>
        `;
        
        // Thêm metadata nếu có
        if (result.metadata && Object.keys(result.metadata).length > 0) {
            const metadata = result.metadata;
            let metadataBadges = '';
            
            if (metadata.dates && metadata.dates.length > 0) {
                metadata.dates.forEach(date => {
                    metadataBadges += `<div class="metadata-badge date"><i class="fas fa-calendar"></i> ${date}</div>`;
                });
            }
            
            if (metadata.numbers && metadata.numbers.length > 0) {
                metadata.numbers.forEach(number => {
                    metadataBadges += `<div class="metadata-badge number"><i class="fas fa-hashtag"></i> ${number}</div>`;
                });
            }
            
            if (metadata.money && metadata.money.length > 0) {
                metadata.money.forEach(money => {
                    metadataBadges += `<div class="metadata-badge money"><i class="fas fa-money-bill"></i> ${money}</div>`;
                });
            }
            
            if (metadataBadges) {
                detailContent += `
                    <div class="result-detail-metadata">
                        <div class="metadata-title">
                            <i class="fas fa-tags"></i> Thông tin bổ sung
                        </div>
                        <div class="metadata-badges">
                            ${metadataBadges}
                        </div>
                    </div>
                `;
            }
        }

        // Thêm thông tin nguồn
        let sourceContent = '';
        if (result.source === 'pdf') {
            sourceContent = `
                <div class="source-title">
                    <i class="fas fa-file-pdf"></i> Thông tin file PDF
                </div>
                <div class="source-info">
                    <div class="source-info-item">
                        <i class="fas fa-file"></i> Tên file: <strong>${result.filename}</strong>
                    </div>
                    <div class="source-info-item">
                        <i class="fas fa-book-open"></i> Trang: <strong>${result.page}</strong>
                    </div>
                    <div class="source-info-item">
                        <i class="fas fa-percentage"></i> Độ tương đồng: <strong>${(result.similarity * 100).toFixed(1)}%</strong>
                    </div>
                    <div class="source-info-item">
                        <i class="fas fa-list-ol"></i> Thứ tự: <strong>#${result.rank}</strong>
                    </div>
                </div>
            `;
        } else {
            sourceContent = `
                <div class="source-title">
                    <i class="fas fa-database"></i> Thông tin dữ liệu
                </div>
                <div class="source-info">
                    <div class="source-info-item">
                        <i class="fas fa-hashtag"></i> ID: <strong>${result.index}</strong>
                    </div>
                    <div class="source-info-item">
                        <i class="fas fa-list-ol"></i> Thứ tự: <strong>#${result.rank}</strong>
                    </div>
                    <div class="source-info-item">
                        <i class="fas fa-chart-line"></i> Score: <strong>${result.score ? result.score.toFixed(3) : 'N/A'}</strong>
                    </div>
                    <div class="source-info-item">
                        <i class="fas fa-database"></i> Nguồn: <strong>Dữ liệu mặc định</strong>
                    </div>
                </div>
            `;
        }

        detailContent += `
                <div class="result-detail-source">
                    ${sourceContent}
                </div>
            </div>
        `;

        // Hiển thị modal và nội dung
        modalContent.innerHTML = detailContent;
        modal.show();
    }



    // Generate modal content với 8 documents + mở rộng đến delimiter
    function generateModalContent(result) {
        if (!result.modal_content_list || !result.modal_content_list.length) {
            return result.content;
        }
        
        // Hiển thị tất cả documents (8 cố định + mở rộng đến delimiter)
        let modalContent = `<div class="modal-main-result">
            <div class="modal-main-header">
                <strong>📌 Kết quả chính:</strong>
            </div>
            <div class="modal-main-content">${result.modal_content_list[0]}</div>
        </div>`;
        
        // Hiển thị các documents còn lại
        if (result.modal_content_list.length > 1) {
            modalContent += `<div class="modal-extended-results">
                <div class="modal-extended-header">
                    <strong>Nội dung liên quan:</strong>
                </div>`;
            
            // Hiển thị tất cả documents từ thứ 2 trở đi
            result.modal_content_list.slice(1).forEach((doc, index) => {
                modalContent += `<div class="modal-additional-content">
                    ${doc}
                </div>`;
            });
            modalContent += `</div>`;
        }
        
        return modalContent;
    }

    // Format expanded content function
    function formatExpandedContent(expandedContent) {
        if (!expandedContent) return '';
        
        // Chia nội dung thành các đoạn văn
        const paragraphs = expandedContent.split(/\n+/).filter(p => p.trim().length > 0);
        
        let formattedContent = '';
        
        paragraphs.forEach((paragraph, index) => {
            const trimmedParagraph = paragraph.trim();
            
            // Kiểm tra xem đoạn có phải là tiêu đề không
            if (isHeadingParagraph(trimmedParagraph)) {
                formattedContent += `<h6 class="text-primary fw-bold mt-3 mb-2">
                    <i class="fas fa-bookmark"></i> ${trimmedParagraph}
                </h6>`;
            } else {
                // Đoạn văn bình thường
                formattedContent += `<p class="mb-3 lh-lg" style="text-align: justify;">
                    ${trimmedParagraph}
                </p>`;
            }
        });
        
        return formattedContent;
    }
    
    // Check if paragraph is a heading
    function isHeadingParagraph(paragraph) {
        const upperParagraph = paragraph.toUpperCase();
        
        // Kiểm tra các pattern của tiêu đề
        const headingPatterns = [
            /^PHẦN\s+\d+/,           // PHẦN 1, PHẦN 2
            /^CHƯƠNG\s+[IVX\d]+/,   // CHƯƠNG I, CHƯƠNG 1
            /^ĐIỀU\s+\d+/,          // ĐIỀU 1, ĐIỀU 2
            /^[IVX]+\./,            // I., II., III.
            /^\d+\.\s*[A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ]/,  // 1. ABC, 2. XYZ
            /^[a-z]\)\s*[A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ]/   // a) ABC, b) XYZ
        ];
        
        return headingPatterns.some(pattern => pattern.test(upperParagraph));
    }

    // Initialize page
    console.log('Ứng dụng tìm kiếm đã được khởi tạo thành công!');
});