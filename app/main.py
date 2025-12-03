from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List
import io
import joblib
import pandas as pd
import numpy as np
import os

# ==========================================
# CẤU HÌNH ỨNG DỤNG (APPLICATION CONFIG)
# ==========================================

# Khởi tạo ứng dụng FastAPI với thông tin mô tả chi tiết
app = FastAPI(
    title="Hệ Thống Phân Loại Học Viên (Student Classification API)",
    description="API cung cấp dịch vụ dự đoán xếp loại học viên dựa trên kết quả học tập và trường học, sử dụng thuật toán K-Means Clustering.",
    version="1.0.0",
    docs_url="/docs",  # Đường dẫn tài liệu Swagger UI
    redoc_url="/redoc" # Đường dẫn tài liệu ReDoc
)

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN (PATH CONFIGURATION)
# ==========================================

# Xác định đường dẫn tuyệt đối đến thư mục chứa artifacts (mô hình đã huấn luyện)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "../model/artifacts")

# Đường dẫn chi tiết đến từng file thành phần
MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'kmeans_model.pkl')       # Mô hình K-Means
SCALER_PATH = os.path.join(ARTIFACTS_DIR, 'scaler.pkl')            # Bộ chuẩn hóa dữ liệu (StandardScaler)
ENCODER_PATH = os.path.join(ARTIFACTS_DIR, 'encoder.pkl')          # Bộ mã hóa tên trường (OrdinalEncoder)
MAPPING_PATH = os.path.join(ARTIFACTS_DIR, 'cluster_mapping.pkl')  # Ánh xạ từ cụm sang nhãn xếp loại

# Biến toàn cục để lưu trữ các thành phần mô hình sau khi tải
model_artifacts = {}

# ==========================================
# SỰ KIỆN KHỞI ĐỘNG (STARTUP EVENTS)
# ==========================================

@app.on_event("startup")
def load_artifacts():
    """
    Hàm này được gọi tự động khi server khởi động.
    Nhiệm vụ: Tải toàn bộ các file artifacts (.pkl) vào bộ nhớ để sẵn sàng phục vụ dự đoán.
    """
    try:
        print("Đang tải các thành phần mô hình...")
        model_artifacts['kmeans'] = joblib.load(MODEL_PATH)
        model_artifacts['scaler'] = joblib.load(SCALER_PATH)
        model_artifacts['encoder'] = joblib.load(ENCODER_PATH)
        model_artifacts['mapping'] = joblib.load(MAPPING_PATH)
        print("✅ Đã tải thành công mô hình và các artifacts!")
    except Exception as e:
        print(f"❌ LỖI NGHIÊM TRỌNG: Không thể tải artifacts. Chi tiết: {e}")
        # Lưu ý: Trong môi trường production, có thể cân nhắc dừng server nếu không tải được model.

# ==========================================
# ĐỊNH NGHĨA DỮ LIỆU (DATA MODELS)
# ==========================================

from typing import List, Optional

# ... (imports)

class StudentInput(BaseModel):
    """
    Mô hình dữ liệu đầu vào cho một học viên.
    Sử dụng Pydantic để validate dữ liệu tự động.
    """
    student_id: Optional[str] = None # ID định danh học viên (tùy chọn, để client dễ đối chiếu)
    score1: float       # Điểm môn 1
    score2: float       # Điểm môn 2
    score3: float       # Điểm môn 3
    school_name: str    # Tên trường học (VD: "Trường THPT Nguyễn Thượng Hiền")

# ==========================================
# HÀM HỖ TRỢ (HELPER FUNCTIONS)
# ==========================================

def process_student_data(score1, score2, score3, school_name, student_id=None):
    """
    Hàm xử lý logic dự đoán cốt lõi cho một học viên.
    """
    # 1. Tính điểm trung bình
    final_score = np.mean([score1, score2, score3])
    
    # 2. Xử lý mã hóa tên trường (School Encoding)
    try:
        # Encoder trả về giá trị số tương ứng với rank của trường.
        # Nếu trường không có trong danh sách huấn luyện (Unknown), 
        # encoder sẽ trả về giá trị mặc định (thường là -1) hoặc gây lỗi tùy cấu hình.
        school_encoded = model_artifacts['encoder'].transform([[school_name]])[0][0]
    except (ValueError, Exception):
        # Fallback an toàn: Nếu gặp trường lạ hoặc lỗi, gán giá trị -1.0.
        # Giá trị -1.0 thấp hơn Rank 10 (0.0), đồng nghĩa với việc xếp trường này vào nhóm thấp nhất.
        school_encoded = -1.0

    # 3. Tạo vector đặc trưng (Feature Vector)
    # Thứ tự features phải KHỚP CHÍNH XÁC với lúc huấn luyện:
    # [score1, score2, score3, final_score, school_encoded]
    features_df = pd.DataFrame([[score1, score2, score3, final_score, school_encoded]], 
                               columns=['score1', 'score2', 'score3', 'final_score', 'school_encoded'])
    
    # 4. Chuẩn hóa dữ liệu (Scaling)
    # Đưa dữ liệu về cùng phân phối với tập huấn luyện
    features_scaled = model_artifacts['scaler'].transform(features_df)
    
    # 5. Dự đoán phân cụm (Clustering Prediction)
    cluster = model_artifacts['kmeans'].predict(features_scaled)[0]
    
    # 6. Gán nhãn xếp loại (Label Mapping)
    rank = model_artifacts['mapping'].get(cluster, "Không xác định")
    
    result = {
        "score1": score1,
        "score2": score2,
        "score3": score3,
        "school_name": school_name,
        "final_score": round(final_score, 2),
        "cluster": int(cluster), # Chuyển về int python thuần để trả về JSON
        "rank": rank
    }
    
    # Nếu có ID thì trả về kèm để client đối chiếu
    if student_id is not None:
        result["student_id"] = student_id
        
    return result

# ==========================================
# CÁC ENDPOINT API (API ENDPOINTS)
# ==========================================

@app.get("/")
def read_root():
    """Kiểm tra trạng thái hoạt động của API (Health Check)."""
    return {
        "status": "active",
        "message": "Chào mừng đến với API Phân Loại Học Viên.",
        "endpoints": {
            "predict": "/predict (POST)",
            "predict_batch": "/predict_batch (POST)",
            "predict_csv": "/predict_csv (POST)"
        }
    }

@app.post("/predict")
def predict_student(student: StudentInput):
    """
    Dự đoán xếp loại cho một học viên duy nhất.
    
    - **Input**: JSON chứa điểm số và tên trường.
    - **Output**: Kết quả phân loại chi tiết.
    """
    if not model_artifacts:
        raise HTTPException(status_code=500, detail="Mô hình chưa được tải (Model not loaded).")
    try:
        result = process_student_data(student.score1, student.score2, student.score3, student.school_name, student.student_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi xử lý: {str(e)}")

@app.post("/predict_batch")
def predict_batch(students: List[StudentInput]):
    """
    Dự đoán cho một danh sách nhiều học viên (Batch Processing).
    
    - **Input**: Mảng JSON chứa thông tin các học viên.
    - **Output**: Mảng kết quả tương ứng.
    """
    if not model_artifacts:
        raise HTTPException(status_code=500, detail="Mô hình chưa được tải.")
    
    results = []
    for student in students:
        try:
            res = process_student_data(student.score1, student.score2, student.score3, student.school_name, student.student_id)
            results.append(res)
        except Exception as e:
            # Nếu một dòng lỗi, trả về thông báo lỗi cho dòng đó nhưng không dừng toàn bộ process
            results.append({"error": str(e), "input": student.dict()})
    return results

@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    """
    Dự đoán từ file CSV tải lên.
    
    - **Input**: File .csv (Multipart/form-data).
    - **Yêu cầu**: File phải có các cột header: 'score1', 'score2', 'score3', 'school_name'.
    - **Output**: JSON chứa kết quả dự đoán cho từng dòng.
    """
    if not model_artifacts:
        raise HTTPException(status_code=500, detail="Mô hình chưa được tải.")
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File tải lên phải có định dạng .csv")
    
    try:
        # Đọc nội dung file
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Kiểm tra các cột bắt buộc
        required_cols = ['score1', 'score2', 'score3', 'school_name']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"File CSV thiếu các cột bắt buộc: {missing_cols}")
            
        results = []
        for index, row in df.iterrows():
            try:
                res = process_student_data(row['score1'], row['score2'], row['score3'], row['school_name'])
                # Giữ lại ID nếu có để dễ đối chiếu
                if 'id' in row:
                    res['id'] = row['id']
                results.append(res)
            except Exception as e:
                results.append({"error": str(e), "row_index": index})
                
        return results
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi khi đọc file CSV: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Chạy server uvicorn khi file được thực thi trực tiếp
    uvicorn.run(app, host="0.0.0.0", port=8000)
