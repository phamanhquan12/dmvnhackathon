def create_quiz_prompt(context_text: str, num_questions: int = 20):
    """
    mấy cái này phải thiết kế bằng tiếng việt chắc v2 lên tiếng anh sau.
    """
    # Định nghĩa cấu trúc JSON mong muốn ngay trong prompt
    json_structure = """
    [
        {
            "question": "Nội dung câu hỏi",
            "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
            "answer": "Đáp án đúng (Text)",
            "explanation": "Giải thích ngắn gọn"
        }
    ]
    """
    
    # Sử dụng f-string để nhúng context vào
    return f"""
    Bạn là chuyên gia đào tạo kỹ thuật. Nhiệm vụ của bạn là tạo {num_questions} câu hỏi trắc nghiệm (Quiz) bằng Tiếng Việt dựa trên văn bản sau.
    
    VĂN BẢN NGUỒN:
    ---
    {context_text}
    ---

    YÊU CẦU:
    1. Tạo đúng {num_questions} câu hỏi.
    2. Mỗi câu hỏi có 4 đáp án lựa chọn A, B, C, D.
    3. Trả về kết quả CHỈ LÀ MỘT DANH SÁCH JSON (JSON Array) theo cấu trúc mẫu dưới đây, không thêm bất kỳ lời dẫn nào khác:
    
    CẤU TRÚC MONG MUỐN:
    {json_structure}
    """

def create_flash_cards_prompt(content_text: str, num_cards: int = 30):
    """
    Hàm này tạo prompt để yêu cầu mô hình tạo flash cards từ văn bản nguồn.
    """
    json_structure = """
    [
        {
            "question": "Nội dung câu hỏi",
            "answer": "Nội dung đáp án"
        }
    ]
    """
    
    return f"""
    Bạn là chuyên gia đào tạo kỹ thuật. Nhiệm vụ của bạn là tạo {num_cards} thẻ học (flash cards) bằng Tiếng Việt dựa trên văn bản sau.
    
    VĂN BẢN NGUỒN:
    ---
    {content_text}
    ---

    YÊU CẦU:
    1. Tạo đúng {num_cards} thẻ học.
    2. Mỗi thẻ học gồm một câu hỏi và một đáp án ngắn gọn.
    3. Trả về kết quả CHỈ LÀ MỘT DANH SÁCH JSON (JSON Array) theo cấu trúc mẫu dưới đây, không thêm bất kỳ lời dẫn nào khác:
    
    CẤU TRÚC MONG MUỐN:
    {json_structure}
    """