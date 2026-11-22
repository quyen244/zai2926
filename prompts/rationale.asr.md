
Lần này câu trợ lý thấy là bản nhận dạng giọng nói (ASR), có thể nghe sai chữ.
Bạn được xem cả câu thật để hiểu ý, nhưng trợ lý không thấy câu thật, nên:

- Chỉ được trích chữ (đặt trong ngoặc kép) từ bản nhận dạng hoặc từ kết quả tool; không được trích chữ chỉ có trong câu thật.
- Khi một giá trị của lời giải khác chữ trong bản nhận dạng, nói ngắn trong "args" vì sao vẫn hiểu ra giá trị đúng, dựa trên thứ trợ lý thấy:
  âm gần giống và ngữ cảnh ("hai triệu tém" là hai triệu tám, tức 2800000), chữ thường cần viết hoa ("sếp dũng" là "Sếp Dũng"),
  hoặc kết quả tool đã thấy (danh bạ ở bước trước có "Sếp Dũng").
- Nếu bản nhận dạng không đủ để suy ra giá trị (mất hẳn tên, số), vẫn viết lý do dựa trên chỗ suy ra được; đừng giả vờ người dùng đã nói rõ.
