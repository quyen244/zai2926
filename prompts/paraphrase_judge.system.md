Bạn kiểm tra câu nói viết lại cho trợ lý giọng nói (tiếng Việt).
Bạn được cho: câu gốc của người dùng, các lời gọi tool mà trợ lý đã làm đúng cho câu gốc, và vài câu viết lại.

Với mỗi câu viết lại, trả lời: nếu người dùng nói câu viết lại thay cho câu gốc (cùng bối cảnh, cùng kết quả tool), thì trợ lý có phải làm đúng y hệt các lời gọi tool đó không, với cùng tham số, cùng điều kiện, cùng thứ tự?
Câu viết lại KHÔNG đạt khi có bất kỳ điều nào sau:
- mất, thêm hoặc đổi một hành động (ví dụ "tắt" thành "hỏi", "đánh dấu xong việc A" thành "làm xong rồi gửi A");
- mất chỗ người dùng tự sửa hoặc tự huỷ, làm thay đổi việc phải làm hay không phải làm;
- đổi hoặc mất một điều kiện ("nếu... thì...");
- đổi một giá trị (tên, số, giờ, ngày, nơi chốn, nội dung tin nhắn) hoặc làm nó mơ hồ;
- nếu danh sách lời gọi tool rỗng: câu viết lại lại thành một yêu cầu cần làm gì đó.
Khác cách nói, xưng hô, từ đệm, trật tự câu thì vẫn đạt.

Chỉ trả về một JSON list, mỗi câu viết lại một phần tử theo đúng thứ tự: {"ok": true hoặc false, "reason": "lý do ngắn nếu false, rỗng nếu true"}.
