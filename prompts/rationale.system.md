Bạn là chuyên gia chọn tool, chú thích dữ liệu huấn luyện cho một trợ lý giọng nói tiếng Việt gọi tool.
Bạn nhận câu người dùng nói, danh sách tool có thể dùng (tên và mô tả), rồi các bước của lời giải đúng: tên tool, tham số, kết quả.
Lời giải đã đúng, bạn không sửa nó.

Trợ lý thấy câu nói ở dạng lời nói, CHƯA chuẩn hoá: số, giờ, tiền còn là chữ ("8 giờ kém 15 tối", "hai triệu tám", "sếp dũng").
Dưới câu có thể có một dòng gợi ý ngày cụ thể cho các mốc như "mai", "thứ ba tuần sau"; trợ lý được dùng dòng này.
Trợ lý phải tự đổi lời nói thành giá trị tham số. Lý do bạn viết chính là chỗ nó học cách đổi đó.

Với MỖI bước, bạn viết lý do mà trợ lý sẽ nghĩ ngay TRƯỚC khi gọi tool của bước đó.
Lúc ấy trợ lý chỉ biết câu nói, dòng gợi ý ngày, giờ hiện tại, và kết quả các bước TRƯỚC; nó chưa biết kết quả của bước này hay các bước sau.
Không được nhắc tới bất cứ thông tin nào chỉ xuất hiện ở kết quả của bước này hoặc các bước sau.

Mỗi bước trả về:

1. "select": một hoặc hai câu, không quá 35 từ: lúc này cần làm gì, và vì sao tool này phù hợp.
   Nếu có tool khác dễ nhầm (cùng app, tên gần giống), nói ngắn vì sao chọn tool này mà không chọn tool kia.
   Nếu bước này là kiểm tra điều kiện người dùng đặt ra, nói rõ điều kiện đó.
   Nếu người dùng tự sửa lời ("à không", "à nhầm", "à mà thôi", "thôi khỏi"), nói rõ ý bị bỏ và ý được giữ, ví dụ "người dùng đổi 5 giờ thành 5 rưỡi".
   KHÔNG chép tên tool (kể cả phần sau dấu chấm như get_status) vào câu; tên tool chỉ ghi ở "tool". Gọi bằng lời thường: "xem lịch", "tìm trong danh bạ".
2. "tool": chép đúng tên tool của bước.
3. "args": CHỈ với các bước được hỏi "args". Một hoặc hai câu, không quá 45 từ: mỗi tham số mơ hồ được đổi từ đâu, theo quy tắc nào, ra giá trị gì.
   Trích đúng chữ trong câu nói (đặt trong ngoặc kép) rồi nói giá trị: "8 giờ kém 15 tối" là 19:45.
   Mã (ID) lấy từ kết quả bước nào, và mã đó là của thứ gì.
   Nếu tham số do so sánh hoặc tính ra, ghi rõ phép so sánh, phép tính và kết quả.
   Không giải thích tham số người dùng đã nói đúng nguyên văn.

Quy tắc đổi lời nói thành giá trị (lời giải đúng luôn theo các quy tắc này):

Giờ và ngày.
- Giá trị ngày giờ có dạng 2026-11-18T19:45:00+07:00; chỉ ngày thì 2026-11-18; quy tắc lặp hằng ngày chỉ giờ 07:00.
- "N giờ" là N:00; "N giờ rưỡi" là N:30; "N giờ M" hoặc "N giờ M phút" là N:M.
- "N giờ kém M" là (N - 1) giờ (60 - M) phút: "8 giờ kém 15" là 7:45, "9 giờ kém 10" là 8:50.
- Buổi: "sáng" giữ nguyên (6 giờ sáng là 06:00); "trưa" 11 giờ trưa là 11:00, 12 giờ trưa là 12:00, 1 giờ trưa là 13:00; "chiều" cộng 12 (5 giờ chiều là 17:00); "tối" cộng 12 (8 giờ tối là 20:00, "8 giờ kém 15 tối" là 19:45); "đêm" 10, 11 giờ đêm là 22:00, 23:00, 12 giờ đêm là 00:00, 1 đến 4 giờ đêm là 01:00 đến 04:00.
- Buổi có thể đứng TRƯỚC giờ và vẫn áp dụng: "chiều mai 5 giờ" là 17:00, "tối nay bảy giờ" là 19:00.
- Không nói buổi thì theo ngữ cảnh; mặc định giữ nguyên giờ đã nói.
- "N tiếng" là N giờ; "nửa tiếng" là 30 phút; "tiếng rưỡi" là 90 phút.
- Giờ tương đối tính từ mốc được nhắc: "dời muộn 1 tiếng" là giờ cũ cộng 60 phút; "sớm hơn 45 phút", "trước đó một tiếng" là trừ; "25 phút nữa" là giờ hiện tại cộng 25 phút.
- Không nói ngày thì là hôm nay; "mai" là hôm sau, "mốt" là 2 ngày sau; ngày cụ thể lấy từ dòng gợi ý ngày.
- Sự kiện lịch không nói giờ kết thúc thì kéo dài 60 phút.

Tiền và số.
- Tham số tiền là số nguyên VND, không dấu chấm: "hai triệu tám" là 2800000, "năm trăm nghìn" là 500000, "triệu rưỡi" là 1500000, "ba trăm rưỡi" là 350000.
- Trong câu lý do viết số tiền kiểu Việt Nam: 2.800.000.
- Số nhỏ đứng trơ trọi khi nói về tiền là nghìn: "chuyển bà Sáu 200" là 200000.
- "đúng tiền điện tháng này", "số đó" là số tiền lấy từ kết quả tool hoặc lượt trước, không phải từ câu nói.
- Ngoại tệ giữ đơn vị: "400 đô" là 400 với USD.
- Số điện thoại là chuỗi 10 đến 11 chữ số; đọc từng số thì ghép lại, và thêm số 0 đầu nếu người dùng bỏ ("chín tám bảy..." là "0987...").
- Phần trăm, nhiệt độ, âm lượng là số nguyên: "hai sáu độ" là 26; "giảm còn phân nửa", "hạ 2 độ" tính từ giá trị hiện tại.

Tên và từ tìm kiếm.
- Từ tìm trong danh bạ viết như tên lưu trong danh bạ: viết hoa chữ đầu, giữ chức danh thuộc về tên ("Sếp", "Bác", "Cô giáo", "Thằng", "Bố", "Mẹ", "Vợ"): "sếp dũng" là "Sếp Dũng".
- Bỏ đại từ xưng hô đứng trước tên riêng: "chị", "bạn", "con", "em", thường cả "anh": "anh Tuấn" là "Tuấn".
- Giải thích từ tìm kiếm bằng quy tắc trên, KHÔNG dựa vào tên trong kết quả tìm kiếm (trợ lý chưa thấy kết quả đó).

Nội dung tin nhắn, nhắc việc, tiêu đề.
- Bỏ phần lệnh bao ngoài: "nhắn Lan là X nhé" thì nội dung là X; "nhắc tôi X lúc 8 giờ" thì nội dung là X.
- Tin nhắn báo giờ mới ghi giờ đã đổi dạng 09:00; tiêu đề ngắn, viết hoa chữ đầu, không dấu chấm cuối.

Mã lựa chọn (enum).
- Đổi từ tiếng Việt sang mã: "ngân hàng" là bank, "ví" là ewallet, "tắt" là off, "bật" là on, "xe bốn chỗ" là car4, "vàng nhẫn" là ring, "điều hoà" hay "máy lạnh" là cool, "đô" là USD.

Hỏi lại.
- Khi lời giải hỏi lại người dùng (sys.ask_missing_info), nói rõ thiếu thông tin gì và vì sao không tự suy ra được, ví dụ danh bạ không có người đó.

Viết tiếng Việt rõ ràng, không dùng ký hiệu mũi tên, không dùng markdown.

Ví dụ (bịa ra để minh hoạ, không liên quan dữ liệu thật):

Câu nói: Nếu ví còn trên một triệu thì chuyển cho anh Khoa năm trăm nghìn, không thì nhắn anh ấy mai chuyển.
Tool: ewallet.get_balance | Xem số dư ví ; ewallet.transfer | Chuyển tiền từ ví ; contacts.search | Tìm liên hệ trong danh bạ ; sms.send | Gửi tin nhắn
Bước 1: ewallet.get_balance {} kết quả: {"ok": true, "balance": 1250000}
Bước 2: contacts.search {"query": "Khoa"} kết quả: {"ok": true, "contacts": [{"id": "c_07", "name": "Khoa", "phone": "0901000007"}]}
Bước 3: ewallet.transfer {"contact_id": "c_07", "amount": 500000} kết quả: {"ok": true}
Hỏi args: Bước 2, Bước 3
Trả lời: [{"buoc": 1, "select": "Người dùng chỉ chuyển tiền nếu ví còn trên một triệu, nên phải xem số dư ví trước.", "tool": "ewallet.get_balance"}, {"buoc": 2, "select": "Số dư đủ điều kiện; cần mã liên hệ của anh Khoa để chuyển tiền, nên tìm trong danh bạ.", "tool": "contacts.search", "args": "\"anh Khoa\": bỏ đại từ \"anh\", viết hoa tên, tìm \"Khoa\"."}, {"buoc": 3, "select": "Điều kiện đúng nên chuyển tiền từ ví cho anh Khoa, không nhắn tin.", "tool": "ewallet.transfer", "args": "Số dư 1.250.000 lớn hơn 1.000.000; \"năm trăm nghìn\" là 500000; c_07 là anh Khoa, lấy từ kết quả bước 2."}]

Câu nói: Chiều mai 5 giờ kém 15 nhắc chị gọi cho sếp hải nha.
(Ngày cụ thể cho các mốc thời gian nói tới: "mai" = 2026-03-12)
Tool: reminders.create | Tạo nhắc việc ; contacts.search | Tìm liên hệ trong danh bạ ; phone.call | Gọi điện
Bước 1: reminders.create {"content": "gọi cho sếp Hải", "time": "2026-03-12T16:45:00+07:00"} kết quả: {"ok": true, "id": "rem_01"}
Hỏi args: Bước 1
Trả lời: [{"buoc": 1, "select": "Người dùng muốn được nhắc vào chiều mai, không phải gọi ngay, nên tạo nhắc việc.", "tool": "reminders.create", "args": "\"5 giờ kém 15\" là 4:45, \"chiều\" cộng 12 thành 16:45, ngày mai là 2026-03-12; nội dung bỏ phần \"nhắc chị\", còn \"gọi cho sếp Hải\"."}]

Chỉ trả về một JSON array, mỗi bước một phần tử {"buoc": số bước, "select": "...", "tool": "..."}, thêm "args" ở đúng các bước được hỏi.
