# Yêu cầu nhiều app, nhiều nhóm trong train

Sinh bởi `scripts/request_groups.py` từ 1545 mẫu train. Chỉ số liệu gộp, không có id hay câu của mẫu (repo public).
Không tính `sys.*` (ask_missing_info) là một app.

## Catalog

Catalog train có 46 app, 147 tool (45 app thường + `sys` luôn bật).
Mỗi mẫu chỉ cài một phần: trung bình 9.2 app, trung vị 9, ít nhất 3, nhiều nhất 18.
Prompt chỉ đưa tool của app đã cài.

Nhóm app tự đặt để đọc cho dễ (không phải phân loại chính thức):

| Nhóm | App (số tool) |
|---|---|
| Nhà thông minh | smart_home (6), home_security (4), door_lock (3), robot_vacuum (3), appliances (3), home_energy (2), home_automation (3), household_chores (2) |
| Thời gian, ghi chú | alarm (4), reminders (4), calendar (4), notes (3), tasks (3) |
| Liên lạc | contacts (3), phone (2), sms (2), chat_ott (5), email (6), video_meeting (3) |
| Tài chính | banking (5), ewallet (4), bill_payment (3), loyalty (3), subscriptions (2) |
| Thông tin | weather (2), gold_price (1), fuel_price (1), lottery (2), exchange_rate (2), news (3), calculator (2) |
| Di chuyển | maps (3), ride_hailing (4), flights (5), bike_rental (2) |
| Mua sắm | food_delivery (5), shopping_list (4), package_tracking (3) |
| Giải trí | music (4), tv (4), podcast_radio (4) |
| Sức khỏe | medication (4), fitness (3) |
| Học | learning (4), library (2) |

## Một yêu cầu cần bao nhiêu app

| Số app trong gold | Mẫu | Tỉ lệ |
|---:|---:|---:|
| 0 | 171 | 11.1% |
| 1 | 659 | 42.7% |
| 2 | 512 | 33.1% |
| 3 | 145 | 9.4% |
| 4 | 57 | 3.7% |
| 5 | 1 | 0.1% |

Từ 2 app trở lên: 715 mẫu (46.3%).
Gold rỗng: 111 mẫu; 0 app nhưng có call: 60 mẫu (chỉ `sys.ask_missing_info`).

## Một yêu cầu trải qua bao nhiêu nhóm

| Số nhóm trong gold | Mẫu | Tỉ lệ |
|---:|---:|---:|
| 0 | 171 | 11.1% |
| 1 | 781 | 50.6% |
| 2 | 525 | 34.0% |
| 3 | 66 | 4.3% |
| 4 | 2 | 0.1% |

Từ 3 nhóm trở lên: 68 mẫu (4.4%), trong đó 25 mẫu cũng dài (>= 34 từ, nhóm 10% dài nhất).

Cặp nhóm hay đi cùng nhau nhất (mẫu đúng 2 nhóm):

| Cặp nhóm | Mẫu |
|---|---:|
| Liên lạc + Thời gian, ghi chú | 86 |
| Liên lạc + Tài chính | 53 |
| Thông tin + Thời gian, ghi chú | 46 |
| Liên lạc + Nhà thông minh | 44 |
| Nhà thông minh + Thời gian, ghi chú | 38 |
| Di chuyển + Thời gian, ghi chú | 34 |
| Liên lạc + Mua sắm | 24 |
| Thông tin + Tài chính | 23 |

## Độ dài và số call

Số từ của câu: trung vị 22, p90 34, dài nhất 67.

| Số call trong gold | Mẫu | Tỉ lệ |
|---:|---:|---:|
| 0 | 111 | 7.2% |
| 1 | 369 | 23.9% |
| 2 | 428 | 27.7% |
| 3 | 359 | 23.2% |
| 4 | 161 | 10.4% |
| 5 | 107 | 6.9% |
| 6 | 10 | 0.6% |

| Số nhóm | Số từ trung bình | Số call trung bình |
|---:|---:|---:|
| 0 | 18.9 | 0.4 |
| 1 | 20.2 | 2.0 |
| 2 | 25.9 | 3.1 |
| 3 | 30.5 | 4.1 |
| 4 | 30.0 | 5.5 |

## Điểm các lượt chạy đã cache, theo số app trong gold

| Lượt | 0 app | 1 app | 2 app | 3+ app |
|---|---:|---:|---:|---:|
| QVikhr-3-4B-Instruction B0 (gold) | 28.6 (n=7) | 11.5 (n=42) | 1.2 (n=27) | 0.0 (n=24) |
| Qwen3-4B-AWQ B0 (gold) | 28.6 (n=7) | 7.9 (n=42) | 0.0 (n=27) | 0.0 (n=24) |
| QVikhr-3-4B-Instruction B1 (asr) | 14.3 (n=7) | 4.0 (n=42) | 1.2 (n=27) | 0.0 (n=24) |
| Qwen3-4B-AWQ B1 (asr) | 14.3 (n=7) | 4.0 (n=42) | 0.0 (n=27) | 0.0 (n=24) |

Ô = điểm trung bình x100 trên các mẫu của lượt đó có số app như cột.

## Đọc

Gần một nửa train cần từ 2 app trở lên; các yêu cầu này thường nối kết quả bước trước sang bước sau (tỉ giá ra số tiền, quãng đường ra số cây số, nhiệt độ quyết định bật quạt).
Có các khuôn câu điều kiện lặp lại nhiều lần với tên hoặc hoá đơn khác nhau: "nếu ví đủ thì trả bằng ví, không thì qua ngân hàng" và "mai mưa thì huỷ lịch, không thì đặt xe".
"Thời gian, ghi chú" hay là bước chốt cuối câu (nhắc, tạo việc, ghi chú): có mặt trong 4 trên 8 cặp nhóm phổ biến nhất ở bảng trên và trong 274 trên 525 mẫu đúng 2 nhóm.
