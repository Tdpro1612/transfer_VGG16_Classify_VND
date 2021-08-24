## transfer_VGG16_Classify_VND

#### tạo dữ liệu
- dùng video quay lại video tờ tiền .tốt nhất đặt tờ tiền trên nền đen,để dưới mặt đất quay lại rồi dùng file frame.py để tách ra các frame.lưu ý nên dùng máy độ phân giải cao tý cho hình ảnh sắc nét.có thể dùng camera điện thoại hoặc laptop đều được
### sau khi có dữ liệu tạo data train
- Sau khi dùng file frame.py tách ra hình ảnh theo từng class chứa forder theo cách sau
 ```
 Data/00000 # đây là không có tiền
     /1000
     /10000
     /100000
     /2000
     ...
 ```
 lưu ý ở đây class name nó chạy theo thứ tự chữ số đầu trước chứ không chạy theo cách 1000,2000,5000...
 
 - Sau khi có các forder theo cây ở trên sử dụng file create_data_vnd.py để tạo data
 ### tải file data lên google drive,train nó
 - Sau khi có file pix.data thì tải lên drive rồi sử dụng file model_train_vnd để train
 - Sau khi train xong thì đã có model,tải về 
 ### test thử với camera thôi nào
 - Sử dụng file test_vnd.py để test thôi
