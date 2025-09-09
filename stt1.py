import os
import shutil

# Đường dẫn đến thư mục nguồn chứa các ảnh từ IMG(1) đến IMG(250)
source_folder = 'new/tramanh(2)'  # Đặt đường dẫn thư mục nguồn

# Đường dẫn tới thư mục đích
destination_folder = "new2/61"

# Tạo thư mục đích nếu chưa tồn tại
os.makedirs(destination_folder, exist_ok=True)

# Đếm ảnh để đánh số thứ tự từ 001 đến 250
image_count = 61

# Duyệt qua các tệp trong thư mục nguồn và sắp xếp theo thứ tự
for file_name in sorted(os.listdir(source_folder)):
    # Kiểm tra nếu file là .jpg hoặc .png
    if file_name.lower().endswith((".jpg", ".png")):
        # Giới hạn đến 250 ảnh
        if image_count > 250:
            break

        # Lấy đuôi mở rộng của file (.jpg hoặc .png)
        file_extension = os.path.splitext(file_name)[1].lower()

        # Đặt tên file mới với số thứ tự từ 001 đến 250
        new_file_name = f"{str(image_count).zfill(3)}{file_extension}"

        # Đường dẫn nguồn và đích
        source_path = os.path.join(source_folder, file_name)
        destination_path = os.path.join(destination_folder, new_file_name)

        # Sao chép ảnh vào thư mục đích với tên mới
        shutil.copy(source_path, destination_path)

        # Tăng image_count để tiếp tục đánh số
        image_count += 1

print("Đổi tên và di chuyển ảnh từ IMG(1) đến IMG(250) thành 001 đến 250 hoàn tất.")
