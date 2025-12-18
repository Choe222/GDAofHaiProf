Backtracking Line Search (Tìm kiếm lùi trên đường thẳng) là một chiến lược tìm kiếm bước nhảy (learning rate) $\lambda$ linh hoạt trong mỗi vòng lặp của các thuật toán tối ưu hóa Gradient Descent. Thay vì chọn một giá trị $\lambda$ cố định, thuật toán này sẽ "thử và sai" để tìm ra bước đi đủ tốt giúp hàm mục tiêu giảm một cách đáng kể.

1. Nguyên lý hoạt động
Ý tưởng cốt lõi của Backtracking Line Search là bắt đầu với một bước nhảy lớn và giảm dần nó (co lại) cho đến khi đạt được một tiêu chuẩn giảm hàm số nhất định, gọi là Điều kiện Armijo.