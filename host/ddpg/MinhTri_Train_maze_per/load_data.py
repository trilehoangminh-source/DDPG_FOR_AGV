import numpy as np
import scipy.io as sio

def load_maze_data(agent, file_name='minhtri_maze.mat'):
    """
    Hàm đọc file dữ liệu đua xe .mat của ông 
    và nạp thẳng vào Replay Buffer của agent
    """
    try:
        mat_contents = sio.loadmat(file_name)
        data = mat_contents['data']
        print(f"============================ LOAD DATA HUMAN: Tìm thấy {len(data)} bước di chuyển mẫu! ============================")
        
        for i in range(len(data)):
            # Bốc tách dữ liệu theo đúng chỉ số dòng code của ông
            current_state = data[i][0:28]
            action        = data[i][28:30]
            reward        = data[i][30]
            new_state     = data[i][31:59]
            done          = int(data[i][59])
            
            # Đổ trực tiếp vào Replay Buffer (bộ nhớ RAM) của Agent
            agent.remember(current_state, action, reward, new_state, done)
            
        print(f"============================ NẠP MỒI THÀNH CÔNG: Buffer hiện tại có {len(agent.buffer._experience)} mẫu ============================")
    except Exception as e:
        print(f"============================ LỖI LOAD FILE .MAT: {e} ============================")