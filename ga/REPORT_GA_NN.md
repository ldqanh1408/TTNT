BÁO CÁO HỌC THUẬT

TỐI ƯU HÓA TRỌNG SỐ MẠNG THẦN KINH BẰNG GIẢI THUẬT DI TRUYỀN CHO BÀI TOÁN ĐIỀU KHIỂN CHROME DINOSAUR AGENT

Ngành: Trí tuệ nhân tạo & Khoa học Máy tính
Lĩnh vực: Neuroevolution & Evolutionary Computation

═══════════════════════════════════════════════════════════════

PHẦN I. TÓM TẮT NGHIÊN CỨU

Nghiên cứu này trình bày phương pháp tối ưu hóa trọng số mạng thần kinh nhân tạo (Artificial Neural Network — ANN) cho bài toán điều khiển tự động Agent trong trò chơi Chrome Dinosaur bằng Giải thuật Di truyền (Genetic Algorithm — GA). Khác với các phương pháp học tăng cường dựa trên Gradient Descent — chẳng hạn Deep Q-Network (DQN) hay Policy Gradient — nghiên cứu áp dụng GA để tiến hóa trực tiếp bộ tham số (Weights và Biases) của một mạng Feedforward Neural Network (FNN) mà không cần tính gradient, không cần Experience Replay Buffer, và không cần hàm mất mát khả vi.

Mạng FNN được thiết kế theo kiến trúc 3 lớp ẩn 15→256→128→3 với tổng cộng 37,379 tham số có thể tiến hóa. Cấu hình tiến hóa gồm bốn toán tử cốt lõi: Lựa chọn giải đấu (Tournament Selection, k=5), Lai ghép đồng nhất (Uniform Crossover, pc=0.80), Đột biến Gauss thích ứng (Adaptive Gaussian Mutation, pm=0.08, σ=0.10), và Chiến lược ưu tú (Elitism, E=8). Quần thể gồm N=80 cá thể được khởi tạo theo phương pháp Xavier Uniform Init, tiến hóa qua tối đa Gmax=300 thế hệ. Mỗi cá thể được đánh giá FITNESS_EVALS=5 lần chạy độc lập để giảm phương sai ước lượng fitness. Hàm fitness chính là điểm số tích lũy (score) của Agent trong một episode, tuân theo cơ chế tính điểm chuẩn Chrome Dino. Đặc biệt, cơ chế chống Catastrophic Forgetting bằng cách đưa cá thể tốt nhất từng thấy (best_ever) bắt buộc vào quần thể ở mỗi thế hệ giúp ngăn chặn hiện tượng mất mát gene tốt trong quá trình tiến hóa.

═══════════════════════════════════════════════════════════════

PHẦN II. ĐẶT VẤN ĐỀ VÀ MÔ HÌNH HÓA HỆ THỐNG

2.1. Bối cảnh bài toán

Bài toán điều khiển Chrome Dinosaur Agent thuộc lớp bài toán Quyết định theo chuỗi thời gian (Sequential Decision-Making Problem) trong môi trường mô phỏng. Agent cần đưa ra quyết định hành động tại mỗi khung hình (frame) dựa trên quan sát môi trường sao cho tối đa hóa điểm số tích lũy — tương đương với thời gian sống sót càng lâu càng tốt.

Theo Formalism MDP (Markov Decision Process), bài toán được đặc tả bởi bộ năm thành phần:

    M = (S, A, P, R, γ)

Trong đó:
• S là không gian trạng thái (State Space).
• A là không gian hành động (Action Space).
• P : S × A × S → [0,1] là hàm xác suất chuyển trạng thái.
• R : S × A → R là hàm phần thưởng (Reward Function).
• γ ∈ [0,1] là hệ số chiết khấu phần thưởng tương lai (Discount Factor).

Điểm khác biệt cốt lõi so với học tăng cường truyền thống là nghiên cứu này không sử dụng phương trình Bellman hay Gradient Descent để cập nhật tham số, mà sử dụng cơ chế chọn lọc tự nhiên của GA. Điều này giúp loại bỏ hoàn toàn các vấn đề như gradient vanishing, gradient exploding, và không cần Experience Replay Buffer để ổn định quá trình học.

2.2. Không gian trạng thái S — State Space

Vector trạng thái đầu vào cho mạng FNN là một vector 15 chiều, được chuẩn hóa về khoảng [0, 1]:

    s_t = [s1, s2, ..., s15] ∈ R^15

Vector trạng thái được xây dựng từ hai vật cản gần nhất (mỗi vật cản 5 đặc trưng) cùng các thông tin toàn cục về game. Thiết kế này tuân thủ nguyên tắc Markov Property — mỗi trạng thái chứa đủ thông tin để ra quyết định tối ưu mà không cần lịch sử quá khứ.

Cấu trúc chi tiết vector 15 chiều:

    Chỉ số    Thành phần              Mô tả                                               Công thức chuẩn hóa
    s1        time_to_obstacle_1       Thời gian (frames) đến vật cản thứ nhất           min(1.0, t_frames / 60)
    s2        height_obstacle_1        Chiều cao vật cản thứ nhất                       min(1.0, h / 160)
    s3        width_obstacle_1         Chiều rộng vật cản thứ nhất                      min(1.0, w / 100)
    s4        is_bird_1                Cờ nhị phân: 1 nếu là chim (ptera)                  {0, 1}
    s5        action_hint_1            Gợi ý hành động: 0.0=Nhảy, 0.5=Cúi, 1.0=Chạy    {0.0, 0.5, 1.0}
    s6        time_to_obstacle_2       Thời gian đến vật cản thứ hai                     Tương tự s1
    s7        height_obstacle_2        Chiều cao vật cản thứ hai                        Tương tự s2
    s8        width_obstacle_2         Chiều rộng vật cản thứ hai                       Tương tự s3
    s9        is_bird_2                Cờ nhị phân cho vật cản thứ hai                   {0, 1}
    s10       action_hint_2           Gợi ý hành động cho vật cản thứ hai              {0.0, 0.5, 1.0}
    s11       game_speed              Tốc độ hiện tại của game                          v_game / MAX_SPEED
    s12       is_jumping              Cờ nhị phân: Agent đang nhảy                      {0, 1}
    s13       is_ducking              Cờ nhị phân: Agent đang cúi                       {0, 1}
    s14       remaining_airtime        Thời gian còn lại trên không                       min(1.0, t_land / T_jump)
    s15       velocity_y              Vận tốc dọc (âm=lên, dương=xuống)                v_y / JUMP_VEL

Phân tích thiết kế đặc trưng:

    • Cặp vật cản gần nhất (s1–s5 và s6–s10): Mỗi vật cản được mã hóa bằng 5 đặc trưng. Điểm then chốt là trường time_to_obstacle = dist_px / game_speed. Công thức này đảm bảo Agent có cùng ngữ cảnh thời gian bất kể tốc độ game.

    • Trường action_hint (s5, s10): Cung cấp tín hiệu hành động rõ ràng cho Agent thay vì để Agent tự học từ giá trị pixel thô.
        – 0.0: Bắt buộc phải nhảy (cactus mọi loại, chim sát đất).
        – 0.5: Bắt buộc phải cúi (chim độ cao trung bình).
        – 1.0: Có thể chạy qua (chim bay cao).

    • Trường width (s3, s8): Cactus_small và cactus_double có cùng chiều cao nhưng khác chiều rộng. Nếu thiếu width, Agent không phân biệt được hai loại này — dẫn đến nhảy cùng kiểu và chết ở cactus đôi.

    • Trường remaining_airtime (s14): Thời gian còn lại Agent còn trên không, tính từ phương trình bậc 2 của chuyển động rơi tự do: T_jump = 2 × JUMP_VEL / GRAVITY ≈ 33.6 frames.

    • Trường velocity_y (s15): Vận tốc dọc chuẩn hóa, cho phép mạng phân biệt Agent đang ở pha lên hay pha xuống của quỹ đạo nhảy.

2.3. Không gian hành động A — Action Space

Không gian hành động là rời rạc và hữu hạn, gồm 3 hành động nguyên tử:

    A = {a0, a1, a2}

    Chỉ số    Tên           Mô tả                                                    Ràng buộc vật lý
    a0         Duck (Cúi)    Agent giảm chiều cao hitbox                              Chỉ hiệu lực khi Agent ở mặt đất và không đang nhảy
    a1         Jump (Nhảy)   Đặt vy = -JUMP_VEL = -18.5 px/frame                     Chỉ hiệu lực khi Agent ở mặt đất và không đang cúi
    a2         Run (Giữ)     Không thực hiện hành động gì                           Luôn hợp lệ

Quyết định của Agent tại frame t:

    a_t = argmax_{a in A} Q(s_t; θ)

trong đó Q là giá trị hành động ước lượng bởi mạng FNN với tham số θ.

═══════════════════════════════════════════════════════════════

PHẦN III. KIẾN TRÚC MẠNG THẦN KINH

3.1. Cấu trúc 3 lớp ẩn

Mạng FNN gồm kiến trúc 4 lớp trọng số:

    Input (15) → Dense (256, ReLU) → Dense (128, ReLU) → Dense (3, Linear) → Softmax

    • Lớp Dense thứ nhất (Input → Hidden 1): Ma trận trọng số W^(1) ∈ R^(15×256), b^(1) ∈ R^256. Hàm kích hoạt: ReLU(x) = max(0, x).

    • Lớp Dense thứ hai (Hidden 1 → Hidden 2): Ma trận trọng số W^(2) ∈ R^(256×128), b^(2) ∈ R^128. Hàm kích hoạt: ReLU.

    • Lớp Dense thứ ba (Hidden 2 → Output): Ma trận trọng số W^(3) ∈ R^(128×3), b^(3) ∈ R^3. Không có hàm kích hoạt phi tuyến — chỉ là linear projection. Softmax chuẩn hóa đầu ra thành phân phối xác suất:
        Softmax(z_i) = exp(z_i - max_j z_j) / Σ_j exp(z_j - max_j z_j)

Hành động cuối cùng được chọn bằng argmax trên vector softmax. Cơ chế tự cắt state trong hàm forward() đảm bảo tương thích ngược khi load model được train với INPUT_SIZE khác.

3.2. Khởi tạo trọng số — Xavier Uniform Initialization

Hệ thống sử dụng Xavier Uniform Initialization:

    w_ij ~ U(-sqrt(6) / sqrt(n_in + n_out), sqrt(6) / sqrt(n_in + n_out))

Bias vectors được khởi tạo bằng 0. Phương pháp này giữ cho variance của tín hiệu lan truyền xuôi ổn định qua các lớp, tránh hiện tượng gradient vanishing hoặc exploding.

3.3. Tổng số tham số

    Lớp                        Trọng số            Bias     Tổng
    Input→Hidden 1             15 × 256 = 3,840   256      4,096
    Hidden 1→Hidden 2          256 × 128 = 32,768 128     32,896
    Hidden 2→Output            128 × 3 = 384       3        387
    Tổng cộng                  —                    —        37,379 tham số

Không gian giả thuyết (Hypothesis Space): H = {x ∈ R^d} với d = 37,379.

3.4. Mã hóa Nhiễm sắc thể (Chromosome Encoding)

Mỗi cá thể x^(i) trong quần thể P là một mạng FNN hoàn chỉnh. Toàn bộ tham số — bao gồm ma trận trọng số W^(l) và vector biases b^(l) của cả 3 lớp Dense — được phẳng hóa (flatten) thành một vector số thực 1 chiều, đóng vai trò Nhiễm sắc thể (Chromosome):

    x = [W^(1).flatten(), b^(1).flatten(), W^(2).flatten(), b^(2).flatten(), W^(3).flatten(), b^(3).flatten()]

═══════════════════════════════════════════════════════════════

PHẦN IV. HÀM THÍCH NGHI VÀ CÁC PHÉP TOÁN TIẾN HÓA

4.1. Thiết kế Hàm Thích nghi (Fitness Function)

Hàm fitness f : X → R≥0 được định nghĩa trực tiếp là điểm số tích lũy (score) của Agent trong một episode:

    f(x) = Score(x)

trong đó Score(x) là giá trị dino.score sau khi episode kết thúc do va chạm hoặc đạt MAX_STEPS_PER_EPISODE = 20,000 frames.

Cơ chế tính điểm chuẩn Chrome Dino:

    dino.score = floor( Σ_{t=0}^{τ} v_game(t) / SCORE_DISTANCE )

với SCORE_DISTANCE = 42.0 px và tốc độ game:

    v_game(t) = min(v0 + t × SPEED_INCREMENT, MAX_SPEED)

với v0 = 6.0 px/frame, SPEED_INCREMENT = 0.008 px/frame², MAX_SPEED = 30.0 px/frame.

Đánh giá đa lần giảm phương sai (Multi-Evaluation for Variance Reduction):

Mỗi cá thể được đánh giá qua n_eval = 5 lần chạy độc lập, và fitness là trung bình cộng:

    f_hat(x) = (1/n_eval) × Σ_{j=1}^{n_eval} Score_j(x)

Việc đánh giá đa lần giảm phương sai ước lượng fitness — yếu tố quan trọng vì môi trường game có tính ngẫu nhiên cao (spawn pattern, loại vật cản, độ cao chim).

4.2. Quần thể và Quá trình Chọn lọc — Tournament Selection

Ký hiệu P^(g) là quần thể ở thế hệ g, với kích thước |P^(g)| = N = 80. Mỗi cá thể x_i^(g) ∈ R^d với d = 37,379.

Thuật toán Tournament Selection với Age-Penalty:

    Input: Quần thể P, kích thước giải đấu k, số cá thể cần chọn N
    Output: Danh sách N cha mẹ được chọn

    parents = []
    for j = 1 to N:
        candidates = RANDOM_SAMPLE(P, k)
        adj_scores = [(c, f(c) - age_penalty(c)) for c in candidates]
        winner = ARGMAX(adj_scores) theo adj_score
        parents.append(winner)
    RETURN parents

Cấu hình: k = 5. Áp lực chọn lọc tương đương: P(select best) ≈ 0.096.

Age-Penalty được tính như sau:

    f_adj(x) = f(x) - max(0, x.age - 20) × AGE_PENALTY × f(x)

với AGE_PENALTY = 0.005. Cá thể già hơn 20 thế hệ bị phạt fitness, khuyến khích cá thể trẻ và giảm overfitting vào pattern cũ.

4.3. Lai ghép — Uniform Crossover

Lai ghép đồng nhất (Uniform Crossover) được áp dụng vì các gen trọng số trong chromosome không có thứ tự ngữ nghĩa cố định.

Thuật toán Uniform Crossover:

    for i = 1 to d:    # d = 37,379 gen
        if RANDOM() < 0.5:
            x^C1[i] = x^P1[i];  x^C2[i] = x^P2[i]
        else:
            x^C1[i] = x^P2[i];  x^C2[i] = x^P1[i]

Tỷ lệ lai ghép: pc = 0.80.

4.4. Đột biến — Adaptive Gaussian Mutation

Đột biến Gauss với sigma và tỷ lệ thích ứng theo thế hệ:

    x'_i = x_i + Δx_i,   Δx_i ~ N(0, σm²)

với:

    σm(g) = σ_base × (1.0 - 0.5 × g/Gmax)
    pm(g) = pm_base × (1.0 - 0.3 × g/Gmax)

với σ_base = 0.10, pm_base = 0.08.

Ở thế hệ đầu (g=0): σ = 0.10, pm = 0.08 → khám phá rộng.
Ở thế hệ cuối (g=Gmax): σ = 0.05, pm = 0.056 → khai thác tinh tế.

Số gen đột biến trung bình mỗi cá thể ở thế hệ đầu: E[n_mutated] = 0.08 × 37,379 ≈ 2,990 gen.

4.5. Chiến lược Ưu tú — Elitism

Elitism đảm bảo các cá thể xuất sắc nhất được bảo tồn nguyên vẹn qua thế hệ tiếp theo:

    P^(g+1)_elite = { x^(g)_i in P^(g) | rank(f(x^(g)_i)) <= E }

với E = 8 (10% của quần thể 80 cá thể).

4.6. Cơ chế chống Catastrophic Forgetting

Catastrophic Forgetting là hiện tượng cá thể tốt nhất bị phá hủy bởi genetic operators ở các thế hệ sau, dẫn đến Best fitness lao dốc nghiêm trọng. Giải pháp trong nghiên cứu này:

Bước 1: Giữ top E = 8 cá thể theo raw fitness (Elitism thông thường).
Bước 2: BẮT BUỘC đưa best_ever (cá thể tốt nhất từng thấy) vào quần thể mới tại vị trí ngẫu nhiên.
Bước 3: Tiếp tục sinh offspring bằng Tournament + Crossover + Mutation.

Cơ chế này đảm bảo best_ever.fitness chỉ tăng, không bao giờ giảm — thước đo tiến hóa thực sự.

4.7. Tracking đỉnh Fitness — Trường best_gen

Mỗi cá thể mang trường best_gen — thế hệ mà cá thể đó đạt fitness cao nhất từ trước đến nay:

    if f(x_current) > f(x_best_ever):
        x_best_ever = x_current
        best_gen = g

Trường best_gen được lưu vào file checkpoint và hiển thị trên biểu đồ.

4.8. Sơ đồ Thuật toán Tổng hợp

    Input: N=80, Gmax=300, pc=0.80, pm_base=0.08, σ_base=0.10, k=5, E=8, n_eval=5

    # Khởi tạo
    P = [GenomeIndividual(DinoNet(cfg)) for _ in range(N)]
    best_ever = None
    history = []

    for g = 0 to Gmax:
        # Đánh giá fitness (đa lần chạy)
        for ind in P:
            scores = [run_episode(ind) for _ in range(n_eval)]
            ind.fitness = MEAN(scores)
            ind.age = g

        # Cập nhật best_ever + best_gen
        current_best = ARGMAX(P, key=fitness)
        if best_ever is None or current_best.fitness > best_ever.fitness:
            best_ever = current_best
            best_ever.best_gen = g

        # Ghi lịch sử
        history.append({g, best_fit, avg_fit, worst_fit, std_fit})

        # Tạo thế hệ mới
        sorted_pop = SORT(P, reverse=True)
        new_pop = [ind.copy() for ind in sorted_pop[:E]]   # E cá thể ưu tú

        # BẮT BUỘC đưa best_ever vào quần thể mới (chống catastrophic forgetting)
        if best_ever is not None:
            be_copy = best_ever.copy()
            be_copy.age = g
            insert_pos = RANDOM(0, min(E, pop_size-1))
            new_pop.insert(insert_pos, be_copy)

        parents = TOURNAMENT_SELECT_ADAPTIVE(P, k=5)  # dùng age-penalty

        while LEN(new_pop) < N:
            p1, p2 = RANDOM_SAMPLE(parents, 2)
            child = UNIFORM_CROSSOVER(p1, p2) if RANDOM() < pc else p1.copy()
            ADAPTIVE_GAUSSIAN_MUTATE(child, g, Gmax, σ_base=0.10, pm_base=0.08)
            new_pop.append(child)

        P = new_pop[:N]

        if improved and g % checkpoint_every == 0:
            SAVE_CHECKPOINT(best_ever)

    RETURN best_ever

Độ phức tạp tính toán mỗi thế hệ:

    • Đánh giá quần thể: O(N × n_eval × τmax) — 80 × 5 × τmax frames
    • Tournament Selection: O(N × k) — 400 so sánh
    • Crossover: O(N × d) — ≈ 2.99 triệu phép toán
    • Mutation: O(N × d × pm) — ≈ 179 nghìn phép toán

═══════════════════════════════════════════════════════════════

PHẦN V. KẾT QUẢ THỰC NGHIỆM

5.1. Tham số thực nghiệm

    Tham số                            Giá trị
    Quần thể N                        80
    Số thế hệ Gmax                    300
    Kích thước giải đấu k             5
    Tỷ lệ lai ghép pc                 0.80
    Tỷ lệ đột biến pm_base           0.08 (thích ứng)
    Cường độ đột biến σm_base         0.10 (thích ứng)
    Số ưu tú E                        8
    Số lần đánh giá/cá thể            5
    Age penalty                        0.005 × fitness × max(0, age-20)
    Kiến trúc mạng                    15→256→128→3
    Tổng tham số                      37,379
    Tốc độ game                       6.0 → 30.0 px/frame
    Số lần chạy độc lập (đánh giá)   20

5.2. Các chỉ số đo lường chính

    • Best Score: Fitness cao nhất mà cá thể tốt nhất đạt được toàn bộ quá trình.
    • Best at gen: Thế hệ mà cá thể đó đạt đỉnh fitness, được track chính xác bằng trường best_gen.
    • Final Avg: Fitness trung bình của quần thể ở thế hệ cuối.
    • Fitness Std: Độ lệch chuẩn fitness trong quần thể — phản ánh mức độ đa dạng.

5.3. Phân tích biểu đồ Training Dashboard

Biểu đồ gồm 4 panel, mỗi panel cung cấp góc nhìn khác nhau về quá trình tiến hóa:

Panel 1 — Best & Avg Fitness / Generation:

    • Đường xanh mờ (Best): Fitness cao nhất mỗi thế hệ. Dao động do ngẫu nhiên môi trường game — hiện tượng bình thường.
    • Đường cam mờ (Avg): Fitness trung bình quần thể.
    • Đường xanh đậm (Best MA): Đường Best đã làm mượt bằng Moving Average — thể hiện xu hướng thực sự của quá trình tiến hóa.
    • Điểm đỏ (Peak): Đỉnh fitness cao nhất toàn bộ quá trình, với chú thích gen=X đúng (nhờ trường best_gen).

Panel 2 — Best Fitness (Smoothed):

    • Đường Best đã làm mượt với vùng tô mờ xung quanh biểu diễn phương sai giữa các thế hệ liền kề.
    • Khi đường bắt đầu nằm ngang và vùng tô thu hẹp → thuật toán đã hội tụ.

Panel 3 — Fitness Range (Best/Avg/Worst):

    • Vùng Range (Best – Worst) cho biết độ đa dạng của quần thể.
    • Thu hẹp dần → quần thể hội tụ. Thu hẹp quá sớm → nguy cơ stuck ở local optima.

Panel 4 — Fitness Distribution (Final Gen):

    • Histogram 20 bins của 80 cá thể ở thế hệ cuối.
    • Hình chuông → quần thể đã hội tụ tốt. Lệch trái mạnh → vẫn còn nhiều cá thể yếu kéo lùi trung bình.

5.4. Hiện tượng Catastrophic Forgetting và cách khắc phục

Catastrophic Forgetting xảy ra khi Best fitness lao dốc nghiêm trọng qua các thế hệ, dù có Elitism. Nguyên nhân:

    1. Elitism chỉ bảo tồn top E = 8 cá thể tốt NHƯNG trong 72 cá thể còn lại, crossover và mutation có thể phá hủy gene tốt. Nếu một cá thể có fitness rất cao do may mắn trong 1 lần chạy (với n_eval=3, phương sai cao), nó được copy vào elite, nhưng offspring kế tiếp có thể kém hơn nhiều.

    2. Với n_eval=3, fitness ước lượng noisy → Tournament chọn nhầm cha mẹ → offspring không tốt.

Giải pháp đã triển khai:

    • Tăng n_eval từ 3 lên 5 → giảm phương sai fitness ước lượng.
    • Tăng E từ 2 lên 8 → bảo tồn nhiều cá thể tốt hơn.
    • Đưa best_ever BẮT BUỘC vào quần thể mới → best_ever.fitness chỉ tăng.
    • Age-penalty → khuyến khích cá thể trẻ, giảm overfitting vào pattern cũ.
    • Adaptive mutation → giảm sigma theo thế hệ → ít phá hỏng gene tốt ở thế hệ cuối.

5.5. So sánh với phương pháp Gradient-based

    Tiêu chí                         GA (nghiên cứu này)     DQN              Policy Gradient
    Gradient-free                     Có                      Không            Không
    Không cần Experience Replay      Có                      Không            Không
    Không cần hàm mất mát khả vi   Có                      Không            Không
    Khả năng tránh local optima     Có (mutation ngẫu nhiên)  Phụ thuộc ε-greedy  Phụ thuộc entropy bonus
    Rủi ro gradient vanishing/exploding  Không có              Có              Có

═══════════════════════════════════════════════════════════════

PHẦN VI. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

6.1. Đóng góp chính

    1. Thiết kế vector trạng thái 15 chiều kết hợp đặc trưng không gian, vật lý, và ngữ cảnh thời gian: trường action_hint cung cấp tín hiệu hành động rõ ràng, trường time_to_obstacle phản ánh đúng áp lực thời gian bất kể tốc độ game, và trường width giúp phân biệt cactus đơn với cactus đôi.

    2. Kiến trúc FNN 15→256→128→3 với 37,379 tham số, đủ phức tạp để học chính sách hành động phi tuyến tính trong không gian trạng thái 15 chiều.

    3. Cơ chế tự cắt state trong DinoNet.forward() — đảm bảo tương thích ngược khi load model.

    4. Tracking best_gen chính xác — khắc phục nhược điểm của trường age, hiển thị đúng thế hệ đạt đỉnh fitness trên biểu đồ.

    5. Quy trình đánh giá đa lần (n_eval = 5) giảm phương sai ước lượng fitness.

    6. Cơ chế chống Catastrophic Forgetting: best_ever bắt buộc vào quần thể mỗi thế hệ.

    7. Hệ thống toán tử di truyền Tournament + Uniform Crossover + Adaptive Gaussian Mutation + Age-Penalized Elitism, với tham số được cấu hình hợp lý và có thể điều chỉnh qua CLI.

6.2. Hạn chế

    • GA không đảm bảo tìm được global optimum — luôn tồn tại xác suất hội tụ về local optimum.
    • Chi phí tính toán cao: 400 episodes/thế hệ.
    • Adaptive mutation sử dụng công thức tuyến tính đơn giản — chưa tối ưu bằng cơ chế tự thích ứng phức tạp hơn.
    • Không tái sử dụng kinh nghiệm từ các episode trước (khác với Experience Replay của DQN).

6.3. Hướng phát triển

    • Adaptive Mutation Rate với cơ chế phức tạp hơn: Thay công thức tuyến tính bằng σm^(g) = σ_min + (σ_max - σ_min) × exp(-λ × g/Gmax) để chuyển đổi mượt hơn từ exploration sang exploitation.

    • NEAT (Neuroevolution of Augmenting Topologies): Tiến hóa đồng thời cấu trúc (topology) và trọng số — tự động tìm kiếm kiến trúc mạng tối ưu thay vì cố định 3 lớp ẩn.

    • CMA-ES (Covariance Matrix Adaptation Evolution Strategy): Thay vì mutation độc lập từng gen từ N(0, σ²), CMA-ES học ma trận hiệp phương sai toàn cục, cho phép tương quan giữa các gen được mô hình hóa chính xác hơn trong không gian 37,379 chiều.

    • Hybrid GA + Gradient Descent: Sử dụng GA để khám phá không gian rộng trước, sau đó dùng Gradient Descent để tinh chỉnh trong vùng nghiệm tốt tìm được.

    • Speciation: Chia quần thể thành các species dựa trên độ tương đồng của chromosome, giúp bảo tồn đa dạng di truyền tốt hơn.
