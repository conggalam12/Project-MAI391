
train_x = [
           'just plain boring',
           'entirely predictable and lacks energy',
           'no surprises and very few laughs',
           'very powerful',
           'the most fun film of the summer'
]
train_y = [0, 0, 0, 1, 1]

"""###1.1. Tiền xử lý dữ liệu cơ bản"""

def basic_preprocess(text):
    """ Tiền xử lý và tách các câu
    Args:
        text: câu đầu vào. 
        VD: "Tôi đi học"
    Output:
        text_clean: danh sách các từ (token) sau khi chuyển sang chữ thường và
            được phân tách bởi khoảng trắng
    """
    text_clean = text.lower()
    return text_clean.split()

basic_preprocess(train_x[0])

"""###1.2.Xây dựng bộ từ điển"""

# Ex 1
def count_freq_words(corpus, labels):
    """ Xây dựng bộ từ điển tần suất xuất hiện của các từ
    Args:
        corpus: tập danh sách các câu
        labels: tập nhãn tương ứng với các câu trong corpus (0 hoặc 1)
    Output:
        model: bộ từ điển ánh xạ mỗi từ và tần suất xuất hiện của từ đó trong corpus
            key: (word, label)
            value: frequency
            VD: {('boring', 0): 2} => từ boring xuất hiện 2 lần trong các sample thuộc class 0
    """
    model = {}
    for label, sentence in zip(labels, corpus):
        for word in basic_preprocess(sentence):
            # Định nghĩa key của model là tuple (word, label)
            pair = (word, label)
            # Nếu key đã tồn tại trong model thì tăng value lên 1
            if pair in model:
                model[pair] += 1
            # Nếu key chưa tồn tại trong model thì bổ sung key vào model với value =1
            else:
                model[pair] = 1
    return model

#Kết quả mong đợi
freqs = count_freq_words(train_x, train_y)
freqs

# Hàm lấy ra tần suất xuất hiện là value trong `freq` dựa vào key (word, label)
def lookup(freqs, word, label):
    '''
    Args:
        freqs: a dictionary with the frequency of each pair
        word: the word to look up
        label: the label corresponding to the word
    Output:
        count: the number of times the word with its corresponding label appears.
    '''
    count = 0

    pair = (word, label)
    if pair in freqs:
        count = freqs[pair]

    return count

lookup(freqs, "boring", 0)

"""###1.3.Thuật toán phân loại Naive Bayes
**Bước 1: Tính xác suất tiên nghiệm của các class**
- Tính $D$, $D_{pos}$, $D_{neg}$
    - Dựa vào `train_y` tính số lượng các sample có trong tập training: $D$, số lượng các sample là positive (nhãn 1): $D_{pos}$ và số lượng nhãn là negative (nhãn 0): $D_{neg}$
    - Tính xác suất tiên nghiệm của class 1 là: $P(D_{pos})=D_{pos}/D$, và class 0 là: $P(D_{pos})=D_{pos}/D$
"""

# Ex 2
def compute_prior_prob(train_y):
    # Tính D, D_pos, D_neg dựa vào x_train
    ### START CODE HERE
    # Tính D, số lượng các sample trong training
    D = len(train_y)

    # Tính D_pos, số lượng các positive sample trong training
    D_pos = len(list(filter(lambda x: x == 1, train_y))) # hàm filter là đếm các biến trong tập thỏa mãn hàm , ở đây ví dụ label = 1(x)

    # Tính D_neg, số lượng các negative sample trong training
    D_neg = len(list(filter(lambda x: x == 0, train_y)))

    ### END CODE HERE
    # Tính xác suất tiên nghiệm cho các class 0 và 1
    p_prior = {0:(D_neg/D), 1:(D_pos/D)}
    return p_prior

# Kết quả mong đợi
compute_prior_prob(train_y)


###3.1.Download Dataset
#Tương tự như mục 2.1


# Tải về tập dữ liệu tweets
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

# Chia thành 2 tập train và test
# train: 4000 samples, test: 1000 samples
train_pos = all_positive_tweets[:4000]
test_pos = all_positive_tweets[4000:]

train_neg = all_negative_tweets[:4000]
test_neg = all_negative_tweets[4000:]

train_x = train_pos + train_neg
test_x = test_pos + test_neg

# Tạo nhãn negative: 0, positive: 1
train_y = np.append(np.ones(len(train_pos)), np.zeros(len(train_neg)))
test_y = np.append(np.ones(len(test_pos)), np.zeros(len(test_neg)))

print(train_y.shape)

###3.2. Tiền xử lý dữ liệu cho tập Tweets"""

# Ex 6
def basic_preprocess(text):
    '''
    Args:
        text: câu đầu vào
    Output:
        text_clean: danh sách các từ (token) sau khi chuyển sang chữ thường và
            được phân tách bởi khoảng trắng
    '''
    ### START CODE
    # xóa bỏ stock market tickers like $GE
    text = re.sub(r'\$\w*', '', text)

    # xóa bỏ old style retweet text "RT"
    text = re.sub(r'^RT[\s]+', '', text)

    # xóa bỏ hyperlinks
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)

    # xóa bỏ hashtags
    text = re.sub(r'#', '', text)

    # tokenize
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    text_tokens = tokenizer.tokenize(text)

    text_clean = []
    for word in text_tokens:
        if word not in string.punctuation:  # remove punctuation
            text_clean.append(word)

    ### END CODE
    return text_clean

# Kết quả đầu ra
example_sentence = "RT @Twitter @chapagain Hello There! Have a great day. #good #morning http://chapagain.com.np"
basic_preprocess(example_sentence)

"""*italicized text*###3.3.Logistic Regression

####Sigmoid
The sigmoid function: 

$$ h(z) = \frac{1}{1+\exp^{-z}} \tag{1}$$
"""

# Ex 7
def sigmoid(z): 
    '''
    Args:
        z: is the input (can be a scalar or an array)
    Output:
        h: the sigmoid of z
    '''
    
    ### START CODE HERE
    # calculate the sigmoid of z
    h = 1 / (1 + np.exp(-z))
    ### END CODE HERE ###
    
    return h

# Kết quả kiểm tra hàm sigmpoid
sigmoid(0) == 0.5, sigmoid(4.92) == 0.9927537604041685

"""#### Gradient Descent Function
* Số vòng lặp huấn luyện mô hình: `num_iters`
* Với mỗi vòng lặp chúng ta sẽ tính `logits-z`, cost và cập nhật trọng số
* Số samples training: `m`, số features trên mỗi sample: `n`
* Trọng số mô hình:  
$$\mathbf{\theta} = \begin{pmatrix}
\theta_0
\\
\theta_1
\\ 
\theta_2 
\\ 
\vdots
\\ 
\theta_n
\end{pmatrix}$$

* Tính `logits-z`:   $$z = \mathbf{x}\mathbf{\theta}$$
    * $\mathbf{x}$ có chiều (m, n+1) 
    * $\mathbf{\theta}$: có chiều (n+1, 1)
    * $\mathbf{z}$: có chiều (m, 1)
* Dự đoán 'y_hat' có chiều (m,1):$$\widehat{y}(z) = sigmoid(z)$$
* Cost function $J$:
$$J = \frac{-1}{m} \times \left(\mathbf{y}^T \cdot log(\widehat{y}) + \mathbf{(1-y)}^T \cdot log(\mathbf{1-\widehat{y}}) \right)$$
* Cập nhật `theta`:
$$\mathbf{\theta} = \mathbf{\theta} - \frac{\alpha}{m} \times \left( \mathbf{x}^T \cdot \left( \mathbf{\widehat{y}-y} \right) \right)$$
"""

# Ex 8
def gradient_descent(x, y, theta, alpha, num_iters):
    '''
    Args:
        x: matrix of features, có chiều (m,n+1)
        y: label tương ứng (m,1)
        theta: vector trọng số (n+1,1)
        alpha: tốc độ học
        num_iters: số vòng lặp
    Output:
        J: final cost
        theta: vector trọng số
    '''
    ### START CODE HERE
    # lấy m số lượng các sample trong matrix x
    m = len(x)
    
    for i in tqdm(range(num_iters)):
        
        # Tính z, phép dot product: x và theta
        z = np.dot(x, theta)
        
        # Tính h: sigmoid của z
        y_hat = sigmoid(z)
        
        # Tính cost function
        J = (-1 / m) * (np.dot(y.T, np.log(y_hat)) + np.dot((1 - y).T, np.log(1 - y_hat)))

        # Cập nhật trọn số theta
        theta = theta - (alpha / m) * (np.dot(x.T, (y_hat - y)))
        
    ### END CODE HERE ###
    return J, theta

# Kiểm qua kết quả
np.random.seed(1)

# X input: 10 x 3, bias là 1
tmp_X = np.append(np.ones((10, 1)), np.random.rand(10, 2) * 2000, axis=1)

# Y label: 10 x 1
tmp_Y = (np.random.rand(10, 1) > 0.5).astype(float)

# Apply gradient descent
tmp_J, tmp_theta = gradient_descent(tmp_X, tmp_Y, np.zeros((3, 1)), 1e-8, 100)
print(f"\nCost {tmp_J.item()}")
print(f"Weight {tmp_theta}")

print(tmp_X.shape)

"""###3.4.Trích xuất các feature
Chuyển từ `tweet` sang feature
Với mỗi `tweet` sẽ được biểu diễn bởi 2 feature (Dựa vào `freq` tương tự ở mục #1 và #2):
- số lượng các positive words
- số lượng các negative words
"""

# Ex 9
def extract_features(text, freqs):
    '''
    Args: 
        text: tweet
        freqs: bộ từ điển tần suất xuất hiện của từ theo label (word, label)
    Output: 
        x: vector feature có chiều (1,3)
    '''
    # tiền xử lý
    word_l = basic_preprocess(text)
    
    # 3 thành phần: bias, feature 1 và feature 2
    x = np.zeros((1, 3)) 
    
    # bias
    x[0,0] = 1 
    
    ### START CODE HERE
    for word in word_l:
        x[0,1] += lookup(freqs, word, 1)
        
        x[0,2] += lookup(freqs, word, 0)
        
    ### END CODE HERE ###
    assert(x.shape == (1, 3))
    return x

# Kiểm tra
# freqs tương tự mục 1.2
freqs = count_freq_words(train_x, train_y)
print(train_x[0])
extract_features(train_x[0], freqs)



# Kiểm tra
# freqs tương tự mục 1.2
# VD: các từ không có trong bộ `freq`
x_test = "việt nam"
x_test1 = "happy new year"
extract_features(x_test1, freqs)

"""###3.5.Huấn luyện mô hình Logistic Regression"""

# Tạo ma trận X có kích thước mxn với n=3 (số features)
X = np.zeros((len(train_x), 3))
for i in range(len(train_x)):
    X[i, :]= extract_features(train_x[i], freqs)

Y = np.expand_dims(train_y, 1)

# Huấn luyện với số vòng lặp 1500, tốc độ học 1e-6
J, theta = gradient_descent(X, Y, np.zeros((3, 1)), 1e-9, 8000)
print(f"Cost {J.item()}.")
print(f"Weight {theta}")
print(X)

print(X)

"""###3.6.Dự đoán
* Tiền xử lý với dữ liệu thử nghiệm
* Tính `logits` dựa vào công thức

$$y_{pred} = sigmoid(\mathbf{x} \cdot \theta)$$
"""

# Ex 10
def predict_tweet(text, freqs, theta):
    '''
    Args: 
        text: tweet
        freqs: bộ từ điển tần suất xuất hiện của từ theo label (word, label)
        theta: (3,1) vector trọng số
    Output: 
        y_pred: xác suất dự đoán
    '''
    ### START CODE HERE
    
    # extract features
    x = extract_features(text, freqs)
    
    # dự đoán
    y_pred = sigmoid(np.dot(x, theta))
    
    ### END CODE HERE ###
    
    return y_pred

tests = ["I'll see you in the future, when I'm old enough\nHope you have a good and happy life. Goodbye My friend"]
for t in tests:
    pred = predict_tweet(t, freqs, theta)
    if pred<0.5:
      pred = 'Negative'
    else:
      pred = 'Positive'
    print(f'{t} -> {pred}')

"""###3.7.Đánh giá độ chính xác trên tập test"""

acc = 0
for sentence, label in zip(test_x, test_y):

    # predic each sentence in test set
    pred = predict_tweet(sentence, freqs, theta)

    if pred > 0.5:
        pred_l = 1
    else:
        pred_l = 0

    # compare predict label with target label
    if int(pred_l) == int(label):
        acc += 1

print('Accuracy: ', acc/len(test_x))