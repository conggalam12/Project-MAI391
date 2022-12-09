
# Tập dữ liệu ví dụ

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

"""**Bước 2: Tính xác suất likelihood**
- Tính $V$: Dựa vào `freqs` tính số lượng các từ duy nhất (uniqe words) - gọi là bộ từ điển

- Tính $N_{pos}$ và $N_{neg}$: Dựa vào `freqs` dictionary, tính tổng số từ (có thể trùng lặp) xuất hiện trong positive samples $N_{pos}$ và negative samples $N_{neg}$.

- Tính tần suất xuất hiện của mỗi từ trong positive samples $freq_{pos}$ và trong negative samples $freq_{neg}$

- Tính xác suất likelihood mỗi từ trong bộ từ điển
    - Sử dụng hàm `lookup` lấy ra tần suất xuất hiện của từ là positive $freq_{pos}$, và tần xuất xuất hiện của từ là negative $freq_{neg}$
- Tính xác suất cho mỗi từ thuộc vào positive sample: $P(W_{pos})$, thuộc vào negative sample $P(W_{neg})$ sử dụng công thức 4 & 5.

$$ P(W_{pos}) = \frac{freq_{pos} + 1}{N_{pos} + V}\tag{4} $$
$$ P(W_{neg}) = \frac{freq_{neg} + 1}{N_{neg} + V}\tag{5} $$

**Note:** Chúng ta lưu trữ likelihood của mỗi từ vào dictionary với key (từ): $W$, value (dictionary): ${0: P(W_{pos}), 1: P(W_{pos})}$
"""

# Ex 3
def compute_likelihood(freqs):
    # Tính xác suất likelihood của mỗi từ trong bộ từ điển

    ### START CODE HERE
    # Tính V các từ duy nhất xuất hiện trong tập train
    vocab = set([pair[0] for pair in freqs.keys()])
    V = len(vocab)

    # Tính N_pos: số lượng từ trong positive samples và N_neg: số từ trong negative sample
    N_pos = N_neg = 0
    for pair in freqs.keys():
        # Nếu như class: 1 tăng N_pos thêm số lần xuất hiện của pair trong freqs
        if pair[1] > 0:
            N_pos += freqs[pair]

        # Nếu như class: 0 tăng N_neg thêm số lần xuất hiện của pair trong freqs
        else:
            N_neg += freqs[pair]
    
    print(f'V: {V}, N_pos: {N_pos}, N_neg: {N_neg}')

    # Tính likelihood cho mỗi từ trong bộ từ điển
    p_likelihood = {}
    for word in vocab:
        # Lấy tần xuất xuất hiện của mỗi từ là positive hoặc negative
        freq_pos = lookup(freqs, word, 1)
        freq_neg = lookup(freqs, word, 0)

        # Tính xác suất likelihood của mỗi từ với class positive và negative
        p_w_pos = (freq_pos + 1) / (N_pos + V)
        p_w_neg = (freq_neg + 1) / (N_neg + V)

        # Lưu vào p_likelihood dictionary
        p_likelihood[word] = {0:p_w_neg, 1:p_w_pos}
    # END CODE HERE
    
    return p_likelihood

# Kết quả mong đợi
compute_likelihood(freqs)

"""**Bước 3: Hoàn thiện `train` function cho Naive Bayes***"""

def train_naive_bayes(train_x, train_y):
    ''' Huấn luyện thuật toán Naive Bayes
    Args:
        train_x: Danh sách các câu
        train_y: Danh sách các nhãn tương ứng (0 hoặc 1)
    Output:
        p_prior: the prior probability (Xác suấ tiên nghiệm)
        p_likelihood: the maximum likelihood of the probability.
    '''
    # Xây dựng từ điển tần suất xuất hiện của từ và nhãn tương ứng
    freqs = count_freq_words(train_x, train_y)

    # Tính xác suất tiên nghiệm
    p_prior = compute_prior_prob(train_y)

    # Tính xác suất likelihood
    p_likelihood = compute_likelihood(freqs)

    return p_prior, p_likelihood

# Kết quả đầu ra thu được khi huấn luận Naive Bayes Classifier
p_prior, p_likelihood = train_naive_bayes(train_x, train_y)
p_prior, p_likelihood

"""###1.4.Dự đoán với các mẫu thử nghiệm
- Tính xác suất của mỗi sample (n từ) dựa vào công thức:
$$P(0).P(S|0) = P(0).P(w_{1}|0).P(w_{2}|0)...P(w_{n}|0)$$ 
"""

# Ex 4
def naive_bayes_predict(sentence, p_prior, p_likelihood):
    '''
    Args:
        sentence: a string
        p_prior: a dictionary of the prior probability
        p_likelihood: a dictionary of words mapping to the probability
    Output:
        p: the probability of sentence with 0: negative, 1: positive

    '''
    # Tiền xử lý dữ liệu
    words = basic_preprocess(sentence)

    # Khởi tạo giá trị xác suất ban đầu là giá trị xác suất tiên nghiệm
    p_neg = p_prior[0]
    p_pos = p_prior[1]

    for word in words:
        # Kiểm tra xem word có tồn tại trong p_likelihood hay không
        if word in p_likelihood:
            ### START CODE HERE
            # nhân xác suất tiên nghiệm với xác suất likelihood của các từ
            p_neg *= p_likelihood[word][0]
            p_pos *= p_likelihood[word][1]
            ### END CODE HERE
    return {'prob': {0: p_neg, 1: p_pos},
            'label': 0 if p_neg > p_pos else 1}

# Kết quả mong đợi
sentence = "predictable with no fun"
naive_bayes_predict(sentence, p_prior, p_likelihood)

"""##2.Naive Bayes Classfier for Sentiment Analysis on Tweets
**Phân tích cảm xúc trên tập 1Tweets1 sử dụng thuật toán phân loại Naive Bayes**
"""

import re
import string
import nltk
import numpy as np
nltk.download('twitter_samples')
from nltk.corpus import twitter_samples
from nltk.tokenize import TweetTokenizer
from tqdm import tqdm

"""###2.1.Dowload Dataset"""

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

print(train_y)

all_positive_tweets[:10]

all_negative_tweets[:10]

"""###2.2.Tiền xử lý dữ liệu cho tập `Tweets`

Dựa vào việc quan sát tập dữ liệu trên chúng ta tiến hàng một số bước tiền xử lý như sau:
- Xóa bỏ các hashtags như #FollowFriday,...
- Xóa bỏ các thẻ gắn nhãn các tài khoản như: @Lamb2ja
- Xóa bỏ các thẻ HTML, CSS,.. có thể có như: https://t.co/smyYriipxI
- Xóa bỏ retweet trong text: "RT"
- Xóa bỏ dấu câu và có thể xóa hết số, ký tự đặc biệt (Với mục đích tập trung ngữ nghĩa các từ)
- Có thể thực hiện một số bước tiền xử lý khác
- Sau khi tiền xử lý xong chúng ta tiến hành tách câu thành các từ (word base tokenizer). Ở đây chúng ta sẽ dùng bộ tách từ có sẵn cho tách từ `tweet` của nltk là `TweetTokenizer`
"""

# Ex 5
def basic_preprocess(text):
    '''
    Args:
        text: câu đầu vào
    Output:
        text_clean: danh sách các từ (token) sau khi chuyển sang chữ thường và
            được phân tách bởi khoảng trắng
    '''
    ### START CODE HERE
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

    ### END CODE HERE
    return text_clean

#demo tu vung
for i in train_neg:
  a = basic_preprocess(str(i))
  for j in a:
    print(j,end = ' ')
  print()

# Kết quả mong đợi
example_sentence = "RT @Twitter @chapagain Hello There! Have a great day. #good #morning http://chapagain.com.np"
basic_preprocess(example_sentence)

"""###2.3.Huấn luyện Naive Bayes Classifier trên tập `Tweets`"""

p_prior, p_likelihood = train_naive_bayes(train_x, train_y)

#Kết quả ví dụ về xác suất tiên nghiệm và likelihood của từ happy
p_prior, p_likelihood['happy']

"""###2.4.Dự đoán"""

test_x[0], test_y[0]

naive_bayes_predict(test_x[0], p_prior, p_likelihood)

"""###2.5.Đánh giá độ chính xác trên tập test"""

acc = 0
for sentence, label in zip(test_x, test_y):

    # predic each sentence in test set
    pred = naive_bayes_predict(sentence, p_prior, p_likelihood)['label']

    # compare predict label with target label
    if int(pred) == int(label):
        acc += 1

print('Accuracy: ', acc/len(test_x))
