import fasttext
from tqdm import tqdm
import time
load_model = fasttext.load_model(r"/ai_jfs/mengying/fastText/python/doc/examples/topic.bin")
test_data = open('/ai_jfs/mengying/data/classify_topic/Test_topic_classify_result_0808.txt','r').readlines()
total_num =len(test_data)
right_num = 0
t1 = time.time()
for line in tqdm(test_data, total=total_num):
    topic = ' '.join(line.strip().split('__label__')[-1].split(' ')[1:])
    predict = load_model.predict(topic)[0][0]
    label = line.strip().split('__label__')[-1].split(' ')[0]
    if predict.replace('__label__', '')==label:
        right_num += 1
    else:
        print('-'*60)
        print("TOPIC:", topic)
        print("PREDICT",predict)
        print("LABEL:", label)
t2 = time.time()
print("TOTAL:", total_num)
print("TIME:", t2-t1)
print("Average time:", (t2-t1)/total_num)
print("ACCURACY:", right_num/total_num)
