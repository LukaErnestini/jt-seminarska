from text_utils import load_data

data = load_data()
data_haikus_ar = ['\n'.join([row[0], row[1], row[2]]) for row in data]
data_haikus_ar_s = '\n\n'.join(data_haikus_ar)

with open('corps.txt', 'wb') as fp:
    fp.write(data_haikus_ar_s.encode("utf8"))
