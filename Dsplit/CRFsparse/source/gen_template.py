def main():
    fp = open('../../../../data/tfidf.scale.5000','r')
    label = {}
    for line in fp:
        line = line.strip().split(' ')
        label[line[0]] = 1
    print(len(label))
    fp = open('template_lshtc1_5000','w')
    fp.write('label:\n')
    for l in label:
        fp.write(l+' ')

if __name__ == '__main__':
    main()
