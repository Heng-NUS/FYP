import sys
from palmettopy.palmetto import Palmetto
palmetto = Palmetto()
# return:    {wid:w, ...}
def read_voca(pt):
    voca = {}
    for l in open(pt):
        wid, w = l.strip().split('\t')[:2]
        voca[int(wid)] = w
    return voca

def read_pz(pt):
    return [float(p) for p in open(pt).readline().split()]
    
# voca = {wid:w,...}
def dispTopics(pt, voca, pz, max_words=10):
    k = 0
    topics = []
    for l in open(pt):
        vs = [float(v) for v in l.split()]
        wvs = zip(range(len(vs)), vs)
        wvs = sorted(wvs, key=lambda d:d[1], reverse=True)
        #tmps = ' '.join(['%s' % voca[w] for w,v in wvs[:10]])
        tmps = ' '.join(['%s' % (voca[w]) for w, v in wvs[:max_words]])
        topics.append((pz[k], tmps))
        k += 1
        
    print('Topics')
    topics = [s for pz, s in sorted(topics, reverse=True)]
    i = 1
    for topic in topics:
        cp = palmetto.get_coherence(topic.split(), coherence_type="cp")
        ca = palmetto.get_coherence(topic.split(), coherence_type="ca")
        allsum = cp+ca
        if allsum > 0.4:
            print(i, topic)
            i += 1
    return topics

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Usage: python %s <model_dir> <K> <voca_pt>' % sys.argv[0])
        print('\tmodel_dir    the output dir of BTM')
        print('\tK    the number of topics')
        print('\tvoca_pt    the vocabulary file')
        exit(1)
        
    model_dir = sys.argv[1]
    K = int(sys.argv[2])
    voca_pt = sys.argv[3]
    voca = read_voca(voca_pt)    
    W = len(voca)

    pz_pt = model_dir + 'pz'
    pz = read_pz(pz_pt)
    
    zw_pt = model_dir + 'pw_z'
    dispTopics(zw_pt, voca, pz)
