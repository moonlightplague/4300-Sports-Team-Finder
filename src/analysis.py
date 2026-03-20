from text_preprocess import tokenize

def merge_postings(term1,term2):
    postings1=inverted_index[term1]
    postings2=inverted_index[term2]
    merged_posting=[]
    i,j=0,0
    while i<len(postings1) and j<len(postings2):
        if postings1[i]==postings2[j]:
          merged_posting.append(postings1[i])
          i+=1
          j+=1
        elif postings1[i]<postings2[j]:
          i+=1
        else:
          j+=1

    return merged_posting