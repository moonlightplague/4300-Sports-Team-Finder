from text_preprocess import tokenize

def merge_postings(postings1, postings2):
    """ 
    merge posting for or
    
    """
    merged_posting=[]
    i,j=0,0
    while i<len(postings1) and j<len(postings2):
        if postings1[i]==postings2[j]:
          merged_posting.append(postings1[i])
          i+=1
          j+=1
        elif postings1[i]<postings2[j]:
          merged_posting.append(postings1[i])
          i+=1
        else:
          merged_posting.append(postings2[j])
          j+=1

    while i < len(postings1):
      merged_posting.append(postings1[i])
      i += 1
    while j < len(postings2):
      merge_postings.append(postings2[j])
      j += 1
    return merged_posting