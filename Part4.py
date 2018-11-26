def data_tuple(file):

    data = []
    lst = []
    with open(file, "r") as f:
        for line in f:
            if line == "\n":
              data.append(lst)
              lst = []
            
            else:
              lines = line.replace("\n",'').split(" ")
              lst.append(tuple(lines))
            
    return data


# Write a function that estimates the emission parameters from the training set using MLE (maximum
# likelihood estimation):
# e(x|y) = Count(y -> x)/Count(y)
# with key <word, tag>

def emission_parameter(data, k=1):

    new_data = []
    new_tweets = []
  
    #word_count = {}
    tags = {} # count of appearance for each tag
    observations = {} # count of appearance for each observation
    smoothed = {} # count of observation appearance after smoothing
    match = {} # count of (X,Y) tuples
    
    # processing data with smoothing
    for lst in data:
        for wordTuple in lst:
          
            observation = wordTuple[0]
            tag = wordTuple[1]
            
            observations[observation] = observations.get(observation,0) + 1
            tags[tag] = tags.get(tag,0) + 1

            if observations[observation] < k:
                smoothed['#UNK#'] = smoothed.get('#UNK#',0) + 1
                match[('#UNK#',tag)] = match.get(('#UNK#',tag),0) + 1         
            else:
                smoothed[observation] = smoothed.get(observation,0) + 1
                match[wordTuple] = match.get(wordTuple,0) + 1
    
    # compute emission parameter          
    em = {i: match[i]/tags[i[1]] for i in match}

    return list(observations), list(tags), em

emission_parameter(data_tuple('SG/train'))


# function that estimates the transition parameters from the training set
# using MLE (maximum likelihood estimation)
# for any state w with previous two states u and v
# q(w|u,v) = Count(u,v,w)/Count(u,v)
# return: a dictionary with tuple of three states as key and their trans as value
def transition_parameter(data):
    double_transitions = {} #(u,v)
    triple_transitions = {} #(u,v,w)

    # processing data
    for lst in data:

        # y-1 = y0 = START
        prev_previous = ''
        previous = '#START#'
        current = '#START#'

        # starting from step 1 to step N
        for wordTuple in lst:

            prev_previous = previous if (previous)
            previous = current

            if lst[-1] == wordTuple: # last item in lst
                current = '#STOP#'
            else:
                current = wordTuple[1]

            double_transitions[(previous,current)] = transitions.get((previous,current),0) + 1
            triple_transitions[(prev_previous,previous,current)] = transitions.get((prev_previous,previous,current),0) + 1

        # final step N+1 --> 'STOP'
        double_transitions[(current,'#STOP#')] = transitions.get((current,'#STOP#'),0) + 1
        triple_transitions[(previous,current,'#STOP#')] = transitions.get((previous,current,'#STOP#'),0) + 1
    
    trans = {i: triple_transitions[i]/tags[i[:2]] for i in triple_transitions}

    return trans


# Viterbi Algorithm
def viterbi(sentense, tags, observations, em, trans):

    pi = [{tag:(0.0,'') for tag in tags} for i in range(len(sentense))]
    
    # j = 0
    pi.insert(0,{'#START#': (1,'')})

    # j = 1
    for v in tags:

        if sentense[0] in training_model:
            if (sentence[0],v) in emission:
                emission = em[(sentence[0],v)]
            else: # emission not found
                emission = 0.0
        else: # this word is #UNK#
            emission = em[('#UNK#',v)]

        pi[j][v] = (trans[('#START#',v)] * emission,'#START#')


    # j = 2 ~ N
    # pi[j][v] = max{ pi[j-1][u] * a(u,v) * b(v,sentence[j]) } 
    # u* = argmax{ pi[j-1][u] * a(u,v) }
    for j in range(2, len(sentence)+1): # loop for each word
        for v in tags: # loop for each possible tag of this word

            score_backward = 0.0
            for u in tags: # loop for each previous tag of current tag

                # assigning emission parameter
                if sentense[j-1] in training_model:
                    if (sentence[j-1],v) in emission:
                        emission = em[(sentence[j-1],v)]
                    else: # emission not found
                        emission = 0.0
                else: # this word is #UNK#
                    emission = em[('#UNK#',v)]

                # maximum score
                score_forward = pi[j-1][u][0] * trans[(u,v)] * emission
                if score_forward > pi[j][v][0]:
                    pi[j][v][0] = score

                # most likely previous tag
                if score_backward < pi[j-1][u][0] * trans[(u,v)]:
                    score_backward = pi[j-1][u][0] * trans[(u,v)]
                    pi[j][v][1] = u

    # j = N+1
    pi.append({'#STOP#': (0.0,'')})
    print (len(pi))    
    for u in tags:

        score = pi[-2][u][0] * trans[(u,'#STOP#')]
        if score > pi[-1]['#STOP#'][0]:
            pi[-1]['#STOP#'][0] = (score,u)

    # Backtracking
    prediction = [pi[-1]['#STOP#'][1]]
    for j in reversed(range(2,len(sentence)+1)):
        prediction.insert(0, pi[j][prediction[0]][1])

    return prediction


if __name__ == '__main__':

    train = data_tuple('machine-learning-project/SG/train')

    test_in = data_tuple('machine-learning-project/SG/dev.in')

    observations, tags, em = emission_parameter(train)
    trans = transition_parameter(train)

    for lst in test_in:

        f = open('machine-learning-project/SG/dev.out')

        sentence = []

        for wordTuple in lst:
            sentence.append(wordTuple[0])

        prediction = viterbi(sentence,tags,observations,em,trans)

        for i in range(len(sentence)):
            if prediction:
                f.write('%s %s\n' % (sentence[i],prediction[i]))
            else: # prediction all zero
                f.write('%s O\n' % (sentence[i]))

        f.write('\n')

    print ('Writing to dev.out')
    f.close()


        
