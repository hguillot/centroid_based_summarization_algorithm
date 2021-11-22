def centroid_based(sentences):
  #criar el word embeddings
  with open('stop_words_portuguese.txt') as infile:
      stopWords = set([line.strip() for line in infile])

  #calcular el tf*idf
  vectorizer = TfidfVectorizer(stop_words=stopWords)
  X = vectorizer.fit_transform(sentences)

  #calcular matrix E
  analyzer = vectorizer.build_analyzer()
  model = Word2Vec([analyzer(sentence) for sentence in sentences], min_count=1)

  # Centroid Embedding
  C = 0
  for i in range(len(sentences)):
    for j in range(len(vectorizer.get_feature_names())):
      #print(vectorizer.get_feature_names()[j], X[i,j])
      if X[i,j] > 0.3:
        C += model.wv[vectorizer.get_feature_names()[j]]

  #Sentence Scoring
  ss = []
  for i in range(len(sentences)):
    temp = 0
    for j in range(len(vectorizer.get_feature_names())):
      #print(vectorizer.get_feature_names()[j], X[i,j])
      if X[i,j] > 0:
        temp += model.wv[vectorizer.get_feature_names()[j]]
    ss.append(temp)

  cosine_scores = cosine_similarity([C], ss)
  
  idxmax = max(range(len(sentences)), key=lambda x: cosine_scores[0][x])
  return sentences[idxmax]