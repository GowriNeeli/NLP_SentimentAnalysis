#######################################################################################
# Natural Language Processing - DATA902

# The project aims to understand and perform sentiment analysis on Farmers market tweets:
#1. Download Farmers Market tweets from Twitter feed text analysis (minimum 2000 tweets)
#2. TF word clouds
      #Unigram
      #Bigram
      #Trigram
#3. TF-IDF word cloud
#4. Sentiment analysis (any one lexicon)
#5. Comparison/Contrast word clouds based on sentiment
#6. Emotional analysis (any one lexicon)

#Author: Gowri N
#Date : 04/21/2018

#########################################################################################

library(twitteR)
library(tm)
library(qdap)
library(tibble)
library(ggplot2)
library(RWeka)
library(wordcloud)
library(lubridate)
library(lexicon)
library(tidytext)
library(lubridate)
library(gutenbergr)
library(stringr)
library(dplyr)
library(radarchart)

install.packages("radarchart", dependencies = TRUE)

#1.Twitter feed text analysis (minimum 2000 tweets)
consumer_key <- "********************************************"
consumer_secret <- "*****************************************"
access_token <- "*******************************************"
access_secret <- "******************************************"


setup_twitter_oauth(consumer_key, consumer_secret, access_token, access_secret)
tw = twitteR::searchTwitter("#farmersmarket", n = 2000)
d = twitteR::twListToDF(tw)

library(qdap)
term_count <- freq_terms(d$text)
plot(term_count)

library(tm)
#making a corpus of a vector source 
review_corpus <- VCorpus(VectorSource(d$text)) #total corpus
print(review_corpus)
d$text <- iconv(d$text, from = "UTF-8", to = "ASCII", sub = "")


library(tm)
#making a corpus of a vector source ; corpus is a bunch of docs
review_corpus <- VCorpus(VectorSource(d$text))
#Cleaning corpus - pre_processing
clean_corpus <- function(cleaned_corpus){
  removeURL <- content_transformer(function(x) gsub("(f|ht)tp(s?)://\\S+", "", x, perl=T))
  cleaned_corpus <- tm_map(cleaned_corpus, removeURL)
  cleaned_corpus <- tm_map(cleaned_corpus, content_transformer(replace_abbreviation))
  cleaned_corpus <- tm_map(cleaned_corpus, content_transformer(tolower))
  cleaned_corpus <- tm_map(cleaned_corpus, removePunctuation)
  cleaned_corpus <- tm_map(cleaned_corpus, removeNumbers)
  cleaned_corpus <- tm_map(cleaned_corpus, removeWords, stopwords("english"))
  # available stopwords
  # stopwords::stopwords()
  custom_stop_words <- c("farmersmarket") #always start with longest #pattern than smallest ; custom stop words
  cleaned_corpus <- tm_map(cleaned_corpus, removeWords, custom_stop_words)
  cleaned_corpus <- tm_map(cleaned_corpus, stemDocument,language = "english")
  cleaned_corpus <- tm_map(cleaned_corpus, stripWhitespace)
  return(cleaned_corpus)
}


cleaned_review_corpus <- clean_corpus((review_corpus))

TDM_reviews <- TermDocumentMatrix(cleaned_review_corpus)
TDM_reviews_m <- as.matrix(TDM_reviews)


# Term Frequency
term_frequency <- rowSums(TDM_reviews_m)
# Sort term_frequency in descending order
term_frequency <- sort(term_frequency,dec=TRUE)
# View the top 20 most common words
top20 <- term_frequency[1:20]
# Plot a barchart of the 20 most common words

barplot(top20,main="Term Frequency - Top 20 most common words ",col="darkblue",las=2)

#########################################################
#(Q2) Term Frequency Word Clouds
############ (a)Unigram Word Cloud

library(wordcloud)
# Create word_freqs
word_freqs <- data.frame(term = names(term_frequency), num = term_frequency)
# Create a wordcloud for the values in word_freqs
wordcloud(word_freqs$term, word_freqs$num,min.freq=5,random.order=FALSE,max.words=500,colors=brewer.pal(8, "Paired"))

#####Colors
# List the available colors
display.brewer.all()
#http://www.stat.columbia.edu/~tzheng/files/Rcolor.pdf



##############bigrams and trigrams

library(RWeka)
tokenizer <- function(x)
  NGramTokenizer(x,Weka_control(min=2,max=2))

bigram_tdm <- TermDocumentMatrix(cleaned_review_corpus,control = list(tokenize=tokenizer))
bigram_tdm_m <- as.matrix(bigram_tdm)

# Term Frequency
term_frequency <- rowSums(bigram_tdm_m)
# Sort term_frequency in descending order
term_frequency <- sort(term_frequency,dec=TRUE)
############Word Cloud
library(wordcloud)
# Create word_freqs
word_freqs <- data.frame(term = names(term_frequency), num = term_frequency)
# Create a wordcloud for the values in word_freqs
wordcloud(word_freqs$term, word_freqs$num,min.freq=5,random.order=FALSE,max.words=1000,colors=brewer.pal(8, "Paired"))

#trigrams
tokenizer <- function(x)
  NGramTokenizer(x,Weka_control(min=3,max=3))

bigram_tdm <- TermDocumentMatrix(cleaned_review_corpus,control = list(tokenize=tokenizer))
bigram_tdm_m <- as.matrix(bigram_tdm)

# Term Frequency
term_frequency <- rowSums(bigram_tdm_m)
# Sort term_frequency in descending order
term_frequency <- sort(term_frequency,dec=TRUE)
############Word Cloud
library(wordcloud)
# Create word_freqs
word_freqs <- data.frame(term = names(term_frequency), num = term_frequency)
# Create a wordcloud for the values in word_freqs
wordcloud(word_freqs$term, word_freqs$num,min.freq=5,max.words=1000,random.order=FALSE,colors=brewer.pal(8, "Paired"))


#(Q3). TF-IDF word cloud
##########tf-idf weighting

tfidf_tdm <- TermDocumentMatrix(cleaned_review_corpus,control=list(weighting=weightTfIdf))
tfidf_tdm_m <- as.matrix(tfidf_tdm)

# Term Frequency
term_frequency <- rowSums(tfidf_tdm_m)
# Sort term_frequency in descending order
term_frequency <- sort(term_frequency,dec=TRUE)
############Word Cloud
library(wordcloud)
# Create word_freqs
word_freqs <- data.frame(term = names(term_frequency), num = term_frequency)
# Create a wordcloud for the values in word_freqs
wordcloud(word_freqs$term, word_freqs$num,min.freq=5,max.words=1000,colors=brewer.pal(8, "Paired"))


#(Q4).Sentiment analysis (any one lexicon)
library(tidyverse)
tidy_mytext <- tidy(TermDocumentMatrix(cleaned_review_corpus)) #rows are the documents
bing_lex <- get_sentiments("bing")

mytext_bing <- inner_join(tidy_mytext, bing_lex, by = c("term" = "word")) 
mytext_bing$sentiment_n <- ifelse(mytext_bing$sentiment=="negative", -1, 1) 
mytext_bing$sentiment_score <- mytext_bing$count*mytext_bing$sentiment_n
aggdata <- aggregate(mytext_bing$sentiment_score, list(index = mytext_bing$document), sum)
sapply(aggdata,typeof)
aggdata$index <- as.numeric(aggdata$index)
colnames(aggdata) <- c("index","bing_score")

ggplot(aggdata, aes(index, bing_score)) + geom_point()
ggplot(aggdata, aes(index, bing_score)) + geom_smooth()

barplot(aggdata$bing_score,col="red")
#(Q5). Comparison/Contrast word clouds based on sentiment

library(reshape2)

tidy_mytext %>%
  inner_join(bing_lex, by = c("term" = "word")) %>%
  count(term, sentiment, sort=TRUE) %>%
  acast(term ~ sentiment, value.var = "n", fill = 0) %>%
  comparison.cloud(colors = c("#FF0099", "#6600CC"),max.word=1000)
                    

#(Q6). Emotional analysis (any one lexicon)
# NRC lexicon plots and build sentiment by document

nrc_lex <- get_sentiments("nrc") 
table(nrc_lex$sentiment)

mytext_nrc <- inner_join(tidy_mytext, nrc_lex, by = c("term" = "word"))

mytext_nrc$sentiment_n <- ifelse(mytext_nrc$sentiment=="negative", -1, 1)
mytext_nrc_noposneg<-mytext_nrc[!(mytext_nrc$sentiment %in% c("positive","negative")),]
aggdata_nrc <- aggregate(mytext_nrc_noposneg$count, list(index = mytext_nrc_noposneg$sentiment), sum) #aggregating

chartJSRadar(aggdata_nrc)
