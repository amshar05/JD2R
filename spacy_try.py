import spacy
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.summarization import summarize
#from transformers import pipeline
import docx2txt
import textract
from pdftotxt import pdf2txt
from manual_list_match import compare_skills_list
from rake_nltk import Rake
import regex as re
from io import BytesIO

#summarizer helper Hugging face

#summarizer = pipeline("summarization",model=r"C:\\Users\\sharmaap\\Documents\\JD2R\\models\\facebook\\bart-large-cnn"
    #tokenizer=r"C:\\Users\\sharmaap\\Documents\\JD2R\\models\\facebook\\bart-large-cnn")


# Initialize spaCy and RAKE
nlp = spacy.load("en_core_web_sm")
rake = Rake()


# Helper function to extract text from various file formats
def extract_text(file):
    filename = file.filename
    file_extension = filename.split(".")[-1]
    
    if file_extension == "pdf":
        return pdf2txt(file).replace('\n', ' ').replace('•', ' ').replace(',', ' ')
    elif file_extension in ["pptx", "ppt"]:
        return textract.process(file).decode('utf-8').replace('\n', ' ').replace(',', ' ')
    elif file_extension == "docx":
        return docx2txt.process(file).replace(',', ' ')
    else:
        raise ValueError("Unsupported file format")

# Function to clean and filter text using POS tags
def clean_and_filter_text(doc, to_check):
    tokens = [token.text for token in doc if token.pos_ in to_check and len(token.text) > 3]
    return " ".join(tokens)

# Function to extract named entities recognized as skills
def extract_entities(doc):
    skills = {ent.text for ent in doc.ents if ent.label_ in ["SKILL"]}
    return skills

#--------------RAKE-----------------
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

#-----------------Improved similarity calculator---------

from sentence_transformers import SentenceTransformer, util

def calculate_similarity_sbert(text1, text2):
    # Load pre-trained Sentence-BERT model
    model = SentenceTransformer(r"C:\\Users\\sharmaap\\Documents\\JD2R\\all-MiniLM-L6-v2")  # Lightweight and efficient model
    
    # Compute embeddings for both texts
    embeddings_text1 = model.encode(text1, convert_to_tensor=True)
    embeddings_text2 = model.encode(text2, convert_to_tensor=True)
    
    # Calculate cosine similarity
    similarity_score = util.pytorch_cos_sim(embeddings_text1, embeddings_text2).item()
    
    # Convert to percentage
    similarity_percentage = round(similarity_score * 100, 2)
    
    return similarity_percentage

#----------------------Summarizer-----------------------------------

   #------Summported functions--------------

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
#nltk.download("punkt")

def chunk_text_by_sentences(text, max_tokens, tokenizer):
    """Chunks text into smaller pieces, ensuring token count stays within limits."""
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        tokenized_sentence = tokenizer.encode(sentence, add_special_tokens=False)
        sentence_length = len(tokenized_sentence)

        if current_length + sentence_length <= max_tokens:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def summarize_large_text(text, tokenizer, model, max_tokens=1024, max_length=130, min_length=30):
    """Summarizes large text by chunking it and summarizing each chunk."""
    # Split the text into manageable chunks
    chunks = chunk_text_by_sentences(text, max_tokens, tokenizer)

    # Initialize the summarization pipeline
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, truncation=True)

    # Summarize each chunk
    summaries = []
    for chunk in chunks:
        try:
            summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
            summaries.append(summary)
        except Exception as e:
            print(f"Error summarizing chunk: {e}")
            summaries.append("")

    # Combine summaries into a single summary
    final_summary = " ".join(summaries)
    return final_summary


# Load BART model and tokenizer
model_path = r"C:\\Users\\sharmaap\\Documents\\JD2R\\models\\facebook\\bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)


#----------------------------------------------------

# Function to extract keywords using RAKE
def extract_keywords(text, top_n=3):
    # Read stopwords
    with open("C:\\Users\\sharmaap\\Documents\\JD2R\\stopwords.txt", 'r') as file:
        stopwords = file.read().splitlines()
    
    # Add domain-specific stopwords
    stopwords.extend(stopwords)
    
    # Split text by line breaks to treat each line as a separate entity
    lines = text.split('\n')
    
    all_keywords = []
    
    for line in lines:
        # Preprocess text
        line = preprocess_text(line)
        
        # Initialize RAKE with stopwords
        rake = Rake(stopwords)
        
        # Extract keywords from each line
        rake.extract_keywords_from_text(line)
        keywords = rake.get_ranked_phrases()
        
        # Further filtering: Ensure phrases have more than one word
        keywords = [kw for kw in keywords if len(kw.split()) > 1]
        
        # Add to all keywords list
        all_keywords.extend(keywords)
    
    # Get top_n keywords
    top_keywords = all_keywords[:top_n]
    
    #print(top_keywords)
    return set(top_keywords)

# Function to calculate cosine similarity between two texts
def calculate_similarity(text1, text2):
    cv = CountVectorizer(stop_words='english')
    count_matrix = cv.fit_transform([text1, text2])
    match_percentage = cosine_similarity(count_matrix)[0][1] * 100
    return round(match_percentage, 2), cv, count_matrix

#---------- Function to split skills containing the "•" symbol and clean the list-----------
def split_and_clean_skills(skill_list):
    cleaned_list = set()
    for skill in skill_list:
        if '•' in skill:
            parts = skill.split('•')
            cleaned_list.update([part.strip() for part in parts if len(part.strip()) > 1])
        else:
            cleaned_list.add(skill)
    return list(cleaned_list)

#-------------Function to remove single-word entries, duplicates, and specific symbols from a list-------
def clean_not_common_list(not_common_list):
    cleaned_list = split_and_clean_skills(not_common_list)
    cleaned_list = list(set([item.replace('•', '').strip() for item in cleaned_list if len(item.split()) > 1]))
    return cleaned_list

#---------Function to extract skills directly from text using regex patterns from a file----------
def extract_direct_skills(text):
    # Read patterns from a text file
    with open("Patterns.txt", 'r') as file:
        patterns = file.read().splitlines()

    # Use straightforward regex patterns without complex replacements
    compiled_patterns = []
    for pattern in patterns:
        try:
            compiled_patterns.append(re.compile(pattern, re.IGNORECASE))
        except re.error as e:
            print(f"Error in compiling pattern: {pattern}\nError message: {e}\n")

    skills = set()
    for pattern in compiled_patterns:
        matches = pattern.findall(text)
        for match in matches:
            cleaned_skill = match.strip()  # Clean the matched skill
            cleaned_skill = re.sub(r'\band\b$', '', cleaned_skill).strip()
            if len(cleaned_skill.split()) > 1:  # Ensure the skill is more than one word
                for skill in cleaned_skill.split('\n'):
                    skills.add(skill.strip())
            

    return skills

#----------New skill ext logic --------------------------
def extract_direct_skills2(text):
    # Load the spaCy model for named entity recognition and part-of-speech tagging
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    # Predefined patterns to identify skill-related phrases
    patterns = [
        r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b"
        r"\b[A-Z][a-z]+ (and|or) [A-Z][a-z]+\b",  # e.g., Banking and Financial Services
        r"\b[A-Z][a-z]+ [A-Z][a-z]+\b",  # e.g., Digital Transformation
        r"\b[A-Z][a-z]+\b"  # e.g., Payments
    ]

    compiled_patterns = [re.compile(pattern) for pattern in patterns]

    # Extract skills using regex patterns
    skills = set()
    for pattern in compiled_patterns:
        matches = pattern.findall(text)
        for match in matches:
            skills.add(match.strip())

    # Use spaCy to extract noun phrases and potential skill keywords
    for chunk in doc.noun_chunks:
        if len(chunk.text.split()) > 1:  # Ensure it's more than one word
            skills.add(chunk.text.strip())

    # Extract skills based on named entities

    return skills


# end--------------------



#--------------Main function to process resumes and job descriptions-----------
def spacy_match(resumes, jd, role):
    scores, not_common_lists, common_lists, xlabels, resume_values, jd_values = [], [], [], [], [], []
    to_check = {"NOUN", "ADJ", "VERB", "ADV"}  # Set of POS tags to check

    # Extract and summarize job description text
    jd_text = extract_text(jd).lower()
    jd_sum = summarize(jd_text, word_count=50).replace('•', '')
    jd_summary = re.sub(r'[^A-Za-z0-9\s]', '', jd_sum)
    #jd_summary = summarize_large_text(jd_text,tokenizer,model)
    jd_doc = nlp(jd_text)
    jd_clean = clean_and_filter_text(jd_doc, to_check)

    # Extract skills from the job description using NER, RAKE, and direct skill extraction
    jd_skills = extract_entities(jd_doc) #entity based
    #print("JD Skills through ent:", jd_skills)
    jd_keywords = extract_keywords(jd_text) #using RAKE
    #print("JD skills through RAKE", jd_keywords)
    jd_direct_skills = extract_direct_skills(jd_text) #using patterns
    #print("JD Skills through direct dkills:", jd_direct_skills)
    jd_skills.update(jd_keywords)
    jd_skills.update(jd_direct_skills)
    #print("JD Skills through direct dkills2:", extract_direct_skills2(jd_text))

    for resume in resumes:
        resume_text = extract_text(resume)
        resume_doc = nlp(resume_text)
        resume_clean = clean_and_filter_text(resume_doc, to_check)
        #print("resume clean:", resume_clean)

        # Extract skills from the resume using NER, RAKE, and direct skill extraction
        resume_skills = extract_entities(resume_doc)
        resume_keywords = extract_keywords(resume_text)
        resume_direct_skills = extract_direct_skills(resume_text)
        resume_skills.update(resume_keywords)
        resume_skills.update(resume_direct_skills)

        # Calculate common and not common lists
        common_skills = jd_skills & resume_skills
        #print("common skills:", common_skills)
        not_common_skills = jd_skills - common_skills

        # Adjust match percentage with manual list matching
        manual_not_present, manual_present, phrase_list = compare_skills_list(role, resume_text)
        adjusted_match_percentage = round((len(common_skills) * 100 / len(jd_skills)), 2)

        # Create final not_common_list
        final_not_common_skills = list(not_common_skills | set(manual_not_present))
        final_not_common_skills = clean_not_common_list(final_not_common_skills)  # Clean the list
        final_common_skills = list(common_skills | set(manual_present))
        
        # Vectorize and calculate top features
        match_percentage, cv, count_matrix = calculate_similarity(resume_text, jd_text)
        match_percentage_new = calculate_similarity_sbert(resume_text,jd_text)
        match_percentage_final= (match_percentage_new+match_percentage)/2
        x_train_counts = cv.fit_transform([resume_clean, jd_clean])
        df = pd.DataFrame(x_train_counts.toarray(), columns=cv.get_feature_names_out(), index=['resume', 'JD']).transpose()
        top_features = df.nlargest(10, ['JD'])

        if len(final_common_skills) + len(final_not_common_skills) > 0:
            final_percentage = (((len(final_common_skills) * 100) / (len(final_common_skills) + len(final_not_common_skills))) + match_percentage+match_percentage_new)/2
        else:
            final_percentage = match_percentage_final

        # Append results to lists
        scores.append(round(final_percentage,2))
        not_common_lists.append(final_not_common_skills)
        common_lists.append(final_common_skills)
        xlabels.append(top_features.index.tolist())
        resume_values.append(top_features['resume'].tolist())
        jd_values.append(top_features['JD'].tolist())

    return scores, not_common_lists, common_lists, xlabels, resume_values, jd_values, jd_summary

#if __name__ == "__main__":
