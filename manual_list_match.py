import spacy
import pandas as pd

# Load SpaCy's larger model with word vectors
nlp = spacy.load('en_core_web_sm')  # or 'en_core_web_lg'

def compare_skills_list(role, resume_text):
    # Load skills data from Excel
    manual_skill_data = pd.read_excel("skills.xlsx")
    phrase_list = []

    # Extract the list of skills for the given role
    if role not in manual_skill_data.columns:
        raise KeyError(f"Role '{role}' not found in the skills data.")
    
    for i in manual_skill_data[role]:
        if str(i) != 'nan':
            phrase_list.append(i.lower())

    # Create phrase patterns using SpaCy
    phrase_docs = [nlp(text) for text in phrase_list]

    # Convert resume text to SpaCy Doc
    resume_doc = nlp(resume_text)
    
    matched_list = []

    # Perform meaning-based comparison
    for phrase_doc in phrase_docs:
        similarity = phrase_doc.similarity(resume_doc)
        if similarity > 0.8:  # Adjust threshold as needed
            matched_list.append(phrase_doc.text)

    manual_present_list = []
    manual_not_present_list = []

    for skill in phrase_list:
        best_match = None
        best_similarity = 0
        for phrase in matched_list:
            similarity = nlp(skill).similarity(nlp(phrase))
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = phrase
        if best_similarity > 0.8:  # Adjust threshold as needed
            
            manual_present_list.append(skill)  # Append a tuple (skill, matched skill, similarity)
        else:
            manual_not_present_list.append(skill)

    return manual_not_present_list, manual_present_list, phrase_list



if __name__ == "__main__":
	resume_text = "/Users/amit/Desktop/mohit_project/MyNewFlaskApp/res.docx"
	resume_doc = nlp(resume_text)
	missing_skills, present_skills, all_skills = compare_skills_list("PM", resume_doc)
	print("Skills not present:", missing_skills)
	print("Skills present:", present_skills)
