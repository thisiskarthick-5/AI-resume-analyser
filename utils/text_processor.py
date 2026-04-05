import re
import string
import json
import os

# Skill list for predefined matching (Requirement 5)
SKILLS_DB = {
    "Data Scientist": ["python", "machine learning", "pytorch", "tensorflow", "scikit-learn", "pandas", "sql", "statistics"],
    "Web Developer": ["html", "css", "javascript", "react", "node.js", "flask", "django", "sql", "bootstrap"],
    "HR": ["recruitment", "interviews", "communication", "management", "hris", "payroll", "onboarding"],
    "Java Developer": ["java", "spring boot", "hibernate", "sql", "maven", "jenkins", "microservices"]
}

def clean_text(text):
    """
    Cleans the input text based on requirements:
    1.  Convert to lowercase
    2.  Remove special characters
    """
    text = text.lower()
    # Remove newlines and non-alphanumeric (keep spaces)
    text = re.sub(r'\n', ' ', text)
    # Keeping alphanumeric and spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

def tokenize(text):
    """
    Tokenizes the text into a list of words.
    """
    return text.split()

def text_to_sequence(tokens, word_to_idx, max_len=200):
    """
    Converts tokens to numerical format (Requirement 2)
    and pads/truncates to max_len.
    """
    sequence = [word_to_idx.get(token, word_to_idx.get("<UNK>", 0)) for token in tokens]
    # Padding/Truncating
    if len(sequence) < max_len:
        sequence += [0] * (max_len - len(sequence))
    else:
        sequence = sequence[:max_len]
    return sequence

def extract_skills(text):
    """
    Extracts skills from text using predefined list (Requirement 5)
    """
    extracted = []
    # Flattening all skills in DB to find all occurrences
    all_skills = set([skill for sublist in SKILLS_DB.values() for skill in sublist])
    
    text_lower = text.lower()
    for skill in all_skills:
        # Using word boundary for precise matching
        if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
            extracted.append(skill.title())
    return extracted

def calculate_match_score(extracted_skills, role_skills):
    """
    Compares extracted skills with job requirements (Requirement 6)
    """
    if not role_skills:
        return 0
    matches = [skill for skill in extracted_skills if skill.lower() in [rs.lower() for rs in role_skills]]
    score = (len(matches) / len(role_skills)) * 100
    return round(score, 2)
