from flask import Flask, request, jsonify, render_template
import torch
from transformers import AutoTokenizer, AutoModel
import sqlite3
import os
import numpy as np
import json
from dotenv import load_dotenv
import math

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure from environment variables
DEBUG_MODE = os.getenv('DEBUG_MODE', 'False').lower() == 'true'
DATABASE_PATH = os.getenv('DATABASE_PATH', 'aar_clinics.db')
KNOWLEDGE_PATH = os.getenv('KNOWLEDGE_PATH', 'medical_knowledge')

# Initialize empty lists for knowledge base
knowledge_texts = []
knowledge_embeddings = []

# Load BioBERT model with error handling
try:
    # BioBERT base model
    model_name = "dmis-lab/biobert-base-cased-v1.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    biobert_model = AutoModel.from_pretrained(model_name)
    print("BioBERT model loaded successfully!")
except Exception as e:
    print(f"BioBERT model loading failed: {e}")
    exit(1)

# Move models to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
biobert_model = biobert_model.to(device)
biobert_model.eval()

# Medical responses dictionary )
medical_responses = {
    "hello": "Hey there I am AAR MED BOT how may I help you today",

      "nausea": "Nausea is the uncomfortable feeling of needing to vomit, often accompanied by dizziness, sweating, or stomach discomfort. It can be caused by various factors, including food poisoning, motion sickness, pregnancy, migraines, infections, or medications. Treatment depends on the cause but may include rest, hydration, or anti-nausea medications. Would you like to visit one of our clinics for further assesment?",

  "headache": "A headache is pain or discomfort in the head or face, ranging from mild to severe. Common types include tension headaches (stress-related), migraines (often with nausea and light sensitivity), and cluster headaches (intense pain on one side). Causes may include dehydration, stress, lack of sleep, or underlying conditions. Pain relievers and rest often help.Would you like to visit one of our clinics for further assesment?",

  "fatigue": "Fatigue is extreme tiredness or lack of energy that doesn’t improve with rest. It can result from physical exertion, poor sleep, stress, anemia, thyroid disorders, chronic illnesses, or mental health conditions like depression. Managing fatigue involves addressing the underlying cause, improving sleep, and maintaining a balanced diet. Would you like to visit one of our clinics for further assesment?",

  "dizziness": "Dizziness is a sensation of lightheadedness, unsteadiness, or feeling faint. It can be caused by dehydration, low blood pressure, inner ear problems, anxiety, or neurological conditions. If dizziness is severe or persistent, medical evaluation is recommended to rule out serious causes. Would you like to visit one of our clinics for further assesment?",

    "fever": "Fever is a temporary increase in body temperature, often due to an infection or illness. A normal fever (up to 100.4°F or 38°C) helps fight infections, but high fevers (above 102°F or 39°C) may require medical attention. Symptoms include chills, sweating, and body aches. Rest, hydration, and fever-reducing medications can help. Would you like to visit one of our clinics for further assesment?",

  "cough": "A cough is a reflex to clear irritants or mucus from the airways. It can be dry (no mucus) or productive (with phlegm). Causes include colds, flu, allergies, asthma, or infections like bronchitis. Persistent coughs (lasting weeks) should be evaluated by a doctor. Would you like to visit one of our clinics for further assesment?",

    "constipation": "Constipation is difficulty passing stools or infrequent bowel movements, often due to low fiber intake, dehydration, or lack of exercise. Increasing fiber, water, and physical activity can help. Chronic constipation may require medical evaluation. Would you like to visit one of our clinics for further assesment?",

  "chest pain": "Chest pain can range from mild discomfort to severe pressure and may stem from heart, lung, digestive, or muscle issues. While some causes (like heartburn) are minor, sudden or severe chest pain (especially with shortness of breath) requires immediate medical attention to rule out heart attack or other serious conditions. Would you like to visit one of our clinics for further assesment?",


  "back pain": "Back pain is discomfort in the upper, middle, or lower back, often due to muscle strain, poor posture, herniated discs, or arthritis. Most cases improve with rest, gentle stretching, and pain relievers. Chronic or severe pain with numbness/weakness may need medical evaluation. Would you like to visit one of our clinics for further assesment?",

  "diarrhea": "Diarrhea is frequent, loose, or watery bowel movements, often caused by infections (viral, bacterial), food intolerance, or digestive disorders. It can lead to dehydration, so drinking fluids with electrolytes is important. If severe or lasting more than 2 days, medical advice is recommended. Would you like to visit one of our clinics for further assesment?",


    "malaria": "Malaria is caused by a parasite transmitted through mosquito bites. Common symptoms include fever, chills, sweating, headache, nausea, and body aches. In severe cases, it may cause yellow skin (jaundice), seizures, or coma. Immediate medical attention is crucial. We recommend visiting an AAR clinic for a blood test and treatment. Would you like help finding the nearest AAR clinic?",
    
    "covid": "COVID-19 is a viral illness with symptoms like fever, dry cough, shortness of breath, fatigue, body aches, loss of taste or smell, sore throat, and headache. In severe cases, it can lead to difficulty breathing or chest pain—seek emergency care if this occurs. AAR clinics offer testing and treatment. Would you like me to help you find the nearest AAR clinic?",
    
    "diabetes": "Diabetes affects blood sugar levels, causing symptoms like increased thirst, frequent urination, extreme hunger, unexplained weight loss, fatigue, irritability, and blurred vision. Long-term management is key to prevent complications. AAR clinics provide diabetes screening and care programs. Would you like me to locate an AAR clinic near you?",
    
    "headache": "Headaches can result from stress, dehydration, lack of sleep, eye strain, or underlying conditions. Most are manageable, but sudden, severe headaches or those with vision changes or confusion may need urgent care. AAR specialists can diagnose and treat persistent headaches. Would you like help finding the nearest AAR clinic?",
    
    "fever": "A fever is a temperature above 38°C (100.4°F), often signaling an infection or illness. Stay hydrated and rest, but seek care if it exceeds 40°C (104°F) or lasts over 48 hours. AAR clinics offer prompt fever evaluation and treatment. Would you like me to find an AAR clinic near you?",
    
    "flu": "The flu (influenza) causes fever, cough, sore throat, runny nose, muscle aches, and fatigue. It spreads easily and can worsen in vulnerable groups like the elderly or children. AAR clinics provide flu testing, treatment, and vaccinations. Would you like me to help you locate the nearest AAR clinic?",
    
    "stomach pain": "Stomach pain can stem from indigestion, gas, infections, or serious issues like appendicitis. Mild pain may improve with rest and hydration, but severe or persistent pain needs attention. AAR clinics can diagnose and treat the cause. Would you like help finding an AAR clinic?",
    
    "allergy": "Allergies may cause sneezing, itchy eyes, rash, or breathing issues, triggered by pollen, food, or other substances. Severe reactions (anaphylaxis) require immediate care. AAR clinics offer allergy testing and management. Would you like me to find the nearest AAR clinic?",
    
    "high blood pressure": "High blood pressure (hypertension) often has no symptoms but can lead to headaches, dizziness, or nosebleeds in severe cases. It's a risk factor for heart disease and stroke. AAR clinics provide screenings and treatment plans. Would you like help locating an AAR clinic?",
    
    "broken bone": "Signs of a broken bone include intense pain, swelling, bruising, or inability to move the area. Avoid moving the injury and seek care immediately. AAR clinics offer X-rays and fracture treatment. Would you like me to find the nearest AAR clinic?",
    
    "chest pain": "Chest pain can range from indigestion to serious conditions like a heart attack. If it's sudden, severe, or paired with shortness of breath, call emergency services now. For non-emergencies, AAR clinics can evaluate and treat. Would you like me to locate an AAR clinic?",
    
    "clinic": "AAR operates multiple clinics nationwide for your healthcare needs. Click the location pin button to find the nearest one, or let me assist you further!",
    
    "location": "To find an AAR clinic near you, click the location pin button at the bottom right of this chat window. I can also help you find one if you tell me your area!",
    
    "appointment": "AAR clinics offer walk-in visits and scheduled appointments for your convenience. To book an appointment, please click the button at the furthest right corner of this chat window. This will help you schedule your visit quickly and easily.",

    "booking": "To book an appointment at an AAR clinic, please click the button at the furthest right corner of this chat window. This will help you schedule your visit quickly and easily.",

    "schedule": "To schedule a visit at an AAR clinic, please click the button at the furthest right corner of this chat window. Our online booking system will help you find a convenient time for your appointment.",

    "emergency": "For life-threatening emergencies (e.g., severe chest pain, heavy bleeding, or difficulty breathing), call emergency services immediately. For urgent but non-critical issues, AAR clinics provide fast care. Would you like me to find the nearest AAR clinic?",
    
    "yeast infection": "Yeast infections are common fungal infections that cause itching, burning, and abnormal discharge, typically in the genital area. They can result from antibiotics, hormonal changes, or weakened immunity. AAR clinics provide confidential diagnosis and treatment options. Would you like help finding a nearby AAR clinic?",
    
    "rash": "Rashes can be caused by allergies, infections, heat, or skin conditions. Symptoms include redness, itching, bumps, or blisters. If a rash is severe, spreading rapidly, or accompanied by fever, seek medical care. AAR clinics can diagnose and treat various skin conditions. Would you like help finding an AAR clinic?",
    
    "urinary tract infection": "UTIs cause painful urination, frequency, urgency, and sometimes blood in urine. Fever or back pain may indicate a kidney infection requiring immediate care. AAR clinics provide testing and antibiotics for UTIs. Would you like me to help you find the nearest AAR clinic?",
    
    "cough": "A cough may be caused by respiratory infections, allergies, asthma, or other conditions. If your cough is persistent (lasting over 3 weeks), produces thick mucus, or comes with breathing difficulty, seek medical attention. AAR clinics can evaluate and treat persistent coughs. Would you like help finding an AAR clinic?",
    
    "diarrhea": "Diarrhea can result from infections, food poisoning, medications, or digestive disorders. Stay hydrated and seek care if it persists over 2 days, contains blood, or causes severe dehydration. AAR clinics can diagnose the cause and provide treatment. Would you like me to locate an AAR clinic near you?",
    
    "pregnancy": "If you think you might be pregnant or have confirmed a pregnancy, regular prenatal care is essential. AAR clinics offer pregnancy testing, prenatal checkups, and guidance throughout your pregnancy journey. Would you like help finding an AAR clinic for maternal care?",
    
    "pregnant": "If you think you might be pregnant or have confirmed a pregnancy, regular prenatal care is essential. AAR clinics offer pregnancy testing, prenatal checkups, and guidance throughout your pregnancy journey. Would you like help finding an AAR clinic for maternal care?",

    "vaccination": "Vaccinations are crucial for preventing serious diseases. AAR clinics provide various vaccines for all age groups, including routine childhood immunizations, flu shots, travel vaccines, and COVID-19 vaccines. Would you like me to help you find an AAR clinic for vaccination services?",

    "thank you": "I am glad I was of help. Let me know when you need anything else",

    "thanks": "I am glad I was of help. Let me know when you need anything else",

    
  "pregnancy": "If you think you might be pregnant or have confirmed a pregnancy, regular prenatal care is essential. AAR clinics offer pregnancy testing, prenatal checkups, and guidance throughout your pregnancy journey. Would you like help finding an AAR clinic for maternal care?",
  
  "yeast infection": "A yeast infection is a fungal infection caused by an overgrowth of yeast. It most commonly affects the vagina in women, causing itching, burning, redness, and a thick white discharge. Yeast infections can also affect other areas of the body, including the mouth (thrush), skin folds, and nail beds. They are usually treated with antifungal medications.",
  
  "asthma": "Asthma is a chronic condition affecting the airways in the lungs. During an asthma attack, the airways become inflamed and narrow, making it difficult to breathe. Symptoms include wheezing, coughing, chest tightness, and shortness of breath. Asthma can be managed with proper medication and by avoiding triggers.",
  
  "hypertension": "Hypertension, or high blood pressure, is a condition where the force of blood against the artery walls is consistently too high. It often has no symptoms but can lead to serious health problems like heart disease and stroke if left untreated. Regular monitoring and lifestyle changes are essential for management.",
  
  "sinusitis": "Sinusitis is an inflammation of the sinuses, often caused by viral, bacterial, or fungal infections. Symptoms include facial pain, pressure, nasal congestion, thick nasal discharge, and reduced sense of smell. Treatment depends on the cause but may include antibiotics for bacterial infections.",
  
  "arthritis": "Arthritis refers to inflammation of one or more joints, causing pain, stiffness, and reduced mobility. Osteoarthritis results from wear and tear of joint cartilage, while rheumatoid arthritis is an autoimmune disorder. Treatment focuses on pain relief, maintaining function, and preventing further joint damage.",
  
  "eczema": "Eczema, also known as atopic dermatitis, is a chronic skin condition characterized by itchy, inflamed skin. It often appears as dry, thickened, scaly patches on the face, hands, elbows, and knees. Triggers may include allergens, irritants, stress, and climate factors. Treatment includes moisturizers and topical medications.",
  
  "migraine": "Migraine is a neurological condition characterized by severe, recurring headaches often accompanied by nausea, vomiting, and sensitivity to light and sound. Some people experience an aura before the headache. Triggers may include stress, certain foods, hormonal changes, and environmental factors.",
  
  "GERD": "Gastroesophageal reflux disease (GERD) occurs when stomach acid frequently flows back into the esophagus. Symptoms include heartburn, chest pain, difficulty swallowing, and regurgitation of food or sour liquid. Lifestyle changes and medications can help manage GERD.",
  
  "anemia": "Anemia is a condition where you don't have enough healthy red blood cells to carry adequate oxygen to your tissues. Symptoms include fatigue, weakness, pale skin, shortness of breath, and dizziness. Causes include iron deficiency, vitamin deficiencies, chronic diseases, and genetic disorders.",
  
  "pneumonia": "Pneumonia is an infection that inflames the air sacs in one or both lungs, which may fill with fluid. Symptoms include cough with phlegm, fever, chills, and difficulty breathing. It can be caused by bacteria, viruses, or fungi and may range from mild to life-threatening.",
  
  "depression": "Depression is a mood disorder that causes a persistent feeling of sadness and loss of interest. It affects how you feel, think, and behave and can lead to various emotional and physical problems. Symptoms include persistent sadness, lack of energy, changes in appetite or sleep, and thoughts of death or suicide.",
  
  "influenza": "Influenza (flu) is a contagious respiratory illness caused by influenza viruses. Symptoms include fever, cough, sore throat, body aches, fatigue, and sometimes vomiting and diarrhea. Complications can be serious, especially in high-risk groups like young children and older adults.",
  
  "allergic rhinitis": "Allergic rhinitis, or hay fever, is an allergic response causing cold-like symptoms. Triggers include pollen, dust mites, pet dander, and mold. Symptoms include runny nose, sneezing, nasal congestion, and itchy eyes. Treatment options include antihistamines, nasal corticosteroids, and allergen avoidance.",
  
  "UTI": "Urinary tract infections (UTIs) are infections affecting any part of the urinary system. They most commonly occur in the bladder and urethra. Symptoms include a strong urge to urinate, burning sensation during urination, cloudy urine, and pelvic pain. Most UTIs are treated with antibiotics.",
  
  "conjunctivitis": "Conjunctivitis, or pink eye, is inflammation of the conjunctiva, the clear tissue covering the white of the eye. It can be caused by viruses, bacteria, allergies, or irritants. Symptoms include redness, itching, grittiness, and discharge. Treatment depends on the cause but may include eye drops or ointments.",
  
  "chickenpox": "Chickenpox is a highly contagious viral infection caused by the varicella-zoster virus. It's characterized by an itchy rash with blisters that usually starts on the face, chest, and back before spreading. Other symptoms include fever, fatigue, and headache. A vaccine is available for prevention.",
  
  "bronchitis": "Bronchitis is inflammation of the bronchial tubes that carry air to and from the lungs. Acute bronchitis is usually caused by viruses and resolves within a few weeks. Chronic bronchitis is a more serious condition often caused by smoking. Symptoms include cough, mucus production, fatigue, and shortness of breath.",
  
  "thyroid disorders": "Thyroid disorders affect the thyroid gland, which produces hormones that regulate metabolism. Hyperthyroidism occurs when the thyroid produces too much hormone, causing symptoms like weight loss, rapid heartbeat, and anxiety. Hypothyroidism occurs when it produces too little, causing symptoms like weight gain, fatigue, and cold intolerance.",
  
  "osteoporosis": "Osteoporosis is a condition where bones become weak and brittle, increasing the risk of fractures. It often develops without symptoms until a fracture occurs. Risk factors include aging, being female, low body weight, low calcium intake, and certain medications. Treatment includes medication, calcium, vitamin D, and weight-bearing exercise.",
  
  "psoriasis": "Psoriasis is a chronic skin condition that speeds up the life cycle of skin cells, causing them to build up rapidly on the surface of the skin. The extra skin cells form scales and red patches that are often itchy and sometimes painful. It's thought to be an immune system problem and can be triggered by infections, stress, and cold weather.",

    
    "blood test": "Blood tests help diagnose conditions, monitor health, and check organ function. AAR clinics offer comprehensive blood testing services with quick results. Our laboratories maintain high quality standards. Would you like help finding an AAR clinic for blood work?"
}

# Expanded medical knowledge base
expanded_knowledge = [
    "A yeast infection is a fungal infection caused by an overgrowth of yeast. It most commonly affects the vagina in women, causing itching, burning, redness, and a thick white discharge. Yeast infections can also affect other areas of the body, including the mouth (thrush), skin folds, and nail beds. They are usually treated with antifungal medications.",
    
    "Asthma is a chronic condition affecting the airways in the lungs. During an asthma attack, the airways become inflamed and narrow, making it difficult to breathe. Symptoms include wheezing, coughing, chest tightness, and shortness of breath. Asthma can be managed with proper medication and by avoiding triggers.",
    
    "Hypertension, or high blood pressure, is a condition where the force of blood against the artery walls is consistently too high. It often has no symptoms but can lead to serious health problems like heart disease and stroke if left untreated. Regular monitoring and lifestyle changes are essential for management.",
    
    "Sinusitis is an inflammation of the sinuses, often caused by viral, bacterial, or fungal infections. Symptoms include facial pain, pressure, nasal congestion, thick nasal discharge, and reduced sense of smell. Treatment depends on the cause but may include antibiotics for bacterial infections.",
    
    "Arthritis refers to inflammation of one or more joints, causing pain, stiffness, and reduced mobility. Osteoarthritis results from wear and tear of joint cartilage, while rheumatoid arthritis is an autoimmune disorder. Treatment focuses on pain relief, maintaining function, and preventing further joint damage.",
    
    "Eczema, also known as atopic dermatitis, is a chronic skin condition characterized by itchy, inflamed skin. It often appears as dry, thickened, scaly patches on the face, hands, elbows, and knees. Triggers may include allergens, irritants, stress, and climate factors. Treatment includes moisturizers and topical medications.",
    
    "Migraine is a neurological condition characterized by severe, recurring headaches often accompanied by nausea, vomiting, and sensitivity to light and sound. Some people experience an aura before the headache. Triggers may include stress, certain foods, hormonal changes, and environmental factors.",
    
    "Gastroesophageal reflux disease (GERD) occurs when stomach acid frequently flows back into the esophagus. Symptoms include heartburn, chest pain, difficulty swallowing, and regurgitation of food or sour liquid. Lifestyle changes and medications can help manage GERD.",
    
    "Anemia is a condition where you don't have enough healthy red blood cells to carry adequate oxygen to your tissues. Symptoms include fatigue, weakness, pale skin, shortness of breath, and dizziness. Causes include iron deficiency, vitamin deficiencies, chronic diseases, and genetic disorders.",
    
    "Pneumonia is an infection that inflames the air sacs in one or both lungs, which may fill with fluid. Symptoms include cough with phlegm, fever, chills, and difficulty breathing. It can be caused by bacteria, viruses, or fungi and may range from mild to life-threatening.",
    
    "Depression is a mood disorder that causes a persistent feeling of sadness and loss of interest. It affects how you feel, think, and behave and can lead to various emotional and physical problems. Symptoms include persistent sadness, lack of energy, changes in appetite or sleep, and thoughts of death or suicide.",
    
    "Influenza (flu) is a contagious respiratory illness caused by influenza viruses. Symptoms include fever, cough, sore throat, body aches, fatigue, and sometimes vomiting and diarrhea. Complications can be serious, especially in high-risk groups like young children and older adults.",
    
    "Allergic rhinitis, or hay fever, is an allergic response causing cold-like symptoms. Triggers include pollen, dust mites, pet dander, and mold. Symptoms include runny nose, sneezing, nasal congestion, and itchy eyes. Treatment options include antihistamines, nasal corticosteroids, and allergen avoidance.",
    
    "Urinary tract infections (UTIs) are infections affecting any part of the urinary system. They most commonly occur in the bladder and urethra. Symptoms include a strong urge to urinate, burning sensation during urination, cloudy urine, and pelvic pain. Most UTIs are treated with antibiotics.",
    
    "Conjunctivitis, or pink eye, is inflammation of the conjunctiva, the clear tissue covering the white of the eye. It can be caused by viruses, bacteria, allergies, or irritants. Symptoms include redness, itching, grittiness, and discharge. Treatment depends on the cause but may include eye drops or ointments.",
    
    "Chickenpox is a highly contagious viral infection caused by the varicella-zoster virus. It's characterized by an itchy rash with blisters that usually starts on the face, chest, and back before spreading. Other symptoms include fever, fatigue, and headache. A vaccine is available for prevention.",
    
    "Bronchitis is inflammation of the bronchial tubes that carry air to and from the lungs. Acute bronchitis is usually caused by viruses and resolves within a few weeks. Chronic bronchitis is a more serious condition often caused by smoking. Symptoms include cough, mucus production, fatigue, and shortness of breath.",
    
    "Thyroid disorders affect the thyroid gland, which produces hormones that regulate metabolism. Hyperthyroidism occurs when the thyroid produces too much hormone, causing symptoms like weight loss, rapid heartbeat, and anxiety. Hypothyroidism occurs when it produces too little, causing symptoms like weight gain, fatigue, and cold intolerance.",
    
    "Osteoporosis is a condition where bones become weak and brittle, increasing the risk of fractures. It often develops without symptoms until a fracture occurs. Risk factors include aging, being female, low body weight, low calcium intake, and certain medications. Treatment includes medication, calcium, vitamin D, and weight-bearing exercise.",
    
    "Psoriasis is a chronic skin condition that speeds up the life cycle of skin cells, causing them to build up rapidly on the surface of the skin. The extra skin cells form scales and red patches that are often itchy and sometimes painful. It's thought to be an immune system problem and can be triggered by infections, stress, and cold weather."
]

# Function to set up the knowledge base
def setup_knowledge_base():
    global knowledge_texts, knowledge_embeddings
    
    # Create directory if it doesn't exist
    if not os.path.exists(KNOWLEDGE_PATH):
        os.makedirs(KNOWLEDGE_PATH)
        
    # Check if we already have knowledge embeddings
    knowledge_file = os.path.join(KNOWLEDGE_PATH, "knowledge_texts.json")
    embeddings_file = os.path.join(KNOWLEDGE_PATH, "knowledge_embeddings.npy")
    
    if os.path.exists(knowledge_file) and os.path.exists(embeddings_file):
        # Load existing knowledge and embeddings
        with open(knowledge_file, 'r') as f:
            knowledge_texts = json.load(f)
        knowledge_embeddings = np.load(embeddings_file)
        print(f"Loaded {len(knowledge_texts)} knowledge chunks from disk")
        return
    
    # If not, create embeddings for our expanded knowledge base
    knowledge_texts = expanded_knowledge
    knowledge_embeddings = []
    
    print("Creating embeddings for knowledge base...")
    for text in knowledge_texts:
        embedding = get_embedding(text)
        knowledge_embeddings.append(embedding)
    
    # Convert to numpy array
    knowledge_embeddings = np.array(knowledge_embeddings)
    
    # Save knowledge and embeddings
    with open(knowledge_file, 'w') as f:
        json.dump(knowledge_texts, f)
    np.save(embeddings_file, knowledge_embeddings)
    
    print(f"Created knowledge base with {len(knowledge_texts)} entries")

# Database setup function
def setup_database():
    """Create database and tables if they don't exist"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Create clinics table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS clinics (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        address TEXT NOT NULL,
        lat REAL NOT NULL,
        lng REAL NOT NULL,
        phone TEXT NOT NULL
    )
    ''')
    
    # Insert sample data if table is empty
    cursor.execute("SELECT COUNT(*) FROM clinics")
    if cursor.fetchone()[0] == 0:
        sample_clinics = [
    (1, "AAR Healthcare Nairobi", "Kiambere Road, Upper Hill, Nairobi", -1.298357, 36.818359, "+254 709 071 000"),
    (2, "AAR Hospital Kisumu", "Oginga Odinga Street, Kisumu", -0.091702, 34.756057, "+254 709 071 100"),
    (3, "AAR Healthcare Mombasa", "Moi Avenue, Mombasa", -4.042605, 39.669025, "+254 709 071 200"),
    (4, "AAR Healthcare Nakuru", "Kenyatta Avenue, Nakuru", -0.283325, 36.065929, "+254 709 071 300"),
    (5, "AAR Healthcare Eldoret", "Uganda Road, Eldoret", 0.514277, 35.269779, "+254 709 071 400"),
    (6, "AAR Healthcare Thika", "General Kago Road, Thika", -1.039278, 37.083969, "+254 709 071 500"),
    (7, "AAR Healthcare Meru", "Makutano Road, Meru", 0.051246, 37.645920, "+254 709 071 600"),
    (8, "AAR Healthcare Nyeri", "Kimathi Way, Nyeri", -0.420131, 36.947586, "+254 709 071 700"),
    (9, "AAR Healthcare Kitale", "Kenyatta Street, Kitale", 1.015783, 35.006439, "+254 709 071 800"),
    (10, "AAR Healthcare Malindi", "Lamu Road, Malindi", -3.217864, 40.116928, "+254 709 071 900"),
    (11, "AAR Healthcare Machakos", "Mumbuni Road, Machakos", -1.517684, 37.263412, "+254 709 072 000"),
    (12, "AAR Healthcare Kakamega", "Mudiri Road, Kakamega", 0.282730, 34.751863, "+254 709 072 100"),
    (13, "AAR Healthcare Kericho", "Kiprotich Road, Kericho", -0.367308, 35.283138, "+254 709 072 200"),
    (14, "AAR Healthcare Embu", "Baricho Road, Embu", -0.537299, 37.457678, "+254 709 072 300"),
    (15, "AAR Healthcare Nanyuki", "Nanyuki Road, Nanyuki", 0.015783, 37.072529, "+254 709 072 400"),
    (16, "AAR Healthcare Garissa", "Kismayu Road, Garissa", -0.456944, 39.658333, "+254 709 072 500"),
    (17, "AAR Healthcare Lamu", "Harambee Avenue, Lamu", -2.269558, 40.900640, "+254 709 072 600"),
    (18, "AAR Healthcare Bungoma", "Kanduyi Road, Bungoma", 0.569525, 34.558376, "+254 709 072 700"),
    (19, "AAR Healthcare Kilifi", "Mombasa-Malindi Road, Kilifi", -3.633333, 39.850000, "+254 709 072 800"),
    (20, "AAR Healthcare Kisii", "Kisii-Migori Road, Kisii", -0.683333, 34.766667, "+254 709 072 900"),
    (21, "AAR Healthcare Voi", "Mombasa-Nairobi Highway, Voi", -3.396050, 38.556089, "+254 709 073 000"),
    (22, "AAR Healthcare Narok", "Narok-Maai Mahiu Road, Narok", -1.083333, 35.866667, "+254 709 073 100"),
    (23, "AAR Healthcare Homa Bay", "Kendu Bay Road, Homa Bay", -0.516667, 34.450000, "+254 709 073 200"),
    (24, "AAR Healthcare Nyahururu", "Nakuru-Nyeri Road, Nyahururu", 0.033333, 36.366667, "+254 709 073 300"),
    (25, "AAR Healthcare Kitui", "Kitui-Kibwezi Road, Kitui", -1.366667, 38.016667, "+254 709 073 400"),
    (26, "AAR Healthcare Busia", "Busia-Malaba Road, Busia", 0.466667, 34.116667, "+254 709 073 500")
        ]
        cursor.executemany(
            "INSERT INTO clinics (id, name, address, lat, lng, phone) VALUES (?, ?, ?, ?, ?, ?)",
            sample_clinics
        )
    
    conn.commit()
    conn.close()

def get_all_clinics():
    """Retrieve all clinics from database"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM clinics")
    clinics = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    return clinics

# Function to calculate distance between two points using Haversine formula
def calculate_distance(lat1, lng1, lat2, lng2):
    # Earth's radius in kilometers
    R = 6371.0
    
    # Convert latitude and longitude from degrees to radians
    lat1_rad = math.radians(lat1)
    lng1_rad = math.radians(lng1)
    lat2_rad = math.radians(lat2)
    lng2_rad = math.radians(lng2)
    
    # Differences in coordinates
    dlat = lat2_rad - lat1_rad
    dlng = lng2_rad - lng1_rad
    
    # Haversine formula
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlng/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c
    
    return distance

# Function to get BioBERT embeddings
def get_embedding(text):
    # Tokenize the text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get model output
    with torch.no_grad():
        outputs = biobert_model(**inputs)
    
    # Mean pooling
    token_embeddings = outputs.last_hidden_state
    attention_mask = inputs['attention_mask']
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    embedding = sum_embeddings / sum_mask
    
    # Convert to NumPy array
    return embedding[0].cpu().numpy()

# Function to retrieve relevant knowledge
def retrieve_knowledge(query, top_k=2):
    # Get query embedding
    query_embedding = get_embedding(query)
    
    # Calculate similarities with all knowledge chunks
    similarities = []
    for i, embedding in enumerate(knowledge_embeddings):
        # Calculate cosine similarity
        sim = np.dot(query_embedding, embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(embedding))
        similarities.append((i, sim))
    
    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Get top k similar chunks
    top_chunks = []
    for i, sim in similarities[:top_k]:
        if sim > 0.5:  # Only include if similarity is above threshold
            top_chunks.append((knowledge_texts[i], sim))
    
    return top_chunks

# Function to format retrieved knowledge into a response
def format_response(query, knowledge_chunks):
    if not knowledge_chunks:
        return None
    
    # Extract the most relevant chunk
    best_chunk, similarity = knowledge_chunks[0]
    
    # Create a response using the knowledge chunk
    response = best_chunk
    
    # Add AAR clinic promotion
    response += " AAR clinics offer diagnosis and treatment for this condition. Would you like help finding the nearest AAR clinic?"
    
    return response

# Function to get response from the chatbot
def get_chatbot_response(query):
    query = query.lower().strip()
    
    # First, check for direct matches in predefined responses
    for keyword, response in medical_responses.items():
        if keyword in query:
            print(f"Direct match found for keyword: {keyword}")
            return response
    
    # If no direct match, try to retrieve relevant knowledge
    print(f"No direct keyword match for: '{query}'. Using knowledge retrieval...")
    knowledge_chunks = retrieve_knowledge(query)
    
    # If we found relevant knowledge, format it into a response
    if knowledge_chunks and knowledge_chunks[0][1] > 0.6:  # Check if similarity is high enough
        print(f"Found knowledge with similarity: {knowledge_chunks[0][1]:.4f}")
        response = format_response(query, knowledge_chunks)
        if response:
            return response
    
    # As a fallback, use BioBERT for semantic similarity with keywords
    print("Using keyword semantic matching as fallback...")
    query_embedding = get_embedding(query)
    similarities = []
    
    for keyword, response in medical_responses.items():
        keyword_embedding = get_embedding(keyword)
        similarity = np.dot(query_embedding, keyword_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(keyword_embedding))
        similarities.append((keyword, similarity))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Print top 3 matches for debugging
    print("Top 3 semantic matches:")
    for keyword, score in similarities[:3]:
        print(f"  {keyword}: {score:.4f}")
    
    best_match_keyword, highest_similarity = similarities[0]
    
    # If similarity is high enough, use that response
    if highest_similarity > 0.6:
        print(f"Using response for '{best_match_keyword}' with similarity {highest_similarity:.4f}")
        return medical_responses[best_match_keyword]
    
    # Last resort fallback
    return "I understand you're asking about a medical condition. While I can provide information on many health topics, I don't have specific details about this condition. I recommend visiting an AAR clinic for personalized medical advice. Would you like me to help you find the nearest AAR clinic?"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat")
def chat_page():
    return render_template("chat.html")

@app.route("/chat", methods=["POST"])
def chat_api():
    try:
        user_input = request.json.get("message", "").strip()
        if not user_input:
            return jsonify({"response": "Please enter a question."})
        
        # Get response from the chatbot
        response = get_chatbot_response(user_input)
        
        return jsonify({"response": response})
    
    except Exception as e:
        print(f"Error in chat_api: {e}")
        return jsonify({"response": "Sorry, I encountered an error processing your request. Please try again."})

@app.route("/find-clinics", methods=["POST"])
def find_clinics():
    try:
        # Validate user location from request
        user_location = request.json.get("location", {})
        user_lat = user_location.get("lat")
        user_lng = user_location.get("lng")
        
        if not isinstance(user_lat, (int, float)) or not isinstance(user_lng, (int, float)):
            return jsonify({"error": "Invalid location data. Latitude and longitude must be numbers."}), 400
        
        # Get clinics from database
        all_clinics = get_all_clinics()
        
        # Calculate distance to each clinic
        clinics_with_distance = []
        for clinic in all_clinics:
            distance = calculate_distance(user_lat, user_lng, clinic["lat"], clinic["lng"])
            clinic_copy = clinic.copy()
            clinic_copy["distance"] = distance
            clinics_with_distance.append(clinic_copy)
        
        # Sort clinics by distance
        clinics_with_distance.sort(key=lambda x: x["distance"])
        
        # Return the 3 closest clinics
        closest_clinics = clinics_with_distance[:3]
        
        return jsonify({"clinics": closest_clinics})
    
    except Exception as e:
        print(f"Error finding clinics: {e}")
        return jsonify({"error": "An error occurred while finding clinics."}), 500

# Initialize before the first request
@app.before_request
def initialize():
    setup_database()
    setup_knowledge_base()

if __name__ == "__main__":
    # Use environment variables for configuration in production
    port = int(os.getenv('PORT', 5000))
    host = os.getenv('HOST', '0.0.0.0')
    
    # Setup the database and knowledge base
    setup_database()
    setup_knowledge_base()
    
    # Run the app
    app.run(host=host, port=port, debug=DEBUG_MODE)