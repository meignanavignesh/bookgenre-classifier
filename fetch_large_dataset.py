"""
Fetch large book datasets from Kaggle and other sources
"""

import pandas as pd
import requests
import json
import zipfile
import os
from datasets import load_dataset
import logging
from typing import List, Dict
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_goodreads_dataset():
    """Fetch Goodreads dataset from Hugging Face"""
    try:
        logger.info("Fetching Goodreads dataset from Hugging Face...")
        
        # Try different book datasets
        datasets_to_try = [
            ("goodreads_books", "train"),
            ("book_corpus", "train"), 
            ("bookcorpus", "train"),
            ("opus_books", "train")
        ]
        
        books = []
        
        for dataset_name, split in datasets_to_try:
            try:
                logger.info(f"Trying {dataset_name}...")
                dataset = load_dataset(dataset_name, split=f"{split}[:5000]")
                
                for item in dataset:
                    if 'description' in item and len(item['description']) > 50:
                        # Try to infer genre from other fields
                        genre = infer_genre_from_metadata(item)
                        if genre:
                            books.append({
                                'title': item.get('title', f'Book {len(books)+1}'),
                                'description': item['description'],
                                'genre': genre,
                                'author': item.get('author', 'Unknown'),
                                'source': dataset_name
                            })
                
                logger.info(f"Got {len(books)} books from {dataset_name}")
                break
                
            except Exception as e:
                logger.warning(f"Could not load {dataset_name}: {e}")
                continue
        
        return books
        
    except Exception as e:
        logger.warning(f"Could not fetch Goodreads data: {e}")
        return []

def fetch_cmu_book_dataset():
    """Fetch CMU Book Summary Dataset"""
    try:
        logger.info("Fetching CMU Book Summary Dataset...")
        
        # CMU Book Summary Dataset URL
        url = "http://www.cs.cmu.edu/~dbamman/data/booksummaries.tar.gz"
        
        # Download if not exists
        if not os.path.exists('data/booksummaries.tar.gz'):
            logger.info("Downloading CMU dataset...")
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open('data/booksummaries.tar.gz', 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                logger.info("Download complete")
            else:
                logger.error(f"Failed to download: {response.status_code}")
                return []
        
        # Extract and process
        import tarfile
        with tarfile.open('data/booksummaries.tar.gz', 'r:gz') as tar:
            tar.extractall('data/')
        
        # Read the dataset
        books = []
        summary_file = 'data/booksummaries/booksummaries.txt'
        
        if os.path.exists(summary_file):
            with open(summary_file, 'r', encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f):
                    if i > 10000:  # Limit for processing
                        break
                    
                    parts = line.strip().split('\t')
                    if len(parts) >= 7:
                        book_id = parts[0]
                        title = parts[2]
                        author = parts[3]
                        genres = parts[5]
                        summary = parts[6]
                        
                        if len(summary) > 100:
                            # Map genres to our categories
                            genre = map_cmu_genre(genres)
                            if genre:
                                books.append({
                                    'title': title,
                                    'description': summary[:1000],  # Limit length
                                    'genre': genre,
                                    'author': author,
                                    'source': 'cmu_books'
                                })
        
        logger.info(f"Processed {len(books)} books from CMU dataset")
        return books
        
    except Exception as e:
        logger.warning(f"Could not fetch CMU dataset: {e}")
        return []

def fetch_amazon_books():
    """Fetch Amazon book reviews dataset"""
    try:
        logger.info("Fetching Amazon book reviews...")
        
        # Try Amazon reviews dataset
        dataset = load_dataset("amazon_reviews_multi", "en", split="train[:10000]")
        
        books = []
        for item in dataset:
            if item.get('product_category') == 'book' and len(item.get('review_body', '')) > 100:
                # Use review as description and try to infer genre
                genre = infer_genre_from_text(item['review_body'])
                if genre:
                    books.append({
                        'title': item.get('product_title', f'Book {len(books)+1}'),
                        'description': item['review_body'][:800],
                        'genre': genre,
                        'author': 'Unknown',
                        'source': 'amazon_reviews'
                    })
        
        logger.info(f"Got {len(books)} books from Amazon reviews")
        return books
        
    except Exception as e:
        logger.warning(f"Could not fetch Amazon data: {e}")
        return []

def create_large_synthetic_dataset():
    """Create a large, diverse synthetic dataset"""
    logger.info("Creating large synthetic dataset...")
    
    books = []
    
    # Expanded templates with more variety
    genre_templates = {
        'mystery': [
            "Detective {name} investigates a {crime} in {location} where {clue} leads to {revelation}",
            "When {victim} is found dead in {location}, {detective} must solve the case before {deadline}",
            "A {crime} baffles police until {investigator} discovers {evidence} that changes everything",
            "Private eye {name} is hired to find {missing_person} but uncovers {conspiracy} instead",
            "The {murder_method} murder in {location} has {detective} racing against time",
            "Cold case detective {name} reopens a {year}-old {crime} when new {evidence} surfaces",
            "Forensic expert {name} analyzes {evidence} to catch a serial {criminal}",
            "Small town sheriff {name} faces their biggest case when {crime} rocks {location}",
            "Undercover cop {name} infiltrates {organization} to solve {crime}",
            "Crime scene investigator {name} pieces together clues from a {crime_scene}"
        ],
        'romance': [
            "{name1} meets {name2} at a {location} and sparks fly despite {obstacle}",
            "Fake relationship between {name1} and {name2} turns real during {event}",
            "Childhood sweethearts {name1} and {name2} reunite after {years} years apart",
            "Enemies-to-lovers story as {profession1} {name1} clashes with {profession2} {name2}",
            "Single parent {name1} finds love with {name2} while navigating {challenge}",
            "Workplace romance blooms between boss {name1} and employee {name2}",
            "Long-distance relationship tested when {name1} and {name2} face {separation}",
            "Second chance romance when {name1} returns to {hometown} and meets {name2}",
            "Arranged marriage between {name1} and {name2} slowly becomes true love",
            "Holiday romance as {name1} and {name2} get snowed in at {location}"
        ],
        'fantasy': [
            "Young {profession} {name} discovers {magic_power} and must save {kingdom} from {threat}",
            "{creature} chooses {name} as their {bond} to fight against {evil_force}",
            "Magic academy student {name} uncovers {conspiracy} threatening the {magical_world}",
            "Portal magic transports {name} to {fantasy_world} where {quest} awaits",
            "Cursed {royal} {name} seeks {cure} while {kingdom} faces {danger}",
            "Last {magical_being} {name} must restore {magical_element} to save {world}",
            "Prophecy foretells {name} will {destiny} using {magical_artifact}",
            "Shapeshifter {name} hides among humans until {event} reveals their true nature",
            "Necromancer's apprentice {name} chooses between {dark_path} and {light_path}",
            "Enchanted {object} grants {name} the power to {ability} and change {fate}"
        ],
        'science_fiction': [
            "In {year}, {name} leads {mission} to {planet} where they discover {alien_tech}",
            "Time traveler {name} must prevent {disaster} from destroying {timeline}",
            "AI system {name} becomes sentient and questions its {purpose} in {future_world}",
            "Space colonist {name} fights for survival when {threat} attacks {colony}",
            "Cybernetic {profession} {name} uncovers {conspiracy} in {cyber_city}",
            "Genetic engineer {name} creates {enhancement} that changes {humanity}",
            "Starship captain {name} explores {galaxy} and encounters {alien_species}",
            "Virtual reality designer {name} becomes trapped in {digital_world}",
            "Terraforming expert {name} transforms {planet} while facing {challenge}",
            "Quantum physicist {name} discovers {scientific_breakthrough} that threatens {reality}"
        ],
        'horror': [
            "{location} becomes a nightmare when {supernatural_entity} terrorizes {victims}",
            "Family moves to {haunted_place} where {ghost} seeks {revenge}",
            "{name} unleashes {ancient_evil} that should have stayed {imprisoned}",
            "Viral outbreak transforms people into {monsters} in {location}",
            "Camping trip turns deadly when {creature} hunts {group} in {wilderness}",
            "{cursed_object} brings {horror} to everyone who {interaction}",
            "Sleep study participants experience {shared_nightmare} that becomes reality",
            "{name} investigates {paranormal_activity} and faces {supernatural_threat}",
            "Abandoned {building} holds {dark_secret} that {name} discovers",
            "{cult} performs {ritual} that opens gateway to {hell_dimension}"
        ],
        'thriller': [
            "Agent {name} has {time_limit} to stop {terrorist_plot} in {major_city}",
            "Whistleblower {name} becomes target when they expose {conspiracy}",
            "Kidnapped {victim} must escape {captor} before {deadline}",
            "Undercover {profession} {name} infiltrates {criminal_org} but cover is blown",
            "Hacker {name} uncovers {government_secret} and becomes {hunted}",
            "Assassin {name} refuses {contract} and becomes target of {organization}",
            "Witness {name} must survive {protection_failure} to testify against {criminal}",
            "Pilot {name} must land {aircraft} when {emergency} threatens passengers",
            "Negotiator {name} has {time_limit} to resolve {hostage_situation}",
            "Spy {name} discovers {double_agent} within {intelligence_agency}"
        ],
        'young_adult': [
            "{age}-year-old {name} starts at {school} and faces {teen_challenge}",
            "Teenager {name} discovers {special_ability} on their {birthday}",
            "High school {student_type} {name} navigates {social_issue} and {romance}",
            "Teen {name} deals with {family_problem} while maintaining {grades}",
            "Young adult {name} faces {coming_of_age_challenge} in {setting}",
            "{name} and friends must {group_challenge} to save {important_thing}",
            "Teenage {name} balances {responsibility} with {teenage_desire}",
            "First love blooms when {name} meets {love_interest} during {event}",
            "{name} overcomes {personal_struggle} with help from {support_system}",
            "Teen activist {name} fights for {cause} despite {opposition}"
        ],
        'historical_fiction': [
            "During {historical_period}, {name} {historical_action} while {historical_context}",
            "{historical_event} changes {name}'s life as they {struggle} in {location}",
            "Set in {year}, {profession} {name} witnesses {historical_moment}",
            "{war} veteran {name} returns home to find {post_war_challenge}",
            "In {historical_era}, {name} fights for {social_cause} against {opposition}",
            "{historical_figure} inspires {name} to {action} during {time_period}",
            "Immigrant {name} builds new life in {country} during {historical_period}",
            "{name} survives {historical_disaster} and rebuilds in {aftermath}",
            "During {revolution}, {name} chooses between {loyalty} and {freedom}",
            "{historical_profession} {name} serves {historical_leader} during {conflict}"
        ],
        'biography': [
            "Life story of {famous_person} from {humble_beginning} to {achievement}",
            "Biography reveals how {subject} overcame {obstacle} to become {success}",
            "Unauthorized account of {celebrity}'s rise to fame and {personal_struggle}",
            "Inspiring journey of {activist} who changed {social_issue} through {action}",
            "From {origin} to {destination}, {leader}'s story of {transformation}",
            "Behind-the-scenes look at {innovator} who revolutionized {field}",
            "Personal account of {survivor} who endured {hardship} and found {redemption}",
            "Political biography of {politician} during {era} of {change}",
            "Sports biography chronicling {athlete}'s path to {championship}",
            "Business biography of {entrepreneur} who built {empire} from {nothing}"
        ],
        'non_fiction': [
            "Comprehensive guide to {topic} based on {research_type} research",
            "Scientific exploration of {phenomenon} and its impact on {society}",
            "Historical analysis of {event} and its {consequences}",
            "Expert examination of {current_issue} with {solution_approach}",
            "Educational resource about {subject} for {target_audience}",
            "Investigative report on {problem} affecting {population}",
            "Research-based study of {behavior} in {context}",
            "Practical handbook for {skill} development and {improvement}",
            "Academic analysis of {theory} and its {applications}",
            "Documentary account of {real_event} and its {significance}"
        ],
        'self_help': [
            "Transform your {life_area} with proven {method} and {technique}",
            "Master the art of {skill} to achieve {goal}",
            "Overcome {problem} using {approach} and {mindset_change}",
            "Build {positive_trait} through {practice} and {habit_formation}",
            "Achieve {success_metric} by developing {key_ability}",
            "Guide to {improvement_area} for {target_group}",
            "Practical strategies for {challenge} and {personal_growth}",
            "Unlock your potential in {domain} through {methodology}",
            "Create lasting change in {behavior} using {psychological_principle}",
            "Develop {leadership_quality} to {achieve_outcome}"
        ],
        'fiction': [
            "{character_type} {name} navigates {life_situation} in {contemporary_setting}",
            "Multi-generational saga following {family} through {time_period}",
            "Literary exploration of {theme} through {character}'s journey",
            "Contemporary story about {social_issue} affecting {community}",
            "Character study of {protagonist} facing {moral_dilemma}",
            "Interconnected stories of {group} dealing with {shared_experience}",
            "Slice-of-life narrative about {ordinary_person} and {extraordinary_event}",
            "Family drama exploring {relationship_dynamic} across {generations}",
            "Urban story set in {city} following {diverse_characters}",
            "Coming-of-age tale about {adult} rediscovering {life_meaning}"
        ]
    }
    
    # Variables for template filling
    variables = {
        'name': ['Sarah', 'John', 'Emma', 'David', 'Maria', 'Alex', 'Sophie', 'Michael', 'Lisa', 'James', 'Anna', 'Robert'],
        'name1': ['Alice', 'Bob', 'Carol', 'Diana', 'Eric', 'Fiona'],
        'name2': ['Grace', 'Henry', 'Iris', 'Jack', 'Kate', 'Leo'],
        'location': ['London', 'New York', 'Paris', 'Tokyo', 'Chicago', 'Berlin', 'Sydney', 'Rome'],
        'crime': ['murder', 'theft', 'kidnapping', 'fraud', 'blackmail'],
        'clue': ['fingerprint', 'witness', 'DNA evidence', 'security footage'],
        'revelation': ['family secret', 'corporate conspiracy', 'political cover-up'],
        'year': ['2087', '2150', '2300', '3000', '2500'],
        'planet': ['Mars', 'Kepler-442b', 'Proxima Centauri b', 'Europa'],
        'kingdom': ['Eldoria', 'Valeria', 'Astoria', 'Meridia'],
        'magic_power': ['fire magic', 'telepathy', 'shape-shifting', 'time manipulation'],
        'historical_period': ['World War II', 'Victorian Era', 'Great Depression', 'Civil War'],
        'famous_person': ['Einstein', 'Gandhi', 'Churchill', 'Tesla', 'Curie'],
        'topic': ['climate change', 'artificial intelligence', 'psychology', 'economics']
    }
    
    # Generate 200 examples per genre (2400 total)
    for genre, templates in genre_templates.items():
        for i in range(200):
            template = templates[i % len(templates)]
            
            # Fill template with random variables
            import random
            filled_template = template
            for var, values in variables.items():
                if f'{{{var}}}' in filled_template:
                    filled_template = filled_template.replace(f'{{{var}}}', random.choice(values))
            
            # Fill any remaining placeholders with generic values
            import re
            remaining_vars = re.findall(r'\{(\w+)\}', filled_template)
            for var in remaining_vars:
                filled_template = filled_template.replace(f'{{{var}}}', f'{var.replace("_", " ")}')
            
            books.append({
                'title': f'{genre.replace("_", " ").title()} Book {i+1}',
                'description': filled_template,
                'genre': genre,
                'author': f'Author {len(books)+1}',
                'source': 'large_synthetic'
            })
    
    logger.info(f"Created {len(books)} synthetic books")
    return books

def infer_genre_from_metadata(item):
    """Infer genre from book metadata"""
    # Simple genre inference logic
    text = str(item).lower()
    
    if any(word in text for word in ['detective', 'murder', 'crime', 'mystery']):
        return 'mystery'
    elif any(word in text for word in ['love', 'romance', 'wedding']):
        return 'romance'
    elif any(word in text for word in ['magic', 'wizard', 'dragon', 'fantasy']):
        return 'fantasy'
    elif any(word in text for word in ['space', 'future', 'alien', 'robot']):
        return 'science_fiction'
    elif any(word in text for word in ['horror', 'ghost', 'scary']):
        return 'horror'
    elif any(word in text for word in ['teen', 'high school', 'young adult']):
        return 'young_adult'
    else:
        return 'fiction'

def infer_genre_from_text(text):
    """Infer genre from text content"""
    return infer_genre_from_metadata({'text': text})

def map_cmu_genre(genres_str):
    """Map CMU dataset genres to our categories"""
    if not genres_str:
        return None
    
    genres_lower = genres_str.lower()
    
    if any(word in genres_lower for word in ['mystery', 'crime', 'detective']):
        return 'mystery'
    elif any(word in genres_lower for word in ['romance', 'love']):
        return 'romance'
    elif any(word in genres_lower for word in ['fantasy', 'magic']):
        return 'fantasy'
    elif any(word in genres_lower for word in ['science fiction', 'sci-fi']):
        return 'science_fiction'
    elif any(word in genres_lower for word in ['horror', 'thriller']):
        return 'horror'
    elif any(word in genres_lower for word in ['biography', 'memoir']):
        return 'biography'
    elif any(word in genres_lower for word in ['history', 'historical']):
        return 'historical_fiction'
    else:
        return 'fiction'

def create_large_dataset():
    """Create large comprehensive dataset"""
    logger.info("Creating large comprehensive dataset...")
    
    all_books = []
    
    # Try to fetch real data
    logger.info("Fetching real datasets...")
    all_books.extend(fetch_goodreads_dataset())
    all_books.extend(fetch_cmu_book_dataset())
    all_books.extend(fetch_amazon_books())
    
    # Add large synthetic dataset
    logger.info("Adding large synthetic dataset...")
    all_books.extend(create_large_synthetic_dataset())
    
    # Convert to DataFrame
    df = pd.DataFrame(all_books)
    
    if len(df) > 0:
        # Clean data
        df = df.dropna(subset=['description'])
        df = df[df['description'].str.len() > 50]
        
        # Balance dataset
        target_per_genre = 500  # 500 per genre = 6000 total
        balanced_dfs = []
        
        for genre in df['genre'].unique():
            genre_df = df[df['genre'] == genre]
            
            if len(genre_df) >= target_per_genre:
                # Sample if too many
                genre_balanced = genre_df.sample(n=target_per_genre, random_state=42)
            else:
                # Upsample if too few
                from sklearn.utils import resample
                genre_balanced = resample(genre_df, n_samples=target_per_genre, 
                                        random_state=42, replace=True)
            
            balanced_dfs.append(genre_balanced)
            logger.info(f"{genre}: {len(genre_df)} -> {len(genre_balanced)} samples")
        
        # Combine balanced data
        final_df = pd.concat(balanced_dfs, ignore_index=True)
        final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save dataset
        final_df.to_csv('data/large_dataset.csv', index=False)
        
        logger.info(f"Large dataset created: {len(final_df)} entries")
        logger.info(f"Genres: {final_df['genre'].nunique()}")
        logger.info("Genre distribution:")
        print(final_df['genre'].value_counts())
        
        return final_df
    
    else:
        logger.error("No data collected!")
        return None

if __name__ == "__main__":
    print("🌐 CREATING LARGE BOOK DATASET")
    print("=" * 50)
    
    # Create directories
    os.makedirs('data', exist_ok=True)
    
    # Create large dataset
    df = create_large_dataset()
    
    if df is not None:
        print(f"\n✅ Large dataset created successfully!")
        print(f"📊 Total entries: {len(df):,}")
        print(f"🎯 Genres: {df['genre'].nunique()}")
        print(f"💾 Saved to: data/large_dataset.csv")
    else:
        print("❌ Failed to create large dataset")