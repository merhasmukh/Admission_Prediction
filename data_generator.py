import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_sample_data(n_samples=1000):
    """
    Generate sample university admission data with the required features
    for admission confirmation, joining, and weekly dropout prediction.
    """
    np.random.seed(42)
    random.seed(42)
    
    data = []
    
    for i in range(n_samples):
        # Basic student information
        application_id = f"APP{str(i+1).zfill(4)}"
        gender = random.choice(['Male', 'Female'])
        
        # Academic performance (correlated with outcomes)
        marks_12th = np.random.normal(75, 12)  # Mean 75%, std 12%
        marks_12th = max(40, min(100, marks_12th))  # Clamp between 40-100
        
        entrance_score = np.random.normal(60, 15)  # Mean 60, std 15
        entrance_score = max(0, min(100, entrance_score))  # Clamp between 0-100
        
        # Form submission date (random date in admission period)
        base_date = datetime(2024, 6, 1)
        days_offset = random.randint(0, 90)  # 3 months admission window
        form_date = base_date + timedelta(days=days_offset)
        
        # Additional features that influence outcomes
        category = random.choice(['General', 'OBC', 'SC', 'ST'])
        admission_round = random.choice([1, 2, 3])  # Which counseling round
        distance_from_home = random.randint(10, 500)  # km
        family_income = random.choice(['Low', 'Middle', 'High'])
        first_preference = random.choice([0, 1])  # Is this their first choice?
        
        # Calculate probabilities based on features
        # Higher marks and entrance scores increase confirmation probability
        confirm_prob = 0.3 + (marks_12th/100) * 0.4 + (entrance_score/100) * 0.3
        confirm_prob += 0.1 if first_preference else 0
        confirm_prob -= 0.1 if admission_round > 1 else 0
        confirm_prob = max(0.1, min(0.95, confirm_prob))
        
        is_confirmed = 1 if random.random() < confirm_prob else 0
        
        # Only confirmed students can join
        if is_confirmed:
            join_prob = 0.4 + (marks_12th/100) * 0.3 + (entrance_score/100) * 0.2
            join_prob += 0.15 if first_preference else 0
            join_prob -= 0.05 if distance_from_home > 200 else 0
            join_prob -= 0.1 if admission_round > 1 else 0
            join_prob = max(0.2, min(0.9, join_prob))
            
            is_joined = 1 if random.random() < join_prob else 0
        else:
            is_joined = 0
        
        # Weekly dropout tracking (only for joined students)
        week1_dropout = week2_dropout = week3_dropout = week4_dropout = 0
        final_dropout = 0
        
        if is_joined:
            # Base dropout probability (lower for better students)
            base_dropout_prob = 0.15 - (marks_12th/100) * 0.1 - (entrance_score/100) * 0.05
            base_dropout_prob += 0.05 if distance_from_home > 300 else 0
            base_dropout_prob += 0.03 if family_income == 'Low' else 0
            base_dropout_prob = max(0.02, min(0.3, base_dropout_prob))
            
            # Week-wise dropout (decreasing probability over time)
            for week in range(1, 5):
                week_prob = base_dropout_prob * (1.5 - week * 0.2)  # Higher in early weeks
                
                if week == 1 and random.random() < week_prob:
                    week1_dropout = 1
                    final_dropout = 1
                    break
                elif week == 2 and random.random() < week_prob:
                    week2_dropout = 1
                    final_dropout = 1
                    break
                elif week == 3 and random.random() < week_prob:
                    week3_dropout = 1
                    final_dropout = 1
                    break
                elif week == 4 and random.random() < week_prob:
                    week4_dropout = 1
                    final_dropout = 1
                    break
        
        # Create record
        record = {
            'Application_ID': application_id,
            'Gender': gender,
            'Marks_12th_Percent': round(marks_12th, 2),
            'Entrance_Exam_Score': round(entrance_score, 2),
            'Form_Submission_Date': form_date.strftime('%Y-%m-%d'),
            'Category': category,
            'Admission_Round': admission_round,
            'Distance_From_Home_KM': distance_from_home,
            'Family_Income': family_income,
            'First_Preference': first_preference,
            'Admission_Confirmed': is_confirmed,
            'Joined': is_joined,
            'Week1_Dropout': week1_dropout,
            'Week2_Dropout': week2_dropout,
            'Week3_Dropout': week3_dropout,
            'Week4_Dropout': week4_dropout,
            'Final_Dropout_Label': final_dropout
        }
        
        data.append(record)
    
    return pd.DataFrame(data)

def save_sample_data():
    """Generate and save sample data to CSV file"""
    print("Generating sample university admission data...")
    
    # Generate data
    df = generate_sample_data(1000)
    
    # Save to CSV
    output_path = 'data/sample_data.csv'
    df.to_csv(output_path, index=False)
    
    print(f"Sample data saved to {output_path}")
    print(f"Dataset shape: {df.shape}")
    print("\nDataset summary:")
    print(f"- Total applications: {len(df)}")
    print(f"- Confirmed admissions: {df['Admission_Confirmed'].sum()}")
    print(f"- Students joined: {df['Joined'].sum()}")
    print(f"- Students dropped out: {df['Final_Dropout_Label'].sum()}")
    
    # Weekly dropout breakdown
    joined_students = df[df['Joined'] == 1]
    if len(joined_students) > 0:
        print(f"\nWeekly dropout breakdown (among {len(joined_students)} joined students):")
        print(f"- Week 1 dropouts: {joined_students['Week1_Dropout'].sum()}")
        print(f"- Week 2 dropouts: {joined_students['Week2_Dropout'].sum()}")
        print(f"- Week 3 dropouts: {joined_students['Week3_Dropout'].sum()}")
        print(f"- Week 4 dropouts: {joined_students['Week4_Dropout'].sum()}")
    
    print(f"\nFirst 5 rows:")
    print(df.head())
    
    return df

if __name__ == "__main__":
    save_sample_data()
